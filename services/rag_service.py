"""
RAG (Retrieval-Augmented Generation) service.

This service orchestrates the complete RAG pipeline:
1. Embed user query
2. Retrieve relevant emails from Pinecone
3. Generate answer using Claude API
"""

from typing import Dict, Optional
from services.query_service import embed_query, search, filter_by_metadata
from services.claude_service import ask_with_emails


def ask(
    query: str,
    top_k: int = 5,
    account: Optional[str] = None,
    from_email: Optional[str] = None,
    to_email: Optional[str] = None,
    date: Optional[str] = None,
    max_context_emails: int = 3,
    language: str = "auto",
    stream: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Ask a question about your emails using RAG.

    Args:
        query: Natural language question
        top_k: Number of emails to retrieve
        account: Filter by account name
        from_email: Filter by sender
        to_email: Filter by recipient
        date: Filter by date
        max_context_emails: Maximum emails to use as context for LLM (should be <= top_k)
        language: Response language ("zh", "en", or "auto")
        stream: Enable streaming response
        verbose: Print progress information

    Returns:
        Dictionary containing:
        - answer: Generated answer
        - sources: List of source emails used
        - context_used: Number of emails used in context
        - retrieval_results: Raw retrieval results
    """
    if verbose:
        print("=" * 60)
        print(f"Question: \"{query}\"")
        print("=" * 60 + "\n")

    # Step 1: Embed query
    if verbose:
        print("[1/3] Embedding query...")
    query_vector = embed_query(query)
    if verbose:
        print(f"      ✓ Query embedded as {len(query_vector)}-dimensional vector\n")

    # Step 2: Retrieve from Pinecone
    if verbose:
        print(f"[2/3] Searching in Pinecone (top_k={top_k})...")
    results = search(query_vector, top_k=top_k)
    if verbose:
        print(f"      ✓ Found {len(results['matches'])} results from Pinecone")

    # Apply filters if specified
    if any([account, from_email, to_email, date]):
        if verbose:
            print("      ✓ Applying metadata filters...")
        filtered_matches = filter_by_metadata(
            results,
            account=account,
            froms=from_email,
            tos=to_email,
            date=date
        )
        results['matches'] = filtered_matches
        if verbose:
            print(f"      ✓ {len(results['matches'])} results after filtering\n")
    else:
        if verbose:
            print()

    # Step 3: Generate answer with OpenAI
    if verbose:
        print(f"[3/3] Generating answer with OpenAI API (using top {max_context_emails} emails)...")

    response = ask_with_emails(
        query=query,
        search_results=results,
        max_context_emails=max_context_emails,
        language=language,
        stream=stream
    )

    if verbose and not stream:
        print("      ✓ Answer generated\n")

    # Combine results
    return {
        **response,
        'retrieval_results': results
    }


def format_rag_response(response: Dict, show_sources: bool = True) -> None:
    """
    Pretty print a RAG response.

    Args:
        response: Response from ask() function
        show_sources: Whether to show source emails
    """
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print()
    print(response['answer'])
    print()

    if show_sources and response.get('sources'):
        print("=" * 60)
        print(f"SOURCES ({response['context_used']} emails used)")
        print("=" * 60)
        print()

        for i, source in enumerate(response['sources'], 1):
            print(f"[{i}] {source['subject']}")
            print(f"    From: {source['from']}")
            print(f"    Date: {source['date']}")
            print(f"    Similarity: {source['similarity']:.2%}")
            print()

    print("=" * 60)
