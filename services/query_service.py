# 創建 services/query_service.py 實現以下功能：

from dotenv import load_dotenv
from pinecone import Pinecone
import os
import sys
import contextlib

# Load environment variables
load_dotenv()

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
INDEX_NAME = "email-rag-search"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

_model = None  # Lazy-loaded model cache

def embed_query(query: str) -> list:
    """
    Embed a query string into a vector representation.
    Args:
        query (str): The query text to be embedded.
    Returns:
        list: The embedded query as a list of floats.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        with open(os.devnull, 'w') as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            _model = SentenceTransformer(MODEL_NAME)
    model = _model

    queries = model.encode(
            query,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        ).tolist()
    return queries

def search(queries: list, top_k: int = 5) -> dict:
    """
    Search the Pinecone index with the given query vectors.
    Args:
        queries (list): The list of query vectors.
        top_k (int): The number of top results to return.
    Returns:
        dict: The search results from Pinecone.
    """
    results = index.query(
        vector=queries,
        top_k=top_k,
        include_metadata=True
    )
    return results


def filter_by_metadata(results, date=None, account=None, froms=None, tos=None):
    """
    Filter search results based on metadata fields.
    Args:
        results (dict): The search results from Pinecone.
        date (str, optional): Filter by date.
        account (str, optional): Filter by account.
        froms (str, optional): Filter by sender.
        tos (str, optional): Filter by recipient.
    Returns:
        list: The filtered list of matches."""
    filtered = []
    for match in results['matches']:
        metadata = match['metadata']
        if date and metadata.get('date') != date:
            continue
        if account and metadata.get('account') != account:
            continue
        if froms and metadata.get('from') != froms:
            continue
        if tos and metadata.get('to') != tos:
            continue
        filtered.append(match)
    return filtered

def format_results(results):
    """
    Format and display the search results.
    Args:
        results (dict): The search results from Pinecone.
    """
    print(f"Found {len(results['matches'])} results\n")

    # Display results
    for i, match in enumerate(results['matches'], 1):
        score = match['score']
        metadata = match['metadata']

        print(f"   [{i}] Similarity: {score:.4f} ({score*100:.1f}%)")
        print(f"       Subject: {metadata.get('subject', 'N/A')[:60]}...")
        print(f"       From: {metadata.get('from', 'N/A')}")
        print(f"       Date: {metadata.get('date', 'N/A')}")
        print(f"       Account: {metadata.get('account', 'N/A')}")

        # Show content preview
        content = metadata.get('content', '')
        if content:
            preview = content[:150].replace('\n', ' ')
            print(f"       Content: {preview}...")
        print()

