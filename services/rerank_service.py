# Wrap the CrossEncoder and expose the rerank() function.

from sentence_transformers import CrossEncoder

_reranker = None  # module-level cache, prevent reloading

def rerank(query: str, matches: list, top_n: int) -> list:
    """
    Re-rank Pinecone matches using a CrossEncoder.

    Args:
        query:   The original query string
        matches: The list of matches returned by Pinecone (each containing metadata.content)
        top_n:   The number of top results to retain after re-ranking

    Returns:
        The re-ordered matches, length = min(top_n, len(matches))
    """
    # 1. Lazy-load model
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    # 2. (query, content) pairs
    pairs = [(query, match["metadata"].get("content", "")) for match in matches]

    # 3. Batch predict scores
    scores = _reranker.predict(pairs)  # numpy array，shape=(len(matches),)

    # 4. Sort by score (descending) and take top_n
    ranked = sorted(zip(scores, matches), key=lambda x: x[0], reverse=True)
    return [match for _, match in ranked[:top_n]]