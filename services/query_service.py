# 創建 services/query_service.py 實現以下功能：

from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

INDEX_NAME = "email-rag-search-v2"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# Load pre-trained BM25 model (absolute path — safe regardless of CWD)
_BM25_MODEL_PATH = Path(__file__).parent.parent / "data" / "bm25_model.json"
if not _BM25_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"BM25 model not found at {_BM25_MODEL_PATH}. "
        "Run: python scripts/train_bm25.py"
    )
bm25 = BM25Encoder().load(str(_BM25_MODEL_PATH))

_openai_client = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def embed_query(query: str) -> list:
    """
    Embed a query string using OpenAI text-embedding-3-small.

    Args:
        query (str): The query text to be embedded.
    Returns:
        list: The embedded query as a list of 1536 floats.
    """
    client = _get_client()
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search(query_text: str, dense_vector: list, top_k: int = 5, filter: dict = None, alpha: float = 0.5) -> dict:
    """
    Hybrid search: combines dense (semantic) and sparse (BM25 keyword) vectors.

    alpha controls the dense/sparse balance:
        alpha=1.0  →  pure dense  (semantic only, original behaviour)
        alpha=0.5  →  equal weight (recommended starting point)
        alpha=0.0  →  pure sparse (BM25 keyword only)

    Pinecone does not accept alpha as a query parameter.
    Weighting is applied by scaling the vectors before querying:
        scaled_dense[i]   = dense[i]  * alpha
        scaled_sparse.values[i] = sparse[i] * (1 - alpha)
    """
    sparse_vec = bm25.encode_queries([query_text])[0]

    # Scale vectors to implement alpha weighting
    scaled_dense = [v * alpha for v in dense_vector]
    scaled_sparse = {
        "indices": sparse_vec["indices"],
        "values":  [v * (1 - alpha) for v in sparse_vec["values"]],
    }

    results = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
        filter=filter if filter else None
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

