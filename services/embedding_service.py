# Embedding service to generate embeddings for chunked email documents

from pathlib import Path
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict
import tiktoken

# text-embedding-3-small hard limit; leave ~2000-token headroom to stay well
# under the per-request total-token limit when batching multiple items.
_MAX_TOKENS = 6000
_enc = tiktoken.get_encoding("cl100k_base")


def _truncate(text: str) -> str:
    """Truncate text to _MAX_TOKENS if it exceeds the OpenAI embedding limit."""
    tokens = _enc.encode(text)
    if len(tokens) <= _MAX_TOKENS:
        return text
    return _enc.decode(tokens[:_MAX_TOKENS])

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


def generate_embeddings(
    chunked_emails: List[Dict],
    batch_size: int = 100,
    show_progress: bool = True
) -> List[Dict]:
    """
    Generate embeddings for chunked email documents using OpenAI text-embedding-3-small.

    Args:
        chunked_emails: List of chunked email documents
        batch_size: Number of documents to embed per API call (max 2048)
        show_progress: Print progress during embedding

    Returns:
        List of documents with embeddings added
    """
    client = _get_client()
    contents = [_truncate(email['content']) for email in chunked_emails]

    # Generate dense embeddings via OpenAI API in batches.
    # If a batch fails (e.g. one item slips past _truncate), fall back to
    # per-item calls with progressively smaller token caps.
    embeddings = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            embeddings.extend([item.embedding for item in response.data])
        except Exception as batch_err:
            # Batch failed — retry each item individually
            print(f"  [WARN] Batch {i//batch_size + 1} failed ({batch_err}). Retrying item-by-item...")
            for j, item_text in enumerate(batch):
                try:
                    r = client.embeddings.create(
                        input=item_text,
                        model="text-embedding-3-small"
                    )
                    embeddings.append(r.data[0].embedding)
                except Exception:
                    # Still failing — hard-truncate to 1000 tokens and retry
                    safe_text = _enc.decode(_enc.encode(item_text)[:1000])
                    r = client.embeddings.create(
                        input=safe_text,
                        model="text-embedding-3-small"
                    )
                    embeddings.append(r.data[0].embedding)
                    print(f"    [WARN] Item {i + j} hard-truncated to 1000 tokens for embedding")
        if show_progress:
            done = min(i + batch_size, len(contents))
            print(f"  Embedded {done}/{len(contents)} chunks")

    # Generate sparse embeddings (BM25) for hybrid search
    sparse_embeddings = bm25.encode_documents(contents)

    # Add embeddings to documents
    docs_with_embeddings = []
    for email, embedding, sparse_embedding in zip(chunked_emails, embeddings, sparse_embeddings):
        docs_with_embeddings.append({
            **email,
            'embedding': embedding,           # dense vector (list of 1536 floats)
            'sparse_embedding': sparse_embedding,  # {"indices": [...], "values": [...]}
        })

    return docs_with_embeddings
