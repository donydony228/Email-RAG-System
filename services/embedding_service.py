# Embedding service to generate embeddings for chunked email documents

from pathlib import Path
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict

# Reuse the model from email_chunker
from services.email_chunker import get_model, MODEL_NAME

# Load pre-trained BM25 model (absolute path — safe regardless of CWD)
_BM25_MODEL_PATH = Path(__file__).parent.parent / "data" / "bm25_model.json"
if not _BM25_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"BM25 model not found at {_BM25_MODEL_PATH}. "
        "Run: python scripts/train_bm25.py"
    )
bm25 = BM25Encoder().load(str(_BM25_MODEL_PATH))

def generate_embeddings(
    chunked_emails: List[Dict],
    batch_size: int = 32,
    show_progress: bool = True
) -> List[Dict]:
    """
    Generate embeddings for chunked email documents.
    
    Args:
        chunked_emails: List of chunked email documents
        batch_size: Number of documents to process at once
        show_progress: Show progress bar
    
    Returns:
        List of documents with embeddings added
    """
    # Load the model (uses lazy loading from email_chunker)
    model = get_model()
    
    # Extract all content texts
    contents = [email['content'] for email in chunked_emails]
    
    # Generate embeddings in batches
    embeddings = model.encode(
        contents,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # For cosine similarity
    )
    
    # Generate sparse embeddings (BM25) for hybrid search
    sparse_embeddings = bm25.encode_documents(contents)

    # Add embeddings to documents
    docs_with_embeddings = []
    for email, embedding, sparse_embedding in zip(chunked_emails, embeddings, sparse_embeddings):
        docs_with_embeddings.append({
            **email,
            'embedding': embedding.tolist(),      # dense vector
            'sparse_embedding': sparse_embedding, # {"indices": [...], "values": [...]}
        })

    return docs_with_embeddings
