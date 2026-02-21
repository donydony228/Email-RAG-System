"""
Train and save a BM25Encoder on the existing email corpus in Pinecone.

This script fetches all email content stored in Pinecone metadata, fits a
BM25Encoder (computing IDF weights for the corpus), and saves the model to
data/bm25_model.json.

The saved model is used by:
  - Ingestion pipeline: to generate sparse vectors during upsert
  - Query pipeline:     to encode queries for hybrid search

Run from project root:
    python scripts/train_bm25.py

Output:
    data/bm25_model.json
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from tqdm import tqdm


INDEX_NAME = "email-rag-search"
MODEL_PATH = Path(__file__).parent.parent / "data" / "bm25_model.json"
FETCH_BATCH_SIZE = 100  # Pinecone fetch uses GET params; keep small to avoid 414


def fetch_all_contents(index) -> list:
    """
    Fetch the content field from every vector's metadata in the Pinecone index.

    Strategy:
      1. Use index.list() to page through all vector IDs (no vector data returned).
      2. Fetch vectors in batches of FETCH_BATCH_SIZE to retrieve metadata.
      3. Extract the 'content' field from each vector's metadata.

    Returns:
        List of content strings (one per chunk stored in Pinecone).
    """
    print("Listing all vector IDs from Pinecone...")
    all_ids = []
    for id_page in index.list():
        all_ids.extend(id_page)

    print(f"Found {len(all_ids)} vectors. Fetching metadata...")

    contents = []
    for i in tqdm(range(0, len(all_ids), FETCH_BATCH_SIZE), desc="Fetching"):
        batch_ids = all_ids[i:i + FETCH_BATCH_SIZE]
        result = index.fetch(ids=batch_ids)
        for vec in result["vectors"].values():
            content = vec.get("metadata", {}).get("content", "")
            if content.strip():
                contents.append(content)

    return contents


def main():
    print("=== BM25 Encoder Training ===\n")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    total_vectors = stats["total_vector_count"]
    print(f"Index  : {INDEX_NAME}")
    print(f"Vectors: {total_vectors}\n")

    if total_vectors == 0:
        print("Error: No vectors found. Run the ingestion pipeline first.")
        sys.exit(1)

    contents = fetch_all_contents(index)
    print(f"\nCollected {len(contents)} content strings for training.\n")

    if not contents:
        print("Error: No content found in vector metadata.")
        sys.exit(1)

    print("Fitting BM25Encoder on corpus...")
    encoder = BM25Encoder()
    encoder.fit(contents)
    print("Fitting complete.\n")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(MODEL_PATH))
    print(f"Model saved: {MODEL_PATH}")
    print("\nNext steps:")
    print("  1. Re-ingest emails so sparse vectors are added to Pinecone.")
    print("  2. Update query_service.py to load the model and pass sparse_vector to index.query().")


if __name__ == "__main__":
    main()
