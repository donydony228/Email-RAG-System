"""
Pinecone vector database service for email RAG system.
"""

from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
from tqdm import tqdm
import os
import time

# Configuration
INDEX_NAME = "email-rag-search-v2"
DIMENSION = 1536
METRIC = "dotproduct"  # Required for hybrid search (sparse + dense); cosine does not support sparse_values
BATCH_SIZE = 100

class PineconeService:
    """Service for managing Pinecone vector database operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pinecone service.
        
        Args:
            api_key: Pinecone API key (defaults to env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
    
    def create_index_if_not_exists(self) -> None:
        """Create Pinecone index if it doesn't exist."""
        if INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"Created index: {INDEX_NAME}")
            # Wait for index to be ready
            time.sleep(5)
        
        self.index = self.pc.Index(INDEX_NAME)
    
    def optimize_metadata(self, metadata: dict, content: str) -> dict:
        """Optimize metadata to fit Pinecone limits."""
        return {
            'account': metadata.get('account', ''),
            'subject': metadata.get('subject', '')[:200],
            'from': metadata.get('from', ''),
            'to': metadata.get('to', ''),
            'date': metadata.get('date', ''),
            'timestamp': metadata.get('timestamp', 0),
            'message_id': metadata.get('message_id', ''),
            'thread_id': metadata.get('thread_id', ''),
            'chunk_index': metadata.get('chunk_index', 0),
            'total_chunks': metadata.get('total_chunks', 1),
            'is_chunked': metadata.get('is_chunked', False),
            'content': content[:1000],
            'labels': ','.join(metadata.get('labels', [])),
        }
    
    def prepare_for_pinecone(self, docs: List[Dict]) -> List[Dict]:
        """Convert documents to Pinecone format (dict with optional sparse_values)."""
        pinecone_data = []

        for doc in docs:
            metadata = self.optimize_metadata(
                doc['metadata'],
                doc['content']
            )

            pinecone_vector = {
                "id": doc['id'],
                "values": doc['embedding'],
                "metadata": metadata,
            }

            # Include sparse vector when available (hybrid search)
            if doc.get('sparse_embedding'):
                pinecone_vector["sparse_values"] = doc['sparse_embedding']

            pinecone_data.append(pinecone_vector)

        return pinecone_data
    
    def upsert(
        self,
        docs_with_embeddings: List[Dict],
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True
    ) -> Dict:
        """
        Upsert documents to Pinecone.
        
        Args:
            docs_with_embeddings: Documents with embeddings
            batch_size: Vectors per batch
            show_progress: Show progress bar
        
        Returns:
            Upload statistics
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index_if_not_exists() first.")
        
        pinecone_data = self.prepare_for_pinecone(docs_with_embeddings)
        
        total_uploaded = 0
        failed_ids = []
        
        iterator = range(0, len(pinecone_data), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Uploading to Pinecone")
        
        for i in iterator:
            batch = pinecone_data[i:i + batch_size]
            
            try:
                response = self.index.upsert(vectors=batch)
                total_uploaded += response['upserted_count']
            except Exception as e:
                print(f"Error uploading batch: {e}")
                failed_ids.extend([vec["id"] for vec in batch])
        
        return {
            'total_uploaded': total_uploaded,
            'total_documents': len(pinecone_data),
            'failed_ids': failed_ids,
            'success_rate': total_uploaded / len(pinecone_data) * 100
        }
    
    def check_exists(self, email_ids: List[str]) -> Dict[str, bool]:
        """
        Check which email IDs already exist in Pinecone.

        Args:
            email_ids: List of email IDs to check (e.g., ["email_abc123", ...])

        Returns:
            Dictionary mapping email_id to existence status
            Example: {"email_abc123": True, "email_xyz789": False}
        """
        if self.index is None:
            raise ValueError("Index not initialized.")

        if not email_ids:
            return {}

        exists_map = {}

        # Keep batches small to avoid 414 URL-Too-Large from AWS API Gateway
        batch_size = 100

        for i in range(0, len(email_ids), batch_size):
            batch_ids = email_ids[i:i + batch_size]

            try:
                # Fetch vectors by ID
                result = self.index.fetch(ids=batch_ids)

                # Mark which IDs exist
                for email_id in batch_ids:
                    exists_map[email_id] = email_id in result['vectors']
            except Exception as e:
                print(f"Error checking existence for batch: {e}")
                # Assume they don't exist if error occurs
                for email_id in batch_ids:
                    exists_map[email_id] = False

        return exists_map

    def filter_new_emails(
        self,
        formatted_emails: List[Dict],
        show_progress: bool = True
    ) -> tuple[List[Dict], Dict]:
        """
        Filter out emails that already exist in Pinecone.

        Args:
            formatted_emails: List of formatted email documents
            show_progress: Show progress information

        Returns:
            Tuple of (new_emails, stats)
            - new_emails: List of emails not in Pinecone
            - stats: Dictionary with filtering statistics
        """
        if not formatted_emails:
            return [], {'total': 0, 'existing': 0, 'new': 0}

        # Extract email IDs
        email_ids = [email['id'] for email in formatted_emails]

        if show_progress:
            print(f"Checking {len(email_ids)} emails against Pinecone...")

        # Check which emails exist
        exists_map = self.check_exists(email_ids)

        # Filter out existing emails
        new_emails = [
            email for email in formatted_emails
            if not exists_map.get(email['id'], False)
        ]

        stats = {
            'total': len(formatted_emails),
            'existing': len(formatted_emails) - len(new_emails),
            'new': len(new_emails)
        }

        if show_progress:
            print(f"Found {stats['new']} new emails (skipped {stats['existing']} existing)")

        return new_emails, stats

    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        if self.index is None:
            raise ValueError("Index not initialized.")

        return self.index.describe_index_stats()

    def delete_all(self) -> None:
        """Delete all vectors from the index."""
        if self.index is None:
            raise ValueError("Index not initialized.")

        self.index.delete(delete_all=True)
        print("Deleted all vectors from index")
