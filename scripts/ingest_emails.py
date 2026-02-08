"""
Command-line tool for ingesting emails into Pinecone.

Usage:
    # Incremental update (default, skip existing emails)
    python scripts/ingest_emails.py --time-range 1d --max-emails 100

    # Full re-ingestion (process all emails)
    python scripts/ingest_emails.py --time-range 30d --max-emails 1000 --no-incremental
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingestion_pipeline import EmailIngestionPipeline
import json


def main():
    parser = argparse.ArgumentParser(
        description="Ingest emails from Gmail into Pinecone vector database"
    )
    
    parser.add_argument(
        "--time-range",
        type=str,
        default="1d",
        help="Time range for fetching emails (e.g., 7d, 30d, 90d)"
    )
    
    parser.add_argument(
        "--max-emails",
        type=int,
        default=100,
        help="Maximum number of emails to fetch per account"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )

    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental mode (process all emails, even if they exist)"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show current index statistics"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EmailIngestionPipeline()
    
    # Show stats only
    if args.stats_only:
        stats = pipeline.get_index_stats()
        print("\n" + "="*50)
        print("PINECONE INDEX STATISTICS")
        print("="*50)
        # Convert to dict if needed
        if hasattr(stats, 'to_dict'):
            stats = stats.to_dict()
        elif not isinstance(stats, dict):
            stats = dict(stats)
        print(json.dumps(stats, indent=2, default=str))
        return
    
    # Run ingestion
    incremental_mode = not args.no_incremental

    print("\n" + "="*50)
    print("EMAIL INGESTION PIPELINE")
    print("="*50)
    print(f"Time range:      {args.time_range}")
    print(f"Max emails:      {args.max_emails}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Incremental:     {'Yes (skip existing)' if incremental_mode else 'No (process all)'}")
    print("="*50 + "\n")

    try:
        stats = pipeline.run(
            time_range=args.time_range,
            max_emails=args.max_emails,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
            incremental=incremental_mode
        )
        
        # Print results
        print("\n" + "="*50)
        print("PIPELINE COMPLETED")
        print("="*50)
        print(f"Emails fetched:        {stats['emails_fetched']}")

        if incremental_mode:
            print(f"Emails skipped:        {stats['emails_skipped']} (already exist)")
            print(f"Emails new:            {stats['emails_new']}")

        print(f"Chunks created:        {stats['chunks_created']}")
        print(f"Embeddings generated:  {stats['embeddings_generated']}")
        print(f"Vectors uploaded:      {stats['vectors_uploaded']}")
        print(f"Success rate:          {stats['success_rate']:.2f}%")
        print(f"Duration:              {stats['duration']:.2f}s")
        
        if stats['failed_ids']:
            print(f"\nWarning: {len(stats['failed_ids'])} vectors failed to upload")
        
        print("="*50)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
