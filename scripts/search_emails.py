"""
CLI tool for searching emails using semantic search.

Usage:
    # Basic search
    python scripts/search_emails.py "找到關於面試的郵件"

    # Search with filters
    python scripts/search_emails.py "scholarship" --account "個人信箱" --top-k 10

    # Search with sender filter
    python scripts/search_emails.py "work opportunities" --from "linkedin.com"

    # Interactive mode
    python scripts/search_emails.py --interactive
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.query_service import embed_query, search, filter_by_metadata, format_results


def search_emails(
    query: str,
    top_k: int = 5,
    account: str = None,
    from_email: str = None,
    to_email: str = None,
    date: str = None
):
    """
    Search for emails using semantic search with optional filters.

    Args:
        query: Natural language search query
        top_k: Number of results to return
        account: Filter by account name
        from_email: Filter by sender email
        to_email: Filter by recipient email
        date: Filter by date
    """
    print("\n" + "="*60)
    print(f"Searching: \"{query}\"")
    print("="*60 + "\n")

    # Step 1: Embed the query
    print("[1/3] Embedding query...")
    query_vector = embed_query(query)
    print(f"      ✓ Query embedded as {len(query_vector)}-dimensional vector")

    # Step 2: Search in Pinecone
    print(f"[2/3] Searching in Pinecone (top_k={top_k})...")
    results = search(query_vector, top_k=top_k)
    print(f"      ✓ Found {len(results['matches'])} results from Pinecone")

    # Step 3: Apply metadata filters if specified
    if any([account, from_email, to_email, date]):
        print("[3/3] Applying filters...")
        filtered_matches = filter_by_metadata(
            results,
            account=account,
            froms=from_email,
            tos=to_email,
            date=date
        )
        results['matches'] = filtered_matches
        print(f"      ✓ Found {len(results['matches'])} results after filtering")
    else:
        print("[3/3] No filters applied")

    print("\n" + "="*60)
    print("Search Results")
    print("="*60 + "\n")

    # Format and display results
    if results['matches']:
        format_results(results)
    else:
        print("   No matching emails found.")

    print("="*60 + "\n")


def interactive_mode():
    """Interactive search mode."""
    print("\n" + "="*60)
    print("Semantic Email Search - Interactive Mode")
    print("="*60)
    print("\nTips:")
    print("   - Enter natural language queries (in Chinese or English)")
    print("   - Type 'quit' or 'exit' to leave")
    print("   - Type 'help' for assistance")
    print("\n" + "="*60 + "\n")

    while True:
        try:
            query = input("Please enter your search query: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if query.lower() == 'help':
                print("\nHelp Information:")
                print("  - Enter any natural language query to search emails")
                print("  - For example: '找到關於面試的郵件'")
                print("  - For example: 'scholarship opportunities'")
                print("  - For example: '工作機會相關的信件'")
                print()
                continue

            # Perform search
            search_emails(query, top_k=5)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Search emails using semantic search with optional filters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本搜索
  python scripts/search_emails.py "找到關於面試的郵件"

  # 搜索並返回更多結果
  python scripts/search_emails.py "scholarship" --top-k 10

  # 按帳號過濾
  python scripts/search_emails.py "工作機會" --account "個人信箱"

  # 按發件人過濾
  python scripts/search_emails.py "meeting" --from "boss@company.com"

  # 互動模式
  python scripts/search_emails.py --interactive
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="自然語言搜索查詢（中文或英文）"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回結果數量（默認：5）"
    )

    parser.add_argument(
        "--account",
        type=str,
        help="按帳號過濾（例如：個人信箱、工作信箱、其他信箱）"
    )

    parser.add_argument(
        "--from",
        dest="from_email",
        type=str,
        help="按發件人過濾（部分匹配）"
    )

    parser.add_argument(
        "--to",
        dest="to_email",
        type=str,
        help="按收件人過濾（部分匹配）"
    )

    parser.add_argument(
        "--date",
        type=str,
        help="按日期過濾（格式：YYYY-MM-DD）"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="進入互動模式"
    )

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return

    # Validate query
    if not args.query:
        parser.print_help()
        print("\nError: 請提供搜索查詢或使用 --interactive 模式")
        sys.exit(1)

    # Perform search
    try:
        search_emails(
            query=args.query,
            top_k=args.top_k,
            account=args.account,
            from_email=args.from_email,
            to_email=args.to_email,
            date=args.date
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
