"""
CLI tool for asking questions about your emails using RAG.

Usage:
    # Basic question
    python scripts/ask_emails.py "誰給我發過關於面試的郵件？"

    # Question with filters
    python scripts/ask_emails.py "有什麼獎學金機會？" --account "個人信箱" --top-k 10

    # English question
    python scripts/ask_emails.py "What job opportunities have I received?" --language en

    # Interactive mode
    python scripts/ask_emails.py --interactive

    # Streaming mode (show answer as it's generated)
    python scripts/ask_emails.py "總結最近的工作機會" --stream
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rag_service import ask, format_rag_response


def ask_question(
    query: str,
    top_k: int = 5,
    account: str = None,
    from_email: str = None,
    to_email: str = None,
    date: str = None,
    max_context: int = 3,
    language: str = "auto",
    stream: bool = False,
    show_sources: bool = True
):
    """
    Ask a question and display the answer.

    Args:
        query: Question to ask
        top_k: Number of emails to retrieve
        account: Filter by account
        from_email: Filter by sender
        to_email: Filter by recipient
        date: Filter by date
        max_context: Max emails to use as context
        language: Response language preference
        stream: Enable streaming
        show_sources: Show source emails
    """
    try:
        # Get answer
        response = ask(
            query=query,
            top_k=top_k,
            account=account,
            from_email=from_email,
            to_email=to_email,
            date=date,
            max_context_emails=max_context,
            language=language,
            stream=stream,
            verbose=True
        )

        # Display results
        if not stream:
            print()  # Add spacing before answer
        format_rag_response(response, show_sources=show_sources)

    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)


def interactive_mode():
    """Interactive question-answering mode."""
    print("\n" + "=" * 60)
    print("RAG Email Assistant - Interactive Mode")
    print("=" * 60)

    while True:
        try:
            query = input("Ask a question: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!\n")
                break

            if query.lower() == 'help':
                print("\nHelp:")
                print("   Ask any question about your emails.")
                print("   The system will:")
                print("   1. Search for relevant emails")
                print("   2. Use Claude AI to generate an answer")
                print("   3. Show sources used for the answer")
                print()
                continue

            # Ask question
            print()  # Add spacing
            ask_question(
                query=query,
                top_k=5,
                max_context=3,
                language="auto",
                stream=False,
                show_sources=True
            )
            print()  # Add spacing for next question

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your emails using RAG (Retrieval-Augmented Generation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  # 基本問題
  python scripts/ask_emails.py "誰給我發過關於面試的郵件？"

  # 檢索更多郵件
  python scripts/ask_emails.py "有什麼獎學金機會？" --top-k 10

  # 使用更多上下文
  python scripts/ask_emails.py "總結最近的工作機會" --max-context 5

  # 按帳號過濾
  python scripts/ask_emails.py "重要通知" --account "工作信箱"

  # 按發件人過濾
  python scripts/ask_emails.py "開會時間" --from "boss@company.com"

  # 指定語言
  python scripts/ask_emails.py "What opportunities are there?" --language en

  # 串流模式（即時顯示答案）
  python scripts/ask_emails.py "總結郵件" --stream

  # 互動模式
  python scripts/ask_emails.py --interactive
        """
    )

    parser.add_argument(
        "question",
        nargs="?",
        type=str,
        help="問題（自然語言）"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="檢索的郵件數量（默認：5）"
    )

    parser.add_argument(
        "--max-context",
        type=int,
        default=3,
        help="用作上下文的郵件數量（默認：3，應該 <= top-k）"
    )

    parser.add_argument(
        "--account",
        type=str,
        help="按帳號過濾"
    )

    parser.add_argument(
        "--from",
        dest="from_email",
        type=str,
        help="按發件人過濾"
    )

    parser.add_argument(
        "--to",
        dest="to_email",
        type=str,
        help="按收件人過濾"
    )

    parser.add_argument(
        "--date",
        type=str,
        help="按日期過濾（格式：YYYY-MM-DD）"
    )

    parser.add_argument(
        "--language",
        type=str,
        choices=["auto", "zh", "en"],
        default="auto",
        help="回答語言（auto: 自動, zh: 中文, en: English）"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="啟用串流模式（即時顯示答案）"
    )

    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="不顯示來源郵件"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="進入互動模式"
    )

    args = parser.parse_args()

    # Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found in environment variables")
        print("Please add it to your .env file:")
        print("   OPENAI_API_KEY=your_api_key_here\n")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return

    # Validate question
    if not args.question:
        parser.print_help()
        print("\nError: 請提供問題或使用 --interactive 模式\n")
        sys.exit(1)

    # Validate max_context <= top_k
    if args.max_context > args.top_k:
        print(f"\nWarning: --max-context ({args.max_context}) > --top-k ({args.top_k})")
        print(f"    Setting --max-context to {args.top_k}\n")
        args.max_context = args.top_k

    # Ask question
    ask_question(
        query=args.question,
        top_k=args.top_k,
        account=args.account,
        from_email=args.from_email,
        to_email=args.to_email,
        date=args.date,
        max_context=args.max_context,
        language=args.language,
        stream=args.stream,
        show_sources=not args.no_sources
    )


if __name__ == "__main__":
    main()
