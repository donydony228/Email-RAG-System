"""
RAG evaluation using RAGAS framework.

Run from project root:
    python scripts/ragas_evaluation.py
    python scripts/ragas_evaluation.py --notes "Added server-side Pinecone filter"
    python scripts/ragas_evaluation.py --notes "Hybrid search alpha=0.5"

Results are appended to data/ragas_results.csv on every run.
"""

import sys
import os
import asyncio
import argparse
import csv
import logging
import warnings
from datetime import datetime
from pathlib import Path

# Suppress noisy ML library output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, AnswerRelevancy, Faithfulness
from services.rag_service import ask


CSV_PATH = Path(__file__).parent.parent / "data" / "ragas_results.csv"
CSV_COLUMNS = [
    "Timestamp", "Notes", "Question", "Ground_Truth", "Answer",
    "Context_Precision", "Context_Recall", "Answer_Relevancy", "Faithfulness", "Average_Score",
]

# ---------------------------------------------------------------------------
# Evaluation questions organised by scenario type.
# Fill in each "ground_truth" before running.
# ---------------------------------------------------------------------------
# Insert current time into EVAL_QUESTIONS
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
EVAL_QUESTIONS = [

    # --- Factual / Precise Information ---
    # Tests whether the system can extract specific facts from a single email.
    {
        "question": "What is the general check-in and check-out dates and times for the Airbnb stay at 'Diva sonata no.7'?",
        "ground_truth": "Check-in: 16:00, Check-out: 11:00"
    },
    {
        "question": "What is the monthly amount paid to Anthropic?",
        "ground_truth": "Monthly amount: $20"
    },
    {
        "question": "What verification code was given for my most recent login in China Airline?",
        "ground_truth": "568393, 811012, 348447"
    },

    # --- Named Entity (Person / Company) ---
    # Tests keyword-level matching for proper nouns — primary target for Hybrid Search.
    {
        "question": "Who is Vera Kang? I remembered she is a HR from a tech company",
        "ground_truth": "Vera Kang is a HR from Entegris, who contacted me regarding my job application."
    },
    {
        "question": "Which company's recruiter last contacted me about a Gen AI Internship role?",
        "ground_truth": "The recruiter from Cotiviti last contacted me about a Gen AI Internship role."
    },

    # --- Multi-document Aggregation ---
    # Tests whether the system can synthesise information across multiple emails.
    {
        "question": "List all the airlines that used to send me emails?",
        "ground_truth": "Airlines that used to send emails include: China Airlines, Jetblue, SAS, Wizz Air"
    },
    {
        "question": "Which companies have sent me promotional discount emails?",
        "ground_truth": "Delta Air, Weee!, Trip.com, Playstation, Best Buy"
    },
    {
        "question": "What subscription services am I currently paying for, based on billing emails?",
        "ground_truth": "Anthropic, Zapier, Amazon"
    },

    # --- Pure Semantic (No Exact Keywords) ---
    # Tests dense vector retrieval with no precise terms to anchor on.
    {
        "question": "Were there any emails asking me to take action or respond urgently?",
        "ground_truth": "No such emails were found."
    },
    {
        "question": "Did anyone send me feedback or comments about my work?",
        "ground_truth": "No feedback or comments were found."
    },

    # --- Time-related ---
    # Tests whether retrieval works correctly with temporal context.
    {
        "question": "What is the most recent newsletter I received?",
        "ground_truth": f"Based on current time: {current_time}"
    },
    {
        "question": "What were the last travel-related emails I received?",
        "ground_truth": f"Based on current time: {current_time}"
    },

    # --- Non-existent Information (Faithfulness) ---
    # The correct answer is "no such information found".
    # Tests whether the LLM hallucinates when context is missing.
    {
        "question": "What is my passport number mentioned in any email?",
        "ground_truth": "No passport number has been mentioned in any email."
    },
    {
        "question": "Has anyone emailed me about a real estate purchase?",
        "ground_truth": "No emails related to real estate purchases were found."
    },
]


async def evaluate_questions(notes: str = "", alpha: float = 0.5,
                              rerank: bool = True, top_k: int = 20) -> list:
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    faithfulness = Faithfulness(llm=llm)

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results = []

    pending = [q for q in EVAL_QUESTIONS if not q["ground_truth"]]
    active  = [q for q in EVAL_QUESTIONS if q["ground_truth"]]

    if pending:
        print(f"\n[INFO] Skipping {len(pending)} question(s) with empty ground_truth (marked TODO).")
    print(f"[INFO] Running {len(active)} question(s).\n")

    for i, item in enumerate(active, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n{'='*60}")
        print(f"[{i}/{len(active)}] Q: {question}")
        print(f"Expected: {ground_truth}")
        print('='*60)

        response = ask(
            query=question,
            top_k=top_k,
            max_context_emails=3,
            language="auto",
            stream=False,
            verbose=False,
            alpha=alpha,
            rerank=rerank,
            rerank_top_n=5,
        )

        contexts = []
        for match in response['retrieval_results']['matches'][:3]:
            metadata = match.get('metadata', {})
            contexts.append(
                f"Subject: {metadata.get('subject', 'N/A')}\n"
                f"From: {metadata.get('from', 'N/A')}\n"
                f"Date: {metadata.get('date', 'N/A')}\n"
                f"Content: {metadata.get('content', 'N/A')}"
            )

        answer = response['answer']

        print(f"A: {answer}\n")
        print("Scores:")
        result_precision = await context_precision.ascore(
            user_input=question,
            reference=ground_truth,
            retrieved_contexts=contexts,
        )
        result_recall = await context_recall.ascore(
            user_input=question,
            reference=ground_truth,
            retrieved_contexts=contexts,
        )
        result_relevancy = await answer_relevancy.ascore(
            user_input=question,
            response=answer,
        )
        result_faithfulness = await faithfulness.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        avg = (result_precision.value + result_recall.value +
               result_relevancy.value + result_faithfulness.value) / 4

        print(f"  Context Precision : {result_precision.value:.3f}")
        print(f"  Context Recall    : {result_recall.value:.3f}")
        print(f"  Answer Relevancy  : {result_relevancy.value:.3f}")
        print(f"  Faithfulness      : {result_faithfulness.value:.3f}")
        print(f"  ─────────────────────")
        print(f"  Average           : {avg:.3f}")

        all_results.append({
            "Timestamp": run_timestamp,
            "Notes": notes,
            "Question": question,
            "Ground_Truth": ground_truth,
            "Answer": answer,
            "Context_Precision": round(result_precision.value, 4),
            "Context_Recall": round(result_recall.value, 4),
            "Answer_Relevancy": round(result_relevancy.value, 4),
            "Faithfulness": round(result_faithfulness.value, 4),
            "Average_Score": round(avg, 4),
        })

    return all_results


def print_summary(results: list) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    metrics = ["Context_Precision", "Context_Recall", "Answer_Relevancy", "Faithfulness", "Average_Score"]
    for m in metrics:
        avg = sum(r[m] for r in results) / len(results)
        print(f"  {m:22s}: {avg:.4f}")
    print("=" * 60)


def save_to_csv(results: list, path: Path) -> None:
    """Append evaluation results to CSV. Creates the file with headers if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    total_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1  # subtract header
    print(f"\nResults appended to: {path}")
    print(f"Total rows in file : {total_rows} (across all runs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline using RAGAS metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ragas_evaluation.py
  python scripts/ragas_evaluation.py --notes "Hybrid alpha=0.5 baseline" --no-rerank --top-k 5
  python scripts/ragas_evaluation.py --notes "Hybrid alpha=0.5 + CrossEncoder rerank" --rerank --top-k 20
        """
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Description of system changes in this run (recorded in CSV)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid search alpha: 1.0=pure dense, 0.0=pure BM25 (default: 0.5)"
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CrossEncoder re-ranking (default: True). Use --no-rerank to disable."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help="Number of candidates to retrieve from Pinecone (default: 20)"
    )
    args = parser.parse_args()

    results = asyncio.run(evaluate_questions(
        notes=args.notes,
        alpha=args.alpha,
        rerank=args.rerank,
        top_k=args.top_k,
    ))
    print_summary(results)
    save_to_csv(results, CSV_PATH)
