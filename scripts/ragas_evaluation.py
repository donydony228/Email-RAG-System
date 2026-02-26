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
# Dynamic ground truth for time-based questions.
# Called once at the start of each evaluation run so the ground truth always
# reflects the most recently ingested emails in Pinecone.
# ---------------------------------------------------------------------------
def _build_temporal_ground_truth(topic: str, top_n: int = 1) -> str:
    """
    Query Pinecone semantically, sort results by email date (newest first),
    and return a human-readable ground truth string from the top N matches.

    Args:
        topic:  Free-text query used to narrow the search (e.g. "newsletter").
        top_n:  How many emails to include in the returned string.

    Returns:
        e.g. "Subject: 'February Global Newsletter', From: Team Pinecone ..., Date: ..."
    """
    from services.query_service import embed_query, search
    from services.temporal_service import sort_by_date

    dense_vector = embed_query(topic)
    results = search(topic, dense_vector, top_k=30, alpha=0.5)
    matches = sort_by_date(results["matches"])  # newest-first

    if not matches:
        return "No relevant emails found."

    lines = []
    for i, m in enumerate(matches[:top_n], 1):
        meta = m["metadata"]
        subj   = meta.get("subject", "N/A")
        sender = meta.get("from",    "N/A")
        date   = meta.get("date",    "N/A")
        lines.append(f"({i}) Subject: '{subj}', From: {sender}, Date: {date}")

    if top_n == 1:
        return f"The most recent match: {lines[0]}"
    return "The most recent matches: " + "; ".join(lines)


# ---------------------------------------------------------------------------
# Evaluation questions organised by scenario type.
# ---------------------------------------------------------------------------
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
        "question": "What is the Delta Airlines flight confirmation number for the booking on March 3, 2026?",
        "ground_truth": "The Delta Airlines confirmation number is F8EVIK, shared by both CHINGYUAN PENG and TZUTING LIN."
    },
    {
        "question": "What is the JetBlue booking confirmation code for TZU TING LIN's flight?",
        "ground_truth": "The JetBlue booking confirmation code (reference) is LRXFBM."
    },
    {
        "question": "What is the SAS booking reference for the flight booked under PENG / CHING YUAN?",
        "ground_truth": "The SAS booking reference is YQZGCN for passenger PENG / CHING YUAN."
    },
    {
        "question": "What rating did the Airbnb guest Danilyn give in their review?",
        "ground_truth": "Danilyn gave a 5-star rating in their Airbnb review."
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
    {
        "question": "Who forwarded me the JetBlue booking confirmation, and whose flight booking was it for?",
        "ground_truth": "Carol (carol09260926@gmail.com) forwarded the JetBlue booking confirmation for TZU TING LIN."
    },
    {
        "question": "Which institution sent me a Spring 2026 bill notification?",
        "ground_truth": "NYU (New York University) Office of the Bursar sent a Spring 2026 bill notification."
    },

    # --- Multi-document Aggregation ---
    # Tests whether the system can synthesise information across multiple emails.
    # NOTE: Standard single-stage RAG (top-k=3 context) structurally cannot recall all items;
    # these questions are retained to demonstrate multi-doc aggregation as a known RAG limitation.
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
        # Replaced: "urgent action" question always scored CP=0 (no such emails exist + RAGAS can't score empty context)
        "question": "Did I receive any emails about flight seat selection or upgrade options?",
        "ground_truth": "Yes, Wizz Air sent an email about assigned seating options, including Front row and Extra Legroom seats for added comfort."
    },
    {
        "question": "Did anyone send me feedback or comments about my work?",
        "ground_truth": "No feedback or comments were found."
    },
    {
        "question": "Were there any emails about grocery or food delivery services?",
        "ground_truth": "Yes, Weee! sent emails about Asian grocery delivery, including an offer for $10 off the first order."
    },

    # --- Time-related ---
    # Tests whether retrieval works correctly with temporal context.
    # ground_truth is set to None here and populated dynamically at runtime
    # (see _build_temporal_ground_truth) so it stays accurate as new emails arrive.
    {
        "question": "What is the most recent newsletter I received?",
        "ground_truth": None,
        "_ground_truth_query": "newsletter subscription updates",
        "_ground_truth_top_n": 1,
    },
    {
        "question": "What were the last travel-related emails I received?",
        "ground_truth": None,
        "_ground_truth_query": "flight booking travel confirmation itinerary",
        "_ground_truth_top_n": 3,
    },

    # --- Non-existent Information (Faithfulness) ---
    # The correct answer is "no such information found".
    # Tests whether the LLM hallucinates when context is missing.
    {
        "question": "What is my passport number mentioned in any email?",
        "ground_truth": "No passport number has been mentioned in any email."
    },
    {
        # Replaced: "real estate" question always scored CP=0 due to RAGAS scoring artifact on empty context
        "question": "Has anyone sent me an actual job offer letter (not a recruiter outreach)?",
        "ground_truth": "No actual job offer letters were found in the emails, only recruiter outreach messages."
    },
]


async def evaluate_questions(notes: str = "", alpha: float = 0.5,
                              rerank: bool = True, top_k: int = 20,
                              temporal_ranking=None) -> list:
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    faithfulness = Faithfulness(llm=llm)

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results = []

    # Populate dynamic ground truths (time-based questions) before filtering.
    # Each question with ground_truth=None and a "_ground_truth_query" key
    # gets its ground truth generated by querying Pinecone at runtime.
    questions = []
    for q in EVAL_QUESTIONS:
        if q["ground_truth"] is None and "_ground_truth_query" in q:
            print(f"[INFO] Generating temporal ground truth for: \"{q['question']}\"")
            gt = _build_temporal_ground_truth(
                q["_ground_truth_query"],
                top_n=q.get("_ground_truth_top_n", 1),
            )
            print(f"       → {gt}\n")
            questions.append({**q, "ground_truth": gt})
        else:
            questions.append(q)

    pending = [q for q in questions if not q["ground_truth"]]
    active  = [q for q in questions if q["ground_truth"]]

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
            temporal_ranking=temporal_ranking,
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
  python scripts/ragas_evaluation.py --notes "Temporal ranking (auto-detect)" --no-rerank --top-k 5
  python scripts/ragas_evaluation.py --notes "Temporal ranking (forced)" --no-rerank --top-k 5 --temporal-ranking
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
    parser.add_argument(
        "--temporal-ranking",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="temporal_ranking",
        help=(
            "Temporal ranking: sort results by date for recency queries. "
            "Default (None) = auto-detect from query keywords. "
            "Use --temporal-ranking to force on, --no-temporal-ranking to force off."
        )
    )
    args = parser.parse_args()

    results = asyncio.run(evaluate_questions(
        notes=args.notes,
        alpha=args.alpha,
        rerank=args.rerank,
        top_k=args.top_k,
        temporal_ranking=args.temporal_ranking,
    ))
    print_summary(results)
    save_to_csv(results, CSV_PATH)
