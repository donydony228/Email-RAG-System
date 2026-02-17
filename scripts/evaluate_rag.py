"""
RAG System Evaluation Script using OpenAI as judge.

Evaluates the email RAG system using GPT-4o-mini as an evaluator (LLM-as-judge).
No local ML models required - uses only OpenAI API calls.

Metrics:
- Faithfulness: Answer is grounded in retrieved context (no hallucinations)
- Answer Relevancy: Answer addresses the question
- Context Precision: Retrieved emails are relevant to the question
- Context Recall: Retrieved context contains enough info to answer the question

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --test-file test_queries.json
    python scripts/evaluate_rag.py --output results.csv
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Starting evaluation script...", flush=True)

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from services.rag_service import ask

print("✓ Ready\n", flush=True)


DEFAULT_TEST_QUERIES = [
    {
        "question": "哪些公司寄送過關於面試的信件？",
        "ground_truth": "應該包含面試邀請的公司名稱"
    },
    {
        "question": "最近我有哪些航班預訂？",
        "ground_truth": "應該列出最近的航班預訂資訊"
    },
]


async def llm_score(client: AsyncOpenAI, prompt: str) -> float:
    """Ask GPT-4o-mini to score something, returns a float 0.0-1.0."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    text = response.choices[0].message.content.strip()
    try:
        score = float(text)
        return max(0.0, min(1.0, score / 10.0))
    except ValueError:
        # Try to extract first number from response
        import re
        match = re.search(r'\d+\.?\d*', text)
        if match:
            return max(0.0, min(1.0, float(match.group()) / 10.0))
        return 0.5


async def evaluate_faithfulness(client: AsyncOpenAI, question: str, answer: str, context: str) -> float:
    """Score: Is the answer supported by the retrieved context? (no hallucinations)"""
    prompt = f"""You are evaluating a RAG system. Score from 0-10 whether the ANSWER is fully supported by the CONTEXT.
- 10: Every claim in the answer is directly supported by the context
- 5: Some claims are supported, some are not
- 0: The answer contains many claims not found in the context (hallucinations)

QUESTION: {question}

CONTEXT:
{context[:2000]}

ANSWER: {answer[:500]}

Reply with ONLY a number from 0 to 10."""
    return await llm_score(client, prompt)


async def evaluate_answer_relevancy(client: AsyncOpenAI, question: str, answer: str) -> float:
    """Score: Does the answer actually address the question?"""
    prompt = f"""You are evaluating a RAG system. Score from 0-10 whether the ANSWER is relevant and directly addresses the QUESTION.
- 10: The answer fully and directly addresses the question
- 5: The answer partially addresses the question
- 0: The answer is off-topic or does not address the question

QUESTION: {question}

ANSWER: {answer[:500]}

Reply with ONLY a number from 0 to 10."""
    return await llm_score(client, prompt)


async def evaluate_context_precision(client: AsyncOpenAI, question: str, context: str) -> float:
    """Score: Are the retrieved documents relevant to the question?"""
    prompt = f"""You are evaluating a RAG system. Score from 0-10 whether the retrieved CONTEXT is relevant to the QUESTION.
- 10: All retrieved context is highly relevant to answering the question
- 5: Some context is relevant, some is off-topic
- 0: The retrieved context is completely irrelevant to the question

QUESTION: {question}

RETRIEVED CONTEXT:
{context[:2000]}

Reply with ONLY a number from 0 to 10."""
    return await llm_score(client, prompt)


async def evaluate_context_recall(client: AsyncOpenAI, question: str, context: str) -> float:
    """Score: Does the context contain enough information to answer the question?"""
    prompt = f"""You are evaluating a RAG system. Score from 0-10 whether the CONTEXT contains sufficient information to answer the QUESTION.
- 10: The context contains all information needed to fully answer the question
- 5: The context has partial information but is missing some key details
- 0: The context does not contain information relevant to answering the question

QUESTION: {question}

RETRIEVED CONTEXT:
{context[:2000]}

Reply with ONLY a number from 0 to 10."""
    return await llm_score(client, prompt)


async def evaluate_rag_system(
    test_queries: List[Dict],
    top_k: int = 5,
    max_context: int = 3,
) -> List[Dict]:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=" * 60)
    print("RAG SYSTEM EVALUATION (LLM-as-Judge)")
    print("=" * 60)
    print(f"Queries: {len(test_queries)} | top_k={top_k} | max_context={max_context}")
    print("=" * 60 + "\n")

    all_results = []

    for i, query_data in enumerate(test_queries, 1):
        question = query_data["question"]
        ground_truth = query_data.get("ground_truth", "N/A")

        print(f"[{i}/{len(test_queries)}] {question}", flush=True)

        # Run RAG pipeline
        print("  Querying RAG system...", flush=True)
        response = ask(
            query=question,
            top_k=top_k,
            max_context_emails=max_context,
            language="auto",
            stream=False,
            verbose=False,
        )

        answer = response["answer"]

        # Build context string from retrieved emails
        contexts = []
        for match in response["retrieval_results"]["matches"][:max_context]:
            meta = match.get("metadata", {})
            contexts.append(
                f"Subject: {meta.get('subject', 'N/A')}\n"
                f"From: {meta.get('from', 'N/A')}\n"
                f"Date: {meta.get('date', 'N/A')}\n"
                f"Content: {meta.get('content', 'N/A')}"
            )
        context_str = "\n\n---\n\n".join(contexts)

        # Evaluate all 4 metrics concurrently
        print("  Evaluating metrics...", flush=True)
        faithfulness, relevancy, precision, recall = await asyncio.gather(
            evaluate_faithfulness(client, question, answer, context_str),
            evaluate_answer_relevancy(client, question, answer),
            evaluate_context_precision(client, question, context_str),
            evaluate_context_recall(client, question, context_str),
        )

        avg = (faithfulness + relevancy + precision + recall) / 4

        print(f"  Faithfulness:      {faithfulness:.3f}")
        print(f"  Answer Relevancy:  {relevancy:.3f}")
        print(f"  Context Precision: {precision:.3f}")
        print(f"  Context Recall:    {recall:.3f}")
        print(f"  Average:           {avg:.3f}\n")

        all_results.append({
            "Question": question,
            "Ground_Truth": ground_truth,
            "Answer": answer,
            "Faithfulness": faithfulness,
            "Answer_Relevancy": relevancy,
            "Context_Precision": precision,
            "Context_Recall": recall,
            "Average_Score": avg,
        })

    return all_results


def print_summary(results: List[Dict]) -> None:
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    metrics = ["Faithfulness", "Answer_Relevancy", "Context_Precision", "Context_Recall", "Average_Score"]
    for m in metrics:
        avg = sum(r[m] for r in results) / len(results)
        print(f"  {m:25s}: {avg:.4f}")
    print("=" * 60)


async def main_async():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system using OpenAI as judge")
    parser.add_argument("--test-file", type=str, help="JSON file with test queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context", type=int, default=3)
    parser.add_argument("--output", type=str, help="Output CSV file path")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment / .env file")
        sys.exit(1)

    if args.test_file:
        with open(args.test_file, "r", encoding="utf-8") as f:
            test_queries = json.load(f)
    else:
        test_queries = DEFAULT_TEST_QUERIES
        print(f"Using {len(DEFAULT_TEST_QUERIES)} default test queries\n")

    try:
        results = await evaluate_rag_system(
            test_queries=test_queries,
            top_k=args.top_k,
            max_context=args.max_context,
        )
        print_summary(results)

        if args.output:
            import csv
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n✓ Results saved to {args.output}")

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
