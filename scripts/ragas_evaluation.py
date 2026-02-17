"""
RAG evaluation using RAGAS framework.

Run from project root:
    python scripts/test_ragas_simple.py
"""

import sys
import os
import asyncio
import logging
import warnings
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


EVAL_QUESTIONS = [
    # 1. Fact Retrieval (Information Extraction)
    "What are the check-in and check-out dates and times for the Airbnb stay at 'Diva sonata no.7'?",
    "What is the total amount paid to Anthropic, and what is the specific receipt number for this transaction?",
    "What is the one-time passcode (OTP) sent by Acer Group, and how long is it valid for?",
    "這個帳號的擁有者叫什麼名字？"

    # 2. Semantic & Intent Search
    # "Have I received any updates regarding my job applications or internship status recently?",
    # "Are there any notifications indicating that an automated workflow or script failed to run?",
    # "What is the upcoming career event at NYU, and does it mention any specific perks for attendees?",

    # 3. Complex Reasoning & Technical Context
    # "Regarding the DS-GA 1003 course, what specific technical issue did a student report about Problem Set 2, and what step size (eta) did they suggest using instead?",
    # "Based on my recent billing emails, which AI service am I currently subscribed to, and what is the active billing period?",
    # "Summarize my schedule for the upcoming week based on my accommodation bookings and university events."
]


async def evaluate_questions():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    faithfulness = Faithfulness(llm=llm)

    all_results = []

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(EVAL_QUESTIONS)}] Q: {question}")
        print('='*60)

        response = ask(
            query=question,
            top_k=5,
            max_context_emails=3,
            language="auto",
            stream=False,
            verbose=False,
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
            reference=answer,
            retrieved_contexts=contexts,
        )
        result_recall = await context_recall.ascore(
            user_input=question,
            reference=answer,
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
            "Question": question,
            "Answer": answer,
            "Context_Precision": result_precision.value,
            "Context_Recall": result_recall.value,
            "Answer_Relevancy": result_relevancy.value,
            "Faithfulness": result_faithfulness.value,
            "Average_Score": avg,
        })

    return all_results


def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    metrics = ["Context_Precision", "Context_Recall", "Answer_Relevancy", "Faithfulness", "Average_Score"]
    for m in metrics:
        avg = sum(r[m] for r in results) / len(results)
        print(f"  {m:22s}: {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    results = asyncio.run(evaluate_questions())
    print_summary(results)
