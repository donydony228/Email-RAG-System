# MailMind — AI-Powered Gmail Q&A Agent

Ask natural language questions about your personal emails. MailMind ingests emails from multiple Gmail accounts into a Pinecone vector index and answers queries using hybrid retrieval (BM25 + dense vectors) and Claude as the LLM backbone.

```
"Which company's recruiter last contacted me about an internship?"
"What is the Delta Airlines confirmation number for my March flight?"
"What was the most recent newsletter I received?"
```

---

## How It Works

```
Gmail API (3 accounts)
        │
        ▼
  Preprocessing          HTML → plain text, metadata extraction
        │
        ▼
    Chunking             RecursiveCharacterTextSplitter, 300-token chunks
        │
        ├──► OpenAI text-embedding-3-small   dense vectors (1536-dim)
        │
        └──► BM25Encoder                     sparse vectors
                │
                ▼
          Pinecone Index (dotproduct, hybrid)
                │
                ▼
          Query Pipeline
            ├── Hybrid search  (α=0.5 dense + sparse)
            ├── Temporal ranking  (auto-detect recency intent → sort / filter)
            └── Claude claude-sonnet-4-6  → grounded answer
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid search (BM25 + dense)** | BM25 captures exact keyword matches (booking codes, names); dense handles semantic similarity. α=0.5 increased RAGAS Factual CP from 0.806 → 0.972. |
| **Two-tier temporal detection** | Date-sort and server-side timestamp filter use separate keyword sets. Sort false positives are cheap (reorder only); filter false positives are expensive (exclude documents entirely). Prevents Named Entity queries from being misclassified as recency queries. |
| **dotproduct metric** | Required for Pinecone hybrid search with sparse vectors; cosine does not support `sparse_values`. |
| **300-token chunks** | Balances retrieval granularity against context density for `text-embedding-3-small`'s 8192-token limit. |
| **Incremental ingestion** | Each run checks existing Pinecone IDs before embedding, skipping emails already indexed — avoids redundant API costs on daily runs. |

---

## Evaluation

Evaluated with [RAGAS](https://github.com/explodinggradients/ragas) on a fixed 20-question benchmark covering 6 query types (Factual, Named Entity, Multi-doc, Semantic, Time-based, Non-existent).

| Configuration | Avg RAGAS | Δ vs Baseline |
|---------------|-----------|---------------|
| Dense only (baseline) | 0.637 | — |
| + Hybrid search α=0.5 | 0.676 | +6.1% |
| + CrossEncoder rerank *(evaluated, not adopted)* | 0.693 | +8.8% |
| **+ Temporal ranking v2 (production)** | **0.710** | **+11.4%** |

> Full evaluation history including per-category breakdowns and experiment rationale: [`doc/evaluation_history.md`](doc/evaluation_history.md)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector DB | [Pinecone](https://www.pinecone.io/) (serverless, dotproduct) |
| Embedding | OpenAI `text-embedding-3-small` (1536-dim) |
| Sparse retrieval | `pinecone-text` BM25Encoder |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Email source | Gmail API (OAuth 2.0, 3 accounts) |
| Evaluation | RAGAS v0.4+ |
| Automation | GitHub Actions (daily cron ingestion) |

---

## Project Structure

```
services/
  gmail_service.py        # Gmail API multi-account fetch
  email_preprocessor.py   # HTML cleaning, metadata extraction, timestamp parsing
  email_chunker.py        # Token-aware chunking (tiktoken cl100k_base)
  embedding_service.py    # OpenAI embeddings + BM25 sparse vectors
  pinecone_service.py     # Upsert, hybrid search, deduplication
  ingestion_pipeline.py   # Orchestrates full ingestion flow
  query_service.py        # Query embedding + Pinecone hybrid search
  temporal_service.py     # Temporal intent detection, date sort, recency filter
  rag_service.py          # End-to-end RAG pipeline
  claude_service.py       # Answer generation via Claude API (streaming)
  rerank_service.py       # CrossEncoder reranking (evaluated, disabled by default)
  sync_time.py            # Last-sync timestamp persistence

scripts/
  ingest_emails.py        # CLI ingestion runner
  ask_emails.py           # CLI Q&A interface
  search_emails.py        # Raw vector search (debug)
  ragas_evaluation.py     # RAGAS evaluation runner
  train_bm25.py           # BM25 model training on ingested corpus

.github/workflows/
  ingest_emails.yml       # Daily cron ingestion via GitHub Actions

doc/
  evaluation_history.md   # Full RAGAS experiment log (Stage 0–5)
  evaluation_questions.md # 20-question benchmark with ground truth
```

---

## Setup

### Prerequisites

- Python 3.10+
- Pinecone account (free Starter tier works)
- OpenAI API key
- Anthropic API key
- Gmail API credentials (one OAuth app per account)

### Installation

```bash
git clone <repo-url>
cd RAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY
```

### Gmail Authorization

```bash
# Run once per account — opens browser for OAuth consent
python authorize_accounts.py
# Tokens saved to credentials/token_account{1,2,3}.json
```

### First-time Ingestion

```bash
# Train BM25 model on a sample corpus first
python scripts/train_bm25.py

# Ingest emails (incremental by default — safe to re-run)
python scripts/ingest_emails.py --time-range 365d --max-emails 500
```

### Ask a Question

```bash
python scripts/ask_emails.py
# > What is my JetBlue confirmation code?
```

### Run Evaluation

```bash
python scripts/ragas_evaluation.py \
  --notes "my experiment" \
  --alpha 0.5 \
  --no-rerank \
  --top-k 20
```

### Automated Daily Ingestion

Push `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PINECONE_API_KEY`, and Gmail tokens as GitHub Actions secrets. The workflow at `.github/workflows/ingest_emails.yml` runs at 08:00 UTC daily and ingests the past 24 hours of emails incrementally.
