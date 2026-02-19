# Email RAG System

An intelligent email search and Q&A system built with Retrieval-Augmented Generation (RAG). Ask natural language questions about your emails across multiple Gmail accounts — the system retrieves the most relevant emails and generates a grounded answer.

---

## Current Status

### Completed

**Ingestion Pipeline**
- Gmail API integration supporting multiple accounts
- HTML → plain text preprocessing
- Email chunking for long messages
- Embedding with `paraphrase-multilingual-mpnet-base-v2` (local, multilingual)
- Vector storage in Pinecone with full metadata (subject, from, date, account, content)

**Query & Answer Pipeline**
- Query embedding using the same local model
- Semantic search via Pinecone top-k retrieval
- Metadata filtering (account, sender, recipient, date)
- Answer generation via Claude API (streaming supported)
- Language auto-detection for response (Chinese / English)

**Evaluation**
- RAGAS v0.4+ evaluation script (`scripts/ragas_evaluation.py`) with ground truth — measures Context Precision, Context Recall, Answer Relevancy, Faithfulness
- LLM-as-judge evaluation script (`scripts/evaluate_rag.py`) — faster alternative using GPT-4o-mini, runs in under 15 seconds
- Known limitation: Chinese-language queries produce lower scores due to cross-language embedding mismatch (Chinese query vs. English email content)

### Project Structure

```
services/
  gmail_service.py        # Gmail API multi-account fetch
  email_preprocessor.py  # HTML cleaning, metadata extraction
  email_chunker.py        # Long email chunking
  embedding_service.py    # Sentence Transformers embedding
  pinecone_service.py     # Pinecone upsert and index management
  ingestion_pipeline.py   # Orchestrates full ingestion flow
  query_service.py        # Query embedding + Pinecone search
  claude_service.py       # Answer generation via Claude API
  rag_service.py          # End-to-end RAG pipeline

scripts/
  ingest_emails.py        # Run ingestion from CLI
  search_emails.py        # Raw vector search from CLI
  ask_emails.py           # Full RAG Q&A from CLI
  ragas_evaluation.py     # RAGAS evaluation with ground truth
```

---

## Planned Features

### 1. Scheduled Auto-Embedding
Automatically ingest new emails on a schedule so the vector index stays up to date without manual runs.
- Run ingestion via GitHub Actions cron or a local scheduler
- Incremental sync: only embed emails newer than the last ingestion timestamp
- Store last sync time to avoid re-processing old emails

### 2. Slack RAG Chat Interface
Integrate the RAG pipeline into Slack so it can be used alongside the existing email summary AI agent.
- Respond to `@mentions` or `/ask` slash commands in a Slack channel
- Send query to `rag_service.ask()`, return answer + source emails
- Unified Slack workspace as the single interface for all email AI tools

### 3. LLM Query Expansion
Before embedding the user query, use an LLM to rewrite and expand it into multiple semantically richer variants. Retrieve results for all variants and merge — improves recall especially for vague or short queries.
- Generate 3–5 rewritten queries from the original
- Embed and search all variants in parallel
- Deduplicate and re-rank results by combined score

---

## Recommended Next Steps

**Priority 1 — Fix Chinese query performance**
Switch the embedding model from `paraphrase-multilingual-mpnet-base-v2` to OpenAI `text-embedding-3-small`. This model handles cross-language retrieval significantly better (Chinese query → English content). Requires re-indexing all emails in Pinecone.

**Priority 2 — Incremental sync script**
Modify `scripts/ingest_emails.py` to read a last-sync timestamp from a local file or Pinecone metadata, fetch only new emails since that time, and update the timestamp after a successful run. This is a prerequisite for scheduling.

**Priority 3 — Slack bot**
Set up a Slack app with a slash command (`/ask`) that routes to a FastAPI endpoint, calls `rag_service.ask()`, and posts the answer back. The Slack SDK is already in `requirements.txt`.

**Priority 4 — Query expansion**
Add an optional pre-retrieval step in `rag_service.ask()` that calls GPT-4o-mini to generate expanded query variants, then merges and deduplicates results before passing to the LLM.

---

## Additional Ideas

- **Re-ranking**: After top-k retrieval, use a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score results by relevance to the query. Improves precision with minimal added cost.
- **Email thread grouping**: Group retrieved results by `thread_id` so the answer draws from a full conversation, not isolated fragments.
- **Hybrid search**: Combine vector similarity with keyword matching on metadata fields (subject, sender) using Pinecone's metadata filters — useful for queries like "emails from HR about Entegris".
- **Evaluation dashboard**: Save evaluation results to a CSV after each run and track metric trends over time to measure the impact of each improvement.
- **Web UI**: A minimal Streamlit or Gradio interface for local use — text input → answer + expandable source emails. Low effort, useful for demos.

---

## Tech Stack

| Component | Technology |
|---|---|
| Vector DB | Pinecone |
| Embedding | `paraphrase-multilingual-mpnet-base-v2` (local) |
| LLM (answers) | Anthropic Claude (claude-sonnet-4-6) |
| LLM (evaluation) | OpenAI GPT-4o-mini |
| Email source | Gmail API (3 accounts) |
| Evaluation | RAGAS v0.4+, custom LLM-as-judge |
| Slack | slack-sdk |
| API | FastAPI + uvicorn |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Authorize Gmail accounts
python authorize_accounts.py

# Run ingestion
python scripts/ingest_emails.py

# Ask a question
python scripts/ask_emails.py

# Run evaluation
python scripts/ragas_evaluation.py
```
