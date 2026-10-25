# Enterprise RAG & Agentic AI Platform

An AI assistant that answers questions over **company documents and a SQL database**, with every
answer **grounded and cited**. It is built to be **free to run**: local embeddings and re-ranking,
free-tier LLMs (Llama 3 on Groq, Google Gemini), and optional Claude / GPT.

![CI](https://github.com/<your-username>/enterprise-rag-agent/actions/workflows/ci.yml/badge.svg)

## What it does

Ask *"Which products in the recall policy had the most sales in Q2 2026?"* and the agent:
1. classifies the question (documents, database, both, or small talk)
2. searches the policy documents (hybrid search + re-ranking)
3. writes and safely runs a **read-only SQL** query on the sales database
4. writes an answer citing **[1]** document passages and **[DB]** database rows
5. checks its own citations and retries once if they are missing
6. remembers the conversation for follow-up questions

## Features

- **Semantic ingestion** of PDF / DOCX / Markdown with page- and section-level metadata; idempotent re-ingest
- **Hybrid retrieval**: dense vectors (bge-small) + BM25, fused with **Reciprocal Rank Fusion**, then a **cross-encoder re-ranker**
- **Grounded answers**: enforced citations and an explicit "I don't know" when the documents lack the answer
- **Model-agnostic LLM router**: Groq (Llama), Gemini, Claude, GPT; task-based routing, automatic fallback, caching (Redis or in-memory), cost and latency logging
- **LangGraph agent**: classify → tools (vector search, guarded SQL) → synthesize → validate, with SQLite-checkpointed memory
- **FastAPI service**: JWT auth, rate limiting, validated inputs, **SSE streaming** of agent steps and tokens, file upload, `/health`, `/metrics`, web chat UI
- **Evaluation harness**: hit@1, recall@5, MRR across retrieval setups; answer accuracy, citation rate and refusals per model
- **Quality**: 73 offline tests and lint on every push (GitHub Actions); Docker image with a non-root user and baked-in models

## Architecture

```
            ┌───────────── Ingestion ─────────────┐
 PDF/DOCX → │ loaders → semantic chunks → embed   │ → ChromaDB (vectors + metadata)
    MD      └─────────────────────────────────────┘          │
                                                             ▼
 Browser ──JWT──► FastAPI ──► LangGraph agent ──► vector_search (dense + BM25 → RRF → re-rank)
   ▲  SSE: steps + tokens      classify → tools → synthesize → validate
   └──────────────────────────┘        │            sql_query (read-only SQLite)
                                       ▼
                     LLM router: Groq · Gemini · Claude · GPT
                     (task routing · fallback · cache · cost logs)
```

## Quick start (Windows cmd)

```
git clone https://github.com/<your-username>/enterprise-rag-agent.git
cd enterprise-rag-agent
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
copy .env.example .env
```
Add your free keys to `.env` ([Groq](https://console.groq.com/keys), [Gemini](https://aistudio.google.com/apikey)), then:
```
python -m app.ingestion.pipeline data\sample_docs
uvicorn app.api.main:app --reload
```
Open **http://127.0.0.1:8000** (demo login `admin` / `admin123`; change it in `.env`) or **/docs** for the interactive API.

Mac/Linux: use `source .venv/bin/activate`, `cp`, and `/` paths, or `make install` and `make run`.

### With Docker (includes Redis)
```
copy .env.example .env      # add your keys
docker compose up --build
```

> **No API keys?** Everything still runs: the router falls back to a built-in extractive answerer that quotes the best passage.

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /auth/token` | — | Get a JWT |
| `POST /query` | JWT | Grounded RAG answer with citations (`"stream": true` for SSE) |
| `POST /chat` | JWT | LangGraph agent with memory per `thread_id`; streams steps + tokens |
| `POST /ingest` | JWT | Upload PDF / DOCX / TXT / MD |
| `GET /health` | — | Status, indexed chunks, available providers |
| `GET /metrics` | JWT | Counters, cache hits, cost, p50/p95 latency |

## Evaluation

```
python eval\run_eval.py                                     # retrieval
python eval\run_eval.py --generation --providers groq --sleep 2
```

Results on the 50-question golden set (from `eval/results.md`):

| Retrieval config | hit@1 | recall@5 | MRR@5 |
|---|---|---|---|
| dense only | _your number_ | _your number_ | _your number_ |
| hybrid (dense + BM25, RRF) | _your number_ | _your number_ | _your number_ |
| hybrid + re-rank | _your number_ | _your number_ | _your number_ |

| Provider | Answer accuracy | Citation rate | Refusals (of 3) |
|---|---|---|---|
| groq (llama-3.3-70b) | _your number_ | _your number_ | _your number_ |

## Demo data

`scripts/generate_sample_data.py` builds a fictional grocery retailer, "Harbor Fresh Markets": 21 policy documents,
a SQLite sales database (Jan–Jun 2026) and the golden evaluation set.

## Project layout

```
src/app/
  config.py            settings from .env
  ingestion/           loaders, semantic chunking, embeddings, pipeline CLI
  retrieval/           Chroma vector store, hybrid retriever + RRF, re-rankers
  llm/                 providers, router, cache, prompt templates
  rag.py               single-shot grounded RAG
  agent/               LangGraph state, tools (vector search, guarded SQL), graph
  api/                 FastAPI app, auth + rate limit, schemas, web UI
  observability/       JSON logging, metrics
eval/                  golden set + evaluation harness
scripts/               demo data generator, CLI helpers (ask.py, chat.py)
tests/                 73 offline tests (no network, no keys)
```

## Design decisions

- **Semantic chunking** keeps each chunk on one topic and never crosses page or section boundaries, so citations stay precise.
- **Hybrid + RRF**: BM25 catches exact identifiers (SKUs) that embeddings miss; RRF fuses by rank, so no score calibration is needed.
- **Plain httpx providers** instead of vendor SDKs: fewer dependencies and full control over streaming, caching and fallback.
- **SQL defence in depth**: SELECT-only allowlist + keyword blocklist + a read-only SQLite connection.
- **Offline test doubles** (hash embedder, lexical re-ranker, scripted LLM): CI is fast, free and deterministic.

## Development

```
pytest -q
ruff check src tests scripts eval
```
