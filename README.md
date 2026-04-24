# Company RAG Starter

This project builds a simple RAG assistant for company documents.

It has two steps:

1. `ingest.py` reads markdown files from `knowledge-base/`, chunks them, embeds them, and stores them in ChromaDB.
2. `answer.py` retrieves relevant chunks and asks an LLM to answer with source citations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Groq API key to `.env`:

```bash
GROQ_API_KEY=...
```

## Add Company Documents

Put markdown files under `knowledge-base/`, grouped by topic:

```text
knowledge-base/
  policies/
    pto.md
    security.md
  products/
    platform.md
  sales/
    pricing-faq.md
```

Use clear file names because answers cite the source path.

## Build The Index

```bash
python3 ingest.py
```

The first run downloads the embedding model. It can take a while because `BAAI/bge-m3`
is a large local model.

## Ask Questions

```bash
python3 answer.py
```

Example:

```text
You: What documents can I put in the company knowledge base?
```

## Notes

- The vector database is stored in `preprocessed_db/`.
- Re-run `python3 ingest.py` whenever company documents change.
- If you hit Groq rate limits during ingestion, reduce `WORKERS` in `ingest.py`.
- If your machine is low on memory, reduce `BATCH_SIZE` in `ingest.py`.
