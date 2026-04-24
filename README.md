# FIND_AI

FIND_AI is a Retrieval-Augmented Generation (RAG) assistant for company documents.
It lets you add internal knowledge files, build a searchable vector index, and ask
questions that are answered using retrieved company context.

## What It Does

- Loads company documents from `knowledge-base/`
- Splits documents into useful chunks
- Creates embeddings with a local sentence-transformer model
- Stores vectors in ChromaDB
- Retrieves relevant context for each question
- Uses an LLM through Groq/LiteLLM to generate sourced answers

## Project Structure

```text
FIND_AI/
  answer.py                  # Ask questions using the RAG pipeline
  ingest.py                  # Load, chunk, embed, and index documents
  requirements.txt           # Python dependencies
  .env.example               # Example environment variables
  knowledge-base/
    company/
      overview.md            # Sample company document
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Add your Groq API key to `.env`:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Add Company Documents

Add markdown files inside `knowledge-base/`, grouped by topic:

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

Use clear filenames because answers cite the source documents.

## Build The Index

Run ingestion after adding or updating documents:

```bash
python3 ingest.py
```

This creates a local ChromaDB vector database in `preprocessed_db/`.
The first run may take time because the embedding model is downloaded locally.

## Ask Questions

Start the assistant:

```bash
python3 answer.py
```

Example:

```text
You: What does the company knowledge base support?
```

Type `exit` or `quit` to stop the assistant.

## Notes

- Do not commit `.env`; it contains secrets.
- Do not commit `preprocessed_db/`; it is generated locally.
- Re-run `python3 ingest.py` whenever company documents change.
- If you hit Groq rate limits, reduce `WORKERS` in `ingest.py`.
- If your machine is low on memory, reduce `BATCH_SIZE` in `ingest.py`.
