# FIND_AI

FIND_AI is a local, open-source Retrieval-Augmented Generation (RAG) assistant for
company documents. It indexes markdown files, retrieves the most relevant context,
and answers questions using a local Ollama model through a polished Streamlit UI.

## Features

- Local markdown knowledge base
- Deterministic document chunking
- Local embeddings with Sentence Transformers
- Local vector storage with ChromaDB
- Local chat generation with Ollama
- Streamlit chat interface
- Source-aware answers
- No OpenAI key or hosted LLM required

## Project Structure

```text
FIND_AI/
  answer.py                  # Ask questions using the RAG pipeline
  app.py                     # Streamlit user interface
  ingest.py                  # Load, chunk, embed, and index documents
  requirements.txt           # Python dependencies
  .env.example               # Example local model settings
  knowledge-base/
    company/
      overview.md            # Sample company document
```

## Requirements

- Python 3.10 or newer recommended
- Ollama installed locally

Install Ollama from:

```text
https://ollama.com
```

Pull an open-source chat model:

```bash
ollama pull llama3.2:3b
```

You can use another Ollama model by changing `OLLAMA_MODEL` in `.env`.

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

Default settings:

```bash
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
RETRIEVAL_K=8
CHUNK_WORDS=350
CHUNK_OVERLAP=80
BATCH_SIZE=64
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

## Launch The App

Make sure Ollama is running:

```bash
ollama serve
```

In another terminal, start the assistant:

```bash
streamlit run app.py
```

Example:

```text
You: What does the company knowledge base support?
```

Type `exit` or `quit` to stop the assistant.

You can still use the terminal chat version:

```bash
python3 answer.py
```

## How It Works

1. `ingest.py` reads markdown files from `knowledge-base/`.
2. It splits each document into overlapping word chunks.
3. It embeds chunks using `BAAI/bge-small-en-v1.5`.
4. It stores embeddings and metadata in ChromaDB.
5. `app.py` provides the chat UI and source panel.
6. `answer.py` embeds the user question and retrieves matching chunks.
7. Ollama generates an answer using only the retrieved context.

## Notes

- Do not commit `.env`; it may contain local settings or secrets.
- Do not commit `preprocessed_db/`; it is generated locally.
- Re-run `python3 ingest.py` whenever company documents change.
- Increase `RETRIEVAL_K` if answers need more context.
- Increase `CHUNK_WORDS` for longer sections or decrease it for more precise retrieval.
