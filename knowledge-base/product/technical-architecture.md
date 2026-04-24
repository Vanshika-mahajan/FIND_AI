# Technical Architecture

FIND AI uses a local-first RAG architecture.

## Components

- Knowledge base: markdown files stored under `knowledge-base/`.
- Ingestion script: `ingest.py` loads files, chunks text, embeds chunks, and
  writes them to ChromaDB.
- Embedding model: `BAAI/bge-small-en-v1.5` by default.
- Vector database: ChromaDB stores chunk embeddings and metadata.
- Chat model: Ollama runs `llama3.2:3b` locally by default.
- Interface: Streamlit app in `app.py`.

## Retrieval

When a user asks a question, the app embeds the query and searches ChromaDB for
the most similar chunks. The default retrieval count is eight chunks.

## Generation

Retrieved chunks are formatted as context and sent to the Ollama chat model.
The system prompt instructs the model to answer only from the provided context
and cite source filenames next to factual claims.

## Local Storage

The vector index is stored in `preprocessed_db/`. This folder is generated
locally and should not be committed to GitHub.

## Privacy Assumption

The default project design keeps generation local through Ollama. Documents are
not sent to a hosted LLM provider when using the default configuration.
