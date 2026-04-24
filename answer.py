import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from chromadb import PersistentClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent
DB_NAME = str(PROJECT_ROOT / "preprocessed_db")
COLLECTION_NAME = "docs"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))

SYSTEM_PROMPT = """
You are a careful company knowledge-base assistant.
Answer only from the provided context.
If the context does not contain the answer, say that you do not know from the available documents.
Keep answers clear and concise.
Cite source filenames next to factual claims, like: (source: knowledge-base/company/overview.md).
"""

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model ready.")

chroma = PersistentClient(path=DB_NAME)


@dataclass
class Result:
    page_content: str
    metadata: dict


def get_collection():
    return chroma.get_or_create_collection(COLLECTION_NAME)


def ensure_index_ready() -> None:
    """Fail clearly when the vector database has not been built yet."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "The RAG index is empty. Add markdown files under knowledge-base/ "
            "and run `python3 ingest.py` before asking questions."
        )


def embed_query(text: str) -> list[float]:
    """Embed a single query string using the local model."""
    return embedder.encode([text]).tolist()[0]


def fetch_context(question: str, retrieval_k: int = RETRIEVAL_K) -> list[Result]:
    """Retrieve the most relevant chunks from the local ChromaDB index."""
    ensure_index_ready()
    collection = get_collection()
    query_vector = embed_query(question)
    results = collection.query(query_embeddings=[query_vector], n_results=retrieval_k)

    return [
        Result(page_content=document, metadata=metadata)
        for document, metadata in zip(results["documents"][0], results["metadatas"][0])
    ]


def build_context(chunks: list[Result]) -> str:
    """Format retrieved chunks for the chat model."""
    sections = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk", 0)
        sections.append(
            f"Source: {source} (chunk {chunk_id})\n"
            f"{chunk.page_content}"
        )
    return "\n\n---\n\n".join(sections)


def build_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    context = build_context(chunks)
    user_prompt = f"""
Context:
{context}

Question:
{question}
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_prompt}]


def call_ollama(messages: list[dict], model: str = OLLAMA_MODEL) -> str:
    """Call a local Ollama chat model."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Start it with `ollama serve` and "
            f"pull the model with `ollama pull {model}`."
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(f"Ollama returned an error: {response.text}") from exc

    data = response.json()
    return data["message"]["content"].strip()


def answer_question(
    question: str,
    history: Optional[list[dict]] = None,
    model: str = OLLAMA_MODEL,
    retrieval_k: int = RETRIEVAL_K,
) -> tuple[str, list[Result]]:
    """Answer a question using local RAG and a local open-source chat model."""
    history = history or []
    chunks = fetch_context(question, retrieval_k=retrieval_k)
    messages = build_messages(question, history, chunks)
    answer = call_ollama(messages, model=model)

    sources = sorted({chunk.metadata.get("source", "unknown") for chunk in chunks})
    answer_with_sources = answer + "\n\n---\nSources consulted: " + ", ".join(sources)
    return answer_with_sources, chunks


if __name__ == "__main__":
    print(f"Company RAG assistant using Ollama model `{OLLAMA_MODEL}`.")
    print("Type 'exit' or 'quit' to stop.\n")
    chat_history: list[dict] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        response, _ = answer_question(question, chat_history)
        print(f"\nAssistant: {response}\n")
        chat_history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        )
