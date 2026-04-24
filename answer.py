from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer
from typing import Optional


load_dotenv(override=True)

# Open-source models via Groq (free tier: ~30 req/min)
# Sign up at console.groq.com and set GROQ_API_KEY in your .env
SMALL_MODEL = "groq/llama-3.1-8b-instant"     # query rewriting - fast, low-stakes
LARGE_MODEL = "groq/llama-3.3-70b-versatile"  # reranking + answering - quality matters

PROJECT_ROOT = Path(__file__).resolve().parent
DB_NAME = str(PROJECT_ROOT / "preprocessed_db")
collection_name = "docs"

RETRIEVAL_K = 20   # candidates fetched from vector store
FINAL_K = 10       # chunks sent to the LLM after reranking

wait = wait_exponential(multiplier=1, min=10, max=240)

# Reuse the same embedding model as ingest.py.
# Keep this as a module-level singleton so it is loaded once per process.
print("Loading embedding model...")
embedder = SentenceTransformer("BAAI/bge-m3")
print("Embedding model ready.")

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company.
Your answer will be evaluated for accuracy, relevance, and completeness.
Answer only the question asked, and answer it fully.
If you don't know the answer, say so.

Here are specific extracts from the Knowledge Base that may be directly relevant:
{context}

With this context, please answer the user's question. Be accurate, relevant, and complete.
Cite the source document name next to each fact you use, like: (source: filename).
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk ids ordered from most relevant to least relevant"
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def embed_query(text: str) -> list[float]:
    """Embed a single query string using the local model."""
    return embedder.encode([text]).tolist()[0]


def ensure_index_ready() -> None:
    """Fail clearly when the vector database has not been built yet."""
    if collection.count() == 0:
        raise RuntimeError(
            "The RAG index is empty. Add markdown files under knowledge-base/ "
            "and run `python3 ingest.py` before asking questions."
        )


def fetch_context_unranked(question: str) -> list[Result]:
    """Vector search - returns up to RETRIEVAL_K chunks."""
    ensure_index_ready()
    query_vector = embed_query(question)
    results = collection.query(query_embeddings=[query_vector], n_results=RETRIEVAL_K)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks


def merge_chunks(chunks: list[Result], extra: list[Result]) -> list[Result]:
    """Merge two chunk lists, deduplicating by page_content."""
    merged = chunks[:]
    existing = {c.page_content for c in chunks}
    for chunk in extra:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

@retry(wait=wait)
def rewrite_query(question: str, history: Optional[list[dict]] = None) -> str:
    """
    Rewrite the user's question into a short, precise knowledge-base query.
    Uses the small model - this step is low-stakes and latency-sensitive.
    """
    message = f"""
You are about to search a Knowledge Base to answer a user's question.

Conversation history:
{history or []}

User's question:
{question}

Respond ONLY with a short, specific query (one sentence max) most likely to surface the relevant content.
Do not add any explanation or preamble.
"""
    response = completion(
        model=SMALL_MODEL,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

@retry(wait=wait)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """
    Use the large LLM to reorder chunks by relevance to the question.
    Returns chunks sorted from most to least relevant.
    """
    system_prompt = """
You are a document re-ranker.
You are given a question and a numbered list of text chunks retrieved from a knowledge base.
Rank all chunks by relevance to the question - most relevant first.
Reply ONLY with a JSON object shaped like {"order": [1, 2, 3]}.
Include every id provided.
"""
    user_prompt = f"Question:\n{question}\n\nChunks:\n\n"
    for i, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {i + 1}\n{chunk.page_content}\n\n"
    user_prompt += 'Reply only with JSON in this shape: {"order": [2, 1, 3]}.'

    response = completion(
        model=LARGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RankOrder,
    )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    # Guard against out-of-range ids returned by the model
    valid_order = [i - 1 for i in order if 1 <= i <= len(chunks)]
    missing = [i for i in range(len(chunks)) if i not in valid_order]
    valid_order.extend(missing)
    return [chunks[i] for i in valid_order]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def fetch_context(original_question: str) -> list[Result]:
    """
    Full retrieval pipeline:
      1. Rewrite the query
      2. Fetch chunks for both original and rewritten question
      3. Merge and deduplicate
      4. Rerank with LLM
      5. Return top FINAL_K
    """
    rewritten = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten)
    merged = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, merged)
    return reranked[:FINAL_K]


def make_rag_messages(
    question: str,
    history: list[dict],
    chunks: list[Result],
) -> list[dict]:
    """Assemble the full message list for the LLM, including retrieved context."""
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


# ---------------------------------------------------------------------------
# Main answer function
# ---------------------------------------------------------------------------

@retry(wait=wait)
def answer_question(
    question: str,
    history: Optional[list[dict]] = None,
) -> tuple[str, list[Result]]:
    """
    Answer a question using RAG.

    Returns:
        answer  - the LLM's response string
        chunks  - the retrieved context chunks (for citations / debugging)
    """
    history = history or []
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=LARGE_MODEL, messages=messages)
    answer = response.choices[0].message.content

    # Surface source files alongside the answer for transparency
    sources = list({chunk.metadata["source"] for chunk in chunks})
    answer_with_sources = answer + "\n\n---\n**Sources consulted:** " + ", ".join(sources)

    return answer_with_sources, chunks


if __name__ == "__main__":
    print("Company RAG assistant. Type 'exit' or 'quit' to stop.\n")
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
