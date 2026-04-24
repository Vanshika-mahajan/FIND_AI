import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent
DB_NAME = str(PROJECT_ROOT / "preprocessed_db")
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge-base"
COLLECTION_NAME = "docs"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model ready.")


@dataclass
class Result:
    page_content: str
    metadata: dict


def fetch_documents() -> list[dict]:
    """Load all markdown files from the knowledge base directory tree."""
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base folder not found: {KNOWLEDGE_BASE_PATH}\n"
            "Create it and add company markdown files, for example:\n"
            "  knowledge-base/policies/pto.md\n"
            "  knowledge-base/products/platform.md"
        )

    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            text = file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            documents.append(
                {
                    "type": doc_type,
                    "source": file.relative_to(PROJECT_ROOT).as_posix(),
                    "text": text,
                }
            )

    if not documents:
        raise FileNotFoundError(
            f"No markdown files found under {KNOWLEDGE_BASE_PATH}.\n"
            "Add .md files grouped by folder, then run `python3 ingest.py` again."
        )

    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word chunks without requiring an LLM."""
    words = text.split()
    if not words:
        return []
    if chunk_words <= overlap:
        raise ValueError("CHUNK_WORDS must be greater than CHUNK_OVERLAP")

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def create_chunks(documents: list[dict]) -> list[Result]:
    """Create retrieval chunks from loaded documents."""
    chunks = []
    for document in documents:
        for index, text in enumerate(chunk_text(document["text"])):
            chunks.append(
                Result(
                    page_content=text,
                    metadata={
                        "source": document["source"],
                        "type": document["type"],
                        "chunk": index,
                    },
                )
            )

    print(f"Created {len(chunks)} chunks total")
    return chunks


def batch_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches using the local sentence-transformers model."""
    all_vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[i : i + BATCH_SIZE]
        vectors = embedder.encode(batch, show_progress_bar=False).tolist()
        all_vectors.extend(vectors)
    return all_vectors


def create_embeddings(chunks: list[Result]) -> None:
    """Embed all chunks and store them in ChromaDB."""
    chroma = PersistentClient(path=DB_NAME)

    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]
    vectors = batch_embed(texts)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    ids = [
        hashlib.sha1(f"{chunk.metadata['source']}:{chunk.metadata['chunk']}:{text}".encode("utf-8")).hexdigest()
        for chunk, text in zip(chunks, texts)
    ]
    metadatas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    print(f"Vector store created with {collection.count()} chunks")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
