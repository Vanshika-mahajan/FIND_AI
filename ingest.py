import hashlib
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer


load_dotenv(override=True)

# Open-source models via Groq (free tier: ~30 req/min)
# Sign up at console.groq.com and set GROQ_API_KEY in your .env
SMALL_MODEL = "groq/llama-3.1-8b-instant"     # fast, used for chunking
LARGE_MODEL = "groq/llama-3.3-70b-versatile"  # stronger, used if needed

PROJECT_ROOT = Path(__file__).resolve().parent
DB_NAME = str(PROJECT_ROOT / "preprocessed_db")
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge-base"
collection_name = "docs"
AVERAGE_CHUNK_SIZE = 1200
BATCH_SIZE = 64   # embedding batch size - tune down if you hit memory limits
WORKERS = 3       # parallel document workers - set to 1 if rate limited

wait = wait_exponential(multiplier=1, min=10, max=240)

# Embedding model - downloads ~2GB on first run, cached locally after that.
# BAAI/bge-m3 is multilingual and close in quality to text-embedding-3-large.
print("Loading embedding model (first run downloads ~2 GB)...")
embedder = SentenceTransformer("BAAI/bge-m3")
print("Embedding model ready.")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def fetch_documents():
    """Load all .md files from the knowledge base directory tree."""
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
            with open(file, "r", encoding="utf-8") as f:
                documents.append({
                    "type": doc_type,
                    "source": file.as_posix(),
                    "text": f.read(),
                })
    if not documents:
        raise FileNotFoundError(
            f"No markdown files found under {KNOWLEDGE_BASE_PATH}.\n"
            "Add .md files grouped by folder, then run `python3 ingest.py` again."
        )

    print(f"Loaded {len(documents)} documents")
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def make_prompt(document):
    how_many = max(1, (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1)
    return f"""
You take a document and split it into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company.
The document type is: {document["type"]}
Retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
Divide the document as you see fit, ensuring the entire document is covered - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can use more or fewer as appropriate.
There should be ~25% overlap between adjacent chunks (about 50 words) for best retrieval results.

For each chunk, provide:
- headline: a brief heading (a few words)
- summary: a few sentences summarizing the chunk
- original_text: the original text of the chunk, exactly as written

Together, your chunks must represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [{"role": "user", "content": make_prompt(document)}]


@retry(wait=wait)
def process_document(document):
    messages = make_messages(document)
    response = completion(model=SMALL_MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """
    Chunk documents in parallel.
    Set WORKERS=1 if you hit Groq rate limits.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(process_document, documents),
            total=len(documents),
            desc="Chunking documents",
        ):
            chunks.extend(result)
    print(f"Created {len(chunks)} chunks total")
    return chunks


# ---------------------------------------------------------------------------
# Embeddings - batched to handle large datasets
# ---------------------------------------------------------------------------

def batch_embed(texts: list[str]) -> list[list[float]]:
    """
    Embed texts in batches using the local sentence-transformers model.
    Returns a list of float vectors.
    """
    all_vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[i : i + BATCH_SIZE]
        vecs = embedder.encode(batch, show_progress_bar=False).tolist()
        all_vectors.extend(vecs)
    return all_vectors


def create_embeddings(chunks: list[Result]):
    """Embed all chunks and store them in ChromaDB."""
    chroma = PersistentClient(path=DB_NAME)

    # Drop existing collection so we start fresh on re-ingest
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    vectors = batch_embed(texts)

    collection = chroma.get_or_create_collection(collection_name)

    ids = [
        hashlib.sha1(f"{chunk.metadata['source']}:{i}:{chunk.page_content}".encode("utf-8")).hexdigest()
        for i, chunk in enumerate(chunks)
    ]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
