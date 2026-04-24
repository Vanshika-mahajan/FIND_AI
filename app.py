import html
import subprocess
import sys
from pathlib import Path

import requests
import streamlit as st

from answer import OLLAMA_BASE_URL, OLLAMA_MODEL, answer_question, get_collection


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge-base"


st.set_page_config(
    page_title="FIND_AI",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
  --ink: #17201a;
  --muted: #607066;
  --paper: #fbfaf4;
  --line: #d9ddd1;
  --moss: #4f6f52;
  --clay: #b86b4b;
  --sage: #dce8d4;
  --mist: #eef2e8;
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(184, 107, 75, 0.13), transparent 30rem),
    linear-gradient(135deg, #fbfaf4 0%, #eef2e8 47%, #f8f4ec 100%);
  color: var(--ink);
}

[data-testid="stSidebar"] {
  background: rgba(251, 250, 244, 0.86);
  border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
  color: var(--ink);
}

.main .block-container {
  max-width: 1180px;
  padding-top: 2.5rem;
  padding-bottom: 4rem;
}

.topbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(23, 32, 26, 0.14);
  margin-bottom: 1.2rem;
}

.brand h1 {
  color: var(--ink);
  font-size: 2.55rem;
  line-height: 1.05;
  letter-spacing: 0;
  margin: 0;
}

.brand p {
  color: var(--muted);
  font-size: 1rem;
  margin: 0.55rem 0 0;
  max-width: 48rem;
}

.status-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 1rem 0 1.25rem;
}

.metric {
  background: rgba(255, 255, 255, 0.58);
  border: 1px solid rgba(23, 32, 26, 0.12);
  border-radius: 8px;
  padding: 0.85rem 0.95rem;
}

.metric span {
  color: var(--muted);
  display: block;
  font-size: 0.78rem;
  margin-bottom: 0.28rem;
}

.metric strong {
  color: var(--ink);
  display: block;
  font-size: 1.16rem;
}

.source-box {
  border-left: 3px solid var(--moss);
  background: rgba(255, 255, 255, 0.48);
  padding: 0.75rem 0.9rem;
  margin-bottom: 0.75rem;
  border-radius: 0 8px 8px 0;
}

.source-box small {
  color: var(--muted);
}

.stButton > button {
  border-radius: 8px;
  border: 1px solid rgba(23, 32, 26, 0.18);
  background: #17201a;
  color: #fbfaf4;
  font-weight: 650;
}

.stButton > button:hover {
  border-color: #17201a;
  background: #2d3d32;
  color: #fbfaf4;
}

[data-testid="stChatMessage"] {
  background: rgba(255, 255, 255, 0.58);
  border: 1px solid rgba(23, 32, 26, 0.10);
  border-radius: 8px;
}

@media (max-width: 760px) {
  .topbar {
    display: block;
  }

  .status-row {
    grid-template-columns: 1fr;
  }

  .brand h1 {
    font-size: 2rem;
  }
}
</style>
"""


def ollama_status() -> str:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2)
        response.raise_for_status()
    except requests.RequestException:
        return "Offline"
    return "Online"


def count_markdown_files() -> int:
    if not KNOWLEDGE_BASE_PATH.exists():
        return 0
    return len(list(KNOWLEDGE_BASE_PATH.rglob("*.md")))


def run_ingestion() -> tuple[bool, str]:
    process = subprocess.run(
        [sys.executable, "ingest.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in [process.stdout, process.stderr] if part.strip())
    return process.returncode == 0, output


def render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">
            <h1>FIND_AI</h1>
            <p>Ask focused questions across company knowledge, with local retrieval and open-source generation.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(model: str, retrieval_k: int) -> None:
    st.markdown(
        f"""
        <div class="status-row">
          <div class="metric"><span>Indexed chunks</span><strong>{get_collection().count()}</strong></div>
          <div class="metric"><span>Documents</span><strong>{count_markdown_files()}</strong></div>
          <div class="metric"><span>Model</span><strong>{html.escape(model)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Retrieving top {retrieval_k} chunks from the local vector store.")


def render_sources(chunks) -> None:
    if not chunks:
        return

    st.subheader("Sources")
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk", 0)
        preview = html.escape(chunk.page_content[:420])
        if len(chunk.page_content) > 420:
            preview += "..."
        source = html.escape(str(source))
        chunk_id = html.escape(str(chunk_id))
        st.markdown(
            f"""
            <div class="source-box">
              <strong>{source}</strong><br>
              <small>Chunk {chunk_id}</small>
              <p>{preview}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_chunks" not in st.session_state:
        st.session_state.last_chunks = []


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
initialize_state()

with st.sidebar:
    st.title("Control Room")
    model = st.text_input("Ollama model", value=OLLAMA_MODEL)
    retrieval_k = st.slider("Retrieved chunks", min_value=3, max_value=15, value=8)

    st.divider()
    st.write("Knowledge base")
    st.caption(str(KNOWLEDGE_BASE_PATH.relative_to(PROJECT_ROOT)))

    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Indexing documents..."):
            ok, output = run_ingestion()
        if ok:
            st.success("Index rebuilt.")
            st.code(output)
            st.rerun()
        else:
            st.error("Indexing failed.")
            st.code(output)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_chunks = []
        st.rerun()

    st.divider()
    st.write("Runtime")
    st.caption(f"Ollama: {ollama_status()}")
    st.caption(f"Endpoint: {OLLAMA_BASE_URL}")

render_header()
render_metrics(model, retrieval_k)

left, right = st.columns([0.68, 0.32], gap="large")

with left:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about your company documents")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base..."):
                try:
                    response, chunks = answer_question(
                        prompt,
                        history=history,
                        model=model,
                        retrieval_k=retrieval_k,
                    )
                except Exception as exc:
                    response = str(exc)
                    chunks = []
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.last_chunks = chunks
        st.rerun()

with right:
    render_sources(st.session_state.last_chunks)
