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
  --void: #03050d;
  --panel: rgba(9, 15, 33, 0.72);
  --panel-strong: rgba(12, 18, 42, 0.9);
  --line: rgba(104, 190, 255, 0.22);
  --line-hot: rgba(158, 112, 255, 0.56);
  --text: #edf7ff;
  --muted: #96a6c9;
  --cyan: #28d7ff;
  --blue: #3b82ff;
  --violet: #9d6cff;
  --pink: #ff5fd2;
  --green: #79ffbf;
}

.stApp {
  color: var(--text);
  background:
    radial-gradient(circle at 12% 10%, rgba(48, 112, 255, 0.34), transparent 24rem),
    radial-gradient(circle at 84% 18%, rgba(157, 108, 255, 0.28), transparent 26rem),
    radial-gradient(circle at 48% 85%, rgba(40, 215, 255, 0.15), transparent 30rem),
    linear-gradient(135deg, #02030a 0%, #070b1f 42%, #110623 100%);
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background-image:
    linear-gradient(rgba(77, 167, 255, 0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(77, 167, 255, 0.055) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(circle at center, black, transparent 82%);
}

.stApp::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(circle, rgba(40, 215, 255, 0.45) 0 1px, transparent 2px) 12% 28% / 190px 190px,
    radial-gradient(circle, rgba(255, 95, 210, 0.42) 0 1px, transparent 2px) 68% 18% / 230px 230px,
    radial-gradient(circle, rgba(121, 255, 191, 0.32) 0 1px, transparent 2px) 82% 72% / 210px 210px;
  animation: particleDrift 18s linear infinite;
  opacity: 0.42;
}

@keyframes particleDrift {
  from { transform: translate3d(0, 0, 0); }
  to { transform: translate3d(-28px, 22px, 0); }
}

@keyframes pulse {
  0%, 100% { opacity: 0.48; transform: scale(0.92); }
  50% { opacity: 1; transform: scale(1.12); }
}

@keyframes scan {
  from { transform: translateX(-100%); }
  to { transform: translateX(100%); }
}

@keyframes floatNode {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

.main .block-container {
  max-width: 1440px;
  padding-top: 1.6rem;
  padding-bottom: 4rem;
  position: relative;
  z-index: 1;
}

[data-testid="stSidebar"] {
  background: rgba(4, 8, 22, 0.82);
  border-right: 1px solid rgba(40, 215, 255, 0.18);
  box-shadow: 18px 0 60px rgba(0, 0, 0, 0.42);
  backdrop-filter: blur(22px);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
  color: var(--text);
}

[data-testid="stSidebar"] .stCaptionContainer,
.stCaptionContainer {
  color: var(--muted);
}

.hero-shell {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(84, 181, 255, 0.24);
  border-radius: 18px;
  padding: 1.2rem 1.25rem;
  background:
    linear-gradient(135deg, rgba(13, 24, 56, 0.82), rgba(20, 9, 46, 0.76)),
    radial-gradient(circle at 20% 10%, rgba(40, 215, 255, 0.18), transparent 18rem);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.04) inset,
    0 24px 70px rgba(0, 0, 0, 0.36),
    0 0 36px rgba(40, 215, 255, 0.12);
  backdrop-filter: blur(24px);
}

.hero-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(40, 215, 255, 0.18), transparent);
  animation: scan 5s linear infinite;
}

.topbar {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: center;
}

.brand-kicker {
  color: var(--green);
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08rem;
  margin-bottom: 0.35rem;
}

.brand h1 {
  color: var(--text);
  font-size: clamp(2.2rem, 5vw, 4.4rem);
  line-height: 1;
  letter-spacing: 0;
  margin: 0;
  text-shadow: 0 0 24px rgba(40, 215, 255, 0.36);
}

.brand p {
  color: var(--muted);
  font-size: 1rem;
  margin: 0.65rem 0 0;
  max-width: 56rem;
}

.status-orbit {
  min-width: 19rem;
  border: 1px solid rgba(40, 215, 255, 0.18);
  border-radius: 14px;
  padding: 0.85rem;
  background: rgba(4, 8, 20, 0.46);
  box-shadow: 0 0 34px rgba(40, 215, 255, 0.12) inset;
}

.status-row-mini {
  display: flex;
  gap: 0.55rem;
  align-items: center;
  justify-content: space-between;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--text);
  font-size: 0.75rem;
  white-space: nowrap;
}

.dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  background: var(--cyan);
  box-shadow: 0 0 16px var(--cyan);
  animation: pulse 1.6s ease-in-out infinite;
}

.dot.violet {
  background: var(--violet);
  box-shadow: 0 0 16px var(--violet);
  animation-delay: 0.28s;
}

.dot.green {
  background: var(--green);
  box-shadow: 0 0 16px var(--green);
  animation-delay: 0.56s;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 1rem 0 1rem;
}

.metric {
  border: 1px solid rgba(104, 190, 255, 0.2);
  border-radius: 16px;
  padding: 0.95rem 1rem;
  background: linear-gradient(145deg, rgba(12, 20, 48, 0.76), rgba(4, 9, 24, 0.62));
  box-shadow:
    8px 10px 32px rgba(0, 0, 0, 0.28),
    -1px -1px 0 rgba(255, 255, 255, 0.08) inset,
    0 0 24px rgba(59, 130, 255, 0.09);
}

.metric span {
  display: block;
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06rem;
}

.metric strong {
  display: block;
  color: var(--text);
  font-size: 1.25rem;
  margin-top: 0.3rem;
}

.glass-panel {
  border: 1px solid rgba(104, 190, 255, 0.18);
  border-radius: 18px;
  background: rgba(5, 10, 26, 0.58);
  box-shadow:
    0 22px 60px rgba(0, 0, 0, 0.28),
    0 0 34px rgba(40, 215, 255, 0.08) inset;
  backdrop-filter: blur(24px);
  padding: 1rem;
}

.panel-title {
  color: var(--text);
  font-weight: 800;
  margin: 0 0 0.75rem;
}

.kb-file {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  border: 1px solid rgba(104, 190, 255, 0.13);
  border-radius: 12px;
  padding: 0.62rem 0.72rem;
  margin-bottom: 0.45rem;
  background: rgba(255, 255, 255, 0.035);
  transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
}

.kb-file:hover {
  transform: translateY(-2px);
  border-color: rgba(40, 215, 255, 0.42);
  box-shadow: 0 0 18px rgba(40, 215, 255, 0.14);
}

.kb-file strong {
  color: var(--text);
  font-size: 0.82rem;
}

.kb-file span {
  color: var(--muted);
  font-size: 0.72rem;
}

.source-card {
  border: 1px solid rgba(104, 190, 255, 0.2);
  border-radius: 16px;
  padding: 0.9rem;
  margin-bottom: 0.8rem;
  background:
    linear-gradient(145deg, rgba(12, 18, 42, 0.82), rgba(4, 9, 24, 0.68)),
    radial-gradient(circle at top right, rgba(157, 108, 255, 0.18), transparent 10rem);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24), 0 0 22px rgba(40, 215, 255, 0.08) inset;
  transition: transform 180ms ease, border-color 180ms ease;
}

.source-card:hover {
  transform: translateY(-3px);
  border-color: rgba(40, 215, 255, 0.48);
}

.source-card strong {
  color: var(--text);
  font-size: 0.9rem;
}

.source-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.55rem 0;
}

.pill {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(40, 215, 255, 0.24);
  border-radius: 999px;
  padding: 0.2rem 0.5rem;
  color: var(--muted);
  background: rgba(40, 215, 255, 0.06);
  font-size: 0.7rem;
}

.source-card p {
  color: #c8d5f2;
  font-size: 0.82rem;
  line-height: 1.5;
  margin: 0.45rem 0 0;
}

.graph {
  position: relative;
  min-height: 260px;
  overflow: hidden;
  border: 1px solid rgba(104, 190, 255, 0.18);
  border-radius: 18px;
  background:
    radial-gradient(circle at 50% 50%, rgba(40, 215, 255, 0.14), transparent 9rem),
    linear-gradient(145deg, rgba(8, 13, 31, 0.84), rgba(12, 6, 34, 0.72));
  box-shadow: 0 0 34px rgba(40, 215, 255, 0.08) inset;
}

.graph svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.graph line {
  stroke: rgba(40, 215, 255, 0.35);
  stroke-width: 1.2;
  stroke-dasharray: 7 8;
  animation: dash 7s linear infinite;
}

@keyframes dash {
  to { stroke-dashoffset: -90; }
}

.node {
  position: absolute;
  display: grid;
  place-items: center;
  min-width: 4.6rem;
  height: 2.35rem;
  padding: 0 0.55rem;
  border: 1px solid rgba(40, 215, 255, 0.38);
  border-radius: 999px;
  color: var(--text);
  font-size: 0.72rem;
  background: rgba(7, 14, 34, 0.86);
  box-shadow: 0 0 18px rgba(40, 215, 255, 0.18), 0 0 22px rgba(157, 108, 255, 0.12) inset;
  animation: floatNode 4.8s ease-in-out infinite;
}

.node.core {
  left: 50%;
  top: 48%;
  transform: translate(-50%, -50%);
  border-color: rgba(121, 255, 191, 0.5);
  box-shadow: 0 0 26px rgba(121, 255, 191, 0.22), 0 0 30px rgba(40, 215, 255, 0.14) inset;
}

.node.n1 { left: 8%; top: 18%; animation-delay: 0.1s; }
.node.n2 { right: 8%; top: 16%; animation-delay: 0.4s; }
.node.n3 { left: 11%; bottom: 16%; animation-delay: 0.7s; }
.node.n4 { right: 10%; bottom: 18%; animation-delay: 1s; }

.stTextInput input,
[data-testid="stChatInput"] textarea {
  color: var(--text);
  background: rgba(5, 12, 32, 0.88);
  border: 1px solid rgba(40, 215, 255, 0.28);
  border-radius: 14px;
  box-shadow: 0 0 22px rgba(40, 215, 255, 0.11);
}

[data-testid="stChatInput"] {
  border-radius: 18px;
}

.stSlider [data-testid="stTickBar"] {
  color: var(--muted);
}

.stButton > button {
  border-radius: 12px;
  border: 1px solid rgba(40, 215, 255, 0.28);
  background: linear-gradient(135deg, rgba(40, 215, 255, 0.22), rgba(157, 108, 255, 0.3));
  color: var(--text);
  font-weight: 800;
  box-shadow: 0 0 24px rgba(40, 215, 255, 0.13);
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

.stButton > button:hover {
  transform: translateY(-2px);
  border-color: rgba(40, 215, 255, 0.6);
  color: white;
  box-shadow: 0 0 32px rgba(40, 215, 255, 0.28);
}

[data-testid="stChatMessage"] {
  border: 1px solid rgba(104, 190, 255, 0.16);
  border-radius: 18px;
  background: rgba(7, 13, 32, 0.58);
  box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22);
  backdrop-filter: blur(18px);
}

[data-testid="stChatMessage"] p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: #dbe7ff;
}

h1, h2, h3 {
  color: var(--text);
}

hr {
  border-color: rgba(104, 190, 255, 0.16);
}

@media (max-width: 900px) {
  .topbar,
  .metric-grid {
    grid-template-columns: 1fr;
  }

  .status-orbit {
    min-width: 0;
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


def markdown_files() -> list[Path]:
    if not KNOWLEDGE_BASE_PATH.exists():
        return []
    return sorted(KNOWLEDGE_BASE_PATH.rglob("*.md"))


def count_markdown_files() -> int:
    return len(markdown_files())


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
    status = html.escape(ollama_status())
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="topbar">
            <div class="brand">
              <div class="brand-kicker">Local neural retrieval workspace</div>
              <h1>FIND_AI</h1>
              <p>Interrogate company knowledge through a local RAG system with luminous retrieval, source tracing, and open-source generation.</p>
            </div>
            <div class="status-orbit">
              <div class="status-row-mini">
                <span class="status-chip"><i class="dot green"></i>Ollama {status}</span>
                <span class="status-chip"><i class="dot"></i>Retrieving</span>
                <span class="status-chip"><i class="dot violet"></i>Generating</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(model: str, retrieval_k: int) -> None:
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric"><span>Indexed Chunks</span><strong>{get_collection().count()}</strong></div>
          <div class="metric"><span>Markdown Files</span><strong>{count_markdown_files()}</strong></div>
          <div class="metric"><span>Active Model</span><strong>{html.escape(model)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Neural retrieval depth: top {retrieval_k} chunks from the local ChromaDB index.")


def render_file_list() -> None:
    files = markdown_files()
    st.markdown('<div class="glass-panel"><p class="panel-title">Knowledge Base Files</p>', unsafe_allow_html=True)
    if not files:
        st.caption("No markdown files found.")
    for path in files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        section = path.parent.name
        st.markdown(
            f"""
            <div class="kb-file">
              <div>
                <strong>{html.escape(path.name)}</strong><br>
                <span>{html.escape(rel)}</span>
              </div>
              <span>{html.escape(section)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_sources(chunks) -> None:
    st.markdown('<div class="glass-panel"><p class="panel-title">Retrieved Sources</p>', unsafe_allow_html=True)
    if not chunks:
        st.caption("Ask a question to see retrieved source cards.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for chunk in chunks:
        source = html.escape(str(chunk.metadata.get("source", "unknown")))
        chunk_id = html.escape(str(chunk.metadata.get("chunk", 0)))
        relevance = html.escape(str(chunk.metadata.get("relevance", "n/a")))
        distance = html.escape(str(chunk.metadata.get("distance", "n/a")))
        doc_type = html.escape(str(chunk.metadata.get("type", "document")))
        preview = html.escape(chunk.page_content[:430])
        if len(chunk.page_content) > 430:
            preview += "..."
        st.markdown(
            f"""
            <div class="source-card">
              <strong>{source}</strong>
              <div class="source-meta">
                <span class="pill">type: {doc_type}</span>
                <span class="pill">chunk: {chunk_id}</span>
                <span class="pill">relevance: {relevance}</span>
                <span class="pill">distance: {distance}</span>
              </div>
              <p>{preview}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_graph(chunks) -> None:
    labels = ["Product", "Policies", "Employees", "Support"]
    if chunks:
        seen = []
        for chunk in chunks:
            label = str(chunk.metadata.get("type", "docs")).title()
            if label not in seen:
                seen.append(label)
        labels = (seen + labels)[:4]

    safe_labels = [html.escape(label) for label in labels]
    while len(safe_labels) < 4:
        safe_labels.append("Docs")

    st.markdown(
        f"""
        <div class="graph">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            <line x1="50" y1="50" x2="18" y2="24"></line>
            <line x1="50" y1="50" x2="82" y2="24"></line>
            <line x1="50" y1="50" x2="20" y2="78"></line>
            <line x1="50" y1="50" x2="80" y2="76"></line>
          </svg>
          <div class="node core">Query Core</div>
          <div class="node n1">{safe_labels[0]}</div>
          <div class="node n2">{safe_labels[1]}</div>
          <div class="node n3">{safe_labels[2]}</div>
          <div class="node n4">{safe_labels[3]}</div>
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
    st.title("Command Deck")
    model = st.text_input("Ollama model", value=OLLAMA_MODEL)
    retrieval_k = st.slider("Retrieved chunks", min_value=3, max_value=15, value=8)

    st.divider()
    st.write("Knowledge base")
    st.caption(str(KNOWLEDGE_BASE_PATH.relative_to(PROJECT_ROOT)))
    st.caption(f"{count_markdown_files()} markdown files detected")

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

chat_col, insight_col = st.columns([0.62, 0.38], gap="large")

with chat_col:
    st.markdown('<div class="glass-panel"><p class="panel-title">AI Conversation</p>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask the knowledge graph...   voice input ready")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Neural retrieval in progress..."):
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

with insight_col:
    render_graph(st.session_state.last_chunks)
    st.write("")
    render_sources(st.session_state.last_chunks)
    st.write("")
    render_file_list()
