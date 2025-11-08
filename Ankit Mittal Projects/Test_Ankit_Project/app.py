"""
app.py — Minimal Streamlit app to chat over your ingested PDFs/web pages.

A. Purpose
-------
Provide a friendly chat UI with:
- Question box + Ask button,
- Summary/Detailed toggle,
- Source snippets expander,
- Stateful chat history per session.

B. How it works
------------
1) On startup: load settings, vector store, retriever, and LLM.
2) User enters a question.
3) We prepend a small style hint (Summary vs Detailed) to the question.
4) We pass (question + history) to the ConversationalRetrievalChain.
5) We render the answer and show source chunks.

C. Run
---
streamlit run app.py
"""
# Annotations for better type hints (Python 3.7+). Notes for humans and tools that explain what kind of data is expected, making code easier to read, debug, and maintain.
from __future__ import annotations

# Enable Logs using Project Logger
from project_logger import init_logging, set_context, enable_global_tracing
init_logging(project_name="Test_Ankit_Project", log_dir="logs", level="INFO")
set_context(component="app", ui="streamlit")
enable_global_tracing(project_root=".")

import logging
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)
logging.getLogger("langchain_chroma").setLevel(logging.DEBUG)
logging.getLogger("chromadb").setLevel(logging.DEBUG)
logging.getLogger("sentence_transformers").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.DEBUG)  # shows request lines

import importlib, os
from project_logger import enable_global_tracing

import streamlit as st
from typing import List, Tuple

from chat_core import (
    get_settings,
    load_vectorstore,
    build_retriever,
    make_llm,
    build_chain,
    answer_question,
)
print("ankit1")
def _pkg_dir(name: str) -> str:
    m = importlib.import_module(name)
    return os.path.dirname(m.__file__)
print("ankit2")
project_root = os.path.abspath(".")
paths = [
    # project_root,
    # _pkg_dir("langchain"),                 # core
    # _pkg_dir("langchain_chroma"),          # chroma integration
    # _pkg_dir("chromadb"),                  # chroma engine
    # _pkg_dir("langchain_huggingface"),     # HF embeddings wrapper
    # _pkg_dir("sentence_transformers"),     # actual embedding model code
]
print("ankit3")
# enable_global_tracing(include_paths=paths)

# -----------------
# 0) Page settings
# -----------------
st.set_page_config(page_title="Ask Your PDFs", page_icon="📚", layout="wide")
st.title("📚 Ask Your PDFs & Web Pages")

# -----------------------
# 1) Initialize everything
# -----------------------

# Saves heavy resources across reruns for streamlit apps
@st.cache_resource(show_spinner=False)
# It stores heavy resources to speed up reruns. On rerun, the cached objects are returned instantly if the inputs/config didn’t change
def bootstrap():
    """
    Cache heavy resources across reruns:
    - vector store + retriever
    - LLM and chain
    """
    # Here cfg and chain are local variables, but we return them to be stored as cached resources
    cfg = get_settings()  
    # Loads the vector store from disk
    vs = load_vectorstore(cfg["db_dir"], cfg["embed_model"])
    # Builds the retriever object from the vector store with top-k setting
    retriever = build_retriever(vs, cfg["top_k"])
    llm = make_llm(
        provider=cfg["llm_provider"],
        temperature=cfg["temperature"],
        openai_model=cfg["openai_model"],
        ollama_model=cfg["ollama_model"],
    )
    chain = build_chain(retriever, llm)
    return cfg, chain

# cfg and chain are kind of global for the app lifetime and stores bootstrap value across reruns
cfg, chain = bootstrap()

# ---------------------------------
# 2) Session-scoped chat transcript
# ---------------------------------

# Checks if history exists in session state, else initializes it to retain the user's context across interactions
if "history" not in st.session_state:
    # store as list of (user_text, ai_text)
    st.session_state.history: List[Tuple[str, str]] = []

# ---------------------
# 3) Controls + inputs
# ---------------------
with st.sidebar:
    st.header("Settings")
    st.write(f"LLM provider: **{cfg['llm_provider']}**")
    st.write(f"Retriever k: **{cfg['top_k']}**")
    st.caption(f"DB: `{cfg['db_dir']}` · Embeddings: `{cfg['embed_model']}`")

style = st.radio("Response style", ["Summary", "Detailed"], horizontal=True)
question = st.text_input("Ask a question about your documents:")

# ---------------
# 4) Ask + render
# ---------------
if st.button("Ask") and question.strip():
  
    # Build a light style hint to bias the LLM without over-engineering prompts.
    style_hint = (
        "Give a brief, 3–5 sentence summary."
        if style == "Summary"
        else "Provide a detailed, step-by-step explanation with brief quotes from sources when useful."
    )
    styled_question = f"[User prefers: {style}] {style_hint}\n\nQuestion: {question.strip()}"

    # Convert chat history to required (human, ai) tuples.Conversion of lc_history not needed as _coerce_history is called inside answer_question   
    # lc_history = [(u, a) for (u, a) in st.session_state.history]

    with st.spinner("Thinking..."):
        # LCEL chain in chat_core returns {"answer": str, "sources": list[Document]}
        result = answer_question(chain, styled_question, st.session_state.history)  # -- no need to convert history here again to lc_history

    # Append to session history
    st.session_state.history.append((question.strip(), result["answer"]))

    # Render answer
    st.markdown("### Answer")
    st.write(result["answer"])

    # Render sources (Documents from retriever)
    sources = result["sources"]
    if sources:
        with st.expander("Show cited source chunks"):
            for i, d in enumerate(sources, 1):
                meta = d.metadata
                source = meta.get("source", "unknown")
                page = meta.get("page", "n/a")
                st.markdown(f"**{i}.** {source}  (p. {page})")
                st.code((d.page_content or "")[:800] + ("…" if len(d.page_content) > 800 else ""))

# ---------------
# 5) Chat transcript
# ---------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("Chat Transcript")
    for i, (u, a) in enumerate(st.session_state.history, 1):
        st.markdown(f"**You:** {u}")
        st.markdown(f"**AI:** {a}")
        if i != len(st.session_state.history):
            st.markdown("---")
