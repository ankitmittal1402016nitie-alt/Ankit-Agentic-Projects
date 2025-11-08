"""
chat_core.py — The retrieval + LLM "brain" for your app.

A. Purpose
-------
Provide small, testable building blocks to:
1) Open your persisted Chroma vector store (built by ingest.py).
2) Build a retriever (top-k semantic search over your chunks).
3) Select an LLM backend (OpenAI or local Ollama) based on env/config.
4) Construct a Conversational Retrieval Chain for Q&A over your library.

B. How this fits
-------------
Streamlit (app.py) will import these helpers to:
- initialize the retriever + chain once,
- pass user questions + chat history,
- render answers + sources.

C. Environment variables (optional)
--------------------------------
OPENAI_API_KEY   -> needed if you choose OpenAI for LLM
LLM_PROVIDER     -> "openai" (default) or "ollama"
OLLAMA_MODEL     -> e.g. "llama3.1:8b" (must be pulled in Ollama)
CHROMA_DB_DIR    -> default "db"
EMBED_MODEL_NAME -> default "BAAI/bge-small-en-v1.5"
LLM_TEMPERATURE  -> default "0.2"
RETRIEVER_K      -> default "4"

You already used HuggingFace embeddings in ingest.py; we reuse the same for retrieval.
"""
# Annotations for better type hints (Python 3.7+). Notes for humans and tools that explain what kind of data is expected, making code easier to read, debug, and maintain.
from __future__ import annotations

# ---- logging bootstrap (FIRST lines in file) ----
from project_logger import init_logging, set_context, enable_global_tracing
init_logging(project_name="Test_Ankit_Project", log_dir="logs", level="DEBUG")
set_context(app_file=__file__)
# Trace your whole repo; add key libs too (see step 3)
enable_global_tracing(project_root=".")
# -----------------------------------------------


import os
from typing import List, Tuple, Dict, Any, Iterable
from operator import itemgetter
from project_logger import get_logger, log_execution
logger = get_logger("chat_core")

# --- Vector store + embeddings ---------------------------------------------

# --- a. Vector store (VECTORS -> PERSISTENCE/RETRIEVAL) -----------------------
# from langchain_community.vectorstores import Chroma -- Old import path that has been deprecated now. 
from langchain_chroma import Chroma    # pip install -U langchain-chroma

# --- b. Embeddings (TEXT -> VECTORS) ------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings


# LLM choices
from langchain_openai import ChatOpenAI                       # hosted and pip install -U langchain-openai   (if using OpenAI)
from langchain_community.chat_models import ChatOllama        # local, requires Ollama running server and pip install -U langchain-community

# Retrieval Q&A chain - LCEL building blocks
# --- minimal retrieval QA chain (LCEL style) ----------------------------
#from langchain.chains import ConversationalRetrievalChain   -- No longer supported module by Langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel 
from callbacks_logging import LoggerCallback

# -------------------------
# 1) Configuration helpers
# -------------------------

def get_settings() -> Dict[str, Any]:
    """Collect settings from env with sane defaults."""
    return {
        "db_dir": os.getenv("CHROMA_DB_DIR", "db"),
        "embed_model": os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        "llm_provider": os.getenv("LLM_PROVIDER", "openai").lower(),   # or "ollama"
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        "top_k": int(os.getenv("RETRIEVER_K", "4")),
    }


# ----------------------------------------
# 2) Load vector store + build a retriever
# ----------------------------------------

def load_vectorstore(db_dir: str, embed_model: str) -> Chroma:
    """
    Connect to your persisted Chroma DB using the same embeddings
    model you used during ingestion (so distances are consistent).
    """
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return Chroma(persist_directory=db_dir, embedding_function=embeddings)

def build_retriever(vs: Chroma, k: int = 4):
    """
    Turn your vector store into a retriever that returns the top-k
    most relevant chunks for a given query.
    """
    return vs.as_retriever(search_kwargs={"k": k})          # converts the vector store into a LangChain-compatible retriever object


# --------------------------
# 3) Choose your LLM backend
# --------------------------

def make_llm(provider: str, temperature: float, openai_model: str, ollama_model: str):
    """
    Create the chat model instance.
    - OpenAI requires OPENAI_API_KEY in your environment.
    - Ollama requires running the local Ollama server and pulling a model:
      e.g., `ollama pull llama3.1:8b`
    """
    if provider == "ollama":
        return ChatOllama(model=ollama_model, temperature=temperature)
    # default: OpenAI
    return ChatOpenAI(model=openai_model, temperature=temperature)


# -----------------------------------
# 4) Build a conversational RAG chain
# -----------------------------------

# This joins the retrieved chunks into a single context string with a visible separator (---). This becomes the {context} string injected into the prompt.
def _format_docs(docs: List) -> str:
    """Join retrieved chunks into a single context string."""
    return "\n\n---\n\n".join(d.page_content for d in docs)

# This builds the actual chain that helps answer questions
def build_chain(retriever, llm):
    """
    LCEL-based RAG chain that:
      - fetches docs via retriever
      - injects chat history + context into the prompt
      - returns both 'answer' and 'sources' (docs)

    Input to chain.invoke():
      { "question": str, "chat_history": list[tuple[str, str]] }
    Output:
      { "answer": str, "sources": list[Document] }
    """

    # Prompt with chat history support
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly, precise assistant. Use ONLY the provided context to answer. "
         "If the answer isn't in the context, say you don't know."),
        MessagesPlaceholder(variable_name="chat_history"),   # accepts list of (human, ai) tuples
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    
    # # Step 1: Retrieval function 
    # Retrieval step that ALSO carries sources forward. This steps run before building the chain with LLM.
    def _retrieve(inputs: Dict[str, Any]) -> Dict[str, Any]:
        q = inputs["question"]                          # the user question that has been entered currently
        history = inputs.get("chat_history", [])        # gets the chat history if any (optional). Context won't be drawn on previous history
        # docs = retriever.get_relevant_documents(q)    # Does 2 things. 1. - Embed the query automatically and 2. Calls the retriever to run vector similarity search, returning top-k docs.
        # Get the underlying vectorstore from retriever if present
        vs = getattr(retriever, "vectorstore", None)
        docs = []
        if vs is not None and hasattr(vs, "similarity_search_with_relevance_scores"):
            # scores in [0..1] (higher = more similar)
            scored = vs.similarity_search_with_relevance_scores(q, k=getattr(retriever, "search_kwargs", {}).get("k", 4))
            docs = []
            for doc, score in scored:
                try:
                    doc.metadata = dict(doc.metadata or {})
                    doc.metadata["relevance_score"] = float(score)
                except Exception:
                    pass
                docs.append(doc)
            logger.info("retrieval-scored", extra={"extra_data": {"query": q, "hits": len(docs),
                                                             "scores": [d.metadata.get("relevance_score") for d in docs]}})
        else:
                # LCEL retriever path
            docs = retriever.invoke(q)
            logger.info("retrieval-basic", extra={"extra_data": {"query": q, "hits": len(docs)}})
            # docs = retriever.invoke(q)                      # Newer method name as per latest Langchain versions.
        return {
                "question": q,
                "chat_history": history,
                "context": _format_docs(docs),          # the joined text for the prompt
                "sources": docs,                        # the original Document objects for citations/UI
            }

    # Wraps the _retrieve function as a Runnable
    retrieval = RunnableLambda(_retrieve)

    # Wiring the callback for logging
    #from callbacks_logging import LoggerCallback
    # Step 2: LLM generation 
    # Generate answer from prompt + LLM, and pass sources through
    generate = prompt | llm | StrOutputParser()

    # Step 3: Combine both
    # Produce both fields in the final output
    # retrieval runs first and produces a dict with context, sources, etc.
    # RunnableParallel runs two branches in parallel on that dict:
    # answer=generate: produces the final answer string
    # sources=itemgetter("sources"): just pulls through the original docs
    # chain = retrieval | RunnableParallel(answer=generate, sources=itemgetter("sources")) -- Old line before adding logging callback
    chain = (retrieval | RunnableParallel(answer=generate, sources=itemgetter("sources"))).with_config(
        callbacks=[LoggerCallback("lc-callback")] ) # Attach logging callback to the chain

    return chain

# -------------------------------------------------
# 5) High-level helper: answer a question with state
# -------------------------------------------------


def answer_question(chain, question: str, history: List[Tuple[str, str]]):
    """
    Call the chain with:
      - question: the user's current query
      - history: list of human, ai) tuples

    Returns: dict with keys "answer" and "sources"
    """
    coerced = _coerce_history(history)
    # Wiring the callback for logging
    # from callbacks_logging import LoggerCallback
    result = chain.invoke({"question": question, "chat_history": coerced}, config={"callbacks": [LoggerCallback("invoke")]})
    return {"answer": result["answer"], "sources": result["sources"]}

# Helper to coerce chat history into required format that MessagesPlaceholder("chat_history") expects such as [("human", u1), ("ai", a1), ("human", u2), ("ai", a2), ...    ]
def _coerce_history(pairs: Iterable[Tuple[str, str]]):
    """
    Convert [(user, ai), ...] into [("human", user), ("ai", ai), ...]
    Skips empty strings gracefully.
    """
    msgs = []
    for u, a in pairs:
        if u:
            msgs.append(("human", u))
        if a:
            msgs.append(("ai", a))
    return msgs