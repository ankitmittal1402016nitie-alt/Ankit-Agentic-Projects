"""
Extended Environment & RAG Health Check
--------------------------------------
What this script verifies:
1) Python version and venv status
2) Required packages import successfully
3) OPENAI_API_KEY is set and usable
4) Embeddings can be created (OpenAIEmbeddings) or (HuggingFaceEmbeddings)
5) ChromaDB can be created, populated, queried (mini end-to-end RAG smoke test)

How to run:
    python check_env_and_rag_health.py

Expected outcome:
    - All checks show ✅
    - A test query returns a relevant chunk from a tiny temp knowledge base

Note:
    This script creates a temporary Chroma directory "./db_healthcheck" and removes it afterwards.
"""

import os
import sys
import shutil
import importlib
from contextlib import suppress 

REQUIRED_PY_VERSION_MAJOR = 3
REQUIRED_PY_VERSION_MINOR = 11

REQUIRED_PACKAGES = [
    "streamlit",
    "langchain",
    "langchain_community",
    "langchain_openai",
    "langchain_huggingface",
    "sentence_transformers",
    "langchain_text_splitters",
    "langchain_chroma",
    "langchain_core",
    "dotenv",
    "chromadb",
    "pypdf",
    "tiktoken",
]

# Verify Python version and venv is active or not
def check_python_and_venv():
    print("🔍 Checking Python & venv...")
    ver = sys.version_info
    print(f"   Python: {ver.major}.{ver.minor}.{ver.micro}")
    if ver.major == REQUIRED_PY_VERSION_MAJOR and ver.minor == REQUIRED_PY_VERSION_MINOR:
        print("   ✅ Python version OK (3.11.x)")
    else:
        print("   ⚠️  Warning: Recommended Python 3.11.x")

    # Heuristic venv detection
    venv_hint = (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or \
                (hasattr(sys, 'real_prefix')) or \
                ('VIRTUAL_ENV' in os.environ)
    if venv_hint:
        print(f"   ✅ Virtual environment active: {sys.prefix}")
    else:
        print("   ⚠️  Not running inside a virtual environment (.venv). Activate it before installing packages.")
    print()

# Verify required packages can be imported/installed with version details
def check_imports():
    print("🔍 Checking package imports...")
    all_ok = True
    versions = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(pkg)
            versions[pkg] = getattr(m, "__version__", "Unknown")
            print(f"   ✅ {pkg:<22} (version: {versions[pkg]})")
        except ModuleNotFoundError:
            print(f"   ❌ {pkg:<22} NOT INSTALLED — run: pip install {pkg}")
            all_ok = False
        except Exception as e:
            print(f"   ⚠️  {pkg:<22} import error: {e}")
            all_ok = False
    print()
    return all_ok

# Verify OPENAI_API_KEY is set and embeddings can be created
def check_openai_key_and_embeddings():
    print("🔍 Checking OpenAI key & Model embeddings...")
    # Set open API Key to "Key" variable removing any space in prefix and suffix
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("   ❌ OPENAI_API_KEY not set. Set it and re-run this script.")
        print("      Windows (PowerShell):  setx OPENAI_API_KEY \"sk-...\"  (then restart terminal)")
        print("      macOS/Linux:           export OPENAI_API_KEY=\"sk-...\"")
        print()
        return False

    try:
        # from langchain_openai import OpenAIEmbeddings # <- remove/disable

        # switch to local or free-tier embeddings for testing such as 
        # HuggingFaceEmbeddings, Cohere, VoyageAI or Jina
        # or OllamaEmbeddings this is local embeddings model if you have ollama installed
        from langchain_huggingface import HuggingFaceEmbeddings
        # If using OpenAI LLM later: 
        # from langchain_openai import ChatOpenAI
        # For local LLMs via Ollama, e.g.: 
        # from langchain_community.chat_models import ChatOllama

        # emb = OpenAIEmbeddings()  # <- remove/disable
        # emb = HuggingFaceEmbeddings() -- For default Model

        # Good starter model: all-MiniLM-L6-v2 (small, fast) or bge-small-en-v1.5 (stronger)
        # or: emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        vec = emb.embed_query("hello world")  # tiny inference call
        # check if Vec is of class List and has content
        if isinstance(vec, list) and len(vec) > 0:
            print("   ✅ Embeddings created successfully")
            print(f"   • Example embedding length: {len(vec)}")
            print()
            return True
        else:
            print("   ❌ Embeddings returned unexpected format")
            print()
            return False
    except Exception as e:
        print(f"   ❌ Failed to create embeddings: {e}")
        print()
        return False

# Verify ChromaDB can be created, populated, queried
def check_chroma_round_trip():
    print("🔍 ChromaDB round-trip (create → add → query) ...")
    try:
        from langchain_community.vectorstores import Chroma
        # from langchain_openai import OpenAIEmbeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        # from langchain.schema import Document -- Schema no longer supported module
        from langchain_core.documents import Document

        print("Ankit Check ChromaDB")

        DB_DIR = "./db_healthcheck"
        # Clean any previous run
        with suppress(Exception):
            shutil.rmtree(DB_DIR)
    
        texts = [
            "LangChain helps build LLM applications with components like prompts, chains, and agents.",
            "Chroma is a local vector database that stores embeddings and supports similarity search.",
            "Streamlit is a Python library for building simple web apps, great for AI demos.",
        ]
        docs = [Document(page_content=t, metadata={"source": f"sample_{i}"}) for i, t in enumerate(texts, 1)]

        # embeddings = OpenAIEmbeddings() # <- remove/disable Open AI Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=DB_DIR)
        # Chroma 0.4.0, the library automatically persists your database whenever you use persist_directory=.... So that explicit vs.persist() call is now unnecessary
        # vs.persist()   # ← safe to delete now

        query = "Which tool can I use as a local vector store for embeddings?"
        hits = vs.similarity_search(query, k=2)

        if not hits:
            print("   ❌ Similarity search returned no results")
            return False

        print("   ✅ Similarity search returned results:")
        for i, d in enumerate(hits, 1):
            preview = d.page_content[:90].replace("\\n", " ")
            print(f"     {i}. {preview}... (source: {d.metadata.get('source')})")

        # Cleanup
        with suppress(Exception):
            shutil.rmtree(DB_DIR)

        print("   ✅ ChromaDB round-trip successful")
        print()
        return True

    except Exception as e:
        print(f"   ❌ ChromaDB test failed: {e}")
        print("      Tip: Ensure chromadb installed and your embeddings are working (needs OPENAI_API_KEY).")
        print()
        return False

def main():
    check_python_and_venv()
    imports_ok = check_imports()
    if not imports_ok:
        print("🚫 Fix import errors above and re-run this script.")
        return

    emb_ok = check_openai_key_and_embeddings()
    if not emb_ok:
        print("🚫 Fix OpenAI key/embeddings before proceeding.")
        return

    chroma_ok = check_chroma_round_trip()
    if not chroma_ok:
        print("🚫 ChromaDB round-trip failed. See tips above.")
        return

    print("🎉 All checks passed! Your environment is ready for RAG + Chat over PDFs/Web.")
    print("   Next: run your ingestion and Streamlit app.\n")

if __name__ == "__main__":
    main()
