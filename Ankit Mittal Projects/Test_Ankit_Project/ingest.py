"""
ingest.py — Build or update a local vector database (Chroma) from PDFs and web pages.

A. GOAL
----
Turn your PDFs (and optionally web pages) into small text chunks, compute vectors
(embeddings) using HuggingFace, and store them in a persistent Chroma DB so your
chat app can retrieve the right context at answer time (RAG).

B. WHAT THIS SCRIPT DOES
---------------------
1) Load:       Read PDFs from a folder and/or load web pages by URL.
2) Split:      Break long texts into smaller overlapping chunks for better recall.
3) Embed:      Convert chunks to high-dimensional vectors with HuggingFace models.
4) Store:      Upsert into a persistent Chroma DB (on-disk at ./db by default).

C. SAFE TO RUN MULTIPLE TIMES
--------------------------
We compute a stable ID for each chunk (using SHA1) so re-running ingestion
won't duplicate the same chunk. New/changed files will be added naturally.

D. USAGE
-----
# Basic: ingest all PDFs in ./data into ./db
python ingest.py

# Custom folders / model
python ingest.py --pdf-folder data --db-folder db --model BAAI/bge-small-en-v1.5

# Also ingest web pages:
python ingest.py --urls https://example.com/a https://example.com/b

# See all options:
python ingest.py --help
"""

# Annotations for better type hints (Python 3.7+). Notes for humans and tools that explain what kind of data is expected, making code easier to read, debug, and maintain.
from __future__ import annotations
import importlib, os, time, glob, argparse
from typing import Iterable, List, Sequence

# Hashlib for stable chunk IDs. Takes any data (like a password, file, or text) and converts it into a unique fixed-length code (called a hash).
import hashlib

# --- LangChain loaders & utils (DOCUMENTS IN) -------------------------------
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# from langchain.schema import Document -- Schema no longer supported module
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Embeddings (TEXT -> VECTORS) ------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
# If using OpenAI LLM later: 
# from langchain_openai import ChatOpenAI
# For local LLMs via Ollama, e.g.: 
# from langchain_community.chat_models import ChatOllama

# --- Vector store (VECTORS -> PERSISTENCE/RETRIEVAL) -----------------------
# from langchain_community.vectorstores import Chroma -- Old import path that has been deprecated now. 
from langchain_chroma import Chroma    # pip install -U langchain-chroma

# >>> LOGGING ADDED
from project_logger import get_logger, log_execution, log_block
logger = get_logger("ingest")
# <<< LOGGING ADDED


#>>>>------------------------------ Code Starts -------------------------->>>

# =========================
# 1) CONFIG & CLI ARGUMENTS
# =========================
# Parse command-line arguments, takes input from user when they run the script to customize behavior and also adds default value when not provided by user
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for flexible ingestion."""
    parser = argparse.ArgumentParser(description="Ingest PDFs/Web pages into Chroma.")

    parser.add_argument(
        "--pdf-folder",
        default="data",
        help="Folder containing PDF files to ingest (default: data).",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="One or more web page URLs to ingest (default: none).",
    )
    parser.add_argument(
        "--db-folder",
        default="db",
        help="Folder where Chroma will persist the vector DB (default: db).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embedding model name (default: BAAI/bge-small-en-v1.5).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Characters per chunk before embedding (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Overlap between consecutive chunks (default: 150).",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Filename pattern for PDFs (default: *.pdf).",
    )
    return parser.parse_args()


# =========================
# 2) LOADING SOURCE CONTENT
# =========================
# Load PDFs from a folder into Document objects
def load_pdfs(pdf_folder: str, pattern: str = "*.pdf") -> List[Document]:
    """
    Load all PDFs in a folder into LangChain Document objects.

    Why use a loader?
      PyPDFLoader handles reading, page splitting, and metadata (like page numbers).
    """
    docs: List[Document] = []
    for path in glob.glob(os.path.join(pdf_folder, pattern)):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs

# Load web pages from URLs into Document objects
def load_webpages(urls: Sequence[str]) -> List[Document]:
    """
    Load one or more web pages (HTML -> text) into Document objects.

    WebBaseLoader takes care of fetching, parsing, and basic cleaning of HTML.
    """
    if not urls:
        return []
    loader = WebBaseLoader(urls)
    return loader.load()


# =================================
# 3) SPLITTING INTO RAG i.e. CHUNKS
# =================================
@log_execution(purpose="split-docs")
def split_documents(
    docs: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Split raw Documents into smaller overlapping chunks.

    Why split?
      - Improves retrieval accuracy (smaller targets).
      - Overlap preserves context across boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# ==================================
# 4) EMBEDDINGS & VECTORSTORE HELPERS
# ==================================
@log_execution(purpose="build-vectorstore")
# Creates embeddings for documents
def make_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """
    Create a HuggingFace embeddings object.

    You can swap models later (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
    Larger models = better quality, slower speed. Start with BGE small.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

# Creates vectorbase (Chroma) with persistence
# Chroma = AI’s searchable, memory-like database for text and knowledge
def create_or_load_chroma(db_folder: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Create (or connect to) a persistent Chroma DB.

    - If db_folder exists, Chroma will re-use it.
    - If it doesn't exist, Chroma will create it on first write.
    """
    return Chroma(persist_directory=db_folder, embedding_function=embeddings)

# hashlib to create unique chunk IDs and avoid duplicates. This helps with idempotent ingestion.
def _chunk_id(doc: Document, index: int) -> str:
    """
    Compute a stable ID for a chunk to avoid duplicates on re-ingest.

    We combine:
      - source (file path or URL)
      - page (if present)
      - index (position within split sequence)
      - text content
    Then hash to a short fixed string.
    """
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", "na"))
    payload = f"{source}||{page}||{index}||{doc.page_content}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

# Assign stable IDs and upsert (Update and Insert in DB Language Terms) chunks into Chroma DB
def upsert_chunks(vs: Chroma, chunks: Sequence[Document]) -> int:
    """
    Upsert Document chunks into Chroma with deterministic IDs.

    Why upsert with IDs?
      - Running ingestion again won't create duplicates (idempotent).
      - If a source file changes, its new chunks get new IDs and are added.
    """
    if not chunks:
        return 0

    # Build parallel lists for Chroma's add_documents
    ids: List[str] = []
    docs: List[Document] = []
    for i, d in enumerate(chunks):
        ids.append(_chunk_id(d, i))
        docs.append(d)

    # Chroma will auto-persist when using a persist_directory (>=0.4.x).
    # Some versions allow vs.persist(); others warn it's no longer needed.
    # We'll rely on auto-persist to avoid the deprecation warning.
    vs.add_documents(documents=docs, ids=ids)
    return len(docs)


# ===========================
# 5) ORCHESTRATION (MAIN FLOW)
# ===========================
def main() -> None:
    args = parse_args()

    print("🔧 Settings")
    print(f"  PDF folder      : {args.pdf_folder}")
    print(f"  URLs            : {len(args.urls)} provided")
    print(f"  DB folder       : {args.db_folder}")
    print(f"  Embedding model : {args.model}")
    print(f"  Chunk size/over : {args.chunk_size}/{args.chunk_overlap}")
    print("-" * 60)

    # 1) Load data sources
    pdf_docs = load_pdfs(args.pdf_folder, pattern=args.glob)   # - Loads all PDF from folder in LangChain Document Objects
    web_docs = load_webpages(args.urls)
    print(f"📥 Loaded {len(pdf_docs)} PDF docs, {len(web_docs)} web docs")

    # 2) Split to chunks
    all_docs = pdf_docs + web_docs
    chunks = split_documents(all_docs, args.chunk_size, args.chunk_overlap)
    total_chars = sum(len(d.page_content or "") for d in chunks)
    avg = int(total_chars / max(1, len(chunks)))
    logger.info("chunk-stats", extra={"extra_data": {
        "chunks": len(chunks), "total_chars": total_chars,
        "avg_chars": avg, "chunk_size": args.chunk_size, "overlap": args.chunk_overlap
    }})
    print(f"✂️  Split into {len(chunks)} chunks")

    # 3) Prepare embeddings + vector store
    t0 = time.time()
    embeddings = make_embeddings(args.model)
     # probe dim - This is to record time taken to embed the docs
    dim = len(embeddings.embed_query("dimension probe"))
    with log_block("chroma-create"):
        vectordb = create_or_load_chroma(args.db_folder, embeddings)
    elapsed = int((time.time() - t0)*1000)

    # Assess the size of the vector DB before upsert
    try: 
        size_bytes = 0
        for root, _, files in os.walk(args.db_folder):
            for f in files:
                size_bytes += os.path.getsize(os.path.join(root, f))
    except Exception as e:
        print("chroma-disk-error", extra={"extra_data": {"error": str(e)}})
        pass
    print(f"🗄️  Chroma vector DB ready at '{args.db_folder}'")

    # 4) Upsert chunks (idempotent)
    added = upsert_chunks(vectordb, chunks)
    print(f"🧱 Upserted {added} chunks into Chroma at '{args.db_folder}'")

    # 5) (Optional) manual persist:
    # For Chroma >= 0.4.x, data is auto-persisted when persist_directory is set.
    # Calling persist() may show a deprecation warning. Uncomment if your version needs it.
    # try:
    #     vectordb.persist()
    # except Exception:
    #     pass

    print("✅ Ingestion complete. You can now run your chat app.")


if __name__ == "__main__":
    main()
