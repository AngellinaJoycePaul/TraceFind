"""
FastAPI Server for TraceFind.
Provides RESTful endpoints for repository indexing, hybrid context search,
question answering with citations, and index diagnostics.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.chunker import CodeChunk, TraceFindChunker
from backend.config import (
    API_HOST,
    API_PORT,
    DEFAULT_TOP_K,
    IGNORE_DIRS,
    IGNORE_EXTENSIONS,
    LANGUAGE_MAP,
    DOC_EXTENSIONS,
)
from backend.hybrid_search import TraceFindHybridSearch
from backend.synthesizer import TraceFindSynthesizer
from backend.vector_store import TraceFindVectorStore

# -----------------------------------------------------------------------------
# Lifespan Event Handler
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load vector store and build BM25 index if chunks exist
    print("[TraceFind] Starting server...")
    vector_store = TraceFindVectorStore()
    search_engine = TraceFindHybridSearch()
    search_engine.build_or_refresh_bm25()
    print("[TraceFind] Server ready to accept requests.")
    yield
    print("[TraceFind] Server shutting down.")


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="TraceFind API",
    description="AI-Powered Context Navigator for Developers (Recursion Edition II - PS2)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for VS Code Extension and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons
chunker = TraceFindChunker()
vector_store = TraceFindVectorStore()
search_engine = TraceFindHybridSearch()
synthesizer = TraceFindSynthesizer()


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class IndexRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the repository directory to index.")
    force_reindex: bool = Field(False, description="Whether to clear existing index before indexing.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query or keyword / natural language description.")
    k: int = Field(DEFAULT_TOP_K, ge=1, le=50, description="Number of results to return.")
    filter_type: Optional[str] = Field(None, description="Filter by chunk type ('function', 'class', 'doc_section').")


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the codebase or architecture.")
    k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of context chunks to retrieve for synthesis.")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint indicating server and dependency status."""
    ollama_status = synthesizer.check_connection()
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "service": "TraceFind Context Navigator",
        "version": "1.0.0",
        "indexed_chunks": stats.get("chunk_count", 0),
        "unique_files": stats.get("unique_files", 0),
        "ollama": ollama_status,
    }


@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    """Returns database and indexing metrics."""
    return vector_store.get_stats()


@app.post("/index")
def index_repository(req: IndexRequest) -> Dict[str, Any]:
    """
    Recursively scans and indexes code, markdown, and documentation in the specified path.
    Chunks code with AST awareness and builds vector embeddings + BM25 index.
    """
    target_path = Path(req.path).resolve()
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory '{req.path}' does not exist or is not a directory.")

    if req.force_reindex:
        print("[Index] Force reindex requested. Clearing existing vector store...")
        vector_store.delete_collection()

    start_time = time.time()
    discovered_files: List[Path] = []

    # Traverse directory ignoring standard build and cache directories
    for root, dirs, files in os.walk(str(target_path)):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            if ext in IGNORE_EXTENSIONS or file.startswith("."):
                continue
            if ext in LANGUAGE_MAP or ext in DOC_EXTENSIONS:
                discovered_files.append(file_path)

    if not discovered_files:
        return {
            "status": "warning",
            "message": "No indexable files discovered in directory.",
            "files_scanned": 0,
            "chunks_created": 0,
        }

    total_chunks: List[CodeChunk] = []
    files_indexed_count = 0

    for f_path in discovered_files:
        try:
            chunks = chunker.chunk_file(str(f_path), repo_root=str(target_path))
            if chunks:
                total_chunks.extend(chunks)
                files_indexed_count += 1
        except Exception as e:
            print(f"[Index] Error chunking file {f_path}: {e}")

    # Add chunks into Vector Store
    chunks_stored = vector_store.add_chunks(total_chunks)

    # Refresh in-memory BM25 index with new corpus
    search_engine.build_or_refresh_bm25()

    elapsed = round(time.time() - start_time, 2)
    return {
        "status": "success",
        "repository_path": str(target_path),
        "files_scanned": len(discovered_files),
        "files_indexed": files_indexed_count,
        "chunks_created": len(total_chunks),
        "chunks_stored": chunks_stored,
        "time_taken_seconds": elapsed,
    }


@app.post("/search")
def search_codebase(req: SearchRequest) -> Dict[str, Any]:
    """
    Executes hybrid search (BM25 + Semantic Vector + RRF Re-ranking)
    returning the minimum relevant context needed for a task.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = search_engine.search(
        query=req.query,
        k=req.k,
        filter_type=req.filter_type,
    )

    return {
        "query": req.query,
        "results_count": len(results),
        "results": results,
    }


@app.post("/ask")
def ask_question(req: AskRequest) -> Dict[str, Any]:
    """
    Answers natural-language developer questions with exact file paths,
    line numbers, entity resolution, and cross-source contradiction detection.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Retrieve top-K relevant chunks via hybrid search
    retrieved = search_engine.search(query=req.question, k=req.k)

    # 2. Synthesize answer with Ollama Llama 3.1 8B
    synthesis = synthesizer.synthesize(question=req.question, retrieved_chunks=retrieved)

    return {
        "question": req.question,
        "answer": synthesis.get("answer", ""),
        "citations": synthesis.get("citations", []),
        "contradictions": synthesis.get("contradictions", []),
        "model": synthesis.get("model", ""),
        "context_used": retrieved,
    }


@app.post("/reset")
def reset_index() -> Dict[str, Any]:
    """Clears the entire index and resets search indices."""
    vector_store.delete_collection()
    search_engine.build_or_refresh_bm25()
    return {"status": "success", "message": "Index successfully cleared."}


# -----------------------------------------------------------------------------
# Direct Execution Helper
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
