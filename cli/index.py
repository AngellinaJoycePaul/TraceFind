"""
TraceFind CLI Indexing Tool.
Recursively indexes source repositories and documents into TraceFind.
Supports both direct local indexing and triggering via the running FastAPI backend.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import requests
from backend.chunker import CodeChunk, TraceFindChunker
from backend.config import (
    DOC_EXTENSIONS,
    IGNORE_DIRS,
    IGNORE_EXTENSIONS,
    LANGUAGE_MAP,
)
from backend.hybrid_search import TraceFindHybridSearch
from backend.vector_store import TraceFindVectorStore


def find_indexable_files(target_dir: Path) -> List[Path]:
    """Recursively collects indexable code and document files."""
    files_to_index: List[Path] = []

    for root, dirs, files in os.walk(str(target_dir)):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

        for file in files:
            p = Path(root) / file
            ext = p.suffix.lower()
            if ext in IGNORE_EXTENSIONS or file.startswith("."):
                continue
            if ext in LANGUAGE_MAP or ext in DOC_EXTENSIONS:
                files_to_index.append(p)

    return files_to_index


def index_via_api(target_path: str, api_url: str, force: bool) -> bool:
    """Attempts to index through the running FastAPI backend service."""
    url = f"{api_url.rstrip('/')}/index"
    payload = {"path": target_path, "force_reindex": force}

    print(f"Connecting to TraceFind backend at {api_url}...")
    try:
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            data = response.json()
            print("\n" + "=" * 60)
            print("  TraceFind Indexing Complete (via Backend Service)")
            print("=" * 60)
            print(f"  Repository:     {data.get('repository_path')}")
            print(f"  Files Scanned:  {data.get('files_scanned')}")
            print(f"  Files Indexed:  {data.get('files_indexed')}")
            print(f"  Chunks Created: {data.get('chunks_created')}")
            print(f"  Chunks Stored:  {data.get('chunks_stored')}")
            print(f"  Time Elapsed:   {data.get('time_taken_seconds')}s")
            print("=" * 60 + "\n")
            return True
        else:
            print(f"Backend API returned error {response.status_code}: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Notice: TraceFind backend is not running at {api_url}.")
        print("Falling back to direct local indexing...\n")
        return False
    except Exception as e:
        print(f"Error communicating with backend: {e}. Falling back to direct indexing.\n")
        return False


def index_direct(target_path: Path, force: bool):
    """Executes indexing directly in-process without needing a running server."""
    start_time = time.time()
    vector_store = TraceFindVectorStore()
    chunker = TraceFindChunker()

    if force:
        print("Resetting existing vector store collection...")
        vector_store.delete_collection()

    print(f"Scanning directory: {target_path} ...")
    files = find_indexable_files(target_path)

    if not files:
        print("No supported code or documentation files found to index.")
        return

    print(f"Found {len(files)} candidate files. Parsing and chunking...")

    total_chunks: List[CodeChunk] = []
    file_iterator = tqdm(files, desc="Chunking files", unit="file") if TQDM_AVAILABLE else files

    for f in file_iterator:
        try:
            chunks = chunker.chunk_file(str(f), repo_root=str(target_path))
            if chunks:
                total_chunks.extend(chunks)
        except Exception as e:
            print(f"\nWarning: Could not process {f}: {e}")

    print(f"\nExtracted {len(total_chunks)} AST and semantic chunks.")
    print("Generating embeddings and writing to ChromaDB...")

    # Embed and write to ChromaDB
    chunks_stored = vector_store.add_chunks(total_chunks, batch_size=200)

    # Rebuild BM25 index
    print("Building BM25 keyword index...")
    search_engine = TraceFindHybridSearch()
    search_engine.build_or_refresh_bm25()

    elapsed = round(time.time() - start_time, 2)

    print("\n" + "=" * 60)
    print("  TraceFind Indexing Complete (Direct Mode)")
    print("=" * 60)
    print(f"  Repository:     {target_path}")
    print(f"  Files Processed:{len(files)}")
    print(f"  Chunks Stored:  {chunks_stored}")
    print(f"  Database Path:  {vector_store.persist_dir}")
    print(f"  Time Taken:     {elapsed} seconds")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="TraceFind Context Navigator: Index codebases and documents for AI retrieval."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the repository or directory to index.",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8000",
        help="TraceFind backend API URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Force direct in-process indexing instead of calling the backend API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing from scratch, clearing previously stored chunks.",
    )

    args = parser.parse_args()
    target = Path(args.path).resolve()

    if not target.exists():
        print(f"Error: Path '{args.path}' does not exist.")
        sys.exit(1)

    if not target.is_dir():
        print(f"Error: Path '{args.path}' is not a directory.")
        sys.exit(1)

    # Attempt API indexing if not explicitly forced direct
    indexed = False
    if not args.direct:
        indexed = index_via_api(str(target), args.api, args.force)

    if not indexed:
        index_direct(target, args.force)


if __name__ == "__main__":
    main()
