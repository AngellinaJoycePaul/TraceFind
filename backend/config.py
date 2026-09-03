"""
Configuration settings for TraceFind.
Provides central settings for paths, models, chunking, and API parameters.
"""

import os
from pathlib import Path
from typing import Dict, Set

# Base directory paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Database & Storage
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models"))).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "tracefind_knowledge")

# Ensure required directories exist
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1:8b")

# Ollama API Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# Server Settings
API_HOST = os.getenv("TRACEFIND_HOST", "0.0.0.0")
API_PORT = int(os.getenv("TRACEFIND_PORT", "8000"))

# Chunking Parameters
# Code chunks (Tree-sitter AST aware)
CODE_CHUNK_MAX_LINES = int(os.getenv("CODE_CHUNK_MAX_LINES", "80"))
CODE_CHUNK_MIN_LINES = int(os.getenv("CODE_CHUNK_MIN_LINES", "3"))

# Document chunks (Markdown, Notes, PDF, Text)
DOC_CHUNK_SIZE = int(os.getenv("DOC_CHUNK_SIZE", "600"))      # Characters per doc chunk
DOC_CHUNK_OVERLAP = int(os.getenv("DOC_CHUNK_OVERLAP", "100"))  # Characters overlap

# Search Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))  # Reciprocal Rank Fusion constant
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() in ("true", "1", "yes")

# Language & File Extension Support
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
}

DOC_EXTENSIONS: Set[str] = {
    ".md",
    ".txt",
    ".rst",
    ".pdf",
    ".json",
    ".yaml",
    ".yml",
}

# Directories & patterns to ignore during recursive indexing
IGNORE_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".vscode",
    ".idea",
    "dist",
    "build",
    "chroma_db",
    "models",
    ".pytest_cache",
    ".mypy_cache",
    ".coverage",
}

IGNORE_EXTENSIONS: Set[str] = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".pyc",
    ".pyo",
    ".pyd",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".zip",
    ".tar",
    ".gz",
    ".lock",
}
