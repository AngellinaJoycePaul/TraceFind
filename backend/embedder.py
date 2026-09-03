"""
SentenceTransformer embedding generator for TraceFind.
Uses all-MiniLM-L6-v2 with batch processing, progress reporting, and hash-based caching.
"""

import hashlib
import os
from typing import Any, Dict, List, Optional
import numpy as np

from backend.config import EMBEDDING_MODEL_NAME, MODELS_DIR

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TraceFindEmbedder:
    _instance: Optional["TraceFindEmbedder"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TraceFindEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if getattr(self, "_initialized", False):
            return
        self.model_name = model_name
        self.model: Optional[Any] = None
        self._cache: Dict[str, List[float]] = {}
        self._initialized = True

    def _ensure_model_loaded(self):
        """Lazy loads the SentenceTransformer model on first invocation."""
        if self.model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                print("[Embedder] Warning: sentence-transformers not installed. Using pseudo-embeddings.")
                return

            print(f"[Embedder] Loading SentenceTransformer model '{self.model_name}'...")
            try:
                # Set cache directory to local models folder
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR)
                self.model = SentenceTransformer(self.model_name, cache_folder=str(MODELS_DIR))
                print(f"[Embedder] Model '{self.model_name}' loaded successfully.")
            except Exception as e:
                print(f"[Embedder] Error loading model '{self.model_name}': {e}. Falling back to CPU/basic.")
                self.model = SentenceTransformer(self.model_name)

    @staticmethod
    def _hash_text(text: str) -> str:
        """Returns SHA256 hash of text for caching."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embeds a list of texts in batches, checking cache to avoid recomputing.
        """
        if not texts:
            return []

        self._ensure_model_loaded()

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache
        for idx, text in enumerate(texts):
            h = self._hash_text(text)
            if h in self._cache:
                results[idx] = self._cache[h]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        # Compute embeddings for uncached texts
        if uncached_texts:
            if self.model is not None:
                # Use progress bar if batch size is large or requested
                use_pbar = show_progress and len(uncached_texts) > batch_size
                embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=use_pbar,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                embeddings_list = [emb.tolist() for emb in embeddings]
            else:
                # Fallback deterministic pseudo-embedding (384-dim matching all-MiniLM-L6-v2)
                embeddings_list = [self._pseudo_embedding(t) for t in uncached_texts]

            for orig_idx, text, emb in zip(uncached_indices, uncached_texts, embeddings_list):
                h = self._hash_text(text)
                self._cache[h] = emb
                results[orig_idx] = emb

        return [r for r in results if r is not None]

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string."""
        return self.embed_texts([query], show_progress=False)[0]

    def _pseudo_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generates deterministic pseudo-embedding when dependencies are absent."""
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
