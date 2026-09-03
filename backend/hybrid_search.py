"""
Hybrid Search Engine for TraceFind.
Combines code-aware BM25 lexical search with vector semantic search using Reciprocal Rank Fusion (RRF).
Includes optional Cross-Encoder re-ranking with resilient fallbacks.
"""

import math
import re
from typing import Any, Dict, List, Optional

from backend.config import (
    BM25_WEIGHT,
    DEFAULT_TOP_K,
    RERANKER_MODEL_NAME,
    RRF_K,
    USE_CROSS_ENCODER,
    VECTOR_WEIGHT,
)
from backend.vector_store import TraceFindVectorStore

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


def tokenize_code(text: str) -> List[str]:
    """
    Code-aware tokenizer: splits camelCase, snake_case, PascalCase, dot-paths,
    while preserving original compound identifiers for exact matching.
    """
    if not text:
        return []

    # Basic regex word split
    words = re.findall(r"[A-Za-z0-9_]+|\S", text)
    tokens: List[str] = []

    for word in words:
        clean = word.lower().strip()
        if not clean:
            continue
        tokens.append(clean)

        # Split camelCase and PascalCase (e.g., 'getUserProfile' -> 'get', 'user', 'profile')
        sub_tokens = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).split()
        if len(sub_tokens) > 1:
            for st in sub_tokens:
                st_clean = st.lower().strip()
                if st_clean and st_clean != clean:
                    tokens.append(st_clean)

        # Split snake_case (e.g., 'user_auth_token' -> 'user', 'auth', 'token')
        if "_" in word:
            for part in word.split("_"):
                part_clean = part.lower().strip()
                if part_clean and part_clean != clean:
                    tokens.append(part_clean)

    return tokens


class TraceFindHybridSearch:
    _instance: Optional["TraceFindHybridSearch"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TraceFindHybridSearch, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.vector_store = TraceFindVectorStore()
        self.bm25: Optional[Any] = None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.reranker: Optional[Any] = None
        self._reranker_attempted = False
        self._initialized = True

    def build_or_refresh_bm25(self):
        """Builds or updates the BM25 index from current ChromaDB chunks."""
        if not BM25_AVAILABLE:
            print("[HybridSearch] Warning: rank_bm25 not installed. BM25 search disabled.")
            return

        print("[HybridSearch] Fetching chunks from vector store to build BM25 corpus...")
        self.corpus_chunks = self.vector_store.get_all_chunks()
        if not self.corpus_chunks:
            print("[HybridSearch] No chunks in vector store. BM25 index empty.")
            self.bm25 = None
            return

        tokenized_corpus = [
            tokenize_code(
                f"{c.get('name', '')} {c.get('file_path', '')} {c.get('content', '')}"
            )
            for c in self.corpus_chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[HybridSearch] BM25 index built with {len(self.corpus_chunks)} chunks.")

    def _ensure_reranker_loaded(self):
        """Loads the CrossEncoder model lazily if enabled."""
        if not self._reranker_attempted:
            self._reranker_attempted = True
            if USE_CROSS_ENCODER and CROSS_ENCODER_AVAILABLE:
                try:
                    print(f"[HybridSearch] Loading CrossEncoder model '{RERANKER_MODEL_NAME}'...")
                    self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
                    print("[HybridSearch] CrossEncoder loaded successfully.")
                except Exception as e:
                    print(f"[HybridSearch] Could not load CrossEncoder: {e}. Falling back to RRF fusion.")
                    self.reranker = None
            else:
                self.reranker = None

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executes BM25 keyword search over tokenized codebase chunks."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokens = tokenize_code(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        for rank, idx in enumerate(top_indices):
            raw_score = float(scores[idx])
            if raw_score <= 0.0:
                continue
            chunk = dict(self.corpus_chunks[idx])
            chunk["score"] = round(raw_score / max_score, 4)
            chunk["bm25_rank"] = rank + 1
            chunk["source"] = "bm25"
            results.append(chunk)

        return results

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search combining BM25 keyword matching and Vector semantic search
        via Reciprocal Rank Fusion (RRF), followed by optional Cross-Encoder re-ranking.
        """
        # Ensure BM25 is available
        if self.bm25 is None and self.vector_store.collection is not None:
            self.build_or_refresh_bm25()

        # Step 1: Retrieve candidate pool from Vector Search
        candidate_k = max(k * 3, 20)
        where_clause = {"chunk_type": filter_type} if filter_type else None
        vector_results = self.vector_store.semantic_search(
            query=query, n_results=candidate_k, where=where_clause
        )

        # Step 2: Retrieve candidate pool from BM25 Search
        bm25_results = self.search_bm25(query=query, top_k=candidate_k)

        # Step 3: Reciprocal Rank Fusion (RRF)
        fused_candidates: Dict[str, Dict[str, Any]] = {}

        # Process vector rankings
        for rank, item in enumerate(vector_results):
            cid = item["id"]
            rrf_score = VECTOR_WEIGHT / (RRF_K + (rank + 1))
            if cid not in fused_candidates:
                fused_candidates[cid] = dict(item)
                fused_candidates[cid]["rrf_score"] = rrf_score
                fused_candidates[cid]["sources"] = ["vector"]
            else:
                fused_candidates[cid]["rrf_score"] += rrf_score
                fused_candidates[cid]["sources"].append("vector")

        # Process BM25 rankings
        for rank, item in enumerate(bm25_results):
            cid = item["id"]
            rrf_score = BM25_WEIGHT / (RRF_K + (rank + 1))
            if cid not in fused_candidates:
                fused_candidates[cid] = dict(item)
                fused_candidates[cid]["rrf_score"] = rrf_score
                fused_candidates[cid]["sources"] = ["bm25"]
            else:
                fused_candidates[cid]["rrf_score"] += rrf_score
                if "bm25" not in fused_candidates[cid]["sources"]:
                    fused_candidates[cid]["sources"].append("bm25")

        candidate_list = list(fused_candidates.values())

        if not candidate_list:
            return []

        # Step 4: Re-ranking (Cross-Encoder or RRF normalization)
        self._ensure_reranker_loaded()

        if self.reranker is not None and len(candidate_list) > 0:
            try:
                pairs = [[query, f"{c.get('name', '')}\n{c.get('content', '')}"] for c in candidate_list]
                ce_scores = self.reranker.predict(pairs)
                for item, score in zip(candidate_list, ce_scores):
                    # Sigmoid transform for cross-encoder logits if needed
                    prob = 1.0 / (1.0 + math.exp(-float(score))) if isinstance(score, (float, int)) else float(score)
                    item["rerank_score"] = round(prob, 4)
                    item["score"] = item["rerank_score"]

                candidate_list.sort(key=lambda x: x["score"], reverse=True)
            except Exception as e:
                print(f"[HybridSearch] Cross-Encoder re-ranking failed: {e}. Falling back to RRF.")
                self._sort_by_rrf(candidate_list)
        else:
            self._sort_by_rrf(candidate_list)

        # Take top K results
        final_results = candidate_list[:k]
        for res in final_results:
            res["search_mode"] = "+".join(res.get("sources", ["hybrid"]))

        return final_results

    @staticmethod
    def _sort_by_rrf(candidates: List[Dict[str, Any]]):
        """Sorts candidates by normalized RRF score with keyword bonuses."""
        max_rrf = max(c["rrf_score"] for c in candidates) if candidates else 1.0
        for c in candidates:
            # Bonus if matched by both vector AND keyword
            bonus = 1.2 if len(c.get("sources", [])) > 1 else 1.0
            normalized = (c["rrf_score"] / max_rrf) * bonus
            c["score"] = round(min(1.0, normalized), 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)
