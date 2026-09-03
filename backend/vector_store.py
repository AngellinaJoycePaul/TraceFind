"""
ChromaDB Vector Store implementation for TraceFind.
Manages persistent embeddings, document chunks, metadata queries, and statistics.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from backend.chunker import CodeChunk
from backend.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from backend.embedder import TraceFindEmbedder

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class TraceFindVectorStore:
    _instance: Optional["TraceFindVectorStore"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TraceFindVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_dir: str = str(CHROMA_PERSIST_DIR), collection_name: str = COLLECTION_NAME):
        if getattr(self, "_initialized", False):
            return
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = TraceFindEmbedder()
        self.client = None
        self.collection = None
        self._init_db()
        self._initialized = True

    def _init_db(self):
        """Initializes persistent ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            print("[VectorStore] Warning: chromadb library not installed. Vector store disabled.")
            return

        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[VectorStore] Connected to ChromaDB at '{self.persist_dir}'. Collection: '{self.collection_name}'")
        except Exception as e:
            print(f"[VectorStore] Error initializing ChromaDB: {e}")

    def add_chunks(self, chunks: List[CodeChunk], batch_size: int = 250) -> int:
        """
        Embeds and upserts code chunks into ChromaDB in safe batch increments.
        Returns the number of successfully added chunks.
        """
        if not chunks or self.collection is None:
            return 0

        total_added = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [
                {
                    "file_path": c.file_path,
                    "file_name": c.file_name,
                    "language": c.language,
                    "chunk_type": c.chunk_type,
                    "name": c.name,
                    "start_line": int(c.start_line),
                    "end_line": int(c.end_line),
                }
                for c in batch
            ]

            # Generate embeddings via TraceFindEmbedder
            embeddings = self.embedder.embed_texts(documents, show_progress=False)

            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_added += len(batch)
            except Exception as e:
                print(f"[VectorStore] Error adding batch {i} to {i + len(batch)}: {e}")

        return total_added

    def semantic_search(
        self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search using cosine similarity.
        Returns matched chunks ordered by relevance score.
        """
        if self.collection is None:
            return []

        try:
            query_embedding = self.embedder.embed_query(query)
            query_args: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_args["where"] = where

            results = self.collection.query(**query_args)

            formatted_results: List[Dict[str, Any]] = []
            if results and results["ids"] and len(results["ids"][0]) > 0:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]

                for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
                    # Chroma returns cosine distance (0 to 2 for cosine space).
                    # Similarity = 1.0 - (dist / 2.0) or 1.0 - dist
                    similarity = max(0.0, 1.0 - float(dist))
                    formatted_results.append({
                        "id": chunk_id,
                        "content": doc,
                        "metadata": meta,
                        "file_path": meta.get("file_path", ""),
                        "file_name": meta.get("file_name", ""),
                        "language": meta.get("language", ""),
                        "chunk_type": meta.get("chunk_type", ""),
                        "name": meta.get("name", ""),
                        "start_line": int(meta.get("start_line", 1)),
                        "end_line": int(meta.get("end_line", 1)),
                        "score": round(similarity, 4),
                        "source": "vector",
                    })

            return formatted_results

        except Exception as e:
            print(f"[VectorStore] Error in semantic search: {e}")
            return []

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Retrieves all documents and metadatas currently in the collection.
        Used to construct or refresh the in-memory BM25 index.
        """
        if self.collection is None:
            return []

        try:
            total_count = self.collection.count()
            if total_count == 0:
                return []

            # Retrieve in full or batch
            data = self.collection.get(include=["documents", "metadatas"])
            chunks: List[Dict[str, Any]] = []

            for chunk_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                chunks.append({
                    "id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "file_path": meta.get("file_path", ""),
                    "file_name": meta.get("file_name", ""),
                    "language": meta.get("language", ""),
                    "chunk_type": meta.get("chunk_type", ""),
                    "name": meta.get("name", ""),
                    "start_line": int(meta.get("start_line", 1)),
                    "end_line": int(meta.get("end_line", 1)),
                })
            return chunks
        except Exception as e:
            print(f"[VectorStore] Error fetching all chunks: {e}")
            return []

    def delete_collection(self):
        """Resets the collection by deleting and re-creating it."""
        if self.client is None:
            return
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            print(f"[VectorStore] Collection '{self.collection_name}' deleted and re-initialized.")
        except Exception as e:
            print(f"[VectorStore] Error resetting collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection summary metrics for monitoring and health check."""
        if self.collection is None:
            return {"status": "offline", "chunk_count": 0, "unique_files": 0}

        try:
            count = self.collection.count()
            all_chunks = self.get_all_chunks()
            files = set(c["file_path"] for c in all_chunks)
            languages: Dict[str, int] = {}
            for c in all_chunks:
                lang = c.get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1

            return {
                "status": "online",
                "chunk_count": count,
                "unique_files": len(files),
                "languages": languages,
                "persist_directory": str(self.persist_dir),
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "chunk_count": 0}
