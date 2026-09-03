"""
LLM Synthesizer and Contradiction Detector for TraceFind.
Integrates with Ollama (Llama 3.1 8B) for local, private generation.
Enforces exact source citations and performs cross-source contradiction detection.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from backend.config import LLM_MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

try:
    import ollama
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False


class TraceFindSynthesizer:
    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = None
        if OLLAMA_LIB_AVAILABLE:
            try:
                self.client = ollama.Client(host=self.base_url)
            except Exception:
                self.client = None

    def check_connection(self) -> Dict[str, Any]:
        """Checks if local Ollama daemon is reachable and lists available models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                has_target = any(self.model_name in m for m in models)
                return {
                    "connected": True,
                    "target_model": self.model_name,
                    "model_available": has_target,
                    "installed_models": models,
                }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "target_model": self.model_name,
                "model_available": False,
                "installed_models": [],
            }
        return {"connected": False, "target_model": self.model_name, "model_available": False}

    def synthesize(
        self, question: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates an answer grounded strictly in the retrieved context chunks.
        Performs entity resolution and checks for contradictions (e.g. docs vs code).
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant context found in the indexed repository to answer this question.",
                "citations": [],
                "contradictions": [],
                "model": self.model_name,
            }

        # Build structured context representation
        formatted_context, citations = self._build_context_block(retrieved_chunks)

        system_prompt = (
            "You are TraceFind, an expert AI context navigator for developers.\n"
            "Your job is to answer natural-language developer questions with high precision.\n"
            "RULES:\n"
            "1. Ground every claim strictly in the provided context snippets.\n"
            "2. Always cite specific files, line numbers, and function/class names using the format: [path:start_line-end_line].\n"
            "3. If different sources contradict each other (e.g. README vs actual code implementation, or mismatched parameters/types/configs), you MUST explicitly report the contradiction.\n"
            "4. Be concise, direct, and technical. Do not invent details not present in the snippets.\n"
            "5. At the end of your response, if any contradiction or discrepancy exists, output a section labeled '### Contradictions Detected:' with details."
        )

        user_prompt = (
            f"Developer Question:\n{question}\n\n"
            f"Retrieved Context Snippets:\n"
            f"{formatted_context}\n\n"
            f"Please provide:\n"
            f"1. A direct, clear answer citing exact files and lines [file:lines].\n"
            f"2. Entity resolution: explain how concepts map between code and documentation if applicable.\n"
            f"3. Any discrepancies or contradictions between documentation and code."
        )

        raw_response = self._call_llm(system_prompt, user_prompt)

        # Parse contradictions from response or heuristics
        contradictions = self._detect_contradictions(retrieved_chunks, raw_response)

        return {
            "answer": raw_response,
            "citations": citations,
            "contradictions": contradictions,
            "model": self.model_name,
            "context_count": len(retrieved_chunks),
        }

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Invokes the Ollama LLM via client library or direct REST API."""
        # Try direct HTTP REST API to Ollama
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            }
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
            else:
                print(f"[Synthesizer] Ollama API returned status {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print("[Synthesizer] Could not connect to Ollama. Daemon may not be running.")
            return self._generate_fallback_response(user_prompt)
        except Exception as e:
            print(f"[Synthesizer] LLM invocation failed: {e}")

        # Try client SDK fallback if available
        if self.client is not None:
            try:
                res = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return res["message"]["content"].strip()
            except Exception as e:
                print(f"[Synthesizer] Ollama SDK error: {e}")

        return self._generate_fallback_response(user_prompt)

    def _build_context_block(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Formats chunks with numbered labels and extracts citation metadata."""
        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, start=1):
            file_path = chunk.get("file_path", "unknown")
            start_line = chunk.get("start_line", 1)
            end_line = chunk.get("end_line", 1)
            name = chunk.get("name", "block")
            chunk_type = chunk.get("chunk_type", "code")
            content = chunk.get("content", "")

            citation = {
                "citation_id": idx,
                "file_path": file_path,
                "file_name": chunk.get("file_name", ""),
                "start_line": start_line,
                "end_line": end_line,
                "name": name,
                "chunk_type": chunk_type,
                "language": chunk.get("language", ""),
                "score": chunk.get("score", 0.0),
                "snippet": content[:200] + "..." if len(content) > 200 else content,
            }
            citations.append(citation)

            block = (
                f"--- [Source {idx}] {file_path}:{start_line}-{end_line} ({chunk_type}: {name}) ---\n"
                f"{content}\n"
            )
            context_parts.append(block)

        return "\n".join(context_parts), citations

    def _detect_contradictions(
        self, chunks: List[Dict[str, Any]], llm_output: str
    ) -> List[Dict[str, Any]]:
        """
        Analyzes whether code vs documentation or multiple snippets contradict each other.
        Combines LLM detection with structural checks between docs and code.
        """
        contradictions: List[Dict[str, Any]] = []

        # 1. Parse explicit contradictions detected in LLM text output
        if "contradiction" in llm_output.lower() or "discrepancy" in llm_output.lower() or "conflict" in llm_output.lower():
            # Look for lines indicating contradiction
            lines = llm_output.splitlines()
            capture = False
            buf = []
            for line in lines:
                if "contradiction" in line.lower() or "discrepanc" in line.lower():
                    capture = True
                if capture and line.strip():
                    buf.append(line.strip())
                elif capture and not line.strip() and len(buf) > 3:
                    break

            if buf:
                desc = " ".join(buf[:4])
                # Find referenced sources
                sources = re.findall(r"\[([A-Za-z0-9_\-\.\/]+:\d+(?:-\d+)?)\]", desc)
                src_a = sources[0] if len(sources) > 0 else "Documentation"
                src_b = sources[1] if len(sources) > 1 else "Implementation"

                contradictions.append({
                    "type": "semantic_mismatch",
                    "source_a": src_a,
                    "source_b": src_b,
                    "description": desc,
                    "confidence": 0.88,
                })

        # 2. Heuristic check: Look for doc files vs code files on the same entity
        doc_chunks = [c for c in chunks if c.get("language") in ("markdown", "text", "pdf")]
        code_chunks = [c for c in chunks if c.get("language") not in ("markdown", "text", "pdf")]

        if doc_chunks and code_chunks:
            # Check for version, timeout, default values or port discrepancies in text
            for dc in doc_chunks:
                for cc in code_chunks:
                    # Look for port mismatch
                    doc_ports = re.findall(r"port\s*[:=]\s*(\d{4,5})", dc.get("content", ""), re.IGNORECASE)
                    code_ports = re.findall(r"port\s*[:=]\s*(\d{4,5})", cc.get("content", ""), re.IGNORECASE)
                    if doc_ports and code_ports and set(doc_ports) != set(code_ports):
                        contradictions.append({
                            "type": "configuration_mismatch",
                            "source_a": f"{dc.get('file_path')}:{dc.get('start_line')}",
                            "source_b": f"{cc.get('file_path')}:{cc.get('start_line')}",
                            "description": f"Port configuration discrepancy: Docs reference port {doc_ports[0]} while code references {code_ports[0]}.",
                            "confidence": 0.92,
                        })

        return contradictions

    def _generate_fallback_response(self, user_prompt: str) -> str:
        """Returns a helpful fallback response when Ollama is not actively running."""
        return (
            "TraceFind retrieved relevant context snippets from the codebase.\n\n"
            "> **Ollama Status Note**: The local Ollama server is currently offline or unreachable at http://localhost:11434. "
            "Please ensure Ollama is running (`ollama serve`) and model `llama3.1:8b` is pulled (`ollama pull llama3.1:8b`).\n\n"
            "Below is the relevant context retrieved via hybrid search (BM25 + Semantic Vector Fusion):\n"
            f"{user_prompt}"
        )
