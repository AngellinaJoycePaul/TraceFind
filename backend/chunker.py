"""
AST-Aware Code and Document Chunker for TraceFind.
Preserves function, class, and method boundaries using Tree-sitter.
Provides fallback mechanisms for non-code files and unparsed syntax.
"""

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import (
    CODE_CHUNK_MAX_LINES,
    CODE_CHUNK_MIN_LINES,
    DOC_CHUNK_OVERLAP,
    DOC_CHUNK_SIZE,
    DOC_EXTENSIONS,
    LANGUAGE_MAP,
)

# Safe import for tree_sitter_languages
try:
    from tree_sitter import Node, Parser
    import tree_sitter_languages
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class CodeChunk:
    id: str
    file_path: str
    file_name: str
    language: str
    chunk_type: str  # 'function', 'class', 'method', 'doc_section', 'block'
    name: str        # e.g., 'getUserData', 'AuthService', 'API Overview'
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceFindChunker:
    def __init__(self):
        self._parsers: Dict[str, Any] = {}
        if TREE_SITTER_AVAILABLE:
            self._init_parsers()

    def _init_parsers(self):
        """Pre-load Tree-sitter parsers for supported languages."""
        supported_langs = ["python", "javascript", "typescript", "java", "go"]
        for lang in supported_langs:
            try:
                self._parsers[lang] = tree_sitter_languages.get_parser(lang)
            except Exception as e:
                # Log parser load error and continue with fallback
                print(f"[Chunker] Warning: Could not initialize tree-sitter parser for {lang}: {e}")

    def chunk_file(self, file_path: str, repo_root: Optional[str] = None) -> List[CodeChunk]:
        """
        Chunks an individual file based on its extension.
        Preserves AST boundaries for code, or structural boundaries for docs.
        """
        path = Path(file_path)
        if not path.is_file():
            return []

        rel_path = os.path.relpath(str(path), repo_root) if repo_root else str(path)
        # Normalize slashes for cross-platform consistency
        rel_path = rel_path.replace("\\", "/")
        file_name = path.name
        ext = path.suffix.lower()

        try:
            # Handle PDF documents
            if ext == ".pdf":
                return self._chunk_pdf(path, rel_path, file_name)

            # Read text content
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                print(f"[Chunker] Error reading {file_path}: {e}")
                return []

            if not content.strip():
                return []

            # Check if recognized code file
            if ext in LANGUAGE_MAP:
                lang = LANGUAGE_MAP[ext]
                return self._chunk_code_file(content, rel_path, file_name, lang)

            # Check if documentation file
            if ext in DOC_EXTENSIONS:
                if ext == ".md":
                    return self._chunk_markdown(content, rel_path, file_name)
                return self._chunk_generic_document(content, rel_path, file_name, ext[1:])

            # Fallback for other text files
            return self._chunk_line_window(
                content, rel_path, file_name, language="text", chunk_type="block"
            )

        except Exception as e:
            print(f"[Chunker] Unexpected error chunking {file_path}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Tree-Sitter AST Chunking
    # -------------------------------------------------------------------------
    def _chunk_code_file(
        self, content: str, rel_path: str, file_name: str, language: str
    ) -> List[CodeChunk]:
        """Parses code file with Tree-sitter and returns AST-aligned chunks."""
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        parser = self._parsers.get(language)
        if not parser:
            # Fallback if tree-sitter not available for this language
            return self._chunk_line_window(content, rel_path, file_name, language, "block")

        try:
            tree = parser.parse(bytes(content, "utf-8"))
            root_node = tree.root_node
            ast_chunks = self._extract_ast_nodes(root_node, content, rel_path, file_name, language)

            if not ast_chunks:
                # If no functions/classes found (e.g. simple script), chunk by line window
                return self._chunk_line_window(content, rel_path, file_name, language, "block")

            # Collect covered line ranges to capture un-nested/module-level code
            covered_ranges = sorted([(c.start_line, c.end_line) for c in ast_chunks])
            all_chunks: List[CodeChunk] = list(ast_chunks)

            # Capture remaining uncovered lines (e.g., header imports, global constants)
            current_line = 1
            for start, end in covered_ranges:
                if start > current_line:
                    gap_content = "".join(lines[current_line - 1 : start - 1]).strip()
                    gap_lines = (start - 1) - current_line + 1
                    if gap_content and gap_lines >= CODE_CHUNK_MIN_LINES:
                        chunk_id = self._generate_id(rel_path, current_line, start - 1, gap_content)
                        all_chunks.append(
                            CodeChunk(
                                id=chunk_id,
                                file_path=rel_path,
                                file_name=file_name,
                                language=language,
                                chunk_type="module",
                                name="module_scope",
                                start_line=current_line,
                                end_line=start - 1,
                                content=gap_content,
                            )
                        )
                current_line = max(current_line, end + 1)

            if current_line <= total_lines:
                remaining_content = "".join(lines[current_line - 1 : total_lines]).strip()
                rem_lines = total_lines - current_line + 1
                if remaining_content and rem_lines >= CODE_CHUNK_MIN_LINES:
                    chunk_id = self._generate_id(rel_path, current_line, total_lines, remaining_content)
                    all_chunks.append(
                        CodeChunk(
                            id=chunk_id,
                            file_path=rel_path,
                            file_name=file_name,
                            language=language,
                            chunk_type="module",
                            name="module_scope",
                            start_line=current_line,
                            end_line=total_lines,
                            content=remaining_content,
                        )
                    )

            # Sort chunks by start_line
            all_chunks.sort(key=lambda c: c.start_line)
            return all_chunks

        except Exception as e:
            print(f"[Chunker] Tree-sitter failed for {rel_path} ({language}): {e}. Falling back.")
            return self._chunk_line_window(content, rel_path, file_name, language, "block")

    def _extract_ast_nodes(
        self, root_node: Any, content: str, rel_path: str, file_name: str, language: str
    ) -> List[CodeChunk]:
        """Extracts functions, methods, and classes from the AST."""
        chunks: List[CodeChunk] = []

        # Target node types per language
        target_types = {
            "python": {
                "function_definition": "function",
                "async_function_definition": "function",
                "class_definition": "class",
            },
            "javascript": {
                "function_declaration": "function",
                "method_definition": "method",
                "class_declaration": "class",
                "arrow_function": "function",
            },
            "typescript": {
                "function_declaration": "function",
                "method_definition": "method",
                "class_declaration": "class",
                "interface_declaration": "class",
                "arrow_function": "function",
            },
            "java": {
                "method_declaration": "method",
                "class_declaration": "class",
                "constructor_declaration": "method",
                "interface_declaration": "class",
            },
            "go": {
                "function_declaration": "function",
                "method_declaration": "method",
                "type_declaration": "class",
            },
        }

        lang_targets = target_types.get(language, {})

        def traverse(node: Any):
            node_type = node.type
            if node_type in lang_targets:
                chunk_type = lang_targets[node_type]
                name = self._extract_node_name(node, language)
                start_line = node.start_point[0] + 1  # 1-indexed
                end_line = node.end_point[0] + 1      # 1-indexed

                # Extract node source text
                node_bytes = content.encode("utf-8")[node.start_byte : node.end_byte]
                node_text = node_bytes.decode("utf-8", errors="replace")

                # If the node is too huge (e.g. a huge 500-line class), recurse into its methods
                line_count = end_line - start_line + 1
                if chunk_type == "class" and line_count > CODE_CHUNK_MAX_LINES:
                    # Recurse children to get individual methods
                    for child in node.children:
                        traverse(child)
                    return

                chunk_id = self._generate_id(rel_path, start_line, end_line, node_text)
                chunks.append(
                    CodeChunk(
                        id=chunk_id,
                        file_path=rel_path,
                        file_name=file_name,
                        language=language,
                        chunk_type=chunk_type,
                        name=name or f"anonymous_{chunk_type}",
                        start_line=start_line,
                        end_line=end_line,
                        content=node_text.strip(),
                    )
                )

                # Do not recurse into already-extracted standalone functions unless large
                if line_count <= CODE_CHUNK_MAX_LINES:
                    return

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return chunks

    def _extract_node_name(self, node: Any, language: str) -> Optional[str]:
        """Extracts identifier/name from an AST node."""
        # Common identifier child lookup
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
                return child.text.decode("utf-8", errors="replace")
            # In variable declarations with arrow functions (JS/TS)
            if child.type == "variable_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        return sub.text.decode("utf-8", errors="replace")
            # In Go type declarations
            if child.type == "type_spec":
                for sub in child.children:
                    if sub.type == "type_identifier":
                        return sub.text.decode("utf-8", errors="replace")
        return None

    # -------------------------------------------------------------------------
    # Document & Markdown Chunking
    # -------------------------------------------------------------------------
    def _chunk_markdown(self, content: str, rel_path: str, file_name: str) -> List[CodeChunk]:
        """Chunks markdown files by headers (# Header) while retaining line numbers."""
        lines = content.splitlines()
        chunks: List[CodeChunk] = []

        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
        sections = []
        current_header = "Introduction"
        start_line = 1
        current_lines: List[str] = []

        for idx, line in enumerate(lines, start=1):
            match = header_pattern.match(line)
            if match and current_lines:
                # Flush existing section
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append((current_header, start_line, idx - 1, text))
                current_header = match.group(2).strip()
                start_line = idx
                current_lines = [line]
            else:
                if match and not current_lines:
                    current_header = match.group(2).strip()
                    start_line = idx
                current_lines.append(line)

        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((current_header, start_line, len(lines), text))

        for header, s_line, e_line, sec_text in sections:
            chunk_id = self._generate_id(rel_path, s_line, e_line, sec_text)
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    file_path=rel_path,
                    file_name=file_name,
                    language="markdown",
                    chunk_type="doc_section",
                    name=header,
                    start_line=s_line,
                    end_line=e_line,
                    content=sec_text,
                )
            )

        return chunks if chunks else self._chunk_generic_document(content, rel_path, file_name, "markdown")

    def _chunk_generic_document(
        self, content: str, rel_path: str, file_name: str, language: str
    ) -> List[CodeChunk]:
        """Chunks generic text/json/yaml documents with overlapping windows."""
        lines = content.splitlines()
        if not lines:
            return []

        chunks: List[CodeChunk] = []
        lines_per_chunk = max(10, CODE_CHUNK_MAX_LINES // 2)
        step = max(5, lines_per_chunk - (DOC_CHUNK_OVERLAP // 20))

        for i in range(0, len(lines), step):
            window_lines = lines[i : i + lines_per_chunk]
            start_line = i + 1
            end_line = min(i + len(window_lines), len(lines))
            chunk_text = "\n".join(window_lines).strip()

            if not chunk_text:
                continue

            chunk_id = self._generate_id(rel_path, start_line, end_line, chunk_text)
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    file_path=rel_path,
                    file_name=file_name,
                    language=language,
                    chunk_type="doc_section",
                    name=f"{file_name}:L{start_line}-{end_line}",
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_text,
                )
            )

        return chunks

    def _chunk_pdf(self, path: Path, rel_path: str, file_name: str) -> List[CodeChunk]:
        """Extracts text from PDF page-by-page using pypdf."""
        chunks: List[CodeChunk] = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue

                chunk_id = self._generate_id(rel_path, page_num, page_num, text)
                chunks.append(
                    CodeChunk(
                        id=chunk_id,
                        file_path=rel_path,
                        file_name=file_name,
                        language="pdf",
                        chunk_type="doc_section",
                        name=f"Page {page_num}",
                        start_line=page_num,
                        end_line=page_num,
                        content=text,
                    )
                )
        except Exception as e:
            print(f"[Chunker] Could not parse PDF {rel_path}: {e}")
        return chunks

    def _chunk_line_window(
        self, content: str, rel_path: str, file_name: str, language: str, chunk_type: str
    ) -> List[CodeChunk]:
        """Sliding window fallback for plain code or unparsed content."""
        lines = content.splitlines()
        chunks: List[CodeChunk] = []
        window_size = CODE_CHUNK_MAX_LINES
        step = max(10, window_size - 10)

        for i in range(0, len(lines), step):
            chunk_lines = lines[i : i + window_size]
            start_line = i + 1
            end_line = min(i + len(chunk_lines), len(lines))
            chunk_content = "\n".join(chunk_lines).strip()

            if not chunk_content:
                continue

            chunk_id = self._generate_id(rel_path, start_line, end_line, chunk_content)
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    file_path=rel_path,
                    file_name=file_name,
                    language=language,
                    chunk_type=chunk_type,
                    name=f"{file_name}:L{start_line}-{end_line}",
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content,
                )
            )

        return chunks

    @staticmethod
    def _generate_id(file_path: str, start_line: int, end_line: int, content: str) -> str:
        """Generates a stable, collision-resistant hash ID for a chunk."""
        key = f"{file_path}:{start_line}:{end_line}:{content[:64]}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
