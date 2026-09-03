# TraceFind ⚡
### AI-Powered Context Navigator for Fragmented & Unstructured Work Environments

> **Hackathon Track:** Recursion Edition II — *PS2: Context Intelligence for Fragmented and Unstructured Work Environments*

TraceFind is a **100% local, zero-API-cost** context navigator and intelligence layer designed for software engineers and engineering teams. It indexes fragmented software repositories, architecture documentation, meeting notes, and issue tracker artifacts (Jira/Slack), answering technical questions with exact source code citations, performing cross-source entity resolution, and detecting critical contradictions (e.g. outdated README documentation vs actual implementation).

---

## 🎯 Key Capabilities

- 🔍 **Exact AST-Aware Citations:** Parses Python, TypeScript, JavaScript, Java, and Go using Tree-sitter grammars to return precise function/class names and exact line numbers (`file.py:24-65`).
- ⚡ **Hybrid Search (BM25 + Semantic Embeddings + RRF):** Fuses exact identifier and symbol matching (BM25 with camelCase/snake_case splitting) with semantic vector search (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF) and Cross-Encoder re-ranking.
- ⚠️ **Cross-Source Contradiction Detection:** Flags divergences where documentation contradicts live code (e.g., deprecated API parameters, mismatched configuration ports, or invalid return structures).
- 🧩 **Minimum Relevant Context:** Trims context down to the exact functions or blocks required for a task, minimizing cognitive load and token bloat.
- 🔒 **Zero Cost & Completely Local:** Runs entirely on-device using ChromaDB and Ollama (`llama3.1:8b`) with no cloud dependencies or API keys required.

---

## 🛠️ Tech Stack

| Component | Tool / Technology | Purpose |
|---|---|---|
| **Code Parser** | Tree-sitter + `tree_sitter_languages` | AST-aware function/class boundary extraction |
| **Vector DB** | ChromaDB (Persistent) | Local vector storage & cosine similarity retrieval |
| **Embeddings** | `all-MiniLM-L6-v2` (`sentence-transformers`) | Fast, lightweight 384-dimensional dense vectors |
| **Keyword Search** | `rank-bm25` | Lexical exact token matching for symbols and paths |
| **Hybrid Ranking** | Reciprocal Rank Fusion (RRF) + Cross-Encoder | Fuses vector and keyword scores with reranking |
| **Local LLM** | Llama 3.1 8B via Ollama | Local context synthesis and contradiction reasoning |
| **Backend** | FastAPI + Uvicorn | Async REST API service |
| **CLI Engine** | Python + `tqdm` + `argparse` | Recursive repo scanner and bulk indexer |
| **Frontend** | VS Code Extension (TypeScript) | Native sidebar navigator with one-click code jumps |

---

## 🏗️ Architecture Flow

```text
  [ Developer Codebase / Docs / Notes ]
                 │
                 ▼
       [ Tree-sitter Parser ]
       (Extracts AST Chunks: Functions, Classes, Line Numbers)
                 │
        ┌────────┴────────┐
        ▼                 ▼
 [ BM25 Lexical Index ]  [ SentenceTransformer Embeddings ]
        │                 │
        │                 ▼
        │        [ ChromaDB Vector Store ]
        │                 │
        └────────┬────────┘
                 ▼
     [ Hybrid Search & RRF ]
   (Reciprocal Rank Fusion k=60)
                 │
                 ▼
     [ Cross-Encoder Re-ranker ]
                 │
                 ▼
      [ Minimum Relevant Context ]
                 │
                 ▼
     [ Ollama: Llama 3.1 8B ]
                 │
  ┌──────────────┴──────────────┐
  ▼                             ▼
[ Grounded Answer + Citations ] [ Contradiction Detection ]
  (e.g., auth.ts:42-88)           (README vs actual code mismatches)
                 │
                 ▼
     [ VS Code Extension UI ]
  (Interactive Jump-to-Line & Copy Context)
```

---

## 📦 Project Structure

```text
tracefind/
│
├── backend/
│   ├── __init__.py           # Package initializer
│   ├── config.py             # Central paths, models, and chunking configuration
│   ├── chunker.py            # Tree-sitter AST and markdown structural chunking
│   ├── embedder.py           # all-MiniLM-L6-v2 sentence embeddings with cache
│   ├── vector_store.py       # ChromaDB persistent client & vector search
│   ├── hybrid_search.py      # BM25 + Vector + RRF fusion + Cross-Encoder
│   ├── synthesizer.py        # Ollama Llama 3.1 8B synthesis & contradiction detector
│   └── main.py               # FastAPI backend REST API
│
├── cli/
│   └── index.py              # CLI bulk repository indexing tool
│
├── vscode-extension/
│   ├── src/
│   │   └── extension.ts      # VS Code extension controller & Webview UI
│   ├── package.json          # Extension manifest & commands
│   └── tsconfig.json         # TypeScript configuration
│
├── models/                   # Local cached transformer models
├── chroma_db/                # Persistent vector database
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git exclusions
```

---

## 🚀 Quick Start Guide

### Step 1: Python Environment Setup

```bash
# 1. Clone or navigate to the project directory
cd tracefind

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Set Up Local LLM with Ollama

1. Download and install Ollama from [https://ollama.com](https://ollama.com).
2. Pull the Llama 3.1 8B model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Ensure Ollama is running in the background (default address: `http://localhost:11434`).

---

### Step 3: Launch the TraceFind Backend

```bash
# From the tracefind root directory:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Verify the server is running by opening:
- Health Check: `http://localhost:8000/health`
- Interactive Swagger API Docs: `http://localhost:8000/docs`

---

### Step 4: Index a Codebase

You can index any repository or project directory using the CLI tool:

```bash
# Direct or API-triggered indexing
python cli/index.py --path /path/to/your/repository

# Force re-indexing from scratch:
python cli/index.py --path /path/to/your/repository --force
```

---

### Step 5: Launch the VS Code Extension

1. Navigate to the `vscode-extension` directory:
   ```bash
   cd vscode-extension
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Compile the TypeScript code:
   ```bash
   npm run compile
   ```
4. Open the `vscode-extension` folder in VS Code.
5. Press **F5** (or go to `Run and Debug` -> `Launch Extension`).
6. A new Extension Development Host window will appear.
7. Click the **TraceFind** icon in the Activity Bar to open the interactive Context Navigator!

---

## 🔌 API Reference

### 1. `POST /index`
Recursively indexes a codebase or document directory.
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "./my-project", "force_reindex": false}'
```

### 2. `POST /search`
Performs hybrid search (BM25 + Semantic Vector + RRF) to retrieve the minimum relevant context.
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "validateSessionToken", "k": 5}'
```

### 3. `POST /ask`
Answers technical questions with exact source citations and contradiction alerts.
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How is authentication handled and is there any mismatch with docs?", "k": 5}'
```

### 4. `GET /health`
Returns system status, Ollama model readiness, and indexed chunk metrics.

### 5. `GET /stats`
Returns language breakdown, total chunks, and unique file metrics.

---

## 💡 Example: Contradiction Detection in Action

Suppose a project contains:
- **`README.md`**: *"The API server runs on port 8080 and requires Bearer JWT authentication."*
- **`server.py`**:
  ```python
  app = FastAPI()
  PORT = 9000
  # Session cookie authentication implemented
  ```

When a developer asks:
> *"What port does the server run on and how do I authenticate?"*

**TraceFind responds:**
1. **Direct Answer:** Identifies the port defined in `server.py:12-18` (Port 9000) and the cookie auth handler.
2. **⚠️ Contradiction Alert:**
   - **Type:** Configuration & Authentication Mismatch
   - **Source A:** `README.md:15` (Port 8080 / JWT)
   - **Source B:** `server.py:14` (Port 9000 / Cookie)
   - **Confidence:** 92%

---

## 🛡️ Troubleshooting

- **`tree_sitter_languages` parser load issues:** Ensure you installed `tree-sitter>=0.21.3` and `tree_sitter_languages>=1.10.2`. Pre-built binaries are provided across Windows, macOS, and Linux without needing MSVC or GCC.
- **Ollama connection warning:** Run `ollama list` in your terminal to verify that `llama3.1:8b` is pulled and that the Ollama service is active.
- **ChromaDB permission errors on Windows:** Ensure your user account has write permissions to the project directory. The Chroma database is saved by default inside `tracefind/chroma_db/`.

---

## 👥 Authors

Built for **Recursion Edition II** under track **PS2: Context Intelligence for Fragmented and Unstructured Work Environments**.
