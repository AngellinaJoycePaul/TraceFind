import * as vscode from "vscode";
import axios from "axios";
import * as path from "path";

export function activate(context: vscode.ExtensionContext) {
  console.log("TraceFind Extension is now active.");

  const provider = new TraceFindViewProvider(context.extensionUri);

  // Register Webview View Provider in Activity Bar Sidebar
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      TraceFindViewProvider.viewType,
      provider,
      {
        webviewOptions: { retainContextWhenHidden: true },
      }
    )
  );

  // Command: Search Context
  context.subscriptions.push(
    vscode.commands.registerCommand("tracefind.search", async () => {
      const query = await vscode.window.showInputBox({
        prompt: "Search codebase context with TraceFind",
        placeHolder: "e.g., auth session token validation",
      });
      if (query) {
        provider.executeSearch(query);
      }
    })
  );

  // Command: Ask Question
  context.subscriptions.push(
    vscode.commands.registerCommand("tracefind.ask", async () => {
      const question = await vscode.window.showInputBox({
        prompt: "Ask a question about the codebase with TraceFind",
        placeHolder: "e.g., Where is the user login route and how does expiry work?",
      });
      if (question) {
        provider.executeAsk(question);
      }
    })
  );

  // Command: Index Current Workspace
  context.subscriptions.push(
    vscode.commands.registerCommand("tracefind.index", async () => {
      provider.indexCurrentWorkspace();
    })
  );
}

export function deactivate() {}

class TraceFindViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "tracefindSidebar";
  private _view?: vscode.WebviewView;

  constructor(private readonly _extensionUri: vscode.Uri) {}

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    // Handle incoming messages from the webview
    webviewView.webview.onDidReceiveMessage(async (data) => {
      switch (data.type) {
        case "search":
          await this._handleSearch(data.query);
          break;
        case "ask":
          await this._handleAsk(data.question);
          break;
        case "indexWorkspace":
          await this.indexCurrentWorkspace();
          break;
        case "openFile":
          await this._openFileAtLocation(
            data.filePath,
            data.startLine,
            data.endLine
          );
          break;
        case "copyToClipboard":
          await vscode.env.clipboard.writeText(data.text);
          vscode.window.showInformationMessage("TraceFind context copied to clipboard!");
          break;
      }
    });
  }

  private getApiUrl(): string {
    const config = vscode.workspace.getConfiguration("tracefind");
    return config.get<string>("apiUrl", "http://localhost:8000");
  }

  public executeSearch(query: string) {
    if (this._view) {
      this._view.show?.(true);
      this._view.webview.postMessage({ type: "setSearchQuery", query });
      this._handleSearch(query);
    }
  }

  public executeAsk(question: string) {
    if (this._view) {
      this._view.show?.(true);
      this._view.webview.postMessage({ type: "setAskQuestion", question });
      this._handleAsk(question);
    }
  }

  public async indexCurrentWorkspace() {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) {
      vscode.window.showErrorMessage("TraceFind: No workspace folder is open to index.");
      return;
    }

    const workspacePath = folders[0].uri.fsPath;
    const apiUrl = this.getApiUrl();

    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "TraceFind: Indexing workspace...",
        cancellable: false,
      },
      async (progress) => {
        try {
          if (this._view) {
            this._view.webview.postMessage({ type: "setLoading", loading: true, message: "Indexing workspace..." });
          }

          const response = await axios.post(`${apiUrl}/index`, {
            path: workspacePath,
            force_reindex: false,
          });

          const data = response.data;
          vscode.window.showInformationMessage(
            `TraceFind: Successfully indexed ${data.files_indexed} files (${data.chunks_created} chunks) in ${data.time_taken_seconds}s!`
          );

          if (this._view) {
            this._view.webview.postMessage({
              type: "indexCompleted",
              stats: data,
            });
          }
        } catch (error: any) {
          const msg = error.response?.data?.detail || error.message;
          vscode.window.showErrorMessage(`TraceFind indexing failed: ${msg}`);
          if (this._view) {
            this._view.webview.postMessage({
              type: "setError",
              error: `Indexing failed: ${msg}. Is the backend running?`,
            });
          }
        } finally {
          if (this._view) {
            this._view.webview.postMessage({ type: "setLoading", loading: false });
          }
        }
      }
    );
  }

  private async _handleSearch(query: string) {
    if (!query || !query.trim()) return;

    const apiUrl = this.getApiUrl();
    const config = vscode.workspace.getConfiguration("tracefind");
    const k = config.get<number>("defaultTopK", 5);

    try {
      this._postToWebview({ type: "setLoading", loading: true, message: "Searching hybrid index..." });

      const response = await axios.post(`${apiUrl}/search`, {
        query: query.trim(),
        k: k,
      });

      this._postToWebview({
        type: "searchResults",
        data: response.data,
      });
    } catch (error: any) {
      const msg = error.response?.data?.detail || error.message;
      this._postToWebview({
        type: "setError",
        error: `Search request failed: ${msg}. Check if backend is active at ${apiUrl}.`,
      });
    } finally {
      this._postToWebview({ type: "setLoading", loading: false });
    }
  }

  private async _handleAsk(question: string) {
    if (!question || !question.trim()) return;

    const apiUrl = this.getApiUrl();
    const config = vscode.workspace.getConfiguration("tracefind");
    const k = config.get<number>("defaultTopK", 5);

    try {
      this._postToWebview({
        type: "setLoading",
        loading: true,
        message: "Synthesizing answer & checking contradictions with Ollama...",
      });

      const response = await axios.post(`${apiUrl}/ask`, {
        question: question.trim(),
        k: k,
      });

      this._postToWebview({
        type: "askResults",
        data: response.data,
      });
    } catch (error: any) {
      const msg = error.response?.data?.detail || error.message;
      this._postToWebview({
        type: "setError",
        error: `Ask request failed: ${msg}. Ensure FastAPI backend and Ollama are running.`,
      });
    } finally {
      this._postToWebview({ type: "setLoading", loading: false });
    }
  }

  private async _openFileAtLocation(filePath: string, startLine: number, endLine: number) {
    try {
      let targetPath = filePath;

      // Resolve relative path against workspace root if necessary
      if (!path.isAbsolute(targetPath)) {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
          targetPath = path.join(workspaceFolders[0].uri.fsPath, filePath);
        }
      }

      const fileUri = vscode.Uri.file(targetPath);
      const document = await vscode.workspace.openTextDocument(fileUri);
      const editor = await vscode.window.showTextDocument(document, {
        preview: false,
        viewColumn: vscode.ViewColumn.One,
      });

      // 1-indexed to 0-indexed line numbers
      const sLine = Math.max(0, (startLine || 1) - 1);
      const eLine = Math.max(0, (endLine || startLine || 1) - 1);

      const startPos = new vscode.Position(sLine, 0);
      const endPos = new vscode.Position(eLine, document.lineAt(Math.min(eLine, document.lineCount - 1)).text.length);
      const range = new vscode.Range(startPos, endPos);

      editor.selection = new vscode.Selection(startPos, endPos);
      editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
    } catch (err: any) {
      vscode.window.showErrorMessage(`Could not open file: ${filePath} (${err.message})`);
    }
  }

  private _postToWebview(msg: any) {
    if (this._view) {
      this._view.webview.postMessage(msg);
    }
  }

  private _getHtmlForWebview(_webview: vscode.Webview): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TraceFind Navigator</title>
  <style>
    :root {
      --bg: var(--vscode-sideBar-background);
      --fg: var(--vscode-foreground);
      --input-bg: var(--vscode-input-background);
      --input-fg: var(--vscode-input-foreground);
      --input-border: var(--vscode-input-border);
      --btn-bg: var(--vscode-button-background);
      --btn-fg: var(--vscode-button-foreground);
      --btn-hover: var(--vscode-button-hoverBackground);
      --card-bg: var(--vscode-editor-background);
      --badge-bg: var(--vscode-badge-background);
      --badge-fg: var(--vscode-badge-foreground);
      --warning-border: #d97706;
      --warning-bg: rgba(217, 119, 6, 0.15);
      --accent-blue: #3b82f6;
    }
    body {
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      color: var(--fg);
      background-color: var(--bg);
      padding: 10px;
      margin: 0;
      box-sizing: border-box;
    }
    .header-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .brand-title {
      font-size: 1.15em;
      font-weight: 700;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .brand-title span {
      color: var(--accent-blue);
    }
    .btn-index {
      background: transparent;
      border: 1px solid var(--btn-bg);
      color: var(--fg);
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.85em;
    }
    .btn-index:hover {
      background: var(--btn-bg);
      color: var(--btn-fg);
    }
    .tabs {
      display: flex;
      gap: 4px;
      margin-bottom: 12px;
    }
    .tab-btn {
      flex: 1;
      padding: 6px 10px;
      background: rgba(128, 128, 128, 0.15);
      border: none;
      color: var(--fg);
      cursor: pointer;
      font-weight: 600;
      border-radius: 4px;
      font-size: 0.9em;
    }
    .tab-btn.active {
      background: var(--btn-bg);
      color: var(--btn-fg);
    }
    .input-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }
    textarea, input[type="text"] {
      width: 100%;
      box-sizing: border-box;
      background: var(--input-bg);
      color: var(--input-fg);
      border: 1px solid var(--input-border);
      border-radius: 4px;
      padding: 8px;
      font-family: inherit;
      resize: vertical;
    }
    .btn-submit {
      background: var(--btn-bg);
      color: var(--btn-fg);
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 600;
    }
    .btn-submit:hover {
      background: var(--btn-hover);
    }
    .loader {
      display: none;
      align-items: center;
      gap: 8px;
      padding: 10px;
      background: rgba(128, 128, 128, 0.1);
      border-radius: 4px;
      margin-bottom: 12px;
      font-size: 0.9em;
    }
    .spinner {
      border: 2px solid rgba(255, 255, 255, 0.2);
      border-top: 2px solid var(--accent-blue);
      border-radius: 50%;
      width: 14px;
      height: 14px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    .error-box {
      display: none;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid #ef4444;
      padding: 8px;
      border-radius: 4px;
      margin-bottom: 12px;
      font-size: 0.85em;
    }
    /* Contradiction Alert Box */
    .contradiction-box {
      background: var(--warning-bg);
      border-left: 4px solid var(--warning-border);
      padding: 10px;
      border-radius: 0 4px 4px 0;
      margin-bottom: 12px;
    }
    .contradiction-header {
      font-weight: 700;
      color: var(--warning-border);
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
      font-size: 0.9em;
    }
    .contradiction-item {
      font-size: 0.85em;
      margin-top: 6px;
      line-height: 1.4;
    }
    .contradiction-sources {
      font-family: monospace;
      font-size: 0.8em;
      margin-top: 4px;
      opacity: 0.85;
    }
    /* AI Answer Card */
    .answer-card {
      background: var(--card-bg);
      border: 1px solid rgba(128, 128, 128, 0.2);
      border-radius: 6px;
      padding: 12px;
      margin-bottom: 14px;
      line-height: 1.5;
    }
    .answer-header {
      font-weight: 700;
      font-size: 0.95em;
      margin-bottom: 8px;
      color: var(--accent-blue);
    }
    .answer-text {
      font-size: 0.9em;
      white-space: pre-wrap;
    }
    /* Context Results List */
    .results-header {
      font-size: 0.85em;
      font-weight: 600;
      text-transform: uppercase;
      opacity: 0.8;
      margin-bottom: 8px;
    }
    .chunk-card {
      background: var(--card-bg);
      border: 1px solid rgba(128, 128, 128, 0.2);
      border-radius: 4px;
      padding: 10px;
      margin-bottom: 8px;
      font-size: 0.85em;
    }
    .chunk-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .chunk-path {
      color: var(--accent-blue);
      cursor: pointer;
      font-weight: 600;
      text-decoration: underline;
      font-family: monospace;
      word-break: break-all;
    }
    .chunk-path:hover {
      opacity: 0.8;
    }
    .badge {
      background: var(--badge-bg);
      color: var(--badge-fg);
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.75em;
      font-weight: 600;
    }
    .chunk-code {
      background: rgba(0, 0, 0, 0.25);
      border-radius: 3px;
      padding: 6px;
      font-family: monospace;
      font-size: 0.85em;
      white-space: pre-wrap;
      max-height: 120px;
      overflow-y: auto;
      border: 1px solid rgba(128, 128, 128, 0.1);
    }
    .chunk-actions {
      display: flex;
      justify-content: flex-end;
      margin-top: 6px;
    }
    .btn-copy {
      background: transparent;
      border: 1px solid rgba(128, 128, 128, 0.3);
      color: var(--fg);
      font-size: 0.75em;
      padding: 2px 6px;
      border-radius: 3px;
      cursor: pointer;
    }
    .btn-copy:hover {
      background: rgba(128, 128, 128, 0.2);
    }
  </style>
</head>
<body>

  <div class="header-bar">
    <div class="brand-title">
      <span>⚡</span> TraceFind
    </div>
    <button class="btn-index" id="btnIndex">Index Workspace</button>
  </div>

  <div class="tabs">
    <button class="tab-btn active" id="tabAsk">Ask AI</button>
    <button class="tab-btn" id="tabSearch">Search Context</button>
  </div>

  <!-- Ask View -->
  <div id="viewAsk">
    <div class="input-group">
      <textarea id="askInput" rows="3" placeholder="Ask a question about the code or documentation..."></textarea>
      <button class="btn-submit" id="btnAskSubmit">Ask TraceFind</button>
    </div>
  </div>

  <!-- Search View -->
  <div id="viewSearch" style="display: none;">
    <div class="input-group">
      <input type="text" id="searchInput" placeholder="Search keywords, functions, classes..." />
      <button class="btn-submit" id="btnSearchSubmit">Search Hybrid</button>
    </div>
  </div>

  <div class="loader" id="loader">
    <div class="spinner"></div>
    <span id="loaderMsg">Thinking...</span>
  </div>

  <div class="error-box" id="errorBox"></div>

  <!-- Contradictions Alert Container -->
  <div id="contradictionsContainer"></div>

  <!-- AI Answer Display -->
  <div id="answerContainer"></div>

  <!-- Chunks / Sources Display -->
  <div id="resultsContainer"></div>

  <script>
    const vscode = acquireVsCodeApi();

    const tabAsk = document.getElementById("tabAsk");
    const tabSearch = document.getElementById("tabSearch");
    const viewAsk = document.getElementById("viewAsk");
    const viewSearch = document.getElementById("viewSearch");

    const askInput = document.getElementById("askInput");
    const searchInput = document.getElementById("searchInput");
    const btnAskSubmit = document.getElementById("btnAskSubmit");
    const btnSearchSubmit = document.getElementById("btnSearchSubmit");
    const btnIndex = document.getElementById("btnIndex");

    const loader = document.getElementById("loader");
    const loaderMsg = document.getElementById("loaderMsg");
    const errorBox = document.getElementById("errorBox");
    const contradictionsContainer = document.getElementById("contradictionsContainer");
    const answerContainer = document.getElementById("answerContainer");
    const resultsContainer = document.getElementById("resultsContainer");

    // Tab Switching
    tabAsk.addEventListener("click", () => {
      tabAsk.classList.add("active");
      tabSearch.classList.remove("active");
      viewAsk.style.display = "block";
      viewSearch.style.display = "none";
    });

    tabSearch.addEventListener("click", () => {
      tabSearch.classList.add("active");
      tabAsk.classList.remove("active");
      viewSearch.style.display = "block";
      viewAsk.style.display = "none";
    });

    // Submissions
    btnAskSubmit.addEventListener("click", () => {
      const q = askInput.value.trim();
      if (q) {
        clearContainers();
        vscode.postMessage({ type: "ask", question: q });
      }
    });

    askInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        btnAskSubmit.click();
      }
    });

    btnSearchSubmit.addEventListener("click", () => {
      const q = searchInput.value.trim();
      if (q) {
        clearContainers();
        vscode.postMessage({ type: "search", query: q });
      }
    });

    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        btnSearchSubmit.click();
      }
    });

    btnIndex.addEventListener("click", () => {
      clearContainers();
      vscode.postMessage({ type: "indexWorkspace" });
    });

    function clearContainers() {
      errorBox.style.display = "none";
      contradictionsContainer.innerHTML = "";
      answerContainer.innerHTML = "";
      resultsContainer.innerHTML = "";
    }

    // Message handler from Extension Host
    window.addEventListener("message", (event) => {
      const msg = event.data;
      switch (msg.type) {
        case "setLoading":
          loader.style.display = msg.loading ? "flex" : "none";
          if (msg.message) loaderMsg.innerText = msg.message;
          break;
        case "setError":
          errorBox.innerText = msg.error;
          errorBox.style.display = "block";
          break;
        case "setAskQuestion":
          tabAsk.click();
          askInput.value = msg.question;
          break;
        case "setSearchQuery":
          tabSearch.click();
          searchInput.value = msg.query;
          break;
        case "askResults":
          renderAskResults(msg.data);
          break;
        case "searchResults":
          renderSearchResults(msg.data);
          break;
        case "indexCompleted":
          renderIndexCompleted(msg.stats);
          break;
      }
    });

    function renderIndexCompleted(stats) {
      answerContainer.innerHTML = \`
        <div class="answer-card">
          <div class="answer-header">✅ Workspace Indexed Successfully</div>
          <div class="answer-text">
            Scanned: \${stats.files_scanned} files<br/>
            Indexed: \${stats.files_indexed} files<br/>
            Chunks: \${stats.chunks_created}<br/>
            Duration: \${stats.time_taken_seconds}s
          </div>
        </div>
      \`;
    }

    function renderAskResults(data) {
      clearContainers();

      // Render Contradictions if any
      if (data.contradictions && data.contradictions.length > 0) {
        let conflictsHtml = "";
        for (const c of data.contradictions) {
          conflictsHtml += \`
            <div class="contradiction-item">
              ⚠️ \${escapeHtml(c.description)}
              <div class="contradiction-sources">
                Sources: <b>\${escapeHtml(c.source_a)}</b> vs <b>\${escapeHtml(c.source_b)}</b>
                (Confidence: \${Math.round(c.confidence * 100)}%)
              </div>
            </div>
          \`;
        }
        contradictionsContainer.innerHTML = \`
          <div class="contradiction-box">
            <div class="contradiction-header">⚡ Cross-Source Contradiction Detected</div>
            \${conflictsHtml}
          </div>
        \`;
      }

      // Render Synthesized Answer
      answerContainer.innerHTML = \`
        <div class="answer-card">
          <div class="answer-header">💡 TraceFind Answer (\${escapeHtml(data.model || "Llama 3.1")})</div>
          <div class="answer-text">\${escapeHtml(data.answer)}</div>
        </div>
      \`;

      // Render Source Context Chunks
      if (data.citations && data.citations.length > 0) {
        renderChunks(data.citations, "CITED CONTEXT");
      } else if (data.context_used && data.context_used.length > 0) {
        renderChunks(data.context_used, "RETRIEVED CONTEXT");
      }
    }

    function renderSearchResults(data) {
      clearContainers();
      if (data.results && data.results.length > 0) {
        renderChunks(data.results, \`MATCHES (\${data.results_count} FOUND)\`);
      } else {
        resultsContainer.innerHTML = "<p>No matching code or documentation found.</p>";
      }
    }

    function renderChunks(chunks, headerTitle) {
      let html = \`<div class="results-header">\${headerTitle}</div>\`;

      chunks.forEach((c, idx) => {
        const filePath = c.file_path || "";
        const sLine = c.start_line || 1;
        const eLine = c.end_line || sLine;
        const name = c.name || "block";
        const lang = c.language || "code";
        const snippet = c.snippet || c.content || "";

        html += \`
          <div class="chunk-card">
            <div class="chunk-meta">
              <span class="chunk-path" onclick="openLocation('\${escapeQuotes(filePath)}', \${sLine}, \${eLine})">
                \${escapeHtml(filePath)}:\${sLine}-\${eLine}
              </span>
              <span class="badge">\${escapeHtml(name)} (\${escapeHtml(lang)})</span>
            </div>
            <div class="chunk-code">\${escapeHtml(snippet)}</div>
            <div class="chunk-actions">
              <button class="btn-copy" onclick="copyContext('\${escapeQuotes(snippet)}')">Copy Context</button>
            </div>
          </div>
        \`;
      });

      resultsContainer.innerHTML = html;
    }

    function openLocation(filePath, startLine, endLine) {
      vscode.postMessage({
        type: "openFile",
        filePath: filePath,
        startLine: startLine,
        endLine: endLine
      });
    }

    function copyContext(text) {
      vscode.postMessage({
        type: "copyToClipboard",
        text: text
      });
    }

    function escapeHtml(str) {
      if (!str) return "";
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function escapeQuotes(str) {
      if (!str) return "";
      return String(str).replace(/\\\\/g, "\\\\\\\\").replace(/'/g, "\\\\'");
    }
  </script>
</body>
</html>`;
  }
}
