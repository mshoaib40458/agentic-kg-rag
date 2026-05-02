/**
 * KG-RAG Enterprise Assistant — Chat UI
 * WebSocket streaming, JWT auth, citations panel, upload, graph explorer
 */

const API_BASE = "http://localhost:8000";
const WS_BASE  = "ws://localhost:8000";

let authToken   = null;
let currentUser = null;
let sessionId   = generateUUID();
let isStreaming = false;
let conversationHistory = [];
let currentCitations = [];

// ── Utilities ─────────────────────────────────────────────────
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function formatTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ── Auth ──────────────────────────────────────────────────────
document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;
  const btn      = document.getElementById('login-btn');
  const spinner  = document.getElementById('login-spinner');
  const errEl    = document.getElementById('login-error');

  btn.disabled = true;
  spinner.classList.remove('hidden');
  errEl.classList.add('hidden');

  try {
    const form = new FormData();
    form.append('username', username);
    form.append('password', password);

    const res = await fetch(`${API_BASE}/api/auth/token`, { method: 'POST', body: form });
    if (!res.ok) throw new Error('Invalid credentials');
    const data = await res.json();

    authToken   = data.access_token;
    currentUser = { username, role: data.role, user_id: data.user_id };

    // Update UI
    document.getElementById('user-name-display').textContent = username;
    document.getElementById('user-role-display').textContent = data.role;
    document.getElementById('user-avatar').textContent = username[0].toUpperCase();
    document.getElementById('session-id-display').textContent = 'Session: ' + sessionId.slice(0, 8);

    document.getElementById('login-modal').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');

    loadStats();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
  }
});

function logout() {
  authToken = null; currentUser = null;
  document.getElementById('app').classList.add('hidden');
  document.getElementById('login-modal').classList.remove('hidden');
}

// ── Navigation ────────────────────────────────────────────────
function switchView(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const viewEl = document.getElementById(`view-${view}`);
  if (viewEl) viewEl.classList.add('active');
  const navEl = document.getElementById(`nav-${view}`);
  if (navEl) navEl.classList.add('active');
  if (view === 'graph') loadStats();
}

// ── Query (WebSocket Streaming) ───────────────────────────────
async function sendQuery() {
  const input = document.getElementById('query-input');
  const query = input.value.trim();
  if (!query || isStreaming) return;

  isStreaming = true;
  document.getElementById('send-btn').disabled = true;

  // Hide welcome screen
  document.getElementById('welcome-screen').style.display = 'none';

  // Add user message
  appendMessage('user', query);
  conversationHistory.push({ role: 'user', content: query });
  input.value = '';
  autoResize(input);

  // Reset reasoning panel
  clearReasoningSteps();
  const pulseEl = document.getElementById('reasoning-pulse');
  pulseEl.classList.add('active');
  document.getElementById('reasoning-panel').classList.remove('collapsed');

  // Add assistant message shell
  const assistantMsgId = 'msg-' + generateUUID();
  appendAssistantShell(assistantMsgId);

  // Open WebSocket
  const wsUrl = `${WS_BASE}/api/stream?token=${authToken}`;
  let ws;

  try {
    ws = new WebSocket(wsUrl);
  } catch (e) {
    // Fallback to REST API if WebSocket fails
    await sendQueryREST(query, assistantMsgId);
    return;
  }

  let answerBuffer = '';
  let finalData = null;

  ws.onopen = () => {
    ws.send(JSON.stringify({
      query,
      session_id: sessionId,
      conversation_history: conversationHistory.slice(-8),
    }));
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'reasoning_step') {
      addReasoningStep(data.content);
    } else if (data.type === 'final_answer') {
      finalData = data;
      renderAnswer(assistantMsgId, data);
    } else if (data.type === 'done') {
      finalizeMessage(assistantMsgId, finalData);
    } else if (data.type === 'error') {
      renderError(assistantMsgId, data.content);
    }
  };

  ws.onerror = () => {
    sendQueryREST(query, assistantMsgId);
  };

  ws.onclose = () => {
    isStreaming = false;
    document.getElementById('send-btn').disabled = false;
    pulseEl.classList.remove('active');
  };
}

// REST fallback
async function sendQueryREST(query, msgId) {
  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        conversation_history: conversationHistory.slice(-8),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Query failed');

    renderAnswer(msgId, {
      content: data.answer,
      confidence: data.confidence,
      citations: data.citations,
      sources: data.sources,
      conflicting_info: data.conflicting_info,
    });

    conversationHistory.push({ role: 'assistant', content: data.answer });
  } catch (err) {
    renderError(msgId, err.message);
  } finally {
    isStreaming = false;
    document.getElementById('send-btn').disabled = false;
    document.getElementById('reasoning-pulse').classList.remove('active');
  }
}

// ── Message Rendering ─────────────────────────────────────────
function appendMessage(role, content) {
  const area = document.getElementById('messages-area');
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="message-avatar">${role === 'user' ? (currentUser?.username[0].toUpperCase() || 'U') : '🤖'}</div>
    <div class="message-body">
      <div class="message-content">${escapeHtml(content)}</div>
      <div class="message-meta">${formatTime()}</div>
    </div>`;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

function appendAssistantShell(msgId) {
  const area = document.getElementById('messages-area');
  const div = document.createElement('div');
  div.id = msgId;
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-body">
      <div class="message-content" id="${msgId}-content">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

function renderAnswer(msgId, data) {
  const content = data.content || data.final_answer || '';
  const confidence = data.confidence || 0;
  const citations = data.citations || [];
  const conflicts = data.conflicting_info || [];
  const sources = data.sources || [];

  currentCitations = sources;

  const contentEl = document.getElementById(`${msgId}-content`);
  if (!contentEl) return;

  // Format citations as clickable tags
  let formatted = escapeHtml(content).replace(
    /\[Source:\s*([^\]]+)\]/g,
    '<span class="citation-tag" title="$1">📎 $1</span>'
  );

  let conflictHtml = '';
  conflicts.forEach(c => {
    conflictHtml += `<div class="conflict-banner">⚠️ ${escapeHtml(c)}</div>`;
  });

  const confLevel = confidence > 0.75 ? 'high' : confidence > 0.5 ? 'medium' : 'low';
  const confPct = Math.round(confidence * 100);

  const bodyEl = document.getElementById(msgId)?.querySelector('.message-body');
  if (bodyEl) {
    bodyEl.innerHTML = `
      <div class="message-content" id="${msgId}-content">${formatted}${conflictHtml}</div>
      <div class="message-meta">
        ${formatTime()}
        <span class="confidence-badge ${confLevel}">◆ ${confPct}% confidence</span>
        ${citations.length > 0 ? `<button class="show-sources-btn" onclick="showCitations()">📎 ${citations.length} sources</button>` : ''}
      </div>`;
  }

  // Add to conversation history
  conversationHistory.push({ role: 'assistant', content });

  document.getElementById('messages-area').scrollTop = document.getElementById('messages-area').scrollHeight;
}

function finalizeMessage(msgId, finalData) {
  if (finalData) renderAnswer(msgId, finalData);
}

function renderError(msgId, message) {
  const contentEl = document.getElementById(`${msgId}-content`);
  if (contentEl) {
    contentEl.innerHTML = `<span style="color:var(--error)">⚠️ ${escapeHtml(message)}</span>`;
  }
}

// ── Reasoning Steps ──────────────────────────────────────────
function addReasoningStep(step) {
  const container = document.getElementById('reasoning-steps');
  const el = document.createElement('div');
  el.className = 'reasoning-step ' + getStepClass(step);
  el.textContent = step;
  container.appendChild(el);
  const body = document.getElementById('reasoning-body');
  body.scrollTop = body.scrollHeight;
}

function getStepClass(step) {
  if (step.includes('[PLANNER]')) return 'planner';
  if (step.includes('[EXECUTOR]')) return 'executor';
  if (step.includes('[VALIDATOR]')) return 'validator';
  if (step.includes('[SYNTHESIZER]')) return 'synthesizer';
  return '';
}

function clearReasoningSteps() {
  document.getElementById('reasoning-steps').innerHTML = '';
}

function toggleReasoning() {
  const panel = document.getElementById('reasoning-panel');
  panel.classList.toggle('collapsed');
}

// ── Citations Panel ───────────────────────────────────────────
function showCitations() {
  const panel = document.getElementById('citations-panel');
  const list  = document.getElementById('citations-list');
  panel.classList.remove('hidden');

  list.innerHTML = currentCitations.map((s, i) => `
    <div class="citation-item">
      <div class="citation-filename">📄 ${escapeHtml(s.filename || 'Document')}</div>
      <div class="citation-snippet">${escapeHtml(s.content_snippet || '').slice(0, 200)}...</div>
      <div class="citation-score">Score: ${(s.score * 100).toFixed(1)}% · ${s.source_type}</div>
    </div>`).join('') || '<p style="color:var(--text-muted);font-size:12px;padding:12px">No sources available</p>';
}

function closeCitations() {
  document.getElementById('citations-panel').classList.add('hidden');
}

// ── Input Helpers ─────────────────────────────────────────────
function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function useExample(btn) {
  const input = document.getElementById('query-input');
  input.value = btn.textContent;
  autoResize(input);
  input.focus();
}

// ── Upload ────────────────────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.add('drag-over');
}
function handleDragLeave(e) {
  document.getElementById('upload-zone').classList.remove('drag-over');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag-over');
  uploadFiles(e.dataTransfer.files);
}
function handleFileSelect(e) {
  uploadFiles(e.target.files);
}

async function uploadFiles(files) {
  const queue = document.getElementById('upload-queue');
  for (const file of files) {
    const itemId = 'upload-' + generateUUID();
    const item = document.createElement('div');
    item.id = itemId; item.className = 'upload-item';
    item.innerHTML = `
      <span class="upload-item-name">${escapeHtml(file.name)}</span>
      <span class="upload-item-status uploading" id="${itemId}-status">Uploading…</span>`;
    queue.appendChild(item);

    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${API_BASE}/api/ingest`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${authToken}` },
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      document.getElementById(`${itemId}-status`).textContent = `✓ ${data.chunks} chunks | ${data.entities} entities`;
      document.getElementById(`${itemId}-status`).className = 'upload-item-status success';
    } catch (err) {
      document.getElementById(`${itemId}-status`).textContent = '✗ ' + err.message;
      document.getElementById(`${itemId}-status`).className = 'upload-item-status error';
    }
  }
}

// ── Graph Stats ───────────────────────────────────────────────
async function loadStats() {
  if (!authToken) return;
  try {
    const res = await fetch(`${API_BASE}/api/admin/stats`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById('stat-entities').textContent = data.graph_store?.entity_count ?? '–';
    document.getElementById('stat-relations').textContent = data.graph_store?.relationship_count ?? '–';
    document.getElementById('stat-vectors').textContent = data.vector_store?.total_vectors ?? '–';
  } catch (e) { /* Not admin — stats hidden */ }
}

async function queryGraph() {
  const entity = document.getElementById('graph-entity-input').value.trim();
  if (!entity) return;
  const display = document.getElementById('graph-results-display');
  display.innerHTML = '<p style="color:var(--text-muted);font-size:12px">Querying graph…</p>';
  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ query: `Show all relationships for entity: ${entity}`, session_id: generateUUID() }),
    });
    const data = await res.json();
    const paths = data.reasoning_trace?.filter(t => t.includes('Graph')) || [];
    display.innerHTML = paths.length
      ? paths.map(p => `<div class="graph-path-item">${escapeHtml(p)}</div>`).join('')
      : `<div class="graph-path-item">No specific graph paths found. Answer: ${escapeHtml(data.answer?.slice(0, 200) || '')}</div>`;
  } catch (e) {
    display.innerHTML = `<p style="color:var(--error);font-size:12px">Error: ${e.message}</p>`;
  }
}
