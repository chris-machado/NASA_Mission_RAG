/* ==========================================================================
   Ask NASA — Chat Frontend
   Handles SSE streaming, message rendering, and markdown
   ========================================================================== */

const messagesEl = document.getElementById('messages');
const emptyState = document.getElementById('emptyState');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const infoToggle = document.getElementById('infoToggle');
const infoPanel = document.getElementById('infoPanel');

let isStreaming = false;

/* ---------- Info Panel Toggle ---------- */

infoToggle.addEventListener('click', () => {
  const isOpen = !infoPanel.hidden;
  infoPanel.hidden = isOpen;
  infoToggle.classList.toggle('active', !isOpen);
});

/* ---------- Submit ---------- */

function handleSubmit(e) {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question || isStreaming) return;
  sendMessage(question);
}

function askSuggestion(btn) {
  if (isStreaming) return;
  sendMessage(btn.textContent.trim());
}

/* ---------- Send & Stream ---------- */

async function sendMessage(question) {
  isStreaming = true;
  sendBtn.disabled = true;
  questionInput.value = '';

  // Hide empty state
  if (emptyState) {
    emptyState.style.display = 'none';
  }

  // Add user message
  addMessage(question, 'user');

  // Add thinking indicator
  const thinkingEl = addThinking();

  try {
    const response = await fetch('api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Server error (${response.status})`);
    }

    // Remove thinking indicator, add assistant message
    thinkingEl.remove();
    const assistantEl = addMessage('', 'assistant');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';
    let sourcesData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        const data = JSON.parse(line.slice(6));

        if (data.token) {
          fullText += data.token;
          assistantEl.innerHTML = marked.parse(fullText);
          scrollToBottom();
        }

        if (data.sources) {
          sourcesData = data.sources;
        }

        if (data.error) {
          assistantEl.remove();
          addMessage(data.error, 'error');
        }

        if (data.done) {
          assistantEl.innerHTML = marked.parse(fullText);
          if (sourcesData) {
            addSources(assistantEl, sourcesData);
          }
          scrollToBottom();
        }
      }
    }
  } catch (err) {
    thinkingEl.remove();
    addMessage(err.message || 'Something went wrong. Please try again.', 'error');
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

/* ---------- DOM Helpers ---------- */

function addMessage(text, type) {
  const div = document.createElement('div');
  div.className = `message message-${type}`;

  if (type === 'user') {
    div.textContent = text;
  } else if (type === 'error') {
    div.className = 'message-error';
    div.textContent = text;
  } else {
    div.innerHTML = text ? marked.parse(text) : '';
  }

  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function addThinking() {
  const div = document.createElement('div');
  div.className = 'message message-assistant thinking';
  div.innerHTML = '<div class="thinking-dot"></div><div class="thinking-dot"></div><div class="thinking-dot"></div>';
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function addSources(assistantEl, sources) {
  const container = document.createElement('div');
  container.className = 'message-sources';

  const label = document.createElement('span');
  label.className = 'sources-label';
  label.textContent = 'Sources';
  container.appendChild(label);

  for (const src of sources) {
    const a = document.createElement('a');
    a.href = src.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.className = 'source-link';
    a.textContent = src.title;
    container.appendChild(a);
  }

  assistantEl.appendChild(container);
  scrollToBottom();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------- Keyboard ---------- */

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSubmit(e);
  }
});
