/**
 * Chat Component
 *
 * Displays messages, typing indicator, and per-agent reasoning panels.
 * Reads `internal_reasoning`, `timings`, and `metadata` from the API
 * response to show the full chain-of-thought for each agent.
 */

import { sendMessage } from "../services/api.js";

/* ── Icons ─────────────────────────────────────────────────── */
const CHART_ICON_SM = `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M4 24 L12 12 L18 18 L28 6"/></svg>`;

const SEND_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 L11 13"/><path d="M22 2 L15 22 L11 13 L2 9 L22 2"/></svg>`;

const SUGGESTIONS = [
  "¿Cómo está el mercado de NVIDIA hoy?",
  "¿Es buen momento para invertir en renta fija?",
  "Analiza los fundamentales de Apple",
  "¿Qué sectores recomiendas para este trimestre?",
];

/* ── Agent visual config ───────────────────────────────────── */
const AGENT_STYLE = {
  orchestrator:   { icon: "🧠", color: "#818cf8", label: "Orchestrator" },
  market:         { icon: "📊", color: "#2dd4bf", label: "Market Agent" },
  recommendation: { icon: "💡", color: "#d4a853", label: "Recommendation Agent" },
  critic:         { icon: "🔍", color: "#f472b6", label: "Critic Agent" },
  _default:       { icon: "⚙️", color: "#8b95b0", label: "Agent" },
};

function _agentStyle(name) {
  const key = (name || "").toLowerCase().replace(/[\s_-]+/g, "").replace("agent", "");
  for (const [k, v] of Object.entries(AGENT_STYLE)) {
    if (k !== "_default" && key.startsWith(k)) return v;
  }
  return AGENT_STYLE._default;
}

/* ── State ─────────────────────────────────────────────────── */
let messages = [];
let isLoading = false;
let userProfile = null;
let sessionId = null;

export function setProfile(profile) {
  userProfile = profile;
  sessionId =
    "session_" + Date.now() + "_" + Math.random().toString(36).substring(2, 8);
}

/* ── Render ────────────────────────────────────────────────── */
export function renderChat() {
  const hasMessages = messages.length > 0;

  const messagesHTML = hasMessages
    ? messages.map((msg, i) => _renderMessage(msg, i)).join("")
    : `
      <div class="chat__welcome">
        <h2>¡Perfil configurado! 🎯</h2>
        <p>
          Tu perfil de inversor ha sido registrado. Hazme cualquier pregunta
          sobre mercados, acciones, o estrategias de inversión.
        </p>
        <div class="suggestions">
          ${SUGGESTIONS.map(
            (s) =>
              `<button class="suggestion-chip" data-suggestion="${s}">${s}</button>`
          ).join("")}
        </div>
      </div>
    `;

  const loadingHTML = isLoading
    ? `
      <div class="message message--assistant">
        <div class="message__avatar">${CHART_ICON_SM}</div>
        <div class="message__bubble">
          <div class="typing-indicator">
            <div class="typing-indicator__dots">
              <div class="typing-indicator__dot"></div>
              <div class="typing-indicator__dot"></div>
              <div class="typing-indicator__dot"></div>
            </div>
            <span class="typing-indicator__text">Los agentes están analizando…</span>
          </div>
        </div>
      </div>
    `
    : "";

  return `
    <section class="screen chat" id="chat-screen">
      <header class="chat__header">
        <div class="chat__header-inner">
          <div class="chat__header-logo">${CHART_ICON_SM}</div>
          <span class="chat__header-title">Finance StratAIgist</span>
          <span class="chat__header-badge">● En línea</span>
        </div>
      </header>

      <div class="chat__body">
        <div class="chat__messages" id="chat-messages">
          ${messagesHTML}
          ${loadingHTML}
        </div>
      </div>

      <div class="chat__input-area">
        <div class="chat__input-wrapper">
          <textarea
            class="chat__input"
            id="chat-input"
            placeholder="Escribe tu consulta financiera…"
            rows="1"
            ${isLoading ? "disabled" : ""}
          ></textarea>
          <button class="chat__send-btn" id="btn-send" ${isLoading ? "disabled" : ""}>
            ${SEND_ICON}
          </button>
        </div>
      </div>
    </section>
  `;
}

/* ── Single message ────────────────────────────────────────── */
function _renderMessage(msg, index) {
  const isUser = msg.role === "user";
  const cssClass = isUser ? "message--user" : "message--assistant";
  const avatar = isUser
    ? `<div class="message__avatar">Tú</div>`
    : `<div class="message__avatar">${CHART_ICON_SM}</div>`;

  let bubbleContent = _formatMarkdown(msg.content);

  /* ── Metadata badge (company, ticker, latency) ──────────── */
  let metaBadge = "";
  if (!isUser && msg.metadata) {
    const parts = [];
    if (msg.metadata.company_name) parts.push(msg.metadata.company_name);
    if (msg.metadata.ticker) parts.push(msg.metadata.ticker);
    const lat = msg.metadata.total_latency || msg.metadata.api_total_latency;
    if (lat) parts.push(`${lat.toFixed(1)}s`);
    if (parts.length) {
      metaBadge = `<div class="message__meta">${_escapeHtml(parts.join(" · "))}</div>`;
    }
  }

  /* ── Agent trace with reasoning ─────────────────────────── */
  let traceHTML = "";
  if (!isUser && msg.trace && msg.trace.length > 0) {
    const timings = msg.metadata?.timings || {};
    const reasoning = msg.metadata?.internal_reasoning || {};

    const panelsHTML = msg.trace
      .map((step, si) => {
        const style = _agentStyle(step.agent);
        const panelId = `trace-${index}-${si}`;
        const agentKey = step.agent.toLowerCase().replace(/[\s_-]+/g, "").replace("agent", "");
        const agentTime = timings[agentKey];
        const timeStr = agentTime ? ` · ${agentTime.toFixed(1)}s` : "";

        // Find matching internal reasoning
        const reasoningText = reasoning[agentKey] || "";

        return `
          <div class="trace-panel" style="--agent-color: ${style.color}">
            <button class="trace-panel__header" data-panel-id="${panelId}">
              <span class="trace-panel__icon">${style.icon}</span>
              <span class="trace-panel__name">${_escapeHtml(step.agent)}${timeStr}</span>
              <span class="trace-panel__arrow" id="arrow-${panelId}">▶</span>
            </button>
            <div class="trace-panel__body hidden" id="${panelId}">
              <div class="trace-panel__row">
                <span class="trace-panel__label">Acción</span>
                <span class="trace-panel__value">${_escapeHtml(step.action)}</span>
              </div>
              ${step.result ? `
              <div class="trace-panel__row">
                <span class="trace-panel__label">Resultado</span>
                <span class="trace-panel__value">${_escapeHtml(step.result)}</span>
              </div>` : ""}
              ${reasoningText ? `
              <div class="trace-panel__reasoning">
                <div class="trace-panel__reasoning-label">💭 Razonamiento del modelo</div>
                <div class="trace-panel__reasoning-text">${_formatReasoning(reasoningText)}</div>
              </div>` : ""}
            </div>
          </div>
        `;
      })
      .join("");

    const totalLat = msg.metadata?.total_latency || msg.metadata?.api_total_latency;
    const latStr = totalLat ? ` · ${totalLat.toFixed(1)}s total` : "";
    traceHTML = `
      <div class="trace-container">
        <div class="trace-header">
          <span class="trace-header__label">🔗 Traza de agentes (${msg.trace.length} pasos${latStr})</span>
        </div>
        ${panelsHTML}
      </div>
    `;
  }

  return `
    <div class="message ${cssClass}" style="animation-delay: ${index * 50}ms">
      ${avatar}
      <div class="message__bubble">
        ${bubbleContent}
        ${metaBadge}
        ${traceHTML}
      </div>
    </div>
  `;
}

/* ── Markdown formatting ───────────────────────────────────── */
function _formatMarkdown(text) {
  if (!text) return "";
  let safe = _escapeHtml(text);
  return safe
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}

/** Format reasoning text — trim, add paragraph breaks, keep it readable */
function _formatReasoning(text) {
  if (!text) return "";
  let safe = _escapeHtml(text.trim());
  // Break into paragraphs on double newlines
  return safe
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}

function _escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ── Event listeners ───────────────────────────────────────── */
export function initChat(rerender) {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("btn-send");
  const scrollContainer = document.querySelector(".chat__body");

  if (input) {
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 150) + "px";
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        _handleSend(input, rerender);
      }
    });

    if (!isLoading) input.focus();
  }

  if (sendBtn) {
    sendBtn.addEventListener("click", () => _handleSend(input, rerender));
  }

  document.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (input) {
        input.value = chip.dataset.suggestion;
        _handleSend(input, rerender);
      }
    });
  });

  // Per-agent trace panel toggles
  document.querySelectorAll(".trace-panel__header").forEach((header) => {
    header.addEventListener("click", () => {
      const panelId = header.dataset.panelId;
      const body = document.getElementById(panelId);
      const arrow = document.getElementById(`arrow-${panelId}`);
      if (body) {
        body.classList.toggle("hidden");
        if (arrow) arrow.classList.toggle("open");
      }
    });
  });

  _scrollToBottom(scrollContainer);
}

/* ── Send handler ──────────────────────────────────────────── */
async function _handleSend(input, rerender) {
  const text = input?.value?.trim();
  if (!text || isLoading) return;

  messages.push({ role: "user", content: text });
  input.value = "";
  input.style.height = "auto";
  isLoading = true;
  rerender();

  try {
    const response = await sendMessage(text, userProfile, sessionId);
    messages.push({
      role: "assistant",
      content: response.response,
      trace: response.agent_trace || [],
      metadata: response.metadata || {},
    });
  } catch (err) {
    messages.push({
      role: "assistant",
      content: `**Error de conexión:** No se pudo contactar con el servidor. Asegúrate de que el backend está activo en el puerto 8045.\n\n_Detalle: ${err.message}_`,
      trace: [],
      metadata: {},
    });
  } finally {
    isLoading = false;
    rerender();
  }
}

function _scrollToBottom(container) {
  if (container) {
    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
  }
}

/** Reset chat state */
export function resetChat() {
  messages = [];
  isLoading = false;
  userProfile = null;
  sessionId = null;
}
