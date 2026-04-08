/**
 * Chat Component
 * Message bubbles with typing indicator, agent trace, and input area.
 */

import { sendMessage } from "../services/api.js";

const CHART_ICON_SM = `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M4 24 L12 12 L18 18 L28 6"/></svg>`;

const SEND_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 L11 13"/><path d="M22 2 L15 22 L11 13 L2 9 L22 2"/></svg>`;

const SUGGESTIONS = [
  "¿Cómo está el mercado de NVIDIA hoy?",
  "¿Es buen momento para invertir en renta fija?",
  "Analiza los fundamentales de Apple",
  "¿Qué sectores recomiendas para este trimestre?",
];

let messages = [];
let isLoading = false;
let userProfile = null;
let sessionId = null;

export function setProfile(profile) {
  userProfile = profile;
  sessionId = "session_" + Date.now() + "_" + Math.random().toString(36).substring(2, 8);
}

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
        <div class="typing-indicator">
          <div class="typing-indicator__dots">
            <div class="typing-indicator__dot"></div>
            <div class="typing-indicator__dot"></div>
            <div class="typing-indicator__dot"></div>
          </div>
          <span class="typing-indicator__text">Los agentes están analizando…</span>
        </div>
      </div>
    `
    : "";

  return `
    <section class="screen chat" id="chat-screen">
      <header class="chat__header">
        <div class="chat__header-logo">${CHART_ICON_SM}</div>
        <span class="chat__header-title">Finance StratAIgist</span>
        <span class="chat__header-badge">● En línea</span>
      </header>

      <div class="chat__messages" id="chat-messages">
        ${messagesHTML}
        ${loadingHTML}
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

function _renderMessage(msg, index) {
  const isUser = msg.role === "user";
  const cssClass = isUser ? "message--user" : "message--assistant";
  const avatar = isUser
    ? `<div class="message__avatar">Tú</div>`
    : `<div class="message__avatar">${CHART_ICON_SM}</div>`;

  let bubbleContent = _formatMarkdown(msg.content);

  // Agent trace for assistant messages
  let traceHTML = "";
  if (!isUser && msg.trace && msg.trace.length > 0) {
    const traceId = `trace-${index}`;
    traceHTML = `
      <button class="trace-toggle" data-trace-id="${traceId}">
        <span class="trace-toggle__arrow" id="arrow-${traceId}">▶</span>
        Ver traza de agentes (${msg.trace.length} pasos)
      </button>
      <div class="trace-content hidden" id="${traceId}">
        ${msg.trace
          .map(
            (step) => `
          <div class="trace-step">
            <div class="trace-step__agent">${step.agent}</div>
            <div class="trace-step__action">${step.action}</div>
            <div class="trace-step__result">${step.result}</div>
          </div>
        `
          )
          .join("")}
      </div>
    `;
  }

  return `
    <div class="message ${cssClass}" style="animation-delay: ${index * 50}ms">
      ${avatar}
      <div class="message__bubble">
        ${bubbleContent}
        ${traceHTML}
      </div>
    </div>
  `;
}

/** Basic markdown-like formatting */
function _formatMarkdown(text) {
  if (!text) return "";
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}

export function initChat(rerender) {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("btn-send");
  const messagesContainer = document.getElementById("chat-messages");

  // Auto-resize textarea
  if (input) {
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 150) + "px";
    });

    // Send on Enter (not Shift+Enter)
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        _handleSend(input, rerender);
      }
    });
  }

  // Send button
  if (sendBtn) {
    sendBtn.addEventListener("click", () => _handleSend(input, rerender));
  }

  // Suggestion chips
  document.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (input) {
        input.value = chip.dataset.suggestion;
        _handleSend(input, rerender);
      }
    });
  });

  // Trace toggles
  document.querySelectorAll(".trace-toggle").forEach((toggle) => {
    toggle.addEventListener("click", () => {
      const traceId = toggle.dataset.traceId;
      const content = document.getElementById(traceId);
      const arrow = document.getElementById(`arrow-${traceId}`);
      if (content) {
        content.classList.toggle("hidden");
        if (arrow) arrow.classList.toggle("open");
      }
    });
  });

  // Scroll to bottom
  _scrollToBottom(messagesContainer);
}

async function _handleSend(input, rerender) {
  const text = input?.value?.trim();
  if (!text || isLoading) return;

  // Add user message
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
    });
  } catch (err) {
    messages.push({
      role: "assistant",
      content: `**Error de conexión:** No se pudo contactar con el servidor. Asegúrate de que el backend está activo en el puerto 8045.\n\n_Detalle: ${err.message}_`,
      trace: [],
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
