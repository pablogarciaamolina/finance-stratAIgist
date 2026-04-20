/**
 * Chat Component
 *
 * Displays messages, typing indicator, and per-agent reasoning panels.
 * Reads `internal_reasoning`, `timings`, and `metadata` from the API
 * response to show the full chain-of-thought for each agent.
 */

import { sendMessage } from "../services/api.js";

/* Icons */
const CHART_ICON_SM = `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M4 24 L12 12 L18 18 L28 6"/></svg>`;

const SEND_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 L11 13"/><path d="M22 2 L15 22 L11 13 L2 9 L22 2"/></svg>`;

const SUGGESTIONS = [
  "Como esta el mercado de NVIDIA hoy?",
  "Es buen momento para invertir en renta fija?",
  "Analiza los fundamentales de Apple",
  "Que sectores recomiendas para este trimestre?",
];

/* Agent visual config */
const AGENT_STYLE = {
  orchestrator: { icon: "🧠", color: "#818cf8", label: "Orchestrator" },
  market: { icon: "📊", color: "#2dd4bf", label: "Market Agent" },
  recommendation: { icon: "💡", color: "#d4a853", label: "Recommendation Agent" },
  critic: { icon: "🔍", color: "#f472b6", label: "Critic Agent" },
  _default: { icon: "⚙️", color: "#8b95b0", label: "Agent" },
};

const TICKER_BLACKLIST = new Set(["RAG", "LLM", "API", "JSON", "USA", "ETF"]);

const COMPANY_HINTS = {
  alphabet: { company_name: "Alphabet", ticker: "GOOGL" },
  amazon: { company_name: "Amazon", ticker: "AMZN" },
  apple: { company_name: "Apple", ticker: "AAPL" },
  google: { company_name: "Alphabet", ticker: "GOOGL" },
  meta: { company_name: "Meta", ticker: "META" },
  "meta platforms": { company_name: "Meta", ticker: "META" },
  microsoft: { company_name: "Microsoft", ticker: "MSFT" },
  nvidia: { company_name: "NVIDIA", ticker: "NVDA" },
  tesla: { company_name: "Tesla", ticker: "TSLA" },
};

const TICKER_TO_COMPANY = Object.values(COMPANY_HINTS).reduce((acc, entry) => {
  acc[entry.ticker] = entry.company_name;
  return acc;
}, {});

const COMPANY_OPENERS = new Set([
  "Analiza",
  "Cada",
  "Como",
  "Cuando",
  "Cuanto",
  "Cuantos",
  "Cual",
  "Cuales",
  "Deberia",
  "Donde",
  "Hay",
  "Incluirias",
  "Que",
  "Quien",
  "Se",
  "Si",
  "Tiene",
  "Tendria",
]);

const FOLLOW_UP_MARKERS = [
  "esta accion",
  "esta empresa",
  "esta posicion",
  "esa accion",
  "esa empresa",
  "esa posicion",
  "mantenerla",
  "reducirla",
  "si ya la tengo",
  "venderla",
];

/* State */
let messages = [];
let isLoading = false;
let userProfile = null;
let sessionId = null;
let activeContext = { company_name: null, ticker: null };

export function setProfile(profile) {
  userProfile = profile;
  sessionId =
    "session_" + Date.now() + "_" + Math.random().toString(36).substring(2, 8);
  activeContext = { company_name: null, ticker: null };
}

function _agentStyle(name) {
  const key = (name || "").toLowerCase().replace(/[\s_-]+/g, "").replace("agent", "");
  for (const [k, v] of Object.entries(AGENT_STYLE)) {
    if (k !== "_default" && key.startsWith(k)) return v;
  }
  return AGENT_STYLE._default;
}

function _normalizeText(value) {
  return (value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function _escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function _escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function _escapeAttribute(str) {
  return _escapeHtml(str);
}

function _extractTickerHint(text) {
  const tickerRegex = /\(([A-Z]{1,5})\)|\b([A-Z]{2,5})\b/g;
  let match = tickerRegex.exec(text);

  while (match) {
    const candidate = (match[1] || match[2] || "").toUpperCase();
    if (candidate && !TICKER_BLACKLIST.has(candidate)) {
      return candidate;
    }
    match = tickerRegex.exec(text);
  }

  return null;
}

function _extractCompanyHint(text) {
  const normalizedText = _normalizeText(text);

  for (const [alias, context] of Object.entries(COMPANY_HINTS)) {
    const aliasRegex = new RegExp(`\\b${_escapeRegex(alias)}\\b`, "i");
    if (aliasRegex.test(normalizedText)) {
      return { ...context };
    }
  }

  const titleCaseRegex = /\b([A-Z][a-zA-Z0-9&.-]*(?:\s+[A-Z][a-zA-Z0-9&.-]*){0,3})\b/g;
  let match = titleCaseRegex.exec(text);
  while (match) {
    const candidate = match[1].trim();
    const firstWord = candidate.split(/\s+/)[0];
    if (!COMPANY_OPENERS.has(firstWord)) {
      return { company_name: candidate };
    }
    match = titleCaseRegex.exec(text);
  }

  return {};
}

function _looksLikeFollowUp(text) {
  const normalizedText = _normalizeText(text);
  return FOLLOW_UP_MARKERS.some((marker) => normalizedText.includes(marker));
}

function _resolveRequestContext(text) {
  const explicitContext = _extractCompanyHint(text);
  const ticker = _extractTickerHint(text);

  if (ticker) {
    explicitContext.ticker = ticker;
    explicitContext.company_name =
      explicitContext.company_name || TICKER_TO_COMPANY[ticker] || activeContext.company_name || null;
  }

  if (explicitContext.company_name || explicitContext.ticker) {
    return explicitContext;
  }

  if (_looksLikeFollowUp(text) && (activeContext.company_name || activeContext.ticker)) {
    return { ...activeContext };
  }

  return {};
}

function _rememberContext(metadata, fallback = {}) {
  const nextCompany = metadata?.company_name || fallback.company_name || activeContext.company_name || null;
  const nextTicker = metadata?.ticker || fallback.ticker || activeContext.ticker || null;
  activeContext = {
    company_name: nextCompany,
    ticker: nextTicker,
  };
}

function _responseNeedsResolution(response) {
  const metadata = response?.metadata || {};
  return metadata.status === "failed" && metadata.failure_stage === "orchestrator";
}

function _buildAssistantMessage(response, originalPrompt = "", requestContext = {}) {
  const metadata = response?.metadata || {};
  const needsResolution = _responseNeedsResolution(response);

  return {
    role: "assistant",
    content: response?.response || "",
    trace: response?.agent_trace || [],
    metadata,
    resolution: needsResolution
      ? {
          required: true,
          pending: false,
          company_name: requestContext.company_name || metadata.request_company_name || "",
          ticker: requestContext.ticker || metadata.request_ticker || "",
          originalPrompt,
          error: "",
        }
      : null,
  };
}

function _renderResolutionCard(resolution, index) {
  const retryLabel = resolution.pending ? "Reintentando..." : "Reintentar con contexto";

  return `
    <form class="resolution-card" data-resolution-form="${index}">
      <div class="resolution-card__title">Falta la empresa para continuar</div>
      <p class="resolution-card__hint">
        No he identificado con suficiente claridad la empresa o el ticker.
        Anade al menos uno de los dos campos y reintento la misma consulta.
      </p>
      <div class="resolution-card__fields">
        <label class="resolution-card__field">
          <span class="resolution-card__label">Company Name</span>
          <input
            class="resolution-card__input"
            id="resolution-company-${index}"
            type="text"
            value="${_escapeAttribute(resolution.company_name || "")}"
            placeholder="Apple"
            ${resolution.pending ? "disabled" : ""}
          />
        </label>
        <label class="resolution-card__field">
          <span class="resolution-card__label">Ticker</span>
          <input
            class="resolution-card__input"
            id="resolution-ticker-${index}"
            type="text"
            value="${_escapeAttribute(resolution.ticker || "")}"
            placeholder="AAPL"
            ${resolution.pending ? "disabled" : ""}
          />
        </label>
      </div>
      ${resolution.error ? `<div class="resolution-card__error">${_escapeHtml(resolution.error)}</div>` : ""}
      <div class="resolution-card__actions">
        <button class="resolution-card__button" type="submit" ${resolution.pending ? "disabled" : ""}>
          ${retryLabel}
        </button>
      </div>
    </form>
  `;
}

/* Render */
export function renderChat() {
  const hasMessages = messages.length > 0;

  const messagesHTML = hasMessages
    ? messages.map((msg, i) => _renderMessage(msg, i)).join("")
    : `
      <div class="chat__welcome">
        <h2>Perfil configurado</h2>
        <p>
          Tu perfil de inversor ha sido registrado. Hazme cualquier pregunta
          sobre mercados, acciones o estrategias de inversion.
        </p>
        <div class="suggestions">
          ${SUGGESTIONS.map(
            (suggestion) =>
              `<button class="suggestion-chip" data-suggestion="${suggestion}">${suggestion}</button>`
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
            <span class="typing-indicator__text">Los agentes estan analizando...</span>
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
          <span class="chat__header-badge">● En linea</span>
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
            placeholder="Escribe tu consulta financiera..."
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

/* Single message */
function _renderMessage(msg, index) {
  const isUser = msg.role === "user";
  const cssClass = isUser ? "message--user" : "message--assistant";
  const avatar = isUser
    ? `<div class="message__avatar">Tu</div>`
    : `<div class="message__avatar">${CHART_ICON_SM}</div>`;

  const bubbleContent = _formatMarkdown(msg.content);

  let metaBadge = "";
  if (!isUser && msg.metadata) {
    const parts = [];
    if (msg.metadata.company_name) parts.push(msg.metadata.company_name);
    if (msg.metadata.ticker) parts.push(msg.metadata.ticker);
    const latency = msg.metadata.total_latency || msg.metadata.api_total_latency;
    if (latency) parts.push(`${latency.toFixed(1)}s`);
    if (parts.length) {
      metaBadge = `<div class="message__meta">${_escapeHtml(parts.join(" · "))}</div>`;
    }
  }

  const resolutionHTML =
    !isUser && msg.resolution?.required ? _renderResolutionCard(msg.resolution, index) : "";

  let traceHTML = "";
  if (!isUser && msg.trace && msg.trace.length > 0) {
    const timings = msg.metadata?.timings || {};
    const reasoning = msg.metadata?.internal_reasoning || {};

    const panelsHTML = msg.trace
      .map((step, stepIndex) => {
        const style = _agentStyle(step.agent);
        const panelId = `trace-${index}-${stepIndex}`;
        const agentKey = step.agent.toLowerCase().replace(/[\s_-]+/g, "").replace("agent", "");
        const agentTime = timings[agentKey];
        const timeStr = agentTime ? ` · ${agentTime.toFixed(1)}s` : "";
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
                <span class="trace-panel__label">Accion</span>
                <span class="trace-panel__value">${_escapeHtml(step.action)}</span>
              </div>
              ${step.result ? `
              <div class="trace-panel__row">
                <span class="trace-panel__label">Resultado</span>
                <span class="trace-panel__value">${_escapeHtml(step.result)}</span>
              </div>` : ""}
              ${reasoningText ? `
              <div class="trace-panel__reasoning">
                <div class="trace-panel__reasoning-label">Razonamiento del modelo</div>
                <div class="trace-panel__reasoning-text">${_formatReasoning(reasoningText)}</div>
              </div>` : ""}
            </div>
          </div>
        `;
      })
      .join("");

    const totalLatency = msg.metadata?.total_latency || msg.metadata?.api_total_latency;
    const latencyLabel = totalLatency ? ` · ${totalLatency.toFixed(1)}s total` : "";
    traceHTML = `
      <div class="trace-container">
        <div class="trace-header">
          <span class="trace-header__label">Traza de agentes (${msg.trace.length} pasos${latencyLabel})</span>
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
        ${resolutionHTML}
        ${traceHTML}
      </div>
    </div>
  `;
}

/* Markdown formatting */
function _formatMarkdown(text) {
  if (!text) return "";
  const safe = _escapeHtml(text);
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

function _formatReasoning(text) {
  if (!text) return "";
  const safe = _escapeHtml(text.trim());
  return safe
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}

/* Event listeners */
export function initChat(rerender) {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("btn-send");
  const scrollContainer = document.querySelector(".chat__body");

  if (input) {
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 150) + "px";
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
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

  document.querySelectorAll("[data-resolution-form]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const index = Number(form.dataset.resolutionForm);
      _handleResolutionRetry(index, rerender);
    });
  });

  _scrollToBottom(scrollContainer);
}

/* Send handler */
async function _handleSend(input, rerender) {
  const text = input?.value?.trim();
  if (!text || isLoading) return;

  const requestContext = _resolveRequestContext(text);

  messages.push({ role: "user", content: text });
  input.value = "";
  input.style.height = "auto";
  isLoading = true;
  rerender();

  try {
    const response = await sendMessage(text, userProfile, sessionId, requestContext);
    _rememberContext(response.metadata || {}, requestContext);
    messages.push(_buildAssistantMessage(response, text, requestContext));
  } catch (err) {
    messages.push({
      role: "assistant",
      content:
        `**Error de conexion:** No se pudo contactar con el servidor. ` +
        `Asegurate de que el backend esta activo en el puerto 8045.\n\n` +
        `_Detalle: ${err.message}_`,
      trace: [],
      metadata: {},
      resolution: null,
    });
  } finally {
    isLoading = false;
    rerender();
  }
}

async function _handleResolutionRetry(messageIndex, rerender) {
  const message = messages[messageIndex];
  if (!message?.resolution?.required || isLoading) return;

  const companyInput = document.getElementById(`resolution-company-${messageIndex}`);
  const tickerInput = document.getElementById(`resolution-ticker-${messageIndex}`);

  const company_name = companyInput?.value?.trim() || "";
  const ticker = (tickerInput?.value?.trim() || "").toUpperCase();

  if (!company_name && !ticker) {
    messages[messageIndex] = {
      ...message,
      resolution: {
        ...message.resolution,
        company_name,
        ticker,
        error: "Indica al menos la empresa o el ticker.",
      },
    };
    rerender();
    return;
  }

  messages[messageIndex] = {
    ...message,
    resolution: {
      ...message.resolution,
      company_name,
      ticker,
      pending: true,
      error: "",
    },
  };
  isLoading = true;
  rerender();

  const retryContext = {
    company_name: company_name || undefined,
    ticker: ticker || undefined,
  };

  try {
    const response = await sendMessage(
      message.resolution.originalPrompt,
      userProfile,
      sessionId,
      retryContext
    );

    _rememberContext(response.metadata || {}, retryContext);

    messages[messageIndex] = {
      ...messages[messageIndex],
      resolution: {
        ...messages[messageIndex].resolution,
        company_name,
        ticker,
        pending: false,
        required: false,
        error: "",
      },
    };

    messages.push(
      _buildAssistantMessage(response, message.resolution.originalPrompt, retryContext)
    );
  } catch (err) {
    messages[messageIndex] = {
      ...messages[messageIndex],
      resolution: {
        ...messages[messageIndex].resolution,
        company_name,
        ticker,
        pending: false,
        error: `No se pudo reintentar la consulta: ${err.message}`,
      },
    };
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
  activeContext = { company_name: null, ticker: null };
}
