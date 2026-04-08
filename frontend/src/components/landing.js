/**
 * Landing Page Component
 * Full-screen welcome with animated gradient, project name, and CTA button.
 */

const CHART_ICON = `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 24 L12 12 L18 18 L28 6"/><circle cx="28" cy="6" r="2" fill="currentColor"/></svg>`;

export function renderLanding(onStart) {
  return `
    <div class="bg-gradient"></div>
    <section class="screen landing" id="landing-screen">
      <div class="landing__logo">
        <div class="landing__logo-icon">${CHART_ICON}</div>
      </div>

      <h1 class="landing__title">Finance StratAIgist</h1>

      <p class="landing__subtitle">
        Tu asesor financiero inteligente. Un sistema multiagente que analiza mercados,
        genera recomendaciones personalizadas y valida cada resultado.
      </p>

      <div class="landing__features">
        <div class="landing__feature">
          <span class="landing__feature-icon">📊</span>
          Datos de mercado en tiempo real
        </div>
        <div class="landing__feature">
          <span class="landing__feature-icon">🤖</span>
          Análisis multiagente con IA
        </div>
        <div class="landing__feature">
          <span class="landing__feature-icon">🛡️</span>
          Validación automática
        </div>
      </div>

      <button class="btn-primary" id="btn-start">
        Comenzar análisis
        <span>→</span>
      </button>
    </section>
  `;
}

export function initLanding(onStart) {
  const btn = document.getElementById("btn-start");
  if (btn) {
    btn.addEventListener("click", onStart);
  }
}
