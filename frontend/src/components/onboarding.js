/**
 * Onboarding Component
 * Step-by-step questionnaire to collect the user's investment profile.
 *
 * Steps:
 *  1. Risk tolerance    (card selector)
 *  2. Investment horizon (card selector)
 *  3. Capital amount    (numeric input)
 *  4. Investment goals  (multi-select chips)
 */

const STEPS = [
  {
    key: "risk_level",
    label: "Paso 1 de 4",
    question: "¿Cuál es tu tolerancia al riesgo?",
    type: "cards",
    options: [
      {
        value: "conservative",
        icon: "🛡️",
        title: "Conservador",
        desc: "Priorizo la seguridad de mi capital sobre la rentabilidad.",
      },
      {
        value: "moderate",
        icon: "⚖️",
        title: "Moderado",
        desc: "Busco equilibrio entre riesgo y rentabilidad.",
      },
      {
        value: "aggressive",
        icon: "🚀",
        title: "Agresivo",
        desc: "Acepto alta volatilidad a cambio de mayores rendimientos.",
      },
    ],
  },
  {
    key: "investment_horizon",
    label: "Paso 2 de 4",
    question: "¿Cuál es tu horizonte de inversión?",
    type: "cards",
    options: [
      {
        value: "short",
        icon: "⚡",
        title: "Corto plazo",
        desc: "Menos de 1 año.",
      },
      {
        value: "medium",
        icon: "📅",
        title: "Medio plazo",
        desc: "Entre 1 y 5 años.",
      },
      {
        value: "long",
        icon: "🏔️",
        title: "Largo plazo",
        desc: "Más de 5 años.",
      },
    ],
  },
  {
    key: "capital_amount",
    label: "Paso 3 de 4",
    question: "¿Cuánto capital deseas invertir?",
    type: "capital",
  },
  {
    key: "investment_goals",
    label: "Paso 4 de 4",
    question: "¿Cuáles son tus objetivos de inversión?",
    type: "chips",
    options: [
      { value: "growth", label: "📈 Crecimiento" },
      { value: "income", label: "💰 Ingresos pasivos" },
      { value: "preservation", label: "🏦 Preservación de capital" },
      { value: "speculation", label: "🎯 Especulación" },
    ],
  },
];

let currentStep = 0;
let profile = {
  risk_level: null,
  investment_horizon: null,
  capital_amount: 10000,
  investment_goals: [],
};

export function renderOnboarding() {
  const step = STEPS[currentStep];
  const progress = ((currentStep + 1) / STEPS.length) * 100;

  let contentHTML = "";

  if (step.type === "cards") {
    contentHTML = `
      <div class="card-selector" id="card-selector">
        ${step.options
          .map(
            (opt) => `
          <div class="card-option ${profile[step.key] === opt.value ? "selected" : ""}"
               data-value="${opt.value}">
            <div class="card-option__icon">${opt.icon}</div>
            <div class="card-option__title">${opt.title}</div>
            <div class="card-option__desc">${opt.desc}</div>
          </div>
        `
          )
          .join("")}
      </div>
    `;
  } else if (step.type === "capital") {
    contentHTML = `
      <div class="capital-input-group">
        <span class="capital-currency">€</span>
        <input
          type="number"
          class="capital-input"
          id="capital-input"
          value="${profile.capital_amount}"
          min="0"
          step="1000"
          placeholder="10,000"
        />
      </div>
    `;
  } else if (step.type === "chips") {
    contentHTML = `
      <div class="chip-selector" id="chip-selector">
        ${step.options
          .map(
            (opt) => `
          <button class="chip ${profile.investment_goals.includes(opt.value) ? "selected" : ""}"
                  data-value="${opt.value}">
            ${opt.label}
          </button>
        `
          )
          .join("")}
      </div>
    `;
  }

  const isLast = currentStep === STEPS.length - 1;
  const canProceed = _canProceed();

  return `
    <div class="bg-gradient"></div>
    <section class="screen onboarding" id="onboarding-screen">
      <div class="onboarding__progress">
        <div class="onboarding__progress-bar" style="width: ${progress}%"></div>
      </div>

      <div class="onboarding__step-container" key="${currentStep}">
        <div class="onboarding__step-label">${step.label}</div>
        <h2 class="onboarding__question">${step.question}</h2>
        ${contentHTML}
      </div>

      <div class="onboarding__nav">
        ${
          currentStep > 0
            ? `<button class="btn-secondary" id="btn-prev">← Anterior</button>`
            : `<div></div>`
        }
        <button class="btn-primary" id="btn-next" ${!canProceed ? "disabled" : ""}>
          ${isLast ? "Comenzar chat →" : "Siguiente →"}
        </button>
      </div>
    </section>
  `;
}

function _canProceed() {
  const step = STEPS[currentStep];
  if (step.type === "cards") return profile[step.key] !== null;
  if (step.type === "capital") return profile.capital_amount > 0;
  if (step.type === "chips") return profile.investment_goals.length > 0;
  return true;
}

export function initOnboarding(onComplete, rerender) {
  const step = STEPS[currentStep];

  // Card selector
  if (step.type === "cards") {
    document.querySelectorAll(".card-option").forEach((card) => {
      card.addEventListener("click", () => {
        profile[step.key] = card.dataset.value;
        rerender();
      });
    });
  }

  // Capital input
  if (step.type === "capital") {
    const input = document.getElementById("capital-input");
    if (input) {
      input.addEventListener("input", () => {
        profile.capital_amount = parseFloat(input.value) || 0;
        // Update button state
        const btn = document.getElementById("btn-next");
        if (btn) btn.disabled = !_canProceed();
      });
      input.focus();
    }
  }

  // Chip selector
  if (step.type === "chips") {
    document.querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const val = chip.dataset.value;
        if (profile.investment_goals.includes(val)) {
          profile.investment_goals = profile.investment_goals.filter(
            (g) => g !== val
          );
        } else {
          profile.investment_goals.push(val);
        }
        rerender();
      });
    });
  }

  // Navigation
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");

  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (currentStep > 0) {
        currentStep--;
        rerender();
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (!_canProceed()) return;

      if (currentStep < STEPS.length - 1) {
        currentStep++;
        rerender();
      } else {
        // Final step — pass profile to chat
        onComplete({ ...profile });
      }
    });
  }
}

/** Reset state (for re-entering onboarding) */
export function resetOnboarding() {
  currentStep = 0;
  profile = {
    risk_level: null,
    investment_horizon: null,
    capital_amount: 10000,
    investment_goals: [],
  };
}
