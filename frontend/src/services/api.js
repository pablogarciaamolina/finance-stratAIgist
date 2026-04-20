/**
 * API Service — communicates with the FastAPI backend.
 */

const API_BASE = "/api";

/**
 * Send a chat message to the backend.
 * @param {string} prompt - User message
 * @param {Object} userProfile - User's investment profile
 * @param {string} sessionId - Session identifier
 * @param {Object} [opts] - Optional: { company_name, ticker, mode }
 * @returns {Promise<Object>} Chat response with agent trace
 */
export async function sendMessage(prompt, userProfile, sessionId, opts = {}) {
  // Build user_profile matching backend's Pydantic schema exactly
  const profile = userProfile
    ? {
        risk_level: userProfile.risk_level,
        investment_horizon: userProfile.investment_horizon,
        capital_amount: userProfile.capital_amount,
        investment_goals: userProfile.investment_goals || [],
      }
    : null;

  const body = {
    prompt,
    user_profile: profile,
    session_id: sessionId,
    mode: opts.mode || "advisor",
  };
  if (opts.company_name) body.company_name = opts.company_name;
  if (opts.ticker) body.ticker = opts.ticker;

  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(`Error del servidor (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Check API health.
 * @returns {Promise<Object>}
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}
