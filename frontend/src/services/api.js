/**
 * API Service — communicates with the FastAPI backend.
 */

const API_BASE = "/api";

/**
 * Send a chat message to the backend.
 * @param {string} prompt - User message
 * @param {Object} userProfile - User's investment profile
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Chat response with agent trace
 */
export async function sendMessage(prompt, userProfile, sessionId) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      user_profile: userProfile,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error(`Error del servidor: ${response.status}`);
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
