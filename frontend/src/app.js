/**
 * App Controller — state machine managing the three screens.
 *
 * States:
 *   "landing"     → Landing page
 *   "onboarding"  → User profile questionnaire
 *   "chat"        → Chat interface
 */

import { renderLanding, initLanding } from "./components/landing.js";
import {
  renderOnboarding,
  initOnboarding,
  resetOnboarding,
} from "./components/onboarding.js";
import {
  renderChat,
  initChat,
  setProfile,
  resetChat,
} from "./components/chat.js";

let currentState = "landing";
const appRoot = document.getElementById("app");

/** Main render function — renders the current screen and attaches listeners */
function render() {
  let html = "";

  switch (currentState) {
    case "landing":
      html = renderLanding();
      break;
    case "onboarding":
      html = renderOnboarding();
      break;
    case "chat":
      html = renderChat();
      break;
  }

  appRoot.innerHTML = html;

  // Attach event listeners after DOM update
  switch (currentState) {
    case "landing":
      initLanding(() => {
        currentState = "onboarding";
        resetOnboarding();
        render();
      });
      break;
    case "onboarding":
      initOnboarding(
        (profile) => {
          // Onboarding complete — move to chat
          setProfile(profile);
          currentState = "chat";
          render();
        },
        () => render() // rerender callback for step changes
      );
      break;
    case "chat":
      initChat(() => render());
      break;
  }
}

export function initApp() {
  render();
}
