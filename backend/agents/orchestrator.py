"""
Orchestrator Agent — interpreta la consulta del usuario y prepara
la información estructurada que usará el resto del sistema.

Responsabilidades:
- Extraer ticker o nombre de empresa
- Extraer / normalizar perfil de riesgo
- Extraer / normalizar horizonte temporal
- Construir un plan simple de ejecución
"""
import time
import json
import re
from typing import Any, Dict, Optional

from backend.models.general_model import generate_general_reasoning


class OrchestratorAgent:
    """
    Central coordinator for the multi-agent pipeline.

    Este agente NO ejecuta el pipeline completo.
    Su función es interpretar la query y devolver un plan estructurado
    para que luego lo use el sistema multiagente.
    """

    RISK_KEYWORDS = {
        "conservador": "conservative",
        "prudente": "conservative",
        "moderado": "moderate",
        "medio": "moderate",
        "agresivo": "aggressive",
        "arriesgado": "aggressive",
    }

    HORIZON_PATTERNS = [
        r"(\d+\s*(?:mes|meses|año|años))",
        r"(corto plazo)",
        r"(medio plazo)",
        r"(largo plazo)",
        r"(short term)",
        r"(medium term)",
        r"(long term)",
    ]

    STOP_WORDS = {
        "y", "o", "para", "con", "sin", "de", "del", "ahora", "hoy",
        "mañana", "porque", "si", "tendría", "sentido", "entrar",
        "invertir", "perfil", "moderado", "conservador", "agresivo",
        "mes", "meses", "año", "años", "plazo"
    }

    def __init__(self, model: Any = None, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer

    def _extract_ticker_heuristic(self, query: str) -> Optional[str]:
        match_parenthesis = re.search(r"\(([A-Z]{1,5})\)", query)
        if match_parenthesis:
            return match_parenthesis.group(1)

        candidates = re.findall(r"\b[A-Z]{2,5}\b", query)
        blacklist = {"RAG", "LLM", "API", "JSON", "USA", "ETF"}
        candidates = [c for c in candidates if c not in blacklist]
        return candidates[0] if candidates else None

    def _extract_risk_profile_from_query(self, query: str) -> Optional[str]:
        lower = query.lower()
        for key, value in self.RISK_KEYWORDS.items():
            if key in lower:
                return value
        return None

    def _extract_horizon_from_query(self, query: str) -> Optional[str]:
        lower = query.lower()
        for pattern in self.HORIZON_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                return match.group(1)
        return None

    def _clean_company_candidate(self, candidate: str) -> Optional[str]:
        candidate = candidate.strip(" .,:;¿?¡!()[]{}\"'")
        candidate = re.split(
            r"\b(?:y|o|para|con|sin|porque|si|que|a|en|ahora|hoy|mañana)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        words = candidate.split()
        cleaned_words = []
        for w in words:
            wl = w.lower().strip(" .,:;")
            if wl in self.STOP_WORDS:
                break
            cleaned_words.append(w.strip(" .,:;"))

        candidate = " ".join(cleaned_words).strip()

        if not candidate:
            return None

        if len(candidate.split()) > 5:
            candidate = " ".join(candidate.split()[:5]).strip()

        return candidate or None

    def _extract_company_name(self, query: str) -> Optional[str]:
        patterns = [
            r"(?:analiza|estudia|revisa)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
            r"(?:invertir en|entrada en|comprar)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
            r"(?:empresa|acción de|stock de)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
            r"(?:sobre)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                cleaned = self._clean_company_candidate(match.group(1))
                if cleaned:
                    return cleaned

        return None

    def _extract_json_block(self, text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start:end + 1]

    def _parse_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            return None

        prompt = f"""
Extrae la información clave de esta consulta financiera y responde SOLO en JSON válido.

Campos:
- company_name: string o null
- ticker: string o null
- risk_profile: "conservative", "moderate" o "aggressive"
- horizon: string corto
- user_goal: string corto

Consulta:
{query}

Devuelve SOLO un objeto JSON.
"""
        try:
            raw = generate_general_reasoning(prompt, self.model, self.tokenizer)
            if "ASSISTANT:" in raw:
                raw = raw.split("ASSISTANT:")[-1].strip()
            raw = raw.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(raw)
            if not json_block:
                return None

            parsed = json.loads(json_block)
            company_name = parsed.get("company_name")
            if isinstance(company_name, str):
                company_name = self._clean_company_candidate(company_name)

            return {
                "company_name": company_name,
                "ticker": parsed.get("ticker"),
                "risk_profile": parsed.get("risk_profile"),
                "horizon": parsed.get("horizon"),
                "user_goal": parsed.get("user_goal", "investment analysis"),
            }
        except Exception:
            return None

    def _normalize_profile(self, user_profile: Optional[dict], query: str) -> Dict[str, Any]:
        risk_profile = None
        horizon = None
        capital_amount = None
        investment_goals = []

        if user_profile:
            risk_value = user_profile.get("risk_level")
            horizon_value = user_profile.get("investment_horizon")

            risk_profile = getattr(risk_value, "value", risk_value)
            horizon = getattr(horizon_value, "value", horizon_value)
            capital_amount = user_profile.get("capital_amount")
            raw_goals = user_profile.get("investment_goals", [])
            investment_goals = [getattr(goal, "value", goal) for goal in raw_goals]

        if not risk_profile:
            risk_profile = self._extract_risk_profile_from_query(query) or "moderate"

        if not horizon:
            horizon = self._extract_horizon_from_query(query) or "medium"

        return {
            "risk_profile": risk_profile,
            "horizon": horizon,
            "capital_amount": capital_amount,
            "investment_goals": investment_goals,
        }

    def _derive_user_goal(
        self,
        mode: str,
        llm_goal: Optional[str],
        investment_goals: list,
    ) -> str:
        if mode == "benchmark":
            return "financial QA"

        if llm_goal:
            return llm_goal

        if not investment_goals:
            return "investment analysis"

        if "growth" in investment_goals:
            return "growth-oriented investment analysis"
        if "income" in investment_goals:
            return "income-oriented investment analysis"
        if "preservation" in investment_goals:
            return "capital preservation analysis"
        if "speculation" in investment_goals:
            return "speculative investment analysis"

        return "investment analysis"

    def run(
        self,
        query: str,
        user_profile: dict = None,
        company_name: str = None,
        ticker: str = None,
        mode: str = "advisor",
    ) -> dict:

        start = time.time()

        print("\n[TRACE] Orchestrator START")
        print(f"[TRACE] query={query}")
        print(f"[TRACE] input company_name={company_name} | ticker={ticker} | mode={mode}")

        # Shortcut: benchmark mode con empresa explícita
        if mode == "benchmark" and (company_name or ticker):
            result = {
                "action": "Parsing user query",
                "result": "Benchmark mode orchestration",
                "company_name": company_name,
                "ticker": ticker,
                "risk_profile": None,
                "horizon": None,
                "mode": "benchmark",
            }

            print(f"[TRACE] Orchestrator END ({time.time() - start:.2f}s) [shortcut]")
            return result

        # Construcción prompt
        t0 = time.time()
        prompt = self._build_prompt(query, user_profile)
        print(f"[TRACE] prompt built ({time.time() - t0:.2f}s)")

        # Generación
        t0 = time.time()
        print("[TRACE] Orchestrator generation START")
        raw_output = generate_general_reasoning(prompt, self.model, self.tokenizer)
        print(f"[TRACE] Orchestrator generation END ({time.time() - t0:.2f}s)")

        # Parseo
        t0 = time.time()
        print("[TRACE] Orchestrator parse START")
        parsed = self._parse_json(raw_output)
        print(f"[TRACE] Orchestrator parse END ({time.time() - t0:.2f}s)")

        print(f"[TRACE] extracted company_name={parsed.get('company_name')} | ticker={parsed.get('ticker')}")

        parsed["mode"] = parsed.get("mode") or mode

        print(f"[TRACE] Orchestrator END ({time.time() - start:.2f}s)")
        return parsed