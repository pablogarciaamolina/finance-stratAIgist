"""
Orchestrator Agent — interpreta la consulta del usuario y prepara
la información estructurada que usará el resto del sistema.

Responsabilidades:
- Extraer ticker o nombre de empresa
- Extraer / normalizar perfil de riesgo
- Extraer / normalizar horizonte temporal
- Construir un plan simple de ejecución
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from backend.models.general_model import generate_general_reasoning


class OrchestratorAgent:
    """
    Central coordinator for the multi-agent pipeline.
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

    TICKER_BLACKLIST = {"RAG", "LLM", "API", "JSON", "USA", "ETF"}

    def __init__(self, model: Any = None, tokenizer: Any = None, debug: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.debug = debug

    def _log(self, message: str):
        if self.debug:
            print(f"[OrchestratorAgent] {message}", flush=True)

    def _general_model_ready(self) -> bool:
        if self.model is None:
            return False
        if self.tokenizer is not None:
            return True
        return bool(getattr(self.model, "_is_ollama", False))

    def _extract_ticker_heuristic(self, query: str) -> Optional[str]:
        match_parenthesis = re.search(r"\(([A-Z]{1,5})\)", query)
        if match_parenthesis:
            candidate = match_parenthesis.group(1)
            if candidate not in self.TICKER_BLACKLIST:
                return candidate

        candidates = re.findall(r"\b[A-Z]{2,5}\b", query)
        candidates = [c for c in candidates if c not in self.TICKER_BLACKLIST]
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
        for word in words:
            word_lower = word.lower().strip(" .,:;")
            if word_lower in self.STOP_WORDS:
                break
            cleaned_words.append(word.strip(" .,:;"))

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

    def _normalize_llm_ticker(self, ticker: Any) -> Optional[str]:
        if not isinstance(ticker, str):
            return None
        ticker = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z]{1,5}", ticker):
            return None
        if ticker in self.TICKER_BLACKLIST:
            return None
        return ticker

    def _parse_with_llm(self, query: str) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not self._general_model_ready():
            self._log("LLM parse omitido: modelo/tokenizer no inicializados")
            return None, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency": 0.0,
                "path": "llm_skipped",
            }

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
            self._log("Llamando al modelo general para parsear la consulta")
            llm_start = time.perf_counter()
            raw, token_info = generate_general_reasoning(
                prompt,
                self.model,
                self.tokenizer,
                max_new_tokens=256,
            )
            llm_latency = time.perf_counter() - llm_start
            token_info = token_info or {}
            token_info["latency"] = llm_latency
            token_info["path"] = "llm"
            self._log(f"Respuesta LLM recibida en {llm_latency:.3f}s")

            if "ASSISTANT:" in raw:
                raw = raw.split("ASSISTANT:")[-1].strip()
            raw = raw.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(raw)
            if not json_block:
                self._log("El parseo LLM no devolvió un bloque JSON válido")
                return None, token_info

            parsed = json.loads(json_block)
            company_name = parsed.get("company_name")
            if isinstance(company_name, str):
                company_name = self._clean_company_candidate(company_name)

            return {
                "company_name": company_name,
                "ticker": self._normalize_llm_ticker(parsed.get("ticker")),
                "risk_profile": parsed.get("risk_profile"),
                "horizon": parsed.get("horizon"),
                "user_goal": parsed.get("user_goal", "investment analysis"),
            }, token_info
        except Exception as exc:
            self._log(f"Error en parseo LLM: {exc}")
            return None, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency": 0.0,
                "path": "llm_error",
                "error": str(exc),
            }

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

    def run(self, query: str, user_profile: dict = None) -> tuple[dict, dict]:
        self._log(f"Inicio run | query={query!r}")
        start_total = time.perf_counter()

        llm_parse, token_info = self._parse_with_llm(query)
        company_name = None
        ticker = None
        user_goal = "investment analysis"

        if llm_parse:
            company_name = llm_parse.get("company_name")
            ticker = llm_parse.get("ticker")
            user_goal = llm_parse.get("user_goal", user_goal)

        if not company_name:
            company_name = self._extract_company_name(query)
            self._log(f"Company por heurística: {company_name}")

        if not ticker:
            ticker = self._extract_ticker_heuristic(query)
            self._log(f"Ticker por heurística: {ticker}")

        normalized_profile = self._normalize_profile(user_profile, query)
        plan = ["market_intelligence", "recommendation", "critic_risk_review"]

        total_latency = time.perf_counter() - start_total
        final_token_info = {
            **(token_info or {}),
            "agent_total_latency": total_latency,
        }
        self._log(
            f"Fin run en {total_latency:.3f}s | company={company_name} | ticker={ticker} | "
            f"risk={normalized_profile['risk_profile']} | horizon={normalized_profile['horizon']}"
        )

        return {
            "query": query,
            "company_name": company_name,
            "ticker": ticker,
            "risk_profile": normalized_profile["risk_profile"],
            "horizon": normalized_profile["horizon"],
            "capital_amount": normalized_profile["capital_amount"],
            "investment_goals": normalized_profile["investment_goals"],
            "user_goal": user_goal,
            "plan": plan,
            "action": "Analizando consulta y determinando agentes necesarios",
            "result": "Pipeline: Market Agent → Recommendation Agent → Critic Agent",
        }, final_token_info
