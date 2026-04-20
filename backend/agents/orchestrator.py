"""
Orchestrator Agent - interprets the user query and prepares the
structured context used by the rest of the pipeline.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
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
        r"(\d+\s*(?:mes|meses|ano|anos|año|años))",
        r"(corto plazo)",
        r"(medio plazo)",
        r"(largo plazo)",
        r"(short term)",
        r"(medium term)",
        r"(long term)",
    ]

    STOP_WORDS = {
        "a",
        "agresivo",
        "ahora",
        "ano",
        "anos",
        "año",
        "años",
        "con",
        "conservador",
        "de",
        "del",
        "entrar",
        "hoy",
        "invertir",
        "manana",
        "mañana",
        "mes",
        "meses",
        "moderado",
        "o",
        "para",
        "perfil",
        "plazo",
        "porque",
        "sentido",
        "si",
        "sin",
        "tendria",
        "tendría",
        "y",
    }

    TICKER_BLACKLIST = {"RAG", "LLM", "API", "JSON", "USA", "ETF"}

    KNOWN_COMPANY_ALIASES = {
        "alphabet": ("Alphabet", "GOOGL"),
        "amazon": ("Amazon", "AMZN"),
        "apple": ("Apple", "AAPL"),
        "google": ("Alphabet", "GOOGL"),
        "meta": ("Meta", "META"),
        "meta platforms": ("Meta", "META"),
        "microsoft": ("Microsoft", "MSFT"),
        "nvidia": ("NVIDIA", "NVDA"),
        "tesla": ("Tesla", "TSLA"),
    }

    COMPANY_OPENERS = {
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
    }

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

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def _extract_known_company_alias(self, query: str) -> tuple[Optional[str], Optional[str]]:
        normalized_query = self._normalize_text(query).lower()

        for alias, resolved in sorted(
            self.KNOWN_COMPANY_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
                return resolved

        return None, None

    def _extract_ticker_heuristic(self, query: str) -> Optional[str]:
        match_parenthesis = re.search(r"\(([A-Z]{1,5})\)", query)
        if match_parenthesis:
            candidate = match_parenthesis.group(1)
            if candidate not in self.TICKER_BLACKLIST:
                return candidate

        candidates = re.findall(r"\b[A-Z]{2,5}\b", query)
        candidates = [candidate for candidate in candidates if candidate not in self.TICKER_BLACKLIST]
        return candidates[0] if candidates else None

    def _extract_risk_profile_from_query(self, query: str) -> Optional[str]:
        lower = self._normalize_text(query).lower()
        for key, value in self.RISK_KEYWORDS.items():
            if self._normalize_text(key).lower() in lower:
                return value
        return None

    def _extract_horizon_from_query(self, query: str) -> Optional[str]:
        lower = self._normalize_text(query).lower()
        for pattern in self.HORIZON_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                return match.group(1)
        return None

    def _clean_company_candidate(self, candidate: str) -> Optional[str]:
        candidate = candidate.strip(" .,;:!?()[]{}\"'")
        candidate = re.split(
            r"\b(?:y|o|para|con|sin|porque|si|que|a|en|ahora|hoy|manana|mañana)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        words = candidate.split()
        cleaned_words = []
        for word in words:
            word_lower = self._normalize_text(word).lower().strip(" .,;:")
            if word_lower in self.STOP_WORDS:
                break
            cleaned_words.append(word.strip(" .,;:"))

        candidate = " ".join(cleaned_words).strip()
        if not candidate:
            return None

        if len(candidate.split()) > 5:
            candidate = " ".join(candidate.split()[:5]).strip()

        return candidate or None

    def _extract_company_from_capitalized_sequence(self, query: str) -> Optional[str]:
        connector_words = r"(?:of|and|the|de|del|la|las|los|y)"
        pattern = (
            r"\b([A-Z][a-zA-Z0-9&\-.]*"
            r"(?:\s+(?:"
            + connector_words
            + r")\s+[A-Z][a-zA-Z0-9&\-.]*"
            r"|\s+[A-Z][a-zA-Z0-9&\-.]*){0,3})\b"
        )

        for match in re.finditer(pattern, query):
            candidate = self._clean_company_candidate(match.group(1))
            if not candidate:
                continue

            first_word = candidate.split()[0]
            if first_word in self.COMPANY_OPENERS:
                continue

            return candidate

        return None

    def _ticker_for_company(self, company_name: Optional[str]) -> Optional[str]:
        if not company_name:
            return None

        normalized_company = self._normalize_text(company_name).lower().strip()
        for alias, resolved in self.KNOWN_COMPANY_ALIASES.items():
            resolved_company, resolved_ticker = resolved
            if normalized_company == self._normalize_text(resolved_company).lower():
                return resolved_ticker
            if normalized_company == alias:
                return resolved_ticker

        return None

    def _extract_company_name(self, query: str) -> Optional[str]:
        patterns = [
            r"(?:analiza|estudia|revisa)\s+([A-Z][a-zA-Z0-9&\-. ]+)",
            r"(?:invertir en|entrada en|comprar|incluir|incorporar|anadir|agregar|mantener|usar|vigilar|evitar|abrir posicion en|aumentar posicion en|reducir posicion en)\s+([A-Z][a-zA-Z0-9&\-. ]+)",
            r"(?:empresa|accion de|stock de|posicion en|peso en)\s+([A-Z][a-zA-Z0-9&\-. ]+)",
            r"(?:sobre|frente a|antes de)\s+([A-Z][a-zA-Z0-9&\-. ]+)",
            r"Contexto proporcionado por frontend:\s*empresa:\s*([A-Z][a-zA-Z0-9&\-. ]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                cleaned = self._clean_company_candidate(match.group(1))
                if cleaned:
                    return cleaned

        return self._extract_company_from_capitalized_sequence(query)

    def _extract_json_block(self, text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

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
Extrae la informacion clave de esta consulta financiera y responde SOLO en JSON valido.

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
                self._log("El parseo LLM no devolvio un bloque JSON valido")
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

        known_company_name, known_ticker = self._extract_known_company_alias(query)
        if not company_name and known_company_name:
            company_name = known_company_name
            self._log(f"Company por alias conocido: {company_name}")
        if not ticker and known_ticker:
            ticker = known_ticker
            self._log(f"Ticker por alias conocido: {ticker}")

        if not company_name:
            company_name = self._extract_company_name(query)
            self._log(f"Company por heuristica: {company_name}")

        if not ticker:
            ticker = self._extract_ticker_heuristic(query)
            self._log(f"Ticker por heuristica: {ticker}")

        if not ticker:
            ticker = self._ticker_for_company(company_name)
            if ticker:
                self._log(f"Ticker inferido por company_name: {ticker}")

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
            "result": "Pipeline: Market Agent -> Recommendation Agent -> Critic Agent",
        }, final_token_info
