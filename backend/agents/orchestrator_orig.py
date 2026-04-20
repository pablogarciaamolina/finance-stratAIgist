"""
Orchestrator Agent — interpreta la consulta del usuario y prepara
la información estructurada que usará el resto del sistema.

Responsabilidades:
- Extraer ticker o nombre de empresa
- Extraer / normalizar perfil de riesgo
- Extraer / normalizar horizonte temporal
- Construir un plan simple de ejecución
"""

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
            return None, {}

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
            raw, token_info = generate_general_reasoning(prompt, self.model, self.tokenizer)
            if "ASSISTANT:" in raw:
                raw = raw.split("ASSISTANT:")[-1].strip()
            raw = raw.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(raw)
            if not json_block:
                return None, {}

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
            }, token_info
        except Exception:
            return None, {}
    
    def _build_prompt(
        self,
        query: str,
        recommendation: str = "",
        market_data: dict = None,
        user_profile: dict = None,
    ) -> str:
        compact_market = self._compact_market_data(market_data)
 
        recommendation_data = {}
        if recommendation and isinstance(recommendation, dict):
            # por si alguna vez se pasa el objeto entero accidentalmente
            recommendation_data = recommendation
            recommendation = recommendation.get("response", "")
        elif market_data and isinstance(market_data, dict):
            recommendation_data = {}
 
        compact_recommendation = self._compact_recommendation_data(
            recommendation=recommendation,
            recommendation_data=recommendation_data,
        )
 
        risk_profile = None
        horizon = None
        if user_profile:
            risk_profile = getattr(user_profile.get("risk_level"), "value", user_profile.get("risk_level"))
            horizon = getattr(
                user_profile.get("investment_horizon"),
                "value",
                user_profile.get("investment_horizon"),
            )
 
        return f"""
Actúas como revisor crítico de una recomendación financiera.
Tu función no es rehacer todo el análisis, sino revisar si está bien fundamentado.
 
Consulta original:
{query}
 
Perfil del usuario:
- risk_level: {risk_profile}
- investment_horizon: {horizon}
 
INFORME DE MERCADO:
{compact_market}
 
RECOMENDACIÓN PRELIMINAR:
{compact_recommendation}
 
Responde SOLO entre BEGIN_JSON y END_JSON con un JSON válido de esta estructura:
 
BEGIN_JSON
{{
  "enough_evidence": true,
  "grounded_in_facts": true,
  "missing_risks": ["string"],
  "consistency_issues": ["string"],
  "language_adjustments": ["string"],
  "final_recommendation": "favorable|neutral|desfavorable",
  "final_answer": "string"
}}
END_JSON
 
Reglas:
- enough_evidence debe indicar si existe base suficiente para emitir una recomendación
- si no hay suficiente evidencia, final_answer debe decirlo explícitamente y ser prudente
- grounded_in_facts debe ser true o false
- si no hay problemas, devuelve listas vacías
- no añadas texto fuera del bloque JSON
"""

    def _compact_market_data(self, market_data: Optional[dict]) -> Dict[str, Any]:
        if not market_data:
            return {}
 
        report = market_data.get("data", market_data)
 
        return {
            "company_name": report.get("company_name"),
            "ticker": report.get("ticker"),
            "summary": report.get("summary"),
            "price_data": report.get("price_data"),
            "fundamentals_data": report.get("fundamentals_data"),
            "events_data": report.get("events_data"),
            "has_minimum_evidence": report.get("has_minimum_evidence", False),
            "resolved_ticker": report.get("resolved_ticker", False),
        }
 
    def _compact_recommendation_data(
        self,
        recommendation: str,
        recommendation_data: Optional[dict] = None,
    ) -> Dict[str, Any]:
        recommendation_data = recommendation_data or {}
 
        return {
            "thesis": recommendation_data.get("thesis", recommendation),
            "strengths": recommendation_data.get("strengths", []),
            "risks": recommendation_data.get("risks", []),
            "scenarios": recommendation_data.get("scenarios", []),
            "preliminary_recommendation": recommendation_data.get(
                "preliminary_recommendation"
            ),
            "confidence": recommendation_data.get("confidence"),
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
            investment_goals = user_profile.get("investment_goals", [])

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

    def run(self, query: str, user_profile: dict = None) -> dict:
        """
        Interpreta la query y devuelve un plan estructurado.

        Args:
            query: consulta del usuario
            user_profile: dict con risk_level, investment_horizon, capital_amount, investment_goals

        Returns:
            dict con company_name, ticker, perfil, horizonte y plan
        """
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

        if not ticker:
            ticker = self._extract_ticker_heuristic(query)

        normalized_profile = self._normalize_profile(user_profile, query)

        plan = [
            "market_intelligence",
            "recommendation",
            "critic_risk_review",
        ]

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
        }, token_info