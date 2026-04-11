"""
Critic Agent — validates and reviews the recommendation.

Checks the Recommendation Agent's output for:
- inconsistencies
- missing risk factors
- unsupported claims
- whether there is enough evidence to issue a recommendation

Produces a revised (or confirmed) final response.
"""

import json
from typing import Any, Dict, Optional

from backend.models.general_model import generate_general_reasoning


def recommendation_fallback(text: str) -> str:
    lower = text.lower()
    if "desfavorable" in lower:
        return "desfavorable"
    if "favorable" in lower:
        return "favorable"
    return "neutral"


class CriticAgent:
    """
    Reviews and validates the generated recommendation.

    Checks for:
        - Internal consistency of the analysis
        - Missing risk factors
        - Unsupported claims
        - Whether there is enough evidence
    """

    def __init__(self, model: Any = None, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _extract_json_block(self, text: str) -> Optional[str]:
        if "BEGIN_JSON" in text and "END_JSON" in text:
            start = text.find("BEGIN_JSON") + len("BEGIN_JSON")
            end = text.find("END_JSON", start)
            if end != -1:
                return text[start:end].strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start:end + 1].strip()

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_json(self, raw_output: str) -> Dict[str, Any]:
        cleaned_output = raw_output

        try:
            if "ASSISTANT:" in cleaned_output:
                cleaned_output = cleaned_output.split("ASSISTANT:")[-1].strip()
            cleaned_output = cleaned_output.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(cleaned_output)
            if not json_block:
                raise ValueError("No se encontró JSON en la salida.")

            parsed = json.loads(json_block)

            enough_evidence = parsed.get("enough_evidence", False)
            grounded_in_facts = parsed.get("grounded_in_facts", False)
            missing_risks = parsed.get("missing_risks", [])
            consistency_issues = parsed.get("consistency_issues", [])
            language_adjustments = parsed.get("language_adjustments", [])
            final_recommendation = parsed.get("final_recommendation", "neutral")
            final_answer = parsed.get("final_answer", "")

            if not isinstance(missing_risks, list):
                missing_risks = []
            if not isinstance(consistency_issues, list):
                consistency_issues = []
            if not isinstance(language_adjustments, list):
                language_adjustments = []

            return {
                "action": "Validando coherencia y detectando riesgos no considerados",
                "result": (
                    "Revisión completada."
                    if enough_evidence
                    else "Revisión completada. La evidencia disponible es insuficiente."
                ),
                "revised_response": final_answer,
                "data": {
                    "enough_evidence": enough_evidence,
                    "grounded_in_facts": grounded_in_facts,
                    "missing_risks": missing_risks,
                    "consistency_issues": consistency_issues,
                    "language_adjustments": language_adjustments,
                    "final_recommendation": final_recommendation,
                    "final_answer": final_answer,
                    "raw_output": cleaned_output,
                },
            }
        except Exception:
            fallback_recommendation = recommendation_fallback(cleaned_output)
            fallback_answer = cleaned_output[:1000] or (
                "No se pudo revisar correctamente la recomendación generada."
            )

            return {
                "action": "Validando coherencia y detectando riesgos no considerados",
                "result": "No se pudo parsear correctamente la salida del critic agent.",
                "revised_response": fallback_answer,
                "data": {
                    "enough_evidence": False,
                    "grounded_in_facts": False,
                    "missing_risks": [],
                    "consistency_issues": [
                        "No se pudo parsear la salida del critic agent."
                    ],
                    "language_adjustments": [],
                    "final_recommendation": fallback_recommendation,
                    "final_answer": fallback_answer,
                    "raw_output": cleaned_output,
                },
            }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        recommendation: str = "",
        market_data: dict = None,
        user_profile: dict = None,
    ) -> dict:
        """
        Review and optionally revise a recommendation.

        Args:
            query: Original user query.
            recommendation: Text from the Recommendation Agent.
            market_data: Output from MarketAgent.run(...)
            user_profile: Dict with risk_level, investment_horizon, capital_amount, investment_goals

        Returns:
            Dict with:
                - action
                - result
                - revised_response
                - data (structured critique)
        """
        if self.model is None or self.tokenizer is None:
            fallback = (
                "No se ha podido revisar la recomendación porque el modelo general "
                "no está inicializado."
            )
            return {
                "action": "Validando coherencia y detectando riesgos no considerados",
                "result": "Revisión no realizada.",
                "revised_response": fallback,
                "data": {
                    "enough_evidence": False,
                    "grounded_in_facts": False,
                    "missing_risks": [],
                    "consistency_issues": ["Modelo general no inicializado."],
                    "language_adjustments": [],
                    "final_recommendation": "neutral",
                    "final_answer": fallback,
                    "raw_output": "",
                },
            }

        prompt = self._build_prompt(
            query=query,
            recommendation=recommendation,
            market_data=market_data,
            user_profile=user_profile,
        )

        raw_output = generate_general_reasoning(
            prompt,
            self.model,
            self.tokenizer,
        )

        return self._parse_json(raw_output)