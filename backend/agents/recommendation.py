"""
Recommendation Agent — generates investment analysis and recommendations.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from backend.models.fin_model import generate_financial_reasoning


class RecommendationAgent:
    def __init__(self, model: Any = None, tokenizer: Any = None, debug: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.debug = debug

    def _log(self, message: str):
        if self.debug:
            print(f"[RecommendationAgent] {message}", flush=True)

    def _normalize_user_profile(self, user_profile: Optional[dict]) -> Dict[str, Any]:
        if not user_profile:
            return {
                "risk_level": "moderate",
                "investment_horizon": "medium",
                "capital_amount": None,
                "investment_goals": [],
            }

        risk_value = user_profile.get("risk_level")
        horizon_value = user_profile.get("investment_horizon")
        risk_level = getattr(risk_value, "value", risk_value) or "moderate"
        investment_horizon = getattr(horizon_value, "value", horizon_value) or "medium"
        goals = user_profile.get("investment_goals", [])
        normalized_goals = [getattr(goal, "value", goal) for goal in goals]

        return {
            "risk_level": risk_level,
            "investment_horizon": investment_horizon,
            "capital_amount": user_profile.get("capital_amount"),
            "investment_goals": normalized_goals,
        }

    def _compress_market_report(self, market_data: Optional[dict]) -> Dict[str, Any]:
        if not market_data:
            return {}

        report = market_data.get("data", market_data)

        rag_context = report.get("rag_context", [])[:2]
        compressed_rag = [
            {
                "label": ctx.get("label"),
                "distance": ctx.get("distance"),
                "text": ctx.get("text", "")[:300],
            }
            for ctx in rag_context
        ]

        external_context = report.get("external_context")
        if isinstance(external_context, str):
            external_context = external_context[:800]

        return {
            "company_name": report.get("company_name"),
            "ticker": report.get("ticker"),
            "price_data": report.get("price_data"),
            "fundamentals_data": report.get("fundamentals_data"),
            "historical_financial_data": report.get("historical_financial_data"),
            "events_data": report.get("events_data"),
            "external_context": external_context,
            "rag_context": compressed_rag,
            "summary": report.get("summary"),
            "has_minimum_evidence": report.get("has_minimum_evidence", False),
            "resolved_ticker": report.get("resolved_ticker", False),
        }

    def _build_prompt(self, query: str, market_data: dict = None, user_profile: dict = None) -> str:
        normalized_profile = self._normalize_user_profile(user_profile)
        compact_report = self._compress_market_report(market_data)
        return f"""
Eres un analista financiero especializado en construir tesis de inversión razonadas.
Debes basarte solo en la información proporcionada y usa tu propio conocimiento.


Consulta del usuario:
{query}

Perfil del usuario:
- risk_level: {normalized_profile['risk_level']}
- investment_horizon: {normalized_profile['investment_horizon']}
- capital_amount: {normalized_profile['capital_amount']}
- investment_goals: {normalized_profile['investment_goals']}

INFORME DE MERCADO:
{compact_report}

Responde SOLO entre las etiquetas BEGIN_JSON y END_JSON con un JSON válido de esta estructura exacta:

BEGIN_JSON
{{
  "thesis": "string",
  "strengths": ["string", "string"],
  "risks": ["string", "string"],
  "scenarios": ["string", "string"],
  "preliminary_recommendation": "favorable|neutral|desfavorable",
  "confidence": "baja|media|alta"
}}
END_JSON
"""

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

    def _parse_json(self, raw_output: str, profile_desc: str = "") -> Dict[str, Any]:
        cleaned_output = raw_output
        try:
            if "ASSISTANT:" in cleaned_output:
                cleaned_output = cleaned_output.split("ASSISTANT:")[-1].strip()
            cleaned_output = cleaned_output.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(cleaned_output)
            if not json_block:
                raise ValueError("No se encontró JSON en la salida.")

            parsed = json.loads(json_block)
            thesis = parsed.get("thesis", "")
            strengths = parsed.get("strengths", [])
            risks = parsed.get("risks", [])
            scenarios = parsed.get("scenarios", [])
            preliminary_recommendation = parsed.get("preliminary_recommendation", "neutral")
            confidence = parsed.get("confidence", "media")

            return {
                "action": "Generando análisis y recomendación personalizada",
                "result": f"Análisis realizado. {profile_desc}".strip(),
                "response": thesis,
                "data": {
                    "thesis": thesis,
                    "strengths": strengths if isinstance(strengths, list) else [],
                    "risks": risks if isinstance(risks, list) else [],
                    "scenarios": scenarios if isinstance(scenarios, list) else [],
                    "preliminary_recommendation": preliminary_recommendation,
                    "confidence": confidence,
                    "raw_output": cleaned_output,
                },
            }
        except Exception as exc:
            self._log(f"Fallo parseando JSON del recommendation agent: {exc}")
            fallback_text = cleaned_output[:1000]
            return {
                "action": "Generando análisis y recomendación personalizada",
                "result": f"Análisis realizado. {profile_desc}".strip(),
                "response": fallback_text,
                "data": {
                    "thesis": fallback_text,
                    "strengths": [],
                    "risks": [],
                    "scenarios": [],
                    "preliminary_recommendation": "neutral",
                    "confidence": "media",
                    "raw_output": cleaned_output,
                },
            }

    def run(self, query: str, market_data: dict = None, user_profile: dict = None) -> tuple[dict, dict]:
        normalized_profile = self._normalize_user_profile(user_profile)
        profile_desc = (
            f"Perfil: {normalized_profile['risk_level']}, "
            f"horizonte: {normalized_profile['investment_horizon']}"
        )
        self._log(f"Inicio run | {profile_desc}")
        start_total = time.perf_counter()
        if self.model is None:
            print("sin modelo")
        if self.tokenizer is None:
            print("sin tokenizer")

        if self.model is None or self.tokenizer is None:
            fallback = (
                "No se ha podido generar una tesis de inversión con el modelo financiero "
                "porque el modelo no está inicializado."
            )
            return {
                "action": "Generando análisis y recomendación personalizada",
                "result": f"Análisis no realizado. {profile_desc}",
                "response": fallback,
                "data": {
                    "thesis": fallback,
                    "strengths": [],
                    "risks": [],
                    "scenarios": [],
                    "preliminary_recommendation": "neutral",
                    "confidence": "baja",
                    "raw_output": "",
                },
            }, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "agent_total_latency": 0.0,
                "path": "model_not_initialized",
            }

        prompt = self._build_prompt(
            query=query,
            market_data=market_data,
            user_profile=user_profile,
        )

        self._log("Llamando al modelo financiero Fin-R1")
        raw_output, token_info = generate_financial_reasoning(
            prompt,
            self.model,
            self.tokenizer,
            max_new_tokens=512,
            do_sample=False,
        )

        total_latency = time.perf_counter() - start_total
        final_token_info = {**(token_info or {}), "agent_total_latency": total_latency}
        self._log(f"Generación financiera completada en {total_latency:.3f}s")

        return self._parse_json(raw_output, profile_desc=profile_desc), final_token_info
