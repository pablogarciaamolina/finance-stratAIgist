"""Critic Agent: validates the recommendation and rewrites the final answer."""

from __future__ import annotations

import json
import time
from datetime import date
from typing import Any, Dict, Optional

from backend.agents.output_utils import (
    extract_internal_reasoning,
    normalize_model_output,
    sanitize_visible_answer,
)
from backend.models.general_model import generate_general_reasoning


def recommendation_fallback(text: str) -> str:
    lower = text.lower()
    if "desfavorable" in lower:
        return "desfavorable"
    if "favorable" in lower:
        return "favorable"
    return "neutral"


class CriticAgent:
    def __init__(self, model: Any = None, tokenizer: Any = None, debug: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.debug = debug

    def _log(self, message: str):
        if self.debug:
            print(f"[CriticAgent] {message}", flush=True)

    def _general_model_ready(self) -> bool:
        if self.model is None:
            return False
        if self.tokenizer is not None:
            return True
        return bool(getattr(self.model, "_is_ollama", False))

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
            "historical_financial_data": report.get("historical_financial_data"),
            "events_data": report.get("events_data"),
            "has_minimum_evidence": report.get("has_minimum_evidence", False),
            "has_structured_evidence": report.get("has_structured_evidence", False),
            "structured_signals": report.get("structured_signals", 0),
            "resolved_ticker": report.get("resolved_ticker", False),
        }

    def _compact_recommendation_data(
        self,
        recommendation: str = "",
        recommendation_data: Optional[dict] = None,
    ) -> Dict[str, Any]:
        recommendation_data = recommendation_data or {}

        if "data" in recommendation_data and isinstance(recommendation_data.get("data"), dict):
            payload = recommendation_data["data"]
        else:
            payload = recommendation_data

        return {
            "question_type": payload.get("question_type", "general"),
            "answer": payload.get("answer", recommendation),
            "thesis": payload.get("thesis", recommendation),
            "strengths": payload.get("strengths", []),
            "risks": payload.get("risks", []),
            "watch_items": payload.get("watch_items", []),
            "allocation_guidance": payload.get("allocation_guidance", []),
            "implementation_steps": payload.get("implementation_steps", []),
            "preliminary_recommendation": payload.get("preliminary_recommendation"),
            "confidence": payload.get("confidence"),
            "answered_directly": payload.get("answered_directly", False),
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

    def _safe_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [sanitize_visible_answer(item) for item in value if sanitize_visible_answer(item)]

    def _build_safe_fallback_answer(
        self,
        question_type: str,
        company_name: Optional[str],
        has_structured_evidence: bool,
        recommendation_answer: str,
    ) -> str:
        company_label = company_name or "la empresa analizada"
        if not has_structured_evidence:
            limited_templates = {
                "allocation": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para fijar un peso maximo con precision. "
                    "Sin confirmar precio y fundamentales recientes, evitaria dar un porcentaje exacto y lo trataria solo como una posicion acotada dentro de una cartera diversificada."
                ),
                "rebalance": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para ajustar una cadencia de rebalanceo con precision. "
                    "Antes de decidirla, confirmaria precio, fundamentales y si la tesis ha cambiado materialmente."
                ),
                "entry_plan": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para recomendar una entrada precisa. "
                    "Con esta informacion, seria mas prudente esperar a validar precio, fundamentales y noticias antes de definir tramos concretos."
                ),
                "hedge": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para ajustar una cobertura con precision. "
                    "Antes de introducir coberturas o activos refugio, confirmaria primero el riesgo real de la posicion con datos de mercado actualizados."
                ),
                "geography": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para proponer una redistribucion geografica afinada. "
                    "Antes de mover pesos entre regiones, confirmaria mejor el papel real de esta posicion dentro de la cartera."
                ),
                "risk_watch": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para priorizar riesgos concretos con suficiente confianza. "
                    "Antes de invertir o ampliar posicion, confirmaria fundamentales recientes, precio y noticias materiales."
                ),
                "thesis": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para defender una tesis de inversion solida. "
                    "Antes de tomar una decision, confirmaria precio, fundamentales y noticias recientes en fuentes oficiales."
                ),
                "general": (
                    f"No tengo evidencia estructurada suficiente sobre {company_label} para responder con precision. "
                    "Antes de tomar una decision, confirmaria precio, fundamentales y noticias recientes en fuentes oficiales."
                ),
            }
            return limited_templates.get(question_type, limited_templates["general"])

        if recommendation_answer and len(recommendation_answer.strip()) >= 40:
            return recommendation_answer

        templates = {
            "allocation": (
                f"Con la evidencia disponible, mantendria {company_label} como una posicion acotada dentro de una cartera "
                "moderada y completaria el resto con instrumentos diversificados."
            ),
            "rebalance": (
                f"Para una cartera moderada con peso relevante en {company_label}, usaria un rebalanceo cada 6-12 meses "
                "o antes si la posicion se desvía claramente del objetivo."
            ),
            "entry_plan": (
                f"Para entrar en {company_label} con riesgo de timing contenido, preferiria una entrada escalonada en varios tramos."
            ),
            "hedge": (
                f"Si {company_label} pesa mucho en cartera, puede tener sentido compensarlo con mas diversificacion y una pequena "
                "pata defensiva, sin sobredimensionar la cobertura."
            ),
            "geography": (
                f"Si quieres mantener {company_label} y reducir sesgo geografico, complementaria la cartera con exposicion amplia "
                "a Europa y Asia mediante fondos diversificados."
            ),
            "risk_watch": (
                f"Antes de invertir en {company_label}, vigilaria deterioro de crecimiento, margenes o cash flow y noticias "
                "regulatorias o de guidance."
            ),
            "thesis": (
                f"{company_label} puede tener cabida en cartera, pero la trataria con un peso acorde al perfil moderado y evitaria "
                "sobreconcentrarla."
            ),
            "general": (
                f"Mantendria una postura prudente sobre {company_label} y evitaria una recomendacion tajante sin validar mejor la evidencia."
            ),
        }
        return templates.get(question_type, templates["general"])

    def _build_prompt(
        self,
        query: str,
        recommendation: Any = "",
        market_data: Optional[dict] = None,
        user_profile: Optional[dict] = None,
    ) -> tuple[str, str, bool, Optional[str], str]:
        compact_market = self._compact_market_data(market_data)

        recommendation_data: Dict[str, Any] = {}
        recommendation_text = recommendation if isinstance(recommendation, str) else ""

        if isinstance(recommendation, dict):
            recommendation_data = recommendation
            recommendation_text = recommendation.get("response", "") or recommendation.get("revised_response", "")

        compact_recommendation = self._compact_recommendation_data(
            recommendation=recommendation_text,
            recommendation_data=recommendation_data,
        )

        risk_profile = None
        horizon = None
        if user_profile:
            risk_profile = getattr(
                user_profile.get("risk_level"),
                "value",
                user_profile.get("risk_level"),
            )
            horizon = getattr(
                user_profile.get("investment_horizon"),
                "value",
                user_profile.get("investment_horizon"),
            )

        question_type = compact_recommendation.get("question_type", "general")
        has_structured_evidence = bool(compact_market.get("has_structured_evidence", False))
        company_name = compact_market.get("company_name")
        recommendation_answer = sanitize_visible_answer(compact_recommendation.get("answer", ""))
        today = date.today().isoformat()

        prompt = f"""
Actuas como revisor critico de una recomendacion financiera.
Fecha actual: {today}

Reglas obligatorias:
- Responde siempre en espanol.
- No muestres razonamiento interno ni etiquetas como <think>.
- Tu objetivo es devolver una respuesta final util, breve y prudente para el usuario.
- Si la evidencia estructurada es debil, debes decirlo explicitamente y evitar falsa precision.
- Si la recomendacion preliminar no responde de forma directa a la pregunta, reescribela para que si lo haga.

Consulta original:
{query}

Tipo de consulta:
{question_type}

Perfil del usuario:
- risk_level: {risk_profile}
- investment_horizon: {horizon}

INFORME DE MERCADO:
{compact_market}

RECOMENDACION PRELIMINAR:
{compact_recommendation}

Responde SOLO entre BEGIN_JSON y END_JSON con un JSON valido de esta estructura:

BEGIN_JSON
{{
  "enough_evidence": true,
  "grounded_in_facts": true,
  "missing_risks": ["string"],
  "consistency_issues": ["string"],
  "language_adjustments": ["string"],
  "final_recommendation": "favorable|neutral|desfavorable",
  "answer_quality": "directa|parcial|insuficiente",
  "final_answer": "string"
}}
END_JSON
"""
        return prompt, question_type, has_structured_evidence, company_name, recommendation_answer

    def _parse_json(
        self,
        raw_output: str,
        question_type: str,
        has_structured_evidence: bool,
        company_name: Optional[str],
        fallback_answer: str,
    ) -> Dict[str, Any]:
        cleaned_output = normalize_model_output(raw_output)
        internal_reasoning = extract_internal_reasoning(cleaned_output)
        sanitized_output = sanitize_visible_answer(cleaned_output)

        try:
            json_block = self._extract_json_block(cleaned_output)
            if not json_block:
                raise ValueError("No se encontro JSON en la salida.")

            parsed = json.loads(json_block, strict=False)

            enough_evidence = bool(parsed.get("enough_evidence", False)) and has_structured_evidence
            grounded_in_facts = bool(parsed.get("grounded_in_facts", False)) and has_structured_evidence
            missing_risks = self._safe_list(parsed.get("missing_risks"))
            consistency_issues = self._safe_list(parsed.get("consistency_issues"))
            language_adjustments = self._safe_list(parsed.get("language_adjustments"))
            final_recommendation = recommendation_fallback(str(parsed.get("final_recommendation", "neutral")))
            answer_quality = sanitize_visible_answer(parsed.get("answer_quality", "parcial")) or "parcial"
            final_answer = sanitize_visible_answer(parsed.get("final_answer", ""))

            if len(final_answer) < 30:
                final_answer = self._build_safe_fallback_answer(
                    question_type=question_type,
                    company_name=company_name,
                    has_structured_evidence=has_structured_evidence,
                    recommendation_answer=fallback_answer,
                )

            if answer_quality == "insuficiente":
                enough_evidence = False

            return {
                "action": "Validando coherencia y detectando riesgos no considerados",
                "result": (
                    "Revision completada."
                    if enough_evidence
                    else "Revision completada. La evidencia disponible es insuficiente."
                ),
                "revised_response": final_answer,
                "data": {
                    "enough_evidence": enough_evidence,
                    "grounded_in_facts": grounded_in_facts,
                    "missing_risks": missing_risks,
                    "consistency_issues": consistency_issues,
                    "language_adjustments": language_adjustments,
                    "final_recommendation": final_recommendation,
                    "answer_quality": answer_quality,
                    "final_answer": final_answer,
                    "internal_reasoning": internal_reasoning,
                    "sanitized_output": sanitized_output,
                    "raw_output": cleaned_output,
                },
            }
        except Exception as exc:
            self._log(f"Fallo parseando JSON del critic agent: {exc}")
            safe_answer = self._build_safe_fallback_answer(
                question_type=question_type,
                company_name=company_name,
                has_structured_evidence=has_structured_evidence,
                recommendation_answer=fallback_answer,
            )
            fallback_recommendation = recommendation_fallback(cleaned_output)
            return {
                "action": "Validando coherencia y detectando riesgos no considerados",
                "result": "No se pudo parsear correctamente la salida del critic agent.",
                "revised_response": safe_answer,
                "data": {
                    "enough_evidence": False,
                    "grounded_in_facts": False,
                    "missing_risks": [],
                    "consistency_issues": ["No se pudo parsear la salida del critic agent."],
                    "language_adjustments": [],
                    "final_recommendation": fallback_recommendation,
                    "answer_quality": "insuficiente",
                    "final_answer": safe_answer,
                    "internal_reasoning": internal_reasoning,
                    "sanitized_output": sanitized_output,
                    "raw_output": cleaned_output,
                },
            }

    def run(
        self,
        query: str,
        recommendation: Any = None,
        market_data: Optional[dict] = None,
        user_profile: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        self._log("Inicio run")
        start_total = time.perf_counter()

        prompt, question_type, has_structured_evidence, company_name, recommendation_answer = self._build_prompt(
            query=query,
            recommendation=recommendation,
            market_data=market_data,
            user_profile=user_profile,
        )

        if not self._general_model_ready():
            fallback = self._build_safe_fallback_answer(
                question_type=question_type,
                company_name=company_name,
                has_structured_evidence=has_structured_evidence,
                recommendation_answer=recommendation_answer,
            )
            return {
                "action": "Validando coherencia y detectando riesgos no considerados",
                "result": (
                    "Revision heuristica completada."
                    if has_structured_evidence
                    else "Revision heuristica completada con evidencia insuficiente."
                ),
                "revised_response": fallback,
                "data": {
                    "enough_evidence": has_structured_evidence,
                    "grounded_in_facts": has_structured_evidence,
                    "missing_risks": [],
                    "consistency_issues": ["Modelo general no inicializado; se uso revision heuristica."],
                    "language_adjustments": [],
                    "final_recommendation": recommendation_fallback(recommendation_answer),
                    "answer_quality": "parcial" if has_structured_evidence else "insuficiente",
                    "final_answer": fallback,
                    "internal_reasoning": "",
                    "sanitized_output": fallback,
                    "raw_output": "",
                },
            }, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "agent_total_latency": 0.0,
                "path": "model_not_initialized",
            }

        self._log(
            "Llamando al modelo general para revisar la recomendacion "
            f"| question_type={question_type} | structured={has_structured_evidence}"
        )
        raw_output, token_info = generate_general_reasoning(
            prompt,
            self.model,
            self.tokenizer,
            max_new_tokens=512,
        )

        total_latency = time.perf_counter() - start_total
        final_token_info = {
            **(token_info or {}),
            "agent_total_latency": total_latency,
        }
        self._log(f"Revision completada en {total_latency:.3f}s")

        return self._parse_json(
            raw_output=raw_output,
            question_type=question_type,
            has_structured_evidence=has_structured_evidence,
            company_name=company_name,
            fallback_answer=recommendation_answer,
        ), final_token_info
