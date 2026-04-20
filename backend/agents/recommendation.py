"""Recommendation Agent for advisor-mode investment responses."""

from __future__ import annotations

import json
import re
import time
from datetime import date
from typing import Any, Dict, Optional

from backend.agents.output_utils import (
    extract_internal_reasoning,
    normalize_model_output,
    sanitize_visible_answer,
)
from backend.models.fin_model import generate_financial_reasoning


class RecommendationAgent:
    VALID_QUESTION_TYPES = {
        "allocation",
        "rebalance",
        "entry_plan",
        "hedge",
        "geography",
        "risk_watch",
        "thesis",
        "general",
    }

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
            "has_structured_evidence": report.get("has_structured_evidence", False),
            "structured_signals": report.get("structured_signals", 0),
            "resolved_ticker": report.get("resolved_ticker", False),
        }

    def _detect_question_type(self, query: str) -> str:
        raw_query = query or ""
        if "\n\nContexto proporcionado por frontend:" in raw_query:
            raw_query = raw_query.split("\n\nContexto proporcionado por frontend:", 1)[0]

        lower = raw_query.lower()

        if any(term in lower for term in ["rebalance", "rebalancear"]):
            return "rebalance"
        if any(term in lower for term in ["de golpe", "escalonad", "gradual", "timing", "liquidez", "entrar"]):
            return "entry_plan"
        if any(term in lower for term in ["proteger", "refugio", "oro", "cobertura", "correccion"]):
            return "hedge"
        if (
            re.search(r"\bestados unidos\b", lower)
            or re.search(r"\beuropa\b", lower)
            or re.search(r"\basia\b", lower)
            or "geograf" in lower
            or "sesgo geograf" in lower
        ):
            return "geography"
        if any(term in lower for term in ["riesg", "senal", "señal", "catalizador", "noticias", "vigilar", "evitar"]):
            return "risk_watch"
        if any(term in lower for term in ["porcentaje", "peso", "distribuir", "combinarias", "dividirias", "destinarias", "proporcion"]):
            return "allocation"
        if any(term in lower for term in ["posicion", "posición", "encaja", "papel", "cabida", "calidad", "comprarla"]):
            return "thesis"
        return "general"

    def _question_type_instructions(self, question_type: str) -> str:
        instructions = {
            "allocation": (
                "Da una recomendacion concreta de pesos o rangos orientativos y explica por que. "
                "Si no puedes dar un porcentaje exacto, ofrece un rango prudente."
            ),
            "rebalance": (
                "Indica una cadencia orientativa y al menos un trigger practico adicional "
                "por desviacion de pesos o cambio de tesis."
            ),
            "entry_plan": (
                "Propone una forma de entrada gradual o directa y menciona el papel de la liquidez "
                "o el escalonado de compras."
            ),
            "hedge": (
                "Indica si tiene sentido una cobertura o activo refugio y en que grado aproximado, "
                "sin sobredimensionarlo."
            ),
            "geography": (
                "Responde con una propuesta de distribucion geografica o un enfoque claro para reducir sesgo."
            ),
            "risk_watch": (
                "Enumera las senales o riesgos clave a vigilar antes de invertir o aumentar posicion."
            ),
            "thesis": (
                "Di si la accion encaja como posicion principal, complementaria o tactica para este perfil, "
                "y sustentalo con fortalezas y riesgos."
            ),
            "general": (
                "Responde de forma directa a la pregunta del usuario con una recomendacion prudente y accionable."
            ),
        }
        return instructions.get(question_type, instructions["general"])

    def _normalize_question_type(self, value: Any) -> str:
        normalized = sanitize_visible_answer(value).lower()
        if normalized in self.VALID_QUESTION_TYPES:
            return normalized
        return "general"

    def _build_prompt(self, query: str, market_data: dict = None, user_profile: dict = None) -> tuple[str, str]:
        normalized_profile = self._normalize_user_profile(user_profile)
        compact_report = self._compress_market_report(market_data)
        question_type = self._detect_question_type(query)
        today = date.today().isoformat()
        return f"""
Eres un analista financiero especializado en asesoramiento prudente y fundamentado.
Fecha actual: {today}

Reglas obligatorias:
- Responde siempre en espanol.
- No muestres razonamiento interno ni etiquetas como <think>.
- Basa tu respuesta solo en la evidencia proporcionada.
- Si la evidencia es incompleta, dilo explicitamente y evita falsa precision.
- Debes contestar la pregunta exacta del usuario, no limitarte a una tesis generica.

Tipo de consulta detectado: {question_type}
Instruccion especifica:
{self._question_type_instructions(question_type)}

Consulta del usuario:
{query}

Perfil del usuario:
- risk_level: {normalized_profile['risk_level']}
- investment_horizon: {normalized_profile['investment_horizon']}
- capital_amount: {normalized_profile['capital_amount']}
- investment_goals: {normalized_profile['investment_goals']}

INFORME DE MERCADO:
{compact_report}

Responde SOLO entre las etiquetas BEGIN_JSON y END_JSON con un JSON valido de esta estructura exacta:

BEGIN_JSON
{{
  "question_type": "{question_type}",
  "answer": "string",
  "thesis": "string",
  "strengths": ["string", "string"],
  "risks": ["string", "string"],
  "watch_items": ["string", "string"],
  "allocation_guidance": ["string", "string"],
  "implementation_steps": ["string", "string"],
  "preliminary_recommendation": "favorable|neutral|desfavorable",
  "confidence": "baja|media|alta",
  "answered_directly": true,
  "evidence_basis": "string"
}}
END_JSON
""", question_type

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

    def _normalize_recommendation_label(self, value: Any) -> str:
        lower = str(value or "").lower().strip()
        if lower in {"favorable", "buy", "positive"}:
            return "favorable"
        if lower in {"desfavorable", "sell", "negative"}:
            return "desfavorable"
        return "neutral"

    def _build_generic_fallback_answer(self, question_type: str, company_name: Optional[str]) -> str:
        company_label = company_name or "la empresa analizada"
        templates = {
            "allocation": (
                f"Con la evidencia disponible, trataria {company_label} como una posicion complementaria "
                "y no como el eje unico de la cartera. Para un perfil moderado, priorizaria fondos o ETFs "
                "diversificados y dejaria la accion individual en un rango contenido."
            ),
            "rebalance": (
                f"Para una cartera moderada con peso relevante en {company_label}, rebalancearia cada 6-12 meses "
                "o antes si la posicion se desvia claramente del objetivo inicial."
            ),
            "entry_plan": (
                f"Si quisieras entrar en {company_label} con riesgo de timing contenido, preferiria hacerlo en varios "
                "tramos y mantener parte de la liquidez hasta confirmar la tesis."
            ),
            "hedge": (
                f"Si {company_label} pesa mucho en cartera, puede tener sentido compensarlo con mas diversificacion "
                "o una pequena pata defensiva, pero sin sobredimensionar la cobertura."
            ),
            "geography": (
                f"Si quieres mantener {company_label} pero reducir sesgo geografico, complementaria la cartera con "
                "ETFs amplios de Europa y Asia en lugar de concentrar mas el riesgo en una sola region."
            ),
            "risk_watch": (
                f"Antes de invertir en {company_label}, vigilaria sobre todo el crecimiento del negocio, la evolucion "
                "de margenes y cash flow, y cualquier noticia regulatoria o cambio material en la tesis."
            ),
            "thesis": (
                f"{company_label} puede tener cabida en cartera, pero con la evidencia disponible la trataria como una "
                "posicion proporcionada al perfil de riesgo y no como una apuesta excesivamente concentrada."
            ),
            "general": (
                f"Con la evidencia disponible, mantendria una postura prudente sobre {company_label} y evitaria tomar "
                "decisiones muy precisas sin confirmar fundamentales, precio y noticias recientes."
            ),
        }
        return templates.get(question_type, templates["general"])

    def _format_answer_from_payload(
        self,
        payload: Dict[str, Any],
        question_type: str,
        company_name: Optional[str],
    ) -> str:
        direct_answer = sanitize_visible_answer(payload.get("answer", ""))
        if direct_answer:
            return direct_answer

        sections: list[str] = []

        thesis = sanitize_visible_answer(payload.get("thesis", ""))
        strengths = self._safe_list(payload.get("strengths"))
        risks = self._safe_list(payload.get("risks"))
        watch_items = self._safe_list(payload.get("watch_items"))
        allocation_guidance = self._safe_list(payload.get("allocation_guidance"))
        implementation_steps = self._safe_list(payload.get("implementation_steps"))
        evidence_basis = sanitize_visible_answer(payload.get("evidence_basis", ""))

        if thesis:
            sections.append(thesis)
        if allocation_guidance:
            sections.append("Guia de asignacion: " + "; ".join(allocation_guidance))
        if implementation_steps:
            sections.append("Implementacion: " + "; ".join(implementation_steps))
        if strengths:
            sections.append("Fortalezas: " + "; ".join(strengths))
        if risks:
            sections.append("Riesgos: " + "; ".join(risks))
        if watch_items:
            sections.append("A vigilar: " + "; ".join(watch_items))
        if evidence_basis:
            sections.append("Base de evidencia: " + evidence_basis)

        if not sections:
            return self._build_generic_fallback_answer(question_type, company_name)

        return "\n\n".join(sections)

    def _parse_json(
        self,
        raw_output: str,
        expected_question_type: str,
        company_name: Optional[str],
        profile_desc: str = "",
    ) -> Dict[str, Any]:
        cleaned_output = normalize_model_output(raw_output)
        internal_reasoning = extract_internal_reasoning(cleaned_output)
        sanitized_output = sanitize_visible_answer(cleaned_output)

        try:
            json_block = self._extract_json_block(cleaned_output)
            if not json_block:
                raise ValueError("No se encontro JSON en la salida.")

            parsed = json.loads(json_block, strict=False)
            parsed_question_type = self._normalize_question_type(parsed.get("question_type"))
            question_type = (
                expected_question_type
                if expected_question_type != "general"
                else parsed_question_type
            )
            strengths = self._safe_list(parsed.get("strengths"))
            risks = self._safe_list(parsed.get("risks"))
            watch_items = self._safe_list(parsed.get("watch_items"))
            allocation_guidance = self._safe_list(parsed.get("allocation_guidance"))
            implementation_steps = self._safe_list(parsed.get("implementation_steps"))
            evidence_basis = sanitize_visible_answer(parsed.get("evidence_basis", ""))
            answer = self._format_answer_from_payload(parsed, question_type, company_name)
            answered_directly = bool(parsed.get("answered_directly", bool(answer)))
            preliminary_recommendation = self._normalize_recommendation_label(
                parsed.get("preliminary_recommendation")
            )
            confidence = sanitize_visible_answer(parsed.get("confidence", "media")) or "media"

            return {
                "action": "Generando analisis y recomendacion personalizada",
                "result": f"Analisis realizado. {profile_desc}".strip(),
                "response": answer,
                "data": {
                    "question_type": question_type,
                    "answer": answer,
                    "thesis": sanitize_visible_answer(parsed.get("thesis", "")),
                    "strengths": strengths,
                    "risks": risks,
                    "watch_items": watch_items,
                    "allocation_guidance": allocation_guidance,
                    "implementation_steps": implementation_steps,
                    "preliminary_recommendation": preliminary_recommendation,
                    "confidence": confidence,
                    "answered_directly": answered_directly,
                    "evidence_basis": evidence_basis,
                    "internal_reasoning": internal_reasoning,
                    "sanitized_output": sanitized_output,
                    "raw_output": cleaned_output,
                },
            }
        except Exception as exc:
            self._log(f"Fallo parseando JSON del recommendation agent: {exc}")

            looks_like_plain_answer = (
                sanitized_output
                and "BEGIN_JSON" not in sanitized_output
                and "END_JSON" not in sanitized_output
                and sanitized_output.count("{") == 0
            )
            fallback_answer = (
                sanitized_output
                if looks_like_plain_answer and len(sanitized_output) >= 40
                else self._build_generic_fallback_answer(expected_question_type, company_name)
            )

            return {
                "action": "Generando analisis y recomendacion personalizada",
                "result": f"Analisis realizado. {profile_desc}".strip(),
                "response": fallback_answer,
                "data": {
                    "question_type": expected_question_type,
                    "answer": fallback_answer,
                    "thesis": fallback_answer,
                    "strengths": [],
                    "risks": [],
                    "watch_items": [],
                    "allocation_guidance": [],
                    "implementation_steps": [],
                    "preliminary_recommendation": "neutral",
                    "confidence": "baja",
                    "answered_directly": False,
                    "evidence_basis": "",
                    "internal_reasoning": internal_reasoning,
                    "sanitized_output": sanitized_output,
                    "raw_output": cleaned_output,
                },
            }

    def run(
        self,
        query: str,
        market_data: dict = None,
        user_profile: dict = None,
    ) -> tuple[dict, dict]:
        normalized_profile = self._normalize_user_profile(user_profile)
        profile_desc = (
            f"Perfil: {normalized_profile['risk_level']}, "
            f"horizonte: {normalized_profile['investment_horizon']}"
        )
        self._log(f"Inicio run | {profile_desc}")
        start_total = time.perf_counter()

        compact_report = self._compress_market_report(market_data)
        company_name = compact_report.get("company_name")

        if self.model is None or self.tokenizer is None:
            fallback = (
                "No se ha podido generar una recomendacion porque el modelo financiero no esta inicializado."
            )
            return {
                "action": "Generando analisis y recomendacion personalizada",
                "result": f"Analisis no realizado. {profile_desc}",
                "response": fallback,
                "data": {
                    "question_type": self._detect_question_type(query),
                    "answer": fallback,
                    "thesis": fallback,
                    "strengths": [],
                    "risks": [],
                    "watch_items": [],
                    "allocation_guidance": [],
                    "implementation_steps": [],
                    "preliminary_recommendation": "neutral",
                    "confidence": "baja",
                    "answered_directly": False,
                    "evidence_basis": "",
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

        prompt, question_type = self._build_prompt(
            query=query,
            market_data=market_data,
            user_profile=user_profile,
        )

        self._log(f"Llamando al modelo financiero Fin-R1 | question_type={question_type}")
        raw_output, token_info = generate_financial_reasoning(
            prompt,
            self.model,
            self.tokenizer,
            max_new_tokens=512,
            do_sample=False,
        )

        total_latency = time.perf_counter() - start_total
        final_token_info = {**(token_info or {}), "agent_total_latency": total_latency}
        self._log(f"Generacion financiera completada en {total_latency:.3f}s")

        return self._parse_json(
            raw_output=raw_output,
            expected_question_type=question_type,
            company_name=company_name,
            profile_desc=profile_desc,
        ), final_token_info
