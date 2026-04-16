"""
Benchmark Answer Agent — produces factual financial QA answers for benchmark mode.

This agent is designed for datasets such as FinanceBench, where:
- there is no user profile
- the goal is factual / grounded financial QA
- the company may come from metadata rather than from the query itself

It takes the structured output of the Market Agent and generates a direct answer
to the user's question, avoiding recommendation-style language.
"""

import json
import time
from typing import Any, Dict, Optional

from backend.models.general_model import generate_general_reasoning


class BenchmarkAnswerAgent:
    """
    Answers factual financial questions in benchmark mode.
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

        rag_context = report.get("rag_context", [])[:3]
        compact_rag = [
            {
                "label": ctx.get("label"),
                "distance": ctx.get("distance"),
                "text": ctx.get("text", "")[:500],
            }
            for ctx in rag_context
        ]

        external_context = report.get("external_context")
        if isinstance(external_context, str):
            external_context = external_context[:1200]

        return {
            "company_name": report.get("company_name"),
            "ticker": report.get("ticker"),
            "price_data": report.get("price_data"),
            "fundamentals_data": report.get("fundamentals_data"),
            "events_data": report.get("events_data"),
            "external_context": external_context,
            "rag_context": compact_rag,
            "summary": report.get("summary"),
            "has_minimum_evidence": report.get("has_minimum_evidence", False),
            "resolved_ticker": report.get("resolved_ticker", False),
        }

    def _build_prompt(self, query: str, market_data: dict = None) -> str:
        compact_report = self._compact_market_data(market_data)

        print("report:",compact_report    )

        return f"""
You are a financial QA assistant working in benchmark mode.

Your task is to answer the user's question as directly and factually as possible,
using ONLY the information provided below.

Rules:
- Do NOT give investment advice.
- Do NOT say whether the user should buy or sell.
- Do NOT produce a recommendation thesis.
- If the evidence is insufficient to answer precisely, say so clearly.
- If a calculation is possible from the provided data, explain it briefly.
- Keep the answer factual, concise, and grounded.

User question:
{query}

AVAILABLE EVIDENCE:
{compact_report}

Respond ONLY between BEGIN_JSON and END_JSON with valid JSON in this exact format:

BEGIN_JSON
{{
  "answer": "string",
  "grounded": true,
  "confidence": "low|medium|high"
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

    def _parse_json(self, raw_output: str) -> Dict[str, Any]:
        cleaned_output = raw_output

        try:
            if "ASSISTANT:" in cleaned_output:
                cleaned_output = cleaned_output.split("ASSISTANT:")[-1].strip()
            cleaned_output = cleaned_output.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(cleaned_output)
            if not json_block:
                raise ValueError("No JSON block found.")

            parsed = json.loads(json_block)

            answer = parsed.get("answer", "")
            grounded = parsed.get("grounded", True)
            confidence = parsed.get("confidence", "medium")

            return {
                "action": "Generating factual benchmark answer",
                "result": "Benchmark QA answer generated.",
                "response": answer,
                "data": {
                    "answer": answer,
                    "grounded": bool(grounded),
                    "confidence": confidence,
                    "raw_output": cleaned_output,
                },
            }
        except Exception:
            fallback = cleaned_output[:1200] or (
                "No sufficient grounded evidence was available to answer the question."
            )

            return {
                "action": "Generating factual benchmark answer",
                "result": "Benchmark QA answer generated with fallback parsing.",
                "response": fallback,
                "data": {
                    "answer": fallback,
                    "grounded": False,
                    "confidence": "low",
                    "raw_output": cleaned_output,
                },
            }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, query: str, market_data: dict = None) -> dict:
        """
        Generate a factual benchmark-style financial QA answer.
        """
        if self.model is None or self.tokenizer is None:
            fallback = (
                "No se ha podido generar una respuesta factual del benchmark "
                "porque el modelo general no está inicializado."
            )
            return {
                "action": "Generating factual benchmark answer",
                "result": "Benchmark QA not executed.",
                "response": fallback,
                "data": {
                    "answer": fallback,
                    "grounded": False,
                    "confidence": "low",
                    "raw_output": "",
                },
            }

        t0 = time.time()
        print("[TRACE] BenchmarkAnswerAgent prompt build START")
        prompt = self._build_prompt(query=query, market_data=market_data)
        print(f"[TRACE] BenchmarkAnswerAgent prompt build END ({time.time() - t0:.2f}s)")

        t0 = time.time()
        print("[TRACE] BenchmarkAnswerAgent generation START")
        raw_output = generate_general_reasoning(prompt, self.model, self.tokenizer)
        print(f"[TRACE] BenchmarkAnswerAgent generation END ({time.time() - t0:.2f}s)")

        t0 = time.time()
        print("[TRACE] BenchmarkAnswerAgent parse START")
        parsed = self._parse_json(raw_output)
        print(f"[TRACE] BenchmarkAnswerAgent parse END ({time.time() - t0:.2f}s)")

        return parsed