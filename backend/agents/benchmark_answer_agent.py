"""
Benchmark Answer Agent — produces factual financial QA answers for benchmark mode.

Designed for benchmark-style financial QA (e.g. FinanceBench), where:
- there is no user profile
- the goal is factual / grounded financial QA
- the company may come from metadata
- historical financial values matter more than generic market commentary

Key design choices:
1. First try deterministic extraction from structured evidence
2. Then try simple rule-based calculations (e.g. fixed asset turnover)
3. Only fall back to the LLM if needed
"""

import json
import re
import time
from typing import Any, Dict, Optional

from backend.models.general_model import generate_general_reasoning


class BenchmarkAnswerAgent:
    """
    Answers factual financial questions in benchmark mode.

    Priority order:
    1. Historical financial data
    2. Current fundamentals if directly useful
    3. External / RAG context
    4. LLM fallback
    """

    def __init__(self, model: Any = None, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # JSON / normalization helpers
    # ------------------------------------------------------------------

    def _safe_json_loads(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    def _normalize_market_report(self, market_data: Optional[dict]) -> Dict[str, Any]:
        if not market_data:
            return {}

        report = market_data.get("data", market_data)

        fundamentals = self._safe_json_loads(report.get("fundamentals_data"))
        historical = self._safe_json_loads(report.get("historical_financial_data"))
        price_data = self._safe_json_loads(report.get("price_data"))
        events_data = self._safe_json_loads(report.get("events_data"))

        external_context = report.get("external_context")
        if isinstance(external_context, str):
            external_context = external_context[:1500]

        rag_context = report.get("rag_context", [])[:3]
        compact_rag = [
            {
                "label": ctx.get("label"),
                "distance": ctx.get("distance"),
                "text": ctx.get("text", "")[:600],
            }
            for ctx in rag_context
        ]

        # Si fundamentals_data ya viene "fusionado", intentamos usarlo mejor
        merged_fundamentals = fundamentals
        if isinstance(fundamentals, dict) and "historical_financial_data" in fundamentals:
            if historical is None:
                historical = fundamentals.get("historical_financial_data")
            if fundamentals.get("current_fundamentals") is not None:
                merged_fundamentals = fundamentals.get("current_fundamentals")

        return {
            "company_name": report.get("company_name"),
            "ticker": report.get("ticker"),
            "price_data": price_data,
            "fundamentals_data": merged_fundamentals,
            "historical_financial_data": historical,
            "events_data": events_data,
            "external_context": external_context,
            "rag_context": compact_rag,
            "summary": report.get("summary"),
            "has_minimum_evidence": report.get("has_minimum_evidence", False),
            "resolved_ticker": report.get("resolved_ticker", False),
        }

    # ------------------------------------------------------------------
    # Query understanding
    # ------------------------------------------------------------------

    def _extract_year_from_query(self, query: str) -> Optional[int]:
        if not query:
            return None

        fy_match = re.search(r"\bFY\s?(\d{4})\b", query, re.IGNORECASE)
        if fy_match:
            return int(fy_match.group(1))

        year_match = re.search(r"\b(20\d{2}|19\d{2})\b", query)
        if year_match:
            return int(year_match.group(1))

        return None

    def _query_mentions_capex(self, query: str) -> bool:
        q = query.lower()
        return any(term in q for term in [
            "capital expenditure",
            "capital expenditures",
            "capex",
        ])

    def _query_mentions_revenue(self, query: str) -> bool:
        q = query.lower()
        return any(term in q for term in [
            "revenue",
            "sales",
        ])

    def _query_mentions_net_income(self, query: str) -> bool:
        q = query.lower()
        return any(term in q for term in [
            "net income",
            "earnings",
        ])

    def _query_mentions_operating_cash_flow(self, query: str) -> bool:
        q = query.lower()
        return any(term in q for term in [
            "operating cash flow",
            "cash flow from operations",
            "net cash provided by operating activities",
        ])

    def _query_mentions_fixed_asset_turnover(self, query: str) -> bool:
        q = query.lower()
        return "fixed asset turnover" in q

    # ------------------------------------------------------------------
    # Structured evidence extraction
    # ------------------------------------------------------------------

    def _get_hist_value(self, hist: Optional[dict], key: str):
        if not isinstance(hist, dict):
            return None
        value_block = hist.get(key)
        if isinstance(value_block, dict):
            return value_block.get("value")
        return None

    def _format_usd_millions(self, value) -> str:
        try:
            return f"{float(value) / 1_000_000:.2f} USD millions"
        except Exception:
            return str(value)

    def _try_direct_answer_from_structured_data(
        self,
        query: str,
        report: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        hist = report.get("historical_financial_data")
        fundamentals = report.get("fundamentals_data")

        # 1. Direct yearly questions from historical data
        if hist:
            if self._query_mentions_capex(query):
                capex = self._get_hist_value(hist, "capital_expenditures")
                if capex is not None:
                    fy = hist.get("fiscal_year")
                    company = hist.get("company") or report.get("company_name") or report.get("ticker")
                    answer = (
                        f"The fiscal year {fy} capital expenditure amount for {company} "
                        f"was {self._format_usd_millions(capex)}."
                    )
                    return {
                        "answer": answer,
                        "grounded": True,
                        "confidence": "high",
                    }

            if self._query_mentions_operating_cash_flow(query):
                ocf = self._get_hist_value(hist, "operating_cash_flow")
                if ocf is not None:
                    fy = hist.get("fiscal_year")
                    company = hist.get("company") or report.get("company_name") or report.get("ticker")
                    answer = (
                        f"The fiscal year {fy} operating cash flow for {company} "
                        f"was {self._format_usd_millions(ocf)}."
                    )
                    return {
                        "answer": answer,
                        "grounded": True,
                        "confidence": "high",
                    }

            if self._query_mentions_revenue(query):
                revenue = self._get_hist_value(hist, "revenue")
                if revenue is not None:
                    fy = hist.get("fiscal_year")
                    company = hist.get("company") or report.get("company_name") or report.get("ticker")
                    answer = (
                        f"The fiscal year {fy} revenue for {company} "
                        f"was {self._format_usd_millions(revenue)}."
                    )
                    return {
                        "answer": answer,
                        "grounded": True,
                        "confidence": "high",
                    }

            if self._query_mentions_net_income(query):
                net_income = self._get_hist_value(hist, "net_income")
                if net_income is not None:
                    fy = hist.get("fiscal_year")
                    company = hist.get("company") or report.get("company_name") or report.get("ticker")
                    answer = (
                        f"The fiscal year {fy} net income for {company} "
                        f"was {self._format_usd_millions(net_income)}."
                    )
                    return {
                        "answer": answer,
                        "grounded": True,
                        "confidence": "high",
                    }

            if self._query_mentions_fixed_asset_turnover(query):
                revenue = self._get_hist_value(hist, "revenue")
                total_assets = self._get_hist_value(hist, "total_assets")

                # Approximation warning: true fixed asset turnover usually needs net PP&E.
                # We only answer if your structured tool later provides the right input.
                ppe = self._get_hist_value(hist, "property_plant_equipment_net")

                denominator = ppe if ppe is not None else None
                if revenue is not None and denominator not in (None, 0):
                    ratio = float(revenue) / float(denominator)
                    fy = hist.get("fiscal_year")
                    company = hist.get("company") or report.get("company_name") or report.get("ticker")
                    answer = (
                        f"The fiscal year {fy} fixed asset turnover ratio for {company} "
                        f"was approximately {ratio:.4f}."
                    )
                    return {
                        "answer": answer,
                        "grounded": True,
                        "confidence": "medium",
                    }

        # 2. Direct latest fundamentals if query is generic and historical missing
        if isinstance(fundamentals, dict):
            if self._query_mentions_revenue(query) and fundamentals.get("revenue") is not None:
                answer = (
                    f"The available revenue value for {report.get('company_name') or report.get('ticker')} "
                    f"is {self._format_usd_millions(fundamentals['revenue'])}."
                )
                return {
                    "answer": answer,
                    "grounded": True,
                    "confidence": "medium",
                }

        return None

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, report: Dict[str, Any]) -> str:
        # IMPORTANT: do not emphasize "insufficient evidence" in a way that
        # biases the model too much if there is usable structured data.
        return f"""
You are a financial QA assistant working in benchmark mode.

Your task is to answer the user's question as directly and factually as possible,
using ONLY the evidence provided below.

Rules:
- Do NOT give investment advice.
- Do NOT recommend buying or selling.
- Prefer exact numeric answers when available.
- If a calculation is needed and the necessary numbers are present, perform it.
- If the evidence is genuinely insufficient, say so clearly and briefly.
- Base the answer ONLY on the provided evidence.

User question:
{query}

AVAILABLE EVIDENCE:
{report}

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
        if not isinstance(text, str):
            return None

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
            if not isinstance(cleaned_output, str):
                cleaned_output = str(cleaned_output)

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
            fallback = cleaned_output[:1200] if isinstance(cleaned_output, str) else str(cleaned_output)
            if not fallback:
                fallback = "No sufficient grounded evidence was available to answer the question."

            return {
                "action": "Generating factual benchmark answer",
                "result": "Benchmark QA answer generated with fallback parsing.",
                "response": fallback,
                "data": {
                    "answer": fallback,
                    "grounded": False,
                    "confidence": "low",
                    "raw_output": cleaned_output if isinstance(cleaned_output, str) else str(cleaned_output),
                },
            }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, query: str, market_data: dict = None) -> dict:
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

        normalized_report = self._normalize_market_report(market_data)

        print("[TRACE] BenchmarkAnswerAgent normalized report keys:", list(normalized_report.keys()))
        print("[TRACE] BenchmarkAnswerAgent historical_financial_data:", normalized_report.get("historical_financial_data"))

        # 1. First try deterministic extraction / calculation
        t0 = time.time()
        print("[TRACE] BenchmarkAnswerAgent structured extraction START")
        direct_answer = self._try_direct_answer_from_structured_data(query, normalized_report)
        print(f"[TRACE] BenchmarkAnswerAgent structured extraction END ({time.time() - t0:.2f}s)")

        if direct_answer is not None:
            print("[TRACE] BenchmarkAnswerAgent returning DIRECT structured answer")
            return {
                "action": "Generating factual benchmark answer",
                "result": "Benchmark QA answer generated from structured evidence.",
                "response": direct_answer["answer"],
                "data": {
                    "answer": direct_answer["answer"],
                    "grounded": direct_answer["grounded"],
                    "confidence": direct_answer["confidence"],
                    "raw_output": "DIRECT_STRUCTURED_ANSWER",
                },
            }

        # 2. LLM fallback
        t0 = time.time()
        print("[TRACE] BenchmarkAnswerAgent prompt build START")
        prompt = self._build_prompt(query=query, report=normalized_report)
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