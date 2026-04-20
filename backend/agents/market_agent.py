"""
Market Agent — gathers objective market data for a given query.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from backend.tools.finance import (
    stock_price,
    company_fundamentals,
    company_events,
    company_financial_history,
)
from backend.tools.search import internet_search


class MarketAgent:
    TICKER_BLACKLIST = {
        "RAG", "LLM", "API", "JSON", "USA", "ETF", "CEO", "CFO", "SEC",
        "NASDAQ", "NYSE", "USD", "EUR", "AI", "IPO", "Q1", "Q2", "Q3", "Q4",
        "FY", "GAAP",
    }

    HISTORICAL_KEYWORDS = {
        "capital expenditure",
        "capital expenditures",
        "capex",
        "operating cash flow",
        "cash flow",
        "free cash flow",
        "net income",
        "revenue",
        "sales",
        "total assets",
        "total liabilities",
        "fixed asset turnover",
        "ppe",
        "property plant equipment",
        "fiscal year",
        "historical",
        "history",
        "annual",
        "trend",
        "fy2018",
        "fy2019",
        "fy2020",
        "fy2021",
        "fy2022",
        "fy2023",
        "fy2024",
    }

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        rag_engine: Any = None,
        debug: bool = True,
    ):
        self.rag_engine = rag_engine
        self.debug = debug
        self.tools = tools or [
            stock_price,
            company_fundamentals,
            company_events,
            company_financial_history,
            internet_search,
        ]
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _log(self, message: str):
        if self.debug:
            print(f"[MarketAgent] {message}", flush=True)

    def _invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[str, float]:
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return f"Error: herramienta no encontrada: {tool_name}", 0.0

        self._log(f"Invocando tool={tool_name} | args={arguments}")
        start = time.perf_counter()
        try:
            output = tool.invoke(arguments)
            latency = time.perf_counter() - start
            preview = str(output)[:180].replace("\n", " ")
            self._log(f"Tool {tool_name} completada en {latency:.3f}s | preview={preview}")
            return output, latency
        except Exception as exc:
            latency = time.perf_counter() - start
            self._log(f"Tool {tool_name} falló en {latency:.3f}s | error={exc}")
            return f"Error ejecutando {tool_name}: {str(exc)}", latency

    def _is_error_response(self, value: Optional[str]) -> bool:
        if value is None:
            return True
        lower = value.lower()
        return (
            lower.startswith("error")
            or "no se pudo" in lower
            or "no se encontró" in lower
            or "no está configurada" in lower
        )

    def _extract_ticker_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None

        patterns = [
            r"\(([A-Z]{1,5})\)",
            r"(?:NASDAQ|NYSE)\s*[:\-]?\s*([A-Z]{1,5})",
            r"\b([A-Z]{2,5})\b",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for candidate in matches:
                if candidate not in self.TICKER_BLACKLIST:
                    return candidate
        return None

    def _resolve_ticker(self, company_name: Optional[str]) -> tuple[Optional[str], float]:
        if not company_name:
            return None, 0.0
        query = f"{company_name} stock ticker"
        search_result, latency = self._invoke_tool("internet_search", {"query": query})
        if self._is_error_response(search_result):
            return None, latency
        return self._extract_ticker_from_text(search_result), latency

    def _build_search_query(self, company_name: Optional[str], ticker: Optional[str]) -> Optional[str]:
        if company_name and ticker:
            return f"{company_name} {ticker} latest company news market outlook"
        if ticker:
            return f"{ticker} latest company news market outlook"
        if company_name:
            return f"{company_name} latest company news market outlook"
        return None

    def _build_rag_query(self, company_name: Optional[str], ticker: Optional[str]) -> Optional[str]:
        if company_name and ticker:
            return f"{company_name} {ticker} economía finanzas riesgos crecimiento valoración"
        if ticker:
            return f"{ticker} economía finanzas riesgos crecimiento valoración"
        if company_name:
            return f"{company_name} economía finanzas riesgos crecimiento valoración"
        return None

    def _needs_historical_financials(self, query: str) -> bool:
        lower = query.lower()
        return any(keyword in lower for keyword in self.HISTORICAL_KEYWORDS)

    def _extract_historical_year(self, query: str) -> Optional[str]:
        lower = query.lower()

        fy_match = re.search(r"\bfy\s*([12]\d{3})\b", lower)
        if fy_match:
            return fy_match.group(1)

        year_match = re.search(r"\b(19|20)\d{2}\b", lower)
        if year_match:
            return year_match.group(0)

        return None

    def _build_combined_fundamentals(
        self,
        fundamentals_data: Optional[str],
        historical_financial_data: Optional[str],
        historical_year: Optional[str],
    ) -> Optional[str]:
        parts: List[str] = []

        if fundamentals_data and not self._is_error_response(fundamentals_data):
            parts.append(str(fundamentals_data).strip())

        if historical_financial_data and not self._is_error_response(historical_financial_data):
            header = (
                f"Datos financieros históricos ({historical_year}):"
                if historical_year
                else "Datos financieros históricos:"
            )
            parts.append(f"{header}\n{str(historical_financial_data).strip()}")

        if not parts:
            if fundamentals_data is not None:
                return fundamentals_data
            if historical_financial_data is not None:
                return historical_financial_data
            return None

        return "\n\n".join(parts)

    def _retrieve_rag_context(self, query: Optional[str], top_k_rag: int) -> tuple[List[Dict[str, Any]], float]:
        if not self.rag_engine or not query:
            return [], 0.0

        self._log(f"RAG retrieve_context | query={query!r} | top_k={top_k_rag}")
        start = time.perf_counter()
        try:
            if hasattr(self.rag_engine, "retrieve_context"):
                contexts = self.rag_engine.retrieve_context(
                    query=query,
                    top_k=top_k_rag,
                    similarity_threshold=0.75,
                )
            else:
                return [], time.perf_counter() - start
        except TypeError:
            try:
                contexts = self.rag_engine.retrieve_context(query, top_k=top_k_rag)
            except Exception as exc:
                latency = time.perf_counter() - start
                self._log(f"RAG falló en {latency:.3f}s | error={exc}")
                return [], latency
        except Exception as exc:
            latency = time.perf_counter() - start
            self._log(f"RAG falló en {latency:.3f}s | error={exc}")
            return [], latency

        latency = time.perf_counter() - start
        rag_snippets = [
            {
                "label": ctx.get("label"),
                "distance": ctx.get("distance"),
                "text": ctx.get("text", "")[:800],
            }
            for ctx in contexts
        ]
        self._log(f"RAG completado en {latency:.3f}s | snippets={len(rag_snippets)}")
        return rag_snippets, latency

    def _has_useful_market_evidence(
        self,
        price_data: Optional[str],
        fundamentals_data: Optional[str],
        historical_financial_data: Optional[str],
        events_data: Optional[str],
        external_context: Optional[str],
        rag_snippets: List[Dict[str, Any]],
    ) -> bool:
        signals = 0

        if price_data and not self._is_error_response(price_data):
            signals += 1
        if fundamentals_data and not self._is_error_response(fundamentals_data):
            signals += 1
        if historical_financial_data and not self._is_error_response(historical_financial_data):
            signals += 1
        if events_data and not self._is_error_response(events_data):
            signals += 1
        if external_context and not self._is_error_response(external_context):
            signals += 1
        if rag_snippets:
            signals += 1

        return signals >= 2

    def run(
        self,
        query: str,
        company_name: Optional[str] = None,
        ticker: Optional[str] = None,
        top_k_rag: int = 3,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        self._log(f"Inicio run | company={company_name} | ticker={ticker}")
        start_total = time.perf_counter()
        timings: Dict[str, float] = {}

        if not company_name and not ticker:
            self._log("Fin run | no company/ticker detected")
            return {
                "action": "Recopilando datos de mercado",
                "result": "No se pudo identificar la empresa ni el ticker.",
                "data": {
                    "company_name": None,
                    "ticker": None,
                    "price_data": None,
                    "fundamentals_data": None,
                    "historical_financial_data": None,
                    "events_data": None,
                    "external_context": None,
                    "rag_context": [],
                    "summary": "No se pudo identificar la empresa ni el ticker.",
                    "resolved_ticker": False,
                    "has_minimum_evidence": False,
                    "error": "No se pudo identificar la empresa o ticker.",
                },
            }, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "agent_total_latency": 0.0,
                "tool_timings": {},
            }

        resolved_ticker = False
        if not ticker and company_name:
            resolved, latency = self._resolve_ticker(company_name)
            timings["resolve_ticker"] = latency
            if resolved:
                ticker = resolved
                resolved_ticker = True
                self._log(f"Ticker resuelto automáticamente: {ticker}")

        price_data = None
        fundamentals_data = None
        historical_financial_data = None
        historical_year = self._extract_historical_year(query) if query else None

        events_data = None

        if ticker:
            price_data, timings["stock_price"] = self._invoke_tool("stock_price", {"ticker": ticker})
            fundamentals_data, timings["company_fundamentals"] = self._invoke_tool(
                "company_fundamentals",
                {"ticker": ticker},
            )
            events_data, timings["company_events"] = self._invoke_tool(
                "company_events",
                {"ticker": ticker},
            )

            if self._needs_historical_financials(query):
                history_args: Dict[str, Any] = {"ticker": ticker}
                if historical_year:
                    history_args["year"] = historical_year

                historical_financial_data, timings["company_financial_history"] = self._invoke_tool(
                    "company_financial_history",
                    history_args,
                )

        combined_fundamentals = self._build_combined_fundamentals(
            fundamentals_data=fundamentals_data,
            historical_financial_data=historical_financial_data,
            historical_year=historical_year,
        )

        external_context = None
        search_query = self._build_search_query(company_name, ticker)
        if search_query:
            external_context, timings["internet_search"] = self._invoke_tool(
                "internet_search",
                {"query": search_query},
            )

        rag_query = self._build_rag_query(company_name, ticker)
        rag_snippets, timings["rag_retrieve_context"] = self._retrieve_rag_context(
            rag_query,
            top_k_rag=top_k_rag,
        )

        has_minimum_evidence = self._has_useful_market_evidence(
            price_data=price_data,
            fundamentals_data=combined_fundamentals,
            historical_financial_data=historical_financial_data,
            events_data=events_data,
            external_context=external_context,
            rag_snippets=rag_snippets,
        )

        summary_parts = []
        if company_name:
            summary_parts.append(f"Empresa analizada: {company_name}.")
        if ticker:
            summary_parts.append(f"Ticker analizado: {ticker}.")
        if resolved_ticker:
            summary_parts.append("El ticker se resolvió automáticamente a partir del nombre de la empresa.")
        if price_data and not self._is_error_response(price_data):
            summary_parts.append("Se ha obtenido información de precio.")
        if fundamentals_data and not self._is_error_response(fundamentals_data):
            summary_parts.append("Se han obtenido fundamentales.")
        if historical_financial_data and not self._is_error_response(historical_financial_data):
            if historical_year:
                summary_parts.append(
                    f"Se han obtenido datos financieros históricos del ejercicio fiscal {historical_year}."
                )
            else:
                summary_parts.append("Se han obtenido datos financieros históricos.")
        if events_data and not self._is_error_response(events_data):
            summary_parts.append("Se han recuperado eventos recientes.")
        if external_context and not self._is_error_response(external_context):
            summary_parts.append("Se ha realizado búsqueda externa.")
        if rag_snippets:
            summary_parts.append(f"Se han recuperado {len(rag_snippets)} fragmentos por RAG.")
        if not has_minimum_evidence:
            summary_parts.append("La evidencia recuperada es limitada para emitir una respuesta sólida.")

        market_report = {
            "company_name": company_name,
            "ticker": ticker,
            "price_data": price_data,
            "fundamentals_data": combined_fundamentals,
            "historical_financial_data": historical_financial_data,
            "historical_year": historical_year,
            "events_data": events_data,
            "external_context": external_context,
            "rag_context": rag_snippets,
            "summary": " ".join(summary_parts).strip(),
            "resolved_ticker": resolved_ticker,
            "has_minimum_evidence": has_minimum_evidence,
        }
        if not has_minimum_evidence:
            market_report["error"] = "No hay suficiente evidencia de mercado para emitir una recomendación fiable."

        total_latency = time.perf_counter() - start_total
        result_text = (
            "Datos de mercado recopilados correctamente."
            if has_minimum_evidence
            else "Datos de mercado recopilados, pero la evidencia es limitada."
        )
        self._log(f"Fin run en {total_latency:.3f}s | timings={timings}")

        return {
            "action": f"Recopilando datos de mercado para: «{query[:80]}»",
            "result": result_text,
            "data": market_report,
        }, {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "agent_total_latency": total_latency,
            "tool_timings": timings,
            "successful_signals": sum([
                int(price_data is not None and not self._is_error_response(price_data)),
                int(combined_fundamentals is not None and not self._is_error_response(combined_fundamentals)),
                int(historical_financial_data is not None and not self._is_error_response(historical_financial_data)),
                int(events_data is not None and not self._is_error_response(events_data)),
                int(external_context is not None and not self._is_error_response(external_context)),
                int(bool(rag_snippets)),
            ]),
        }
