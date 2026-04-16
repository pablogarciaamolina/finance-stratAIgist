"""
Market Agent — gathers objective market data for a given query.

Uses LangChain tools and RAG to collect:
- stock prices
- fundamentals
- recent events
- external market context
- retrieved economics context
- historical financial statement data for benchmark-like questions

This agent is intentionally factual: it does not recommend.
"""

import json
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
    """
    Retrieves factual market information relevant to the user's query.
    """

    TICKER_BLACKLIST = {
        "RAG", "LLM", "API", "JSON", "USA", "ETF", "CEO", "CFO", "SEC",
        "NASDAQ", "NYSE", "USD", "EUR", "AI", "IPO", "Q1", "Q2", "Q3", "Q4",
        "FY", "GAAP"
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
        "fy2018",
        "fy2019",
        "fy2020",
        "fy2021",
        "fy2022",
        "fy2023",
    }

    def __init__(self, tools: Optional[List[Any]] = None, rag_engine: Any = None):
        self.rag_engine = rag_engine

        self.tools = tools or [
            stock_price,
            company_fundamentals,
            company_events,
            company_financial_history,
            internet_search,
        ]

        self.tool_map = {tool.name: tool for tool in self.tools}

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def _invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return f"Error: herramienta no encontrada: {tool_name}"

        try:
            t0 = time.time()
            print(f"[TRACE] TOOL START -> {tool_name} | args={arguments}")
            result = tool.invoke(arguments)
            print(f"[TRACE] TOOL END   -> {tool_name} ({time.time() - t0:.2f}s)")
            return result
        except Exception as e:
            print(f"[TRACE] TOOL FAIL  -> {tool_name} | error={e}")
            return f"Error ejecutando {tool_name}: {str(e)}"

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

    def _safe_json_loads(self, text: Optional[str]):
        if not text or not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Query understanding
    # ------------------------------------------------------------------

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

    def _resolve_ticker(self, company_name: Optional[str]) -> Optional[str]:
        if not company_name:
            return None

        query = f"{company_name} stock ticker"
        search_result = self._invoke_tool("internet_search", {"query": query})

        if self._is_error_response(search_result):
            return None

        return self._extract_ticker_from_text(search_result)

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

    def _needs_historical_financials(self, query: str) -> bool:
        lower = (query or "").lower()
        return any(keyword in lower for keyword in self.HISTORICAL_KEYWORDS)

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # RAG integration
    # ------------------------------------------------------------------

    def _retrieve_rag_context(self, query: Optional[str], top_k_rag: int) -> List[Dict[str, Any]]:
        if not self.rag_engine or not query:
            return []

        try:
            t0 = time.time()
            print(f"[TRACE] RAG START  -> query={query}")
            if hasattr(self.rag_engine, "retrieve_context"):
                contexts = self.rag_engine.retrieve_context(
                    query=query,
                    top_k=top_k_rag,
                    similarity_threshold=0.75,
                )
            else:
                return []
            print(f"[TRACE] RAG END    -> ({time.time() - t0:.2f}s) | docs={len(contexts)}")
        except TypeError:
            try:
                t0 = time.time()
                print(f"[TRACE] RAG START  -> query={query}")
                contexts = self.rag_engine.retrieve_context(query, top_k=top_k_rag)
                print(f"[TRACE] RAG END    -> ({time.time() - t0:.2f}s) | docs={len(contexts)}")
            except Exception as e:
                print(f"[TRACE] RAG FAIL   -> error={e}")
                return []
        except Exception as e:
            print(f"[TRACE] RAG FAIL   -> error={e}")
            return []

        rag_snippets = []
        for ctx in contexts:
            rag_snippets.append({
                "label": ctx.get("label"),
                "distance": ctx.get("distance"),
                "text": ctx.get("text", "")[:800],
            })

        return rag_snippets

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        company_name: Optional[str] = None,
        ticker: Optional[str] = None,
        top_k_rag: int = 3,
    ) -> Dict[str, Any]:
        print(f"[TRACE] MarketAgent.run START | company_name={company_name} | ticker={ticker}")

        if not company_name and not ticker:
            print("[TRACE] MarketAgent.run END | no company/ticker detected")
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
            }

        # 1. Resolver ticker si hace falta
        resolved_ticker = False
        if not ticker and company_name:
            t0 = time.time()
            print("[TRACE] Resolving ticker START")
            resolved = self._resolve_ticker(company_name)
            print(f"[TRACE] Resolving ticker END ({time.time() - t0:.2f}s) | resolved={resolved}")
            if resolved:
                ticker = resolved
                resolved_ticker = True

        # 2. Determinar si la query necesita históricos
        historical_year = self._extract_year_from_query(query)
        needs_historical = self._needs_historical_financials(query)
        print(f"[TRACE] historical_year={historical_year} | needs_historical={needs_historical}")

        # 3. Herramientas financieras
        price_data = None
        fundamentals_data = None
        historical_financial_data = None
        events_data = None

        if ticker:
            price_data = self._invoke_tool("stock_price", {"ticker": ticker})
            fundamentals_data = self._invoke_tool("company_fundamentals", {"ticker": ticker})
            events_data = self._invoke_tool("company_events", {"ticker": ticker})

            if needs_historical and historical_year is not None:
                historical_financial_data = self._invoke_tool(
                    "company_financial_history",
                    {"ticker": ticker, "year": historical_year},
                )
                print(f"[TRACE] Historical financial data: {historical_financial_data}")

        # 4. Search externo
        external_context = None
        search_query = self._build_search_query(company_name, ticker)
        if search_query:
            external_context = self._invoke_tool("internet_search", {"query": search_query})
            print(f"[TRACE] External search result length: {external_context[:300] if external_context else None}")

        # 5. RAG
        rag_query = self._build_rag_query(company_name, ticker)
        rag_snippets = self._retrieve_rag_context(rag_query, top_k_rag=top_k_rag)

        # 6. Si hay histórico, lo fusionamos dentro de fundamentals_data
        combined_fundamentals = fundamentals_data
        if historical_financial_data and not self._is_error_response(historical_financial_data):
            current_fundamentals_json = self._safe_json_loads(fundamentals_data)
            historical_json = self._safe_json_loads(historical_financial_data)

            merged = {
                "current_fundamentals": current_fundamentals_json if current_fundamentals_json is not None else fundamentals_data,
                "historical_financial_data": historical_json if historical_json is not None else historical_financial_data,
            }
            combined_fundamentals = json.dumps(merged, ensure_ascii=False, indent=2)

        # 7. Validar evidencia mínima
        has_minimum_evidence = self._has_useful_market_evidence(
            price_data=price_data,
            fundamentals_data=combined_fundamentals,
            historical_financial_data=historical_financial_data,
            events_data=events_data,
            external_context=external_context,
            rag_snippets=rag_snippets,
        )

        # 8. Resumen textual
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
                summary_parts.append(f"Se han obtenido datos financieros históricos del ejercicio fiscal {historical_year}.")
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
            "events_data": events_data,
            "external_context": external_context,
            "rag_context": rag_snippets,
            "summary": " ".join(summary_parts).strip(),
            "resolved_ticker": resolved_ticker,
            "has_minimum_evidence": has_minimum_evidence,
        }

        if not has_minimum_evidence:
            market_report["error"] = (
                "No hay suficiente evidencia de mercado para emitir una recomendación fiable."
            )

        result_text = (
            "Datos de mercado recopilados correctamente."
            if has_minimum_evidence
            else "Datos de mercado recopilados, pero la evidencia es limitada."
        )

        print(f"[TRACE] MarketAgent.run END | has_minimum_evidence={has_minimum_evidence}")
        return {
            "action": f"Recopilando datos de mercado para: «{query[:80]}»",
            "result": result_text,
            "data": market_report,
        }