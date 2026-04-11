"""
Market Agent — gathers objective market data for a given query.

Uses LangChain tools and RAG to collect:
- stock prices
- fundamentals
- recent events
- external market context
- retrieved economics context

This agent is intentionally factual: it does not recommend.
"""

import re
from typing import Any, Dict, List, Optional

from backend.tools.finance import stock_price, company_fundamentals, company_events
from backend.tools.search import internet_search


class MarketAgent:
    """
    Retrieves factual market information relevant to the user's query.

    This agent uses LangChain tools as first-class tools:
    - stock_price
    - company_fundamentals
    - company_events
    - internet_search

    It also uses a RAG engine when available.
    """

    TICKER_BLACKLIST = {
        "RAG", "LLM", "API", "JSON", "USA", "ETF", "CEO", "CFO", "SEC",
        "NASDAQ", "NYSE", "USD", "EUR", "AI", "IPO", "Q1", "Q2", "Q3", "Q4"
    }

    def __init__(self, tools: Optional[List[Any]] = None, rag_engine: Any = None):
        self.rag_engine = rag_engine

        # Si no se pasan tools, usamos las del repo nuevo
        self.tools = tools or [
            stock_price,
            company_fundamentals,
            company_events,
            internet_search,
        ]

        self.tool_map = {tool.name: tool for tool in self.tools}

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def _invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Ejecuta una tool de LangChain por nombre usando .invoke().
        """
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return f"Error: herramienta no encontrada: {tool_name}"

        try:
            return tool.invoke(arguments)
        except Exception as e:
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

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _extract_ticker_from_text(self, text: str) -> Optional[str]:
        """
        Heurística para extraer ticker de texto libre.
        """
        if not text:
            return None

        patterns = [
            r"\(([A-Z]{1,5})\)",                         # (NVDA)
            r"(?:NASDAQ|NYSE)\s*[:\-]?\s*([A-Z]{1,5})", # NASDAQ: NVDA
            r"\b([A-Z]{2,5})\b",                        # NVDA
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for candidate in matches:
                if candidate not in self.TICKER_BLACKLIST:
                    return candidate

        return None

    def _resolve_ticker(self, company_name: Optional[str]) -> Optional[str]:
        """
        Si no tenemos ticker, intentamos inferirlo con internet_search.
        """
        if not company_name:
            return None

        query = f"{company_name} stock ticker"
        search_result = self._invoke_tool("internet_search", {"query": query})

        if self._is_error_response(search_result):
            return None

        return self._extract_ticker_from_text(search_result)

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
            # Caso estilo openqa
            if hasattr(self.rag_engine, "retrieve_context"):
                contexts = self.rag_engine.retrieve_context(
                    query=query,
                    top_k=top_k_rag,
                    similarity_threshold=0.75,
                )
            else:
                return []
        except TypeError:
            # Fallback si el engine tiene otra firma
            try:
                contexts = self.rag_engine.retrieve_context(query, top_k=top_k_rag)
            except Exception:
                return []
        except Exception:
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
        events_data: Optional[str],
        external_context: Optional[str],
        rag_snippets: List[Dict[str, Any]],
    ) -> bool:
        signals = 0

        if price_data and not self._is_error_response(price_data):
            signals += 1
        if fundamentals_data and not self._is_error_response(fundamentals_data):
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
        """
        Gathers factual market information for the given query.

        Args:
            query: User investment question
            company_name: Optional company name already extracted by orchestrator
            ticker: Optional ticker already extracted by orchestrator
            top_k_rag: number of RAG snippets to retrieve

        Returns:
            Dict with action/result/data and structured evidence
        """
        if not company_name and not ticker:
            return {
                "action": "Recopilando datos de mercado",
                "result": "No se pudo identificar la empresa ni el ticker.",
                "data": {
                    "company_name": None,
                    "ticker": None,
                    "price_data": None,
                    "fundamentals_data": None,
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
            resolved = self._resolve_ticker(company_name)
            if resolved:
                ticker = resolved
                resolved_ticker = True

        # 2. Herramientas financieras (solo si hay ticker)
        price_data = None
        fundamentals_data = None
        events_data = None

        if ticker:
            price_data = self._invoke_tool("stock_price", {"ticker": ticker})
            fundamentals_data = self._invoke_tool("company_fundamentals", {"ticker": ticker})
            events_data = self._invoke_tool("company_events", {"ticker": ticker})

        # 3. Search externo
        external_context = None
        search_query = self._build_search_query(company_name, ticker)
        if search_query:
            external_context = self._invoke_tool("internet_search", {"query": search_query})

        # 4. RAG
        rag_query = self._build_rag_query(company_name, ticker)
        rag_snippets = self._retrieve_rag_context(rag_query, top_k_rag=top_k_rag)

        # 5. Validar evidencia mínima
        has_minimum_evidence = self._has_useful_market_evidence(
            price_data=price_data,
            fundamentals_data=fundamentals_data,
            events_data=events_data,
            external_context=external_context,
            rag_snippets=rag_snippets,
        )

        # 6. Resumen textual
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
        if events_data and not self._is_error_response(events_data):
            summary_parts.append("Se han recuperado eventos recientes.")
        if external_context and not self._is_error_response(external_context):
            summary_parts.append("Se ha realizado búsqueda externa.")
        if rag_snippets:
            summary_parts.append(f"Se han recuperado {len(rag_snippets)} fragmentos por RAG.")
        if not has_minimum_evidence:
            summary_parts.append("La evidencia recuperada es limitada para emitir una recomendación sólida.")

        market_report = {
            "company_name": company_name,
            "ticker": ticker,
            "price_data": price_data,
            "fundamentals_data": fundamentals_data,
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

        return {
            "action": f"Recopilando datos de mercado para: «{query[:80]}»",
            "result": result_text,
            "data": market_report,
        }