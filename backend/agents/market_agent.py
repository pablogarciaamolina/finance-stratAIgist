"""
Market Agent — gathers objective market data for a given query.

Uses external tools (financial APIs, web search) and RAG to collect
prices, fundamentals, recent news, and relevant economic context.

TODO: Replace stub with LangChain agent + tool bindings.
"""


class MarketAgent:
    """
    Retrieves factual market information relevant to the user's query.

    Capabilities (planned):
        - Stock prices via Alpha Vantage
        - Company fundamentals via SEC EDGAR
        - Recent events and filings (8-K)
        - Web search for breaking news
        - RAG context from economics knowledge base
    """

    def __init__(self, tools=None, rag_engine=None):
        self.tools = tools or []
        self.rag_engine = rag_engine

    def run(self, query: str) -> dict:
        """
        Gather market data for the given query.

        Args:
            query: User's investment question.

        Returns:
            Dict with 'action', 'result', and 'data' (structured market data).
        """
        # TODO: Implement real market data retrieval
        return {
            "action": f"Recopilando datos de mercado para: «{query[:80]}»",
            "result": "Datos de mercado recopilados (stub).",
            "data": {
                "prices": {},
                "fundamentals": {},
                "news": [],
                "rag_context": [],
            },
        }
