"""
Investment Multi-Agent System.

Orchestrates the full pipeline:
1. Orchestrator
2. Market Agent
3. Recommendation Agent
4. Critic Agent

Returns a final response plus agent trace and internal reports.
"""

from typing import Any, Dict, Optional

from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.market_agent import MarketAgent
from backend.agents.recommendation import RecommendationAgent
from backend.agents.critic import CriticAgent
from backend.agents.benchmark_answer_agent import BenchmarkAnswerAgent
from backend.rag.engine import RAGEngine


class InvestmentMultiAgentSystem:
    """
    Sistema multiagente completo para recomendación de inversiones.

    Soporta dos modos:
    - advisor: recomendación personalizada
    - benchmark: respuesta factual / financial QA
    """

    def __init__(
        self,
        general_model: Any,
        general_tokenizer: Any,
        fin_model: Any,
        fin_tokenizer: Any,
        rag_engine: Optional[RAGEngine] = None,
        market_tools: Optional[list] = None,
    ):
        self.orchestrator = OrchestratorAgent(general_model, general_tokenizer)
        self.market_agent = MarketAgent(tools=market_tools, rag_engine=rag_engine)
        self.recommendation_agent = RecommendationAgent(fin_model, fin_tokenizer)
        self.critic_agent = CriticAgent(general_model, general_tokenizer)
        self.benchmark_answer_agent = BenchmarkAnswerAgent(general_model, general_tokenizer)

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def _append_trace(self, trace: list, agent_name: str, payload: Dict[str, Any]):
        trace.append({
            "agent": agent_name,
            "action": payload.get("action", ""),
            "result": payload.get("result", ""),
        })

    # ------------------------------------------------------------------
    # Failure helpers
    # ------------------------------------------------------------------

    def _build_failure_response(
        self,
        message: str,
        trace: list,
        orchestration: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        recommendation_data: Optional[Dict[str, Any]] = None,
        critic_data: Optional[Dict[str, Any]] = None,
        benchmark_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "response": message,
            "agent_trace": trace,
            "metadata": {
                "pipeline": "multiagent",
                "status": "failed",
                **(metadata or {}),
            },
            "debug": {
                "orchestration": orchestration or {},
                "market_data": market_data or {},
                "recommendation_data": recommendation_data or {},
                "critic_data": critic_data or {},
                "benchmark_data": benchmark_data or {},
            },
        }

    def _recommendation_is_too_weak(self, recommendation_output: Dict[str, Any]) -> bool:
        data = recommendation_output.get("data", {})
        thesis = data.get("thesis", "") or recommendation_output.get("response", "")
        strengths = data.get("strengths", [])
        risks = data.get("risks", [])
        scenarios = data.get("scenarios", [])

        if not thesis or len(thesis.strip()) < 20:
            return True

        if not isinstance(strengths, list):
            strengths = []
        if not isinstance(risks, list):
            risks = []
        if not isinstance(scenarios, list):
            scenarios = []

        if len(strengths) == 0 and len(risks) == 0 and len(scenarios) == 0:
            return True

        return False

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        user_profile: Optional[dict] = None,
        company_name: Optional[str] = None,
        ticker: Optional[str] = None,
        mode: str = "advisor",
    ) -> Dict[str, Any]:
        trace = []

        # 1. Orchestrator
        orchestration = self.orchestrator.run(
            query=query,
            user_profile=user_profile,
            company_name=company_name,
            ticker=ticker,
            mode=mode,
        )
        self._append_trace(trace, "Orchestrator", orchestration)

        company_name = orchestration.get("company_name")
        ticker = orchestration.get("ticker")

        # 2. Market Agent
        market_data = self.market_agent.run(
            query=query,
            company_name=company_name,
            ticker=ticker,
        )
        self._append_trace(trace, "Market Agent", market_data)

        market_report = market_data.get("data", {})

        # Si no hay empresa/ticker y tampoco se ha construido evidencia útil, fallamos
        if (
            not company_name
            and not ticker
            and not market_report.get("has_minimum_evidence", False)
        ):
            return self._build_failure_response(
                message=(
                    "No he podido identificar con suficiente claridad la empresa o el ticker, "
                    "ni reunir suficiente contexto para responder a la consulta."
                ),
                trace=trace,
                orchestration=orchestration,
                market_data=market_data,
                metadata={"failure_stage": "orchestrator_market", "mode": mode},
            )

        # Modo benchmark: factual QA, sin recommendation/critic
        if mode == "benchmark":
            benchmark_output = self.benchmark_answer_agent.run(
                query=query,
                market_data=market_data,
            )
            self._append_trace(trace, "Benchmark Answer Agent", benchmark_output)

            final_answer = (
                benchmark_output.get("response")
                or benchmark_output.get("data", {}).get("answer")
                or "No se pudo generar una respuesta factual para la consulta."
            )

            return {
                "response": final_answer,
                "agent_trace": trace,
                "metadata": {
                    "pipeline": "multiagent",
                    "status": "completed",
                    "mode": "benchmark",
                    "agents_used": ["orchestrator", "market", "benchmark_answer"],
                },
                "debug": {
                    "orchestration": orchestration,
                    "market_data": market_data,
                    "benchmark_data": benchmark_output,
                },
            }

        # Advisor mode: aquí sí exigimos evidencia suficiente para recomendar
        if market_report.get("error") or not market_report.get("has_minimum_evidence", False):
            return self._build_failure_response(
                message=(
                    "No he podido reunir suficiente evidencia estructurada y fiable del mercado "
                    "para emitir una recomendación razonada sobre esta empresa en este momento."
                ),
                trace=trace,
                orchestration=orchestration,
                market_data=market_data,
                metadata={"failure_stage": "market_agent", "mode": "advisor"},
            )

        # 3. Recommendation Agent
        recommendation_output = self.recommendation_agent.run(
            query=query,
            market_data=market_data,
            user_profile=user_profile,
        )
        self._append_trace(trace, "Recommendation Agent", recommendation_output)

        if self._recommendation_is_too_weak(recommendation_output):
            return self._build_failure_response(
                message=(
                    "He podido recuperar información de mercado, pero la tesis de inversión generada "
                    "no tiene suficiente calidad o detalle como para devolver una recomendación fiable."
                ),
                trace=trace,
                orchestration=orchestration,
                market_data=market_data,
                recommendation_data=recommendation_output,
                metadata={"failure_stage": "recommendation_agent", "mode": "advisor"},
            )

        # 4. Critic Agent
        critic_output = self.critic_agent.run(
            query=query,
            recommendation=recommendation_output,
            market_data=market_data,
            user_profile=user_profile,
        )
        self._append_trace(trace, "Critic Agent", critic_output)

        critic_data = critic_output.get("data", {})
        enough_evidence = critic_data.get("enough_evidence", False)

        if not enough_evidence:
            final_answer = (
                critic_output.get("revised_response")
                or critic_data.get("final_answer")
                or (
                    "No hay suficiente evidencia para emitir una recomendación de inversión "
                    "con un nivel razonable de confianza."
                )
            )

            return {
                "response": final_answer,
                "agent_trace": trace,
                "metadata": {
                    "pipeline": "multiagent",
                    "status": "completed_with_warning",
                    "mode": "advisor",
                    "warning": "insufficient_evidence",
                    "agents_used": ["orchestrator", "market", "recommendation", "critic"],
                },
                "debug": {
                    "orchestration": orchestration,
                    "market_data": market_data,
                    "recommendation_data": recommendation_output,
                    "critic_data": critic_output,
                },
            }

        final_answer = (
            critic_output.get("revised_response")
            or critic_data.get("final_answer")
            or recommendation_output.get("response")
            or "No se pudo generar una respuesta final."
        )

        return {
            "response": final_answer,
            "agent_trace": trace,
            "metadata": {
                "pipeline": "multiagent",
                "status": "completed",
                "mode": "advisor",
                "agents_used": ["orchestrator", "market", "recommendation", "critic"],
            },
            "debug": {
                "orchestration": orchestration,
                "market_data": market_data,
                "recommendation_data": recommendation_output,
                "critic_data": critic_output,
            },
        }