"""
Investment Multi-Agent System.
 
Orchestrates the full pipeline:
1. Orchestrator
2. Market Agent
3. Recommendation Agent
4. Critic Agent
 
Returns a final response plus agent trace and internal reports.
"""
 
from __future__ import annotations
 
import time
from typing import Any, Dict, List, Optional
 
from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.output_utils import sanitize_visible_answer
from backend.agents.market_agent import MarketAgent
from backend.agents.recommendation import RecommendationAgent
from backend.agents.critic import CriticAgent
from backend.rag.engine import RAGEngine
 
 
class InvestmentMultiAgentSystem:
    """
    Sistema multiagente completo para recomendación de inversiones.
    """
 
    def __init__(
        self,
        general_model: Any,
        general_tokenizer: Any,
        fin_model: Any,
        fin_tokenizer: Any,
        rag_engine: Optional[RAGEngine] = None,
        market_tools: Optional[list] = None,
        debug: bool = True,
    ):
        self.debug = debug
        self.orchestrator = OrchestratorAgent(
            general_model,
            general_tokenizer,
            debug=debug,
        )
        self.market_agent = MarketAgent(
            tools=market_tools,
            rag_engine=rag_engine,
            debug=debug,
        )
        self.recommendation_agent = RecommendationAgent(
            fin_model,
            fin_tokenizer,
            debug=debug,
        )
        self.critic_agent = CriticAgent(
            general_model,
            general_tokenizer,
            debug=debug,
        )
 
    def _log(self, message: str):
        if self.debug:
            print(f"[InvestmentMultiAgentSystem] {message}", flush=True)
 
    def _append_trace(self, trace: list, agent_name: str, payload: Dict[str, Any]):
        trace.append(
            {
                "agent": agent_name,
                "action": payload.get("action", ""),
                "result": payload.get("result", ""),
            }
        )
 
    def _record_metric(
        self,
        metrics_collector: Optional[List[Dict[str, Any]]],
        *,
        agent: str,
        latency: float,
        tokens: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        if metrics_collector is None:
            return
 
        metric = {
            "agent": agent,
            "latency": latency,
            "tokens": tokens or {},
        }
 
        if extra:
            metric.update(extra)
 
        metrics_collector.append(metric)
 
    def _build_failure_response(
        self,
        message: str,
        trace: list,
        orchestration: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        recommendation_data: Optional[Dict[str, Any]] = None,
        critic_data: Optional[Dict[str, Any]] = None,
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
            },
        }
 
    def _recommendation_is_too_weak(
        self,
        recommendation_output: Dict[str, Any],
    ) -> bool:
        data = recommendation_output.get("data", {})
        thesis = data.get("thesis", "") or recommendation_output.get("response", "")
        answer = data.get("answer", "") or recommendation_output.get("response", "")
        strengths = data.get("strengths", [])
        risks = data.get("risks", [])
        watch_items = data.get("watch_items", [])
        implementation_steps = data.get("implementation_steps", [])
        allocation_guidance = data.get("allocation_guidance", [])
        answered_directly = bool(data.get("answered_directly", False))
 
        if not answer or len(answer.strip()) < 20:
            return True
 
        if not isinstance(strengths, list):
            strengths = []
        if not isinstance(risks, list):
            risks = []
        if not isinstance(watch_items, list):
            watch_items = []
        if not isinstance(implementation_steps, list):
            implementation_steps = []
        if not isinstance(allocation_guidance, list):
            allocation_guidance = []

        if answered_directly and len(answer.strip()) >= 40:
            return False

        supporting_items = (
            len(strengths)
            + len(risks)
            + len(watch_items)
            + len(implementation_steps)
            + len(allocation_guidance)
        )
        return len(thesis.strip()) < 20 and supporting_items == 0
 
    def run(
        self,
        query: str,
        user_profile: Optional[dict] = None,
        metrics_collector: Optional[list] = None,
        context_company_name: Optional[str] = None,
        context_ticker: Optional[str] = None,
    ) -> tuple[Dict[str, Any], list]:
        trace = []
        metrics_collector = metrics_collector if metrics_collector is not None else []
        pipeline_start = time.perf_counter()
        agent_timings: Dict[str, float] = {}
        pipeline_warnings: List[str] = []
 
        self._log(f"Inicio pipeline | query={query!r}")
 
        # 1. Orchestrator
        self._log("-> Ejecutando Orchestrator")
        start_orchestrator = time.perf_counter()
        orchestration, token_info_orchestrator = self.orchestrator.run(
            query=query,
            user_profile=user_profile,
        )
        orchestration = orchestration or {}
        time_orchestrator = time.perf_counter() - start_orchestrator
        agent_timings["orchestrator"] = time_orchestrator
 
        self._record_metric(
            metrics_collector,
            agent="orchestrator",
            latency=time_orchestrator,
            tokens=token_info_orchestrator or {},
            extra={"stage": "pipeline_agent"},
        )
 
        self._append_trace(trace, "Orchestrator", orchestration)
        self._log(
            f"<- Orchestrator completado en {time_orchestrator:.3f}s | "
            f"company={orchestration.get('company_name')} | ticker={orchestration.get('ticker')}"
        )
 
        company_name = orchestration.get("company_name") or context_company_name
        ticker = orchestration.get("ticker") or context_ticker
 
        if not company_name and not ticker:
            self._log("Fallo: el Orchestrator no identificó empresa ni ticker")
            total_latency = time.perf_counter() - pipeline_start
            result = self._build_failure_response(
                message=(
                    "No he podido identificar con suficiente claridad la empresa o el ticker "
                    "sobre el que quieres análisis. Indica el nombre de la empresa o su ticker bursátil."
                ),
                trace=trace,
                orchestration=orchestration,
                metadata={
                    "failure_stage": "orchestrator",
                    "timings": agent_timings,
                    "total_latency": total_latency,
                },
            )
            return result, metrics_collector
 
        # 2. Market Agent
        self._log("-> Ejecutando Market Agent")
        start_market = time.perf_counter()
        market_data, token_info_market = self.market_agent.run(
            query=query,
            company_name=company_name,
            ticker=ticker,
        )
        market_data = market_data or {}
        time_market = time.perf_counter() - start_market
        agent_timings["market"] = time_market
 
        self._record_metric(
            metrics_collector,
            agent="market",
            latency=time_market,
            tokens=token_info_market or {},
            extra={"stage": "pipeline_agent"},
        )
 
        self._append_trace(trace, "Market Agent", market_data)
        market_report = market_data.get("data", {}) or {}
 
        self._log(
            f"<- Market Agent completado en {time_market:.3f}s | "
            f"evidence={market_report.get('has_minimum_evidence', False)} | "
            f"level={market_report.get('evidence_level')}"
        )
 
        # Solo bloqueamos si realmente no se identificó empresa/ticker o hay un error duro.
        if market_report.get("blocking_error", False):
            self._log("Fallo bloqueante en Market Agent")
            total_latency = time.perf_counter() - pipeline_start
            result = self._build_failure_response(
                message=(
                    "No he podido preparar el contexto de mercado porque faltan datos esenciales "
                    "sobre la empresa o el ticker solicitado."
                ),
                trace=trace,
                orchestration=orchestration,
                market_data=market_data,
                metadata={
                    "failure_stage": "market_agent",
                    "timings": agent_timings,
                    "total_latency": total_latency,
                },
            )
            return result, metrics_collector
 
        if not market_report.get("has_minimum_evidence", False):
            self._log("Market Agent con evidencia limitada: se continúa igualmente")
            pipeline_warnings.append("limited_market_evidence")
        if not market_report.get("has_structured_evidence", False):
            self._log("Market Agent sin evidencia estructurada suficiente")
            pipeline_warnings.append("missing_structured_market_data")
 
        # 3. Recommendation Agent
        self._log("-> Ejecutando Recommendation Agent")
        start_recommendation = time.perf_counter()
        recommendation_output, token_info_recommendation = self.recommendation_agent.run(
            query=query,
            market_data=market_data,
            user_profile=user_profile,
        )
        recommendation_output = recommendation_output or {}
        time_recommendation = time.perf_counter() - start_recommendation
        agent_timings["recommendation"] = time_recommendation
 
        self._record_metric(
            metrics_collector,
            agent="recommendation",
            latency=time_recommendation,
            tokens=token_info_recommendation or {},
            extra={"stage": "pipeline_agent"},
        )
 
        self._append_trace(trace, "Recommendation Agent", recommendation_output)
        self._log(
            f"<- Recommendation Agent completado en {time_recommendation:.3f}s"
        )
 
        # Ya no cortamos aquí automáticamente.
        if self._recommendation_is_too_weak(recommendation_output):
            self._log("Recommendation Agent produjo una tesis débil, pero se continúa")
            pipeline_warnings.append("weak_recommendation")
        if not recommendation_output.get("data", {}).get("answered_directly", False):
            pipeline_warnings.append("recommendation_not_direct_enough")
 
        # 4. Critic Agent
        self._log("-> Ejecutando Critic Agent")
        start_critic = time.perf_counter()
        critic_output, token_info_critic = self.critic_agent.run(
            query=query,
            recommendation=recommendation_output,
            market_data=market_data,
            user_profile=user_profile,
        )
        critic_output = critic_output or {}
        time_critic = time.perf_counter() - start_critic
        agent_timings["critic"] = time_critic
 
        self._record_metric(
            metrics_collector,
            agent="critic",
            latency=time_critic,
            tokens=token_info_critic or {},
            extra={"stage": "pipeline_agent"},
        )
 
        self._append_trace(trace, "Critic Agent", critic_output)
        self._log(f"<- Critic Agent completado en {time_critic:.3f}s")
 
        critic_data = critic_output.get("data", {}) or {}
        enough_evidence = critic_data.get("enough_evidence", False)
        grounded_in_facts = critic_data.get("grounded_in_facts", False)
        question_type = recommendation_output.get("data", {}).get("question_type", "general")
        total_latency = time.perf_counter() - pipeline_start
 
        if not enough_evidence:
            self._log("Pipeline completado con warning: critic indica evidencia insuficiente")
            pipeline_warnings.append("critic_flagged_insufficient_evidence")
        if not grounded_in_facts:
            pipeline_warnings.append("critic_not_grounded_in_facts")
 
        final_answer = sanitize_visible_answer(
            critic_output.get("revised_response")
            or critic_data.get("final_answer")
            or recommendation_output.get("response")
            or recommendation_output.get("data", {}).get("answer")
            or recommendation_output.get("data", {}).get("thesis")
            or (
                "No se pudo generar una recomendacion completa, pero conviene diversificar, "
                "ajustar el riesgo al perfil del usuario y evitar sobreconcentracion en un unico activo."
            )
        )
        if len(final_answer) < 20:
            final_answer = sanitize_visible_answer(
                recommendation_output.get("data", {}).get("answer")
                or recommendation_output.get("response")
                or "No he podido construir una respuesta final suficientemente clara con la evidencia disponible."
            )
 
        status = "completed_with_warning" if pipeline_warnings else "completed"
 
        self._log(f"Pipeline completado en {total_latency:.3f}s | status={status}")
 
        return {
            "response": final_answer,
            "agent_trace": trace,
            "metadata": {
                "pipeline": "multiagent",
                "status": status,
                "mode": "advisor",
                "warning": pipeline_warnings if pipeline_warnings else None,
                "company_name": market_report.get("company_name") or company_name,
                "ticker": market_report.get("ticker") or ticker,
                "market_has_minimum_evidence": market_report.get("has_minimum_evidence", False),
                "market_has_structured_evidence": market_report.get("has_structured_evidence", False),
                "market_structured_signals": market_report.get("structured_signals", 0),
                "question_type": question_type,
                "recommendation": recommendation_output.get("data", {}).get("preliminary_recommendation"),
                "recommendation_answered_directly": recommendation_output.get("data", {}).get("answered_directly"),
                "critic_grounded": grounded_in_facts,
                "critic_answer_quality": critic_data.get("answer_quality"),
                "internal_reasoning": {
                    "recommendation": recommendation_output.get("data", {}).get("internal_reasoning"),
                    "critic": critic_data.get("internal_reasoning"),
                },
                "agents_used": [
                    "orchestrator",
                    "market",
                    "recommendation",
                    "critic",
                ],
                "timings": agent_timings,
                "total_latency": total_latency,
            },
            "debug": {
                "orchestration": orchestration,
                "market_data": market_data,
                "recommendation_data": recommendation_output,
                "critic_data": critic_output,
            },
        }, metrics_collector
