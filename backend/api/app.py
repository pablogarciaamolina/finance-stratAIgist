"""
FastAPI application for Finance StratAIgist.

Exposes a unified /api/chat endpoint that runs the real multi-agent
pipeline: Orchestrator -> Market Agent -> Recommendation Agent -> Critic Agent
"""

from __future__ import annotations

import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import ChatRequest, ChatResponse, AgentStep, HealthResponse
from backend.models.general_model import load_general_model
from backend.models.fin_model import load_fin_model
from backend.rag.engine import RAGEngine
from backend.agents.investment_multiagent_system import InvestmentMultiAgentSystem
from backend.metrics.efficiency import compute_efficiency

app = FastAPI(
    title="Finance StratAIgist API",
    description="Multi-agent financial advisory system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERAL_MODEL = None
GENERAL_TOKENIZER = None
FIN_MODEL = None
FIN_TOKENIZER = None
RAG_ENGINE = None
MULTIAGENT_SYSTEM = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@app.on_event("startup")
async def startup_event():
    global GENERAL_MODEL, GENERAL_TOKENIZER
    global FIN_MODEL, FIN_TOKENIZER
    global RAG_ENGINE, MULTIAGENT_SYSTEM

    print("Inicializando Finance StratAIgist API...", flush=True)

    GENERAL_MODEL, GENERAL_TOKENIZER = load_general_model()
    FIN_MODEL, FIN_TOKENIZER = load_fin_model()
    RAG_ENGINE = RAGEngine()

    MULTIAGENT_SYSTEM = InvestmentMultiAgentSystem(
        general_model=GENERAL_MODEL,
        general_tokenizer=GENERAL_TOKENIZER,
        fin_model=FIN_MODEL,
        fin_tokenizer=FIN_TOKENIZER,
        rag_engine=RAG_ENGINE,
        debug=True,
    )

    print("Sistema multiagente cargado correctamente.", flush=True)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse()


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    request_start = time.perf_counter()
    session_id = request.session_id

    if MULTIAGENT_SYSTEM is None:
        return ChatResponse(
            response="Error: el sistema multiagente no está inicializado.",
            agent_trace=[],
            metadata={
                "session_id": session_id,
                "pipeline": "multiagent",
                "status": "error",
                "reason": "system_not_initialized",
                "request_mode": request.mode,
                "request_company_name": request.company_name,
                "request_ticker": request.ticker,
            },
        )

    user_profile = request.user_profile.model_dump() if request.user_profile else None
    metrics_collector: list[dict] = []

    # Compatibilidad con el pipeline actual:
    # - models.py expone company_name, ticker y mode
    # - el sistema multiagente actual trabaja con query + user_profile + metrics_collector
    # Guardamos esos campos en metadata y, si vienen company_name/ticker, los incorporamos
    # suavemente al prompt para ayudar al orchestrator sin cambiar la firma del pipeline.
    effective_prompt = request.prompt
    if request.company_name or request.ticker:
        hint_parts = []
        if request.company_name:
            hint_parts.append(f"empresa: {request.company_name}")
        if request.ticker:
            hint_parts.append(f"ticker: {request.ticker}")
        effective_prompt = f"{request.prompt}\n\nContexto proporcionado por frontend: {', '.join(hint_parts)}"

    try:
        result, metrics_collector = MULTIAGENT_SYSTEM.run(
            query=effective_prompt,
            user_profile=user_profile,
            metrics_collector=metrics_collector,
        )
    except Exception as exc:
        total_time = time.perf_counter() - request_start
        return ChatResponse(
            response="Error interno ejecutando el pipeline multiagente.",
            agent_trace=[],
            metadata={
                "session_id": session_id,
                "pipeline": "multiagent",
                "status": "error",
                "reason": "pipeline_exception",
                "error": str(exc),
                "request_mode": request.mode,
                "request_company_name": request.company_name,
                "request_ticker": request.ticker,
                "api_total_latency": total_time,
                "metrics_collector": metrics_collector,
                "efficiency_metrics": compute_efficiency(metrics_collector, total_time),
            },
        )

    total_time = time.perf_counter() - request_start

    trace = [
        AgentStep(
            agent=step.get("agent", ""),
            action=step.get("action", ""),
            result=step.get("result", ""),
        )
        for step in result.get("agent_trace", [])
    ]

    efficiency_metrics = compute_efficiency(metrics_collector, total_time)

    metadata = dict(result.get("metadata", {}))
    metadata["session_id"] = session_id
    metadata["request_mode"] = request.mode
    metadata["request_company_name"] = request.company_name
    metadata["request_ticker"] = request.ticker
    metadata["effective_prompt"] = effective_prompt
    metadata["efficiency_metrics"] = efficiency_metrics
    metadata["metrics_collector"] = metrics_collector
    metadata["api_total_latency"] = total_time
    metadata["metrics_summary"] = {
        "num_metric_events": len(metrics_collector),
        "agents_seen": [m.get("agent") for m in metrics_collector],
        "sum_agent_latency": round(
            sum(_safe_float(m.get("latency", 0.0), 0.0) for m in metrics_collector),
            6,
        ),
    }

    return ChatResponse(
        response=result.get("response", "No se pudo generar respuesta."),
        agent_trace=trace,
        metadata=metadata,
    )


if __name__ == "__main__":
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8045)
