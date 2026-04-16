"""
FastAPI application for Finance StratAIgist.

Exposes a unified /api/chat endpoint that runs the real multi-agent
pipeline:
Orchestrator -> Market Agent -> Recommendation Agent -> Critic Agent
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .models import (
    ChatRequest,
    ChatResponse,
    AgentStep,
    HealthResponse,
)

from backend.models.general_model import load_general_model
from backend.models.fin_model import load_fin_model
from backend.rag.engine import RAGEngine
from backend.agents.investment_multiagent_system import InvestmentMultiAgentSystem

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


@app.on_event("startup")
async def startup_event():
    global GENERAL_MODEL, GENERAL_TOKENIZER
    global FIN_MODEL, FIN_TOKENIZER
    global RAG_ENGINE, MULTIAGENT_SYSTEM

    print("Inicializando Finance StratAIgist API...")

    GENERAL_MODEL, GENERAL_TOKENIZER = load_general_model()
    FIN_MODEL, FIN_TOKENIZER = load_fin_model()
    RAG_ENGINE = RAGEngine()

    MULTIAGENT_SYSTEM = InvestmentMultiAgentSystem(
        general_model=GENERAL_MODEL,
        general_tokenizer=GENERAL_TOKENIZER,
        fin_model=FIN_MODEL,
        fin_tokenizer=FIN_TOKENIZER,
        rag_engine=RAG_ENGINE,
    )

    print("Sistema multiagente cargado correctamente.")


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse()


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    if MULTIAGENT_SYSTEM is None:
        return ChatResponse(
            response="Error: el sistema multiagente no está inicializado.",
            agent_trace=[],
            metadata={
                "session_id": request.session_id,
                "pipeline": "multiagent",
                "status": "error",
                "reason": "system_not_initialized",
            },
        )

    user_profile = request.user_profile.model_dump() if request.user_profile else None

    result = MULTIAGENT_SYSTEM.run(
        query=request.prompt,
        user_profile=user_profile,
        company_name=request.company_name,
        ticker=request.ticker,
        mode=request.mode,
    )

    trace = [
        AgentStep(
            agent=step.get("agent", ""),
            action=step.get("action", ""),
            result=step.get("result", ""),
        )
        for step in result.get("agent_trace", [])
    ]

    metadata = result.get("metadata", {})
    metadata["session_id"] = request.session_id

    return ChatResponse(
        response=result.get("response", "No se pudo generar respuesta."),
        agent_trace=trace,
        metadata=metadata,
    )


if __name__ == "__main__":
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8045, reload=True)