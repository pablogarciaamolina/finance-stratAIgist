"""
FastAPI application for Finance StratAIgist.

Exposes a unified /api/chat endpoint that orchestrates the multi-agent
pipeline.  During development the endpoint uses a mock pipeline that
simulates agent behaviour with realistic delays.
"""

import asyncio
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .models import (
    ChatRequest,
    ChatResponse,
    AgentStep,
    HealthResponse,
)

# ── App instance ────────────────────────────────────────────────
app = FastAPI(
    title="Finance StratAIgist API",
    description="Multi-agent financial advisory system",
    version="0.1.0",
)

# ── CORS (allow frontend dev server) ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse()


# ── Chat endpoint (mock pipeline) ──────────────────────────────
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Receives a user prompt + profile and runs the multi-agent pipeline.

    Current implementation is a **mock** that simulates the Orchestrator →
    Market Agent → Recommendation Agent → Critic Agent flow.
    """
    # Simulate processing delay (2-4 s feels realistic)
    await asyncio.sleep(3)

    # ── Mock agent trace ────────────────────────────────────────
    trace = [
        AgentStep(
            agent="Orchestrator",
            action="Analizando consulta y determinando agentes necesarios",
            result="Se requiere: Market Agent → Recommendation Agent → Critic Agent",
        ),
        AgentStep(
            agent="Market Agent",
            action=f"Recopilando datos de mercado para: «{request.prompt[:80]}»",
            result="Datos de mercado recopilados: precios, fundamentales y noticias recientes.",
        ),
        AgentStep(
            agent="Recommendation Agent",
            action="Generando análisis y recomendación personalizada",
            result=(
                f"Análisis realizado considerando perfil "
                f"{request.user_profile.risk_level.value} "
                f"con horizonte {request.user_profile.investment_horizon.value}."
            ),
        ),
        AgentStep(
            agent="Critic Agent",
            action="Validando coherencia y detectando riesgos no considerados",
            result="Revisión completada. Sin incoherencias detectadas.",
        ),
    ]

    # ── Mock response ───────────────────────────────────────────
    risk_label = {
        "conservative": "conservador",
        "moderate": "moderado",
        "aggressive": "agresivo",
    }
    horizon_label = {
        "short": "corto plazo",
        "medium": "medio plazo",
        "long": "largo plazo",
    }

    r = risk_label.get(request.user_profile.risk_level.value, "moderado")
    h = horizon_label.get(request.user_profile.investment_horizon.value, "medio plazo")
    capital = f"{request.user_profile.capital_amount:,.0f} €"

    response_text = (
        f"Basándome en tu perfil de inversor **{r}** con un horizonte de **{h}** "
        f"y un capital de **{capital}**, he analizado tu consulta:\n\n"
        f"> *{request.prompt}*\n\n"
        f"**Análisis del mercado:** Los indicadores actuales muestran una "
        f"tendencia moderadamente alcista en los principales índices. "
        f"La volatilidad se mantiene dentro de rangos históricos normales.\n\n"
        f"**Recomendación:** Dado tu perfil, sugiero una estrategia de "
        f"diversificación equilibrada. Es importante mantener una exposición "
        f"controlada al riesgo y revisar la cartera periódicamente.\n\n"
        f"**Nota del analista:** Esta es una respuesta simulada. "
        f"El sistema multi-agente real se conectará próximamente con datos "
        f"de mercado en tiempo real y modelos de lenguaje especializados."
    )

    return ChatResponse(
        response=response_text,
        agent_trace=trace,
        metadata={
            "session_id": request.session_id,
            "pipeline": "mock",
            "agents_used": ["orchestrator", "market", "recommendation", "critic"],
        },
    )


# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8045, reload=True)
