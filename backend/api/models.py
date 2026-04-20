"""
Pydantic models for the Finance StratAIgist API.

Defines request/response schemas for the chat endpoint,
user profile configuration, and agent trace data.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentHorizon(str, Enum):
    SHORT = "short"      # < 1 year
    MEDIUM = "medium"    # 1-5 years
    LONG = "long"        # 5+ years


class InvestmentGoal(str, Enum):
    GROWTH = "growth"
    INCOME = "income"
    PRESERVATION = "preservation"
    SPECULATION = "speculation"


# ---------------------------------------------------------------------------
# User Profile — sent once after onboarding
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    risk_level: RiskLevel = Field(
        ...,
        description="Nivel de tolerancia al riesgo del usuario",
    )
    investment_horizon: InvestmentHorizon = Field(
        ...,
        description="Horizonte temporal de inversión",
    )
    capital_amount: float = Field(
        ...,
        ge=0,
        description="Capital disponible para invertir (EUR)",
    )
    investment_goals: list[InvestmentGoal] = Field(
        default_factory=list,
        description="Objetivos de inversión seleccionados",
    )


# ---------------------------------------------------------------------------
# Chat Request / Response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    prompt: str = Field(
        ...,
        min_length=1,
        description="Mensaje del usuario",
    )
    user_profile: Optional[UserProfile] = Field(
        default=None,
        description="Perfil de inversión del usuario. Opcional si el sistema puede inferir contexto.",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Identificador de sesión generado en frontend",
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Nombre de la empresa si ya viene dado por benchmark o metadata",
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Ticker bursátil si ya viene dado por benchmark o metadata",
    )
    mode: Literal["advisor", "benchmark"] = Field(
        default="advisor",
        description="Modo de ejecución",
    )


class AgentStep(BaseModel):
    agent: str = Field(
        ...,
        min_length=1,
        description="Nombre del agente que ejecutó este paso",
    )
    action: str = Field(
        ...,
        description="Acción realizada",
    )
    result: str = Field(
        default="",
        description="Resultado del paso",
    )


class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="Respuesta final al usuario",
    )
    agent_trace: list[AgentStep] = Field(
        default_factory=list,
        description="Traza de ejecución de los agentes",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos adicionales de la respuesta",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"