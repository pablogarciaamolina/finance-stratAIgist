# Finance StratAIgist

Un sistema **multiagente de recomendación de inversiones** basado en LLMs y LangChain. Combina agentes especializados para analizar datos de mercado, generar tesis de inversión personalizadas y validarlas de forma estructurada y transparente.

---

## Arquitectura

El sistema sigue un flujo multiagente con **dos modos de operación**:

```

Usuario → Orchestrator → Market Agent → (Benchmark Agent | Recommendation Agent → Critic Agent) → Respuesta

```

### Modos de ejecución

| Modo | Descripción |
|------|------------|
| **advisor** | Genera una recomendación de inversión personalizada |
| **benchmark** | Responde preguntas financieras factuales (FinanceBench) |

---

## Agentes

| Agente                              | Responsabilidad |
|-------------------------------------|----------------|
| **Orchestrator**                    | Interpreta la query, detecta empresa/ticker y decide el flujo (advisor vs benchmark). |
| **Market Agent**                    | Obtiene datos objetivos: precio, fundamentales, eventos, contexto web y RAG. |
| **Benchmark Answer Agent**          | Genera respuestas factuales sin recomendación (para evaluación tipo FinanceBench). |
| **Recommendation Agent (Fin-R1)**   | Construye tesis de inversión estructurada (fortalezas, riesgos, escenarios). |
| **Critic Agent**                    | Valida la recomendación, detecta inconsistencias y ajusta la respuesta final. |

---

## Herramientas disponibles

Los agentes utilizan herramientas externas reales:

* **Calculator** — Evaluación matemática (`numexpr`)
* **Internet Search** — Búsqueda web (Tavily API)
* **Stock Price** — Precio actual (Alpha Vantage)
* **Company Fundamentals** — Datos financieros (SEC EDGAR)
* **Company Events** — Eventos recientes (SEC EDGAR 8-K)
* **RAG** — Contexto económico desde ChromaDB

---

## Modelos utilizados

| Tipo                    | Modelo                  |
|-------------------------|-------------------------|
| **General reasoning**   | Qwen / Qwen2.5 + LoRA   |
| **Financial reasoning** | Fin-R1 (SUFE-AIFLM-Lab) |
| **Embeddings**          | all-MiniLM-L6-v2        |

---

## Estructura del Proyecto

```

finance-stratAIgist/
├── README.md
├── .env.example
│
├── backend/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   │
│   ├── api/
│   │   ├── app.py
│   │   └── models.py
│   │
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── market_agent.py
│   │   ├── benchmark_answer_agent.py
│   │   ├── recommendation.py
│   │   ├── critic.py
│   │   └── investment_multiagent_system.py
│   │
│   ├── models/
│   │   ├── general_model.py
│   │   ├── fin_model.py
│   │   ├── inference.py
│   │   └── config.py
│   │
│   ├── tools/
│   │   ├── calculator.py
│   │   ├── search.py
│   │   └── finance.py
│   │
│   └── rag/
│       ├── engine.py
│       └── loader.py
│
└── frontend/

````

---

## Inicio Rápido

### 1. Clonar repo

```bash
git clone https://github.com/pablogarciaamolina/finance-stratAIgist.git
cd finance-stratAIgist
cp .env.example .env
````

---

## Backend (modo local)

```bash
cd backend

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

python -m uvicorn api.app:app --host 0.0.0.0 --port 8045 --reload
```

API en:

```
http://localhost:8045
http://localhost:8045/docs
```

---

## Docker (recomendado)

```bash
cd backend
docker compose up --build
```

✔ Soporte GPU NVIDIA (CUDA 12.1)

---

## API

### POST `/api/chat`

Modo automático (advisor o benchmark según query):

```json
{
  "prompt": "¿Cuál fue el revenue de NVIDIA en 2023?",
  "user_profile": null
}
```

Ejemplo advisor:

```json
{
  "prompt": "Analiza Nvidia para un inversor moderado a 12 meses",
  "user_profile": {
    "risk_level": "moderate",
    "investment_horizon": "medium",
    "capital_amount": 10000,
    "investment_goals": ["growth"]
  }
}
```

---

## Response

```json
{
  "response": "Respuesta final...",
  "agent_trace": [
    {"agent": "Orchestrator", "action": "...", "result": "..."},
    {"agent": "Market Agent", "action": "...", "result": "..."},
    {"agent": "Recommendation Agent", "action": "...", "result": "..."},
    {"agent": "Critic Agent", "action": "...", "result": "..."}
  ],
  "metadata": {
    "pipeline": "multiagent",
    "status": "completed",
    "mode": "advisor | benchmark"
  }
}
```

---

## RAG (base de conocimiento)

```bash
cd backend
python -m rag.loader
```

Carga artículos de economía (WikiCAT_esv2) en ChromaDB.

---

## Variables de entorno

| Variable               | Descripción                  |
| ---------------------- | ---------------------------- |
| `TAVILY_API_KEY`       | Búsqueda web                 |
| `ALPHAVANTAGE_API_KEY` | Precio acciones              |
| `HF_TOKEN`             | Acceso a modelos HuggingFace |

---

## Evaluación

El sistema está diseñado para evaluarse en:

* **FinanceBench (PatronusAI)** → modo benchmark
* Evaluación cualitativa → modo advisor

Métricas relevantes:

* Exactitud factual
* Grounding (uso de evidencia)
* Calidad de la tesis (advisor)
* Cobertura de riesgos
* Latencia / tokens

---

## Estado del Proyecto

* [x] Sistema multiagente completo
* [x] Modo advisor + benchmark
* [x] Integración con LangChain tools
* [x] Integración con Fin-R1
* [x] RAG funcional
* [x] API real (no mock)
* [ ] Optimización de latencia
* [ ] Memoria conversacional
* [ ] Evaluación cuantitativa automática

---

## Tech Stack

| Componente | Tecnología                       |
| ---------- | -------------------------------- |
| Backend    | FastAPI + Uvicorn                |
| Agentes    | LangChain                        |
| Modelos    | HuggingFace (Qwen + Fin-R1)      |
| RAG        | ChromaDB                         |
| Tools      | Tavily, SEC EDGAR, Alpha Vantage |
| Infra      | Docker + CUDA                    |

---

## Nota importante

Este sistema **NO es asesor financiero real**.

Está diseñado con fines educativos, experimentales y de investigación en sistemas multiagente.


