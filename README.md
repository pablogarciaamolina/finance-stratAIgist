# Finance StratAIgist

Sistema multiagente para preguntas de asesoramiento financiero e inversion sobre empresas cotizadas.

El proyecto combina:
- una API FastAPI local
- agentes especializados para analisis de mercado y recomendacion
- herramientas reales de mercado y busqueda
- un pipeline de evaluacion para medir calidad y latencia

## Estado actual

Hoy el flujo mas estable del proyecto es el modo `advisor` sobre la API local:

`Usuario -> Orchestrator -> Market Agent -> Recommendation Agent -> Critic Agent -> Respuesta`

Puntos importantes del estado actual:
- La API local corre en `http://127.0.0.1:8045`
- El endpoint principal es `POST /api/chat`
- La respuesta final ya no muestra `<think>`
- El razonamiento interno se conserva en `metadata.internal_reasoning`
- El `Market Agent` usa precio, fundamentales, eventos SEC, web search y RAG
- Alpha Vantage rota varias API keys si se configuran en `.env`
- Hay scripts de evaluacion y comparacion de resultados en `backend/evaluation`

Nota:
- El repositorio incluye piezas para `benchmark` y preguntas factuales, pero la API local actual esta centrada sobre todo en el pipeline `advisor`

## Arquitectura

### Pipeline principal

1. `Orchestrator`
   Interpreta la consulta, detecta empresa y ticker, y normaliza contexto.
2. `Market Agent`
   Recupera evidencia estructurada y contexto externo.
3. `Recommendation Agent`
   Genera una recomendacion estructurada usando el modelo financiero.
4. `Critic Agent`
   Revisa la recomendacion, detecta debilidades y construye la respuesta final.

### Agentes

| Agente | Rol |
| --- | --- |
| `OrchestratorAgent` | Parseo de la consulta, ticker, perfil y plan |
| `MarketAgent` | Precio, fundamentales, filings SEC, web search, RAG |
| `RecommendationAgent` | Tesis y respuesta principal orientada a inversion |
| `CriticAgent` | Revision final, grounding y tono prudente |
| `BenchmarkAnswerAgent` | Modulo experimental/factual presente en el repo |

## Modelos

### Modelo general

Se usa para `orchestrator` y `critic`.

Comportamiento actual:
- intenta cargar el modelo general local
- si no cabe o falla, puede caer a Ollama
- si aun asi falla, el backend arranca en modo heuristico para esas partes

Por defecto:
- fallback Ollama: `llama3.2`

### Modelo financiero

Se usa para `RecommendationAgent`.

Por defecto en la API:
- backend: `ollama`
- modelo: `mychen76/Fin-R1:Q5`

### Embeddings / RAG

- `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB local
- dataset de base: `PlanTL-GOB-ES/WikiCAT_esv2`

## Herramientas de mercado

Las tools actuales viven en `backend/tools/`.

### Disponibles

- `calculator`
- `internet_search` con Tavily
- `stock_price` con Alpha Vantage
- `company_fundamentals` con SEC EDGAR companyfacts
- `company_events` con SEC submissions
- `company_financial_history` con SEC companyfacts por ejercicio

### Notas operativas

- `stock_price` intenta `GLOBAL_QUOTE` y cae a `TIME_SERIES_DAILY`
- si defines `ALPHAVANTAGE_API_KEY`, `ALPHAVANTAGE_API_KEY2` y `ALPHAVANTAGE_API_KEY3`, la tool rota keys automaticamente
- si una key entra en rate limit, se marca como agotada ese dia y deja de usarse durante la sesion

## Estructura del proyecto

```text
finance-stratAIgist/
|-- README.md
|-- .env.example
|-- backend/
|   |-- api/
|   |   |-- app.py
|   |   `-- models.py
|   |-- agents/
|   |   |-- orchestrator.py
|   |   |-- market_agent.py
|   |   |-- recommendation.py
|   |   |-- critic.py
|   |   |-- benchmark_answer_agent.py
|   |   |-- investment_multiagent_system.py
|   |   `-- output_utils.py
|   |-- tools/
|   |   |-- calculator.py
|   |   |-- search.py
|   |   `-- finance.py
|   |-- models/
|   |   |-- general_model.py
|   |   |-- fin_model.py
|   |   `-- inference.py
|   |-- rag/
|   |   |-- loader.py
|   |   `-- engine.py
|   |-- evaluation/
|   |   |-- eval.py
|   |   `-- summarize_eval.py
|   `-- requirements.txt
`-- frontend/
    |-- package.json
    `-- src/
```

## Setup rapido

### 1. Crear entorno virtual

Desde la raiz del repo, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

### 2. Configurar variables de entorno

Puedes partir de `.env.example`, pero ahora mismo conviene ampliar el fichero con mas claves:

```env
TAVILY_API_KEY=...
ALPHAVANTAGE_API_KEY=...
ALPHAVANTAGE_API_KEY2=...
ALPHAVANTAGE_API_KEY3=...
SEC_CONTACT_EMAIL=tu_email@dominio.com
```

Variables utiles:

| Variable | Uso |
| --- | --- |
| `TAVILY_API_KEY` | Busqueda web |
| `ALPHAVANTAGE_API_KEY` | Precio de mercado |
| `ALPHAVANTAGE_API_KEY2` | Rotacion de precio de mercado |
| `ALPHAVANTAGE_API_KEY3` | Rotacion de precio de mercado |
| `SEC_CONTACT_EMAIL` | User-Agent para SEC EDGAR |
| `GENERAL_MODEL_BACKEND` | `auto`, `ollama` o `huggingface` |
| `GENERAL_OLLAMA_MODEL` | Modelo general alternativo para Ollama |

### 3. Preparar Ollama

La API actual espera Ollama para el modelo financiero y normalmente tambien para el general en fallback.

```powershell
ollama pull llama3.2
ollama pull mychen76/Fin-R1:Q5
```

### 4. Cargar el RAG

Esto descarga el dataset de Hugging Face y lo persiste en ChromaDB local.

```powershell
python -m backend.rag.loader
```

### 5. Arrancar el backend

```powershell
python -m backend.api.app
```

Endpoints:

- `http://127.0.0.1:8045/api/health`
- `http://127.0.0.1:8045/api/chat`
- `http://127.0.0.1:8045/docs`

## Frontend

El frontend existe como app Vite simple en `frontend/`.

Para arrancarlo:

```powershell
cd frontend
npm install
npm run dev
```

## Uso de la API

### Health check

```http
GET /api/health
```

### Chat

```http
POST /api/chat
Content-Type: application/json
```

Ejemplo:

```json
{
  "prompt": "Que peso maximo tendria sentido dar a NVIDIA para aprovechar crecimiento sin disparar demasiado la volatilidad?",
  "company_name": "NVIDIA",
  "ticker": "NVDA",
  "session_id": "demo-001",
  "mode": "advisor",
  "user_profile": {
    "risk_level": "moderate",
    "investment_horizon": "long",
    "capital_amount": 10000,
    "investment_goals": ["growth", "preservation"]
  }
}
```

Respuesta simplificada:

```json
{
  "response": "Respuesta final visible al usuario",
  "agent_trace": [
    {"agent": "Orchestrator", "action": "...", "result": "..."},
    {"agent": "Market Agent", "action": "...", "result": "..."},
    {"agent": "Recommendation Agent", "action": "...", "result": "..."},
    {"agent": "Critic Agent", "action": "...", "result": "..."}
  ],
  "metadata": {
    "pipeline": "multiagent",
    "status": "completed_with_warning",
    "company_name": "NVIDIA",
    "ticker": "NVDA",
    "question_type": "allocation",
    "market_has_structured_evidence": true,
    "critic_grounded": true,
    "internal_reasoning": {
      "recommendation": "...",
      "critic": "..."
    }
  }
}
```

## Evaluacion

El proyecto ya incluye una bateria local de preguntas advisor y un script de resumen.

### Lanzar una evaluacion

Con el backend levantado en otra terminal:

```powershell
python -m backend.evaluation.eval --output backend/evaluation/mi_eval.json
```

Opciones utiles:

```powershell
python -m backend.evaluation.eval `
  --output backend/evaluation/mi_eval.json `
  --timeout 300 `
  --retries 2 `
  --retry-backoff 10 `
  --delay 1.5
```

Caracteristicas del evaluador actual:
- timeout configurable
- reintentos automaticos ante timeout o 5xx
- guardado incremental tras cada pregunta
- reanudacion automatica si el JSON de salida ya existe

### Resumir y comparar evaluaciones

```powershell
python -m backend.evaluation.summarize_eval backend/evaluation/mi_eval.json
python -m backend.evaluation.summarize_eval backend/evaluation/mi_eval_old.json --compare backend/evaluation/mi_eval.json
```

El resumen informa, entre otras cosas, de:
- latencia media, mediana y p90
- grounded / think leaks / respuestas cortas
- distribucion de recomendaciones
- desglose por empresa
- desglose por tipo de pregunta

## Limitaciones conocidas

- La latencia puede ser alta y muy variable porque `RecommendationAgent` y `CriticAgent` usan Ollama local
- Alpha Vantage tiene limites diarios y de frecuencia incluso con varias keys
- Algunas respuestas siguen siendo demasiado verbosas o poco precisas para preguntas de asignacion concreta

## Estado del proyecto

- [x] API local funcional en `8045`
- [x] Pipeline multiagente advisor operativo
- [x] RAG local con ChromaDB
- [x] Integracion con Tavily, SEC y Alpha Vantage
- [x] Rotacion de varias API keys de Alpha Vantage
- [x] Evaluador local con reintentos y reanudacion
- [x] Razonamiento interno oculto en la respuesta final y conservado en metadata
- [ ] Reducir latencia de generacion en Ollama
- [ ] Mejorar precision en respuestas de asignacion y sizing
- [ ] Exponer un flujo benchmark plenamente integrado en la API principal

## Seguridad y uso responsable

Este proyecto no sustituye asesoramiento financiero profesional.

Las respuestas son experimentales y deben tratarse como apoyo analitico. Antes de tomar decisiones reales de inversion, conviene validar precio, fundamentales, noticias y contexto macro con fuentes oficiales.
