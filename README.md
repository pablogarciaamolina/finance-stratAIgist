# Finance StratAIgist

Un sistema **multiagente** de asesoría financiera inteligente construido con LangChain. Combina múltiples agentes especializados para analizar mercados, generar recomendaciones de inversión personalizadas y validar resultados de forma estructurada y transparente.

---

## Arquitectura

El sistema sigue un flujo estructurado donde cada agente tiene una responsabilidad clara:

```
Usuario → Orchestrator → Market Agent → Recommendation Agent → Critic Agent → Respuesta
```

| Agente | Responsabilidad |
|--------|----------------|
| **Orchestrator** | Recibe la consulta del usuario y coordina el flujo entre los sub-agentes. Decide qué pasos son necesarios. |
| **Market Agent** | Recopila datos objetivos del mercado: precios, fundamentales, eventos recientes y contexto económico (RAG). |
| **Recommendation Agent** | Genera un análisis interpretativo y una "tesis" de inversión, considerando el perfil del usuario y el horizonte temporal. |
| **Critic Agent** | Revisa la recomendación en busca de incoherencias, información faltante o riesgos no considerados. Ajusta la respuesta final. |

### Herramientas disponibles

Los agentes tienen acceso a herramientas externas:

- **Calculator** — Evaluación de expresiones matemáticas (`numexpr`)
- **Internet Search** — Búsqueda web via Tavily API
- **Company Fundamentals** — Datos financieros desde SEC EDGAR
- **Company Events** — Filings 8-K recientes desde SEC EDGAR
- **Stock Price** — Precio actual via Alpha Vantage
- **RAG** — Recuperación de contexto desde base de conocimiento económico (ChromaDB + WikiCAT_es)

---

## Estructura del Proyecto

```
finance-stratAIgist/
├── README.md
├── .env.example              # Variables de entorno requeridas
├── .gitignore
│
├── backend/                   # Servidor Python (FastAPI)
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   │
│   ├── api/                   # API REST
│   │   ├── app.py             # FastAPI — endpoints /api/chat y /api/health
│   │   └── models.py          # Schemas Pydantic (UserProfile, ChatRequest/Response)
│   │
│   ├── agents/                # Sistema multiagente
│   │   ├── orchestrator.py    # Coordinador principal
│   │   ├── market_agent.py    # Agente de datos de mercado
│   │   ├── recommendation.py  # Agente de recomendación
│   │   └── critic.py          # Agente de validación
│   │
│   ├── tools/                 # Herramientas externas
│   │   ├── calculator.py      # Calculadora (numexpr)
│   │   ├── search.py          # Búsqueda web (Tavily)
│   │   └── finance.py         # APIs financieras (SEC EDGAR, Alpha Vantage)
│   │
│   ├── rag/                   # Retrieval-Augmented Generation
│   │   ├── engine.py          # RAGEngine — consulta y recuperación desde ChromaDB
│   │   ├── loader.py          # Carga de WikiCAT_esv2 en ChromaDB
│   │   └── chroma_db/         # Base de datos vectorial (auto-generada)
│   │
│   └── models/                # Modelos de lenguaje
│       ├── config.py          # Configuración LoRA
│       ├── inference.py       # Carga e inferencia (HuggingFace + Ollama)
│       └── training/          # Scripts de entrenamiento
│           └── train_sft.py   # Supervised Fine-Tuning en GSM8K
│
└── frontend/                  # Aplicación web (Vite + Vanilla JS)
    ├── package.json
    ├── vite.config.js         # Dev server con proxy al backend
    ├── index.html
    │
    ├── public/
    │   └── favicon.svg
    │
    └── src/
        ├── main.js            # Entry point
        ├── style.css          # Sistema de diseño completo
        ├── app.js             # Controlador (máquina de estados)
        ├── components/
        │   ├── landing.js     # Pantalla de bienvenida
        │   ├── onboarding.js  # Cuestionario de perfil (4 pasos)
        │   └── chat.js        # Interfaz de chat con traza de agentes
        └── services/
            └── api.js         # Cliente HTTP para el backend
```

---

## Inicio Rápido

### Requisitos previos

- **Node.js** ≥ 18 (para el frontend)
- **Python** ≥ 3.10 (para el backend)
- (Opcional) **Docker** + **NVIDIA GPU** para el modelo completo

### 1. Clonar y configurar

```bash
git clone https://github.com/tu-usuario/finance-stratAIgist.git
cd finance-stratAIgist

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

### 2. Backend

```bash
cd backend

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python -m uvicorn api.app:app --host 0.0.0.0 --port 8045 --reload
```

El backend estará disponible en `http://localhost:8045`. La documentación interactiva (Swagger) está en `/docs`.

### 3. Frontend

```bash
cd frontend

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev
```

La aplicación estará en `http://localhost:5173`. El dev server de Vite hace proxy automático de las peticiones `/api/*` al backend.

---

## Flujo de la Aplicación Web

1. **Landing** — Pantalla de bienvenida con descripción del sistema
2. **Onboarding** — Cuestionario de 4 pasos para configurar el perfil de inversor:
   - Tolerancia al riesgo (conservador / moderado / agresivo)
   - Horizonte de inversión (corto / medio / largo plazo)
   - Capital disponible
   - Objetivos de inversión (crecimiento, ingresos, preservación, especulación)
3. **Chat** — Interfaz de conversación donde el usuario hace preguntas financieras
   - Respuestas generadas por el pipeline multiagente
   - Traza de agentes visible (colapsable) mostrando cada paso del análisis

---

## API

### `POST /api/chat`

Envía una consulta al pipeline multiagente.

**Request:**
```json
{
  "prompt": "¿Cómo está NVIDIA hoy?",
  "user_profile": {
    "risk_level": "moderate",
    "investment_horizon": "medium",
    "capital_amount": 10000,
    "investment_goals": ["growth"]
  },
  "session_id": "session_abc123"
}
```

**Response:**
```json
{
  "response": "Análisis del mercado de NVIDIA...",
  "agent_trace": [
    {"agent": "Orchestrator", "action": "...", "result": "..."},
    {"agent": "Market Agent", "action": "...", "result": "..."},
    {"agent": "Recommendation Agent", "action": "...", "result": "..."},
    {"agent": "Critic Agent", "action": "...", "result": "..."}
  ],
  "metadata": {"pipeline": "mock", "session_id": "session_abc123"}
}
```

### `GET /api/health`

Health check del servidor.

---

## Docker

```bash
cd backend
docker compose build
docker compose up -d
```

El contenedor expone el puerto **8045** y soporta GPU NVIDIA.

---

## Variables de Entorno

| Variable | Descripción | Requerida |
|----------|-------------|-----------|
| `TAVILY_API_KEY` | API key de Tavily para búsqueda web | Sí (para search tool) |
| `ALPHAVANTAGE_API_KEY` | API key de Alpha Vantage para precios | Sí (para stock_price tool) |

---

## Base de Conocimiento (RAG)

Para cargar la base de conocimiento económico:

```bash
cd backend
python -m rag.loader
```

Descarga el dataset **PlanTL-GOB-ES/WikiCAT_esv2** (categoría Economía) e indexa los artículos en ChromaDB con embeddings de `sentence-transformers/all-MiniLM-L6-v2`.

---

## Estado del Proyecto

- [x] Estructura de proyecto completa
- [x] Frontend (Landing → Onboarding → Chat)
- [x] API con mock pipeline
- [x] Herramientas externas (calculator, search, finance)
- [x] Motor RAG (ChromaDB + embeddings)
- [ ] Implementación real de agentes con LangChain
- [ ] Integración con modelos de lenguaje (Orchestrator, Market, Recommendation, Critic)
- [ ] Memoria de conversación entre turnos
- [ ] Streaming de respuestas

---

## Tech Stack

| Componente | Tecnología |
|-----------|------------|
| **Frontend** | Vite, Vanilla JS, CSS custom properties |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Agentes** | LangChain (en desarrollo) |
| **Modelos** | Qwen/Qwen2.5-7B-Instruct + LoRA (SFT + GRPO) |
| **RAG** | ChromaDB, SentenceTransformers |
| **Herramientas** | Tavily, SEC EDGAR, Alpha Vantage, numexpr |
| **Infra** | Docker, NVIDIA CUDA 12.1 |
