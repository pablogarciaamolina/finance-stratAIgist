# OpenQA — Reasoning + Tools + RAG Agent

An end-to-end open-domain question-answering system built on **Qwen 2.5-7B-Instruct**, combining chain-of-thought **reasoning** (fine-tuned via SFT + GRPO), **tool use** (calculator, web search, financial APIs), and **RAG** (ChromaDB over Spanish Wikipedia articles). Exposed through a **FastAPI** evaluation API and runnable as a standalone interactive agent.

---

## Project Structure

```
openqa/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── Dockerfile                 # CUDA 12.1 container image
├── docker-compose.yml         # Docker Compose service (GPU, volumes, port 8045)
│
├── api/
│   └── app.py                 # FastAPI server with per-phase evaluation endpoints
│
└── src/
    ├── __init__.py
    ├── main.py                # Unified agent loop (Reasoning + Tools + RAG)
    │
    ├── rlm/                   # Phase 1 — Reasoning Language Model
    │   ├── config.py          # LoRA hyperparameters
    │   ├── train_sft.py       # Supervised Fine-Tuning (SFT) with GSM8K
    │   ├── train_grpo.py      # Group Relative Policy Optimization (GRPO) on SFT checkpoint
    │   ├── inference.py       # Model loading & generation (HuggingFace + Ollama)
    │   ├── eval_phase1_adapters.py  # Compare SFT vs GRPO accuracy on GSM8K
    │   └── weights/           # Saved LoRA adapter checkpoints
    │
    ├── tool_use/              # Phase 2 — Tool Use
    │   ├── tools.py           # Tool implementations (calculator, search, finance)
    │   └── tool_handler.py    # JSON parsing, tool dispatch, and ReAct agent loop
    │
    └── rag/                   # Phase 3 — Retrieval-Augmented Generation
        ├── load_dataset.py    # Load WikiCAT_esv2 (Economía) into ChromaDB
        ├── rag_engine.py      # RAGEngine class — query, retrieve, format context
        └── chroma_db/         # Persisted ChromaDB vector store (auto-generated)
```

---

## Module Descriptions

### `src/rlm/` — Reasoning Language Model (Phase 1)

Fine-tunes **Qwen 2.5-7B-Instruct** to perform step-by-step chain-of-thought (CoT) reasoning on math problems.

| File | Purpose |
|------|---------|
| `config.py` | Defines LoRA parameters (`r=10`, `lora_alpha=8`, task `QUESTION_ANS`). |
| `train_sft.py` | Supervised Fine-Tuning on the **GSM8K** dataset. Trains the model to produce `<think>…</think>` reasoning followed by a `Final answer:`. Uses `SFTTrainer` from TRL with LoRA. |
| `train_grpo.py` | **GRPO** reinforcement-learning stage. Loads the SFT adapter, samples `N=4` responses per question, computes a correctness reward, normalises advantages, and applies a REINFORCE-style update. |
| `inference.py` | Loads the trained model (base + LoRA adapter) for inference. Also provides an **Ollama wrapper** to run lightweight local models (e.g. `llama3.2`, `qwen2.5:3b`) for quick testing without GPU. |
| `eval_phase1_adapters.py` | Evaluation script that compares SFT vs GRPO adapters on GSM8K accuracy (exact match of extracted numbers). |

---

### `src/tool_use/` — Tool Use (Phase 2)

Gives the model the ability to call external tools via JSON-formatted function calls.

| File | Purpose |
|------|---------|
| `tools.py` | **5 tool implementations** using LangChain's `@tool` decorator: `calculator` (via `numexpr`), `internet_search` (via Tavily), `company_fundamentals` (SEC EDGAR), `company_events` (SEC 8-K filings), `stock_price` (Alpha Vantage). |
| `tool_handler.py` | Defines tool schemas, parses model output for JSON tool calls (`{"nombre": "...", "argumentos": {...}}`), dispatches execution, and runs a **ReAct agent loop** (generate → detect tool → execute → inject result → re-generate). |

---

### `src/rag/` — Retrieval-Augmented Generation (Phase 3)

Adds a knowledge base from Spanish Wikipedia articles (Economics domain) to ground model answers with real documents.

| File | Purpose |
|------|---------|
| `load_dataset.py` | Downloads the **PlanTL-GOB-ES/WikiCAT_esv2** dataset from HuggingFace, filters for the **Economía** category (label 5), and batches inserts into a **ChromaDB** persistent collection using `sentence-transformers/all-MiniLM-L6-v2` embeddings. |
| `rag_engine.py` | `RAGEngine` class: connects to the persisted ChromaDB collection, performs cosine-similarity retrieval (`retrieve_context`), and provides `format_rag_prompt` to inject retrieved documents into the LLM prompt. |

---

### `src/main.py` — Unified Agent (Phase 4)

The orchestrator that combines all three capabilities in a single agent loop:

1. **RAG retrieval** — queries ChromaDB for relevant economy documents and injects them as context.
2. **Model generation** — uses the fine-tuned RLM (or Ollama) to generate a response that may include reasoning and/or a tool call.
3. **Tool execution** — if a tool call is detected, executes it and feeds the result back to the model.
4. Iterates up to `max_iterations` until the model produces a final answer.

---

### `api/app.py` — FastAPI Evaluation API

Exposes four phase-specific endpoints for testing and grading:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/phase1/reasoning` | POST | Evaluates CoT reasoning. Returns the reasoning trace and final answer. |
| `/phase2/tools` | POST | Evaluates tool calling. Runs the ReAct agent loop and returns the tool execution trace. |
| `/phase3/rag` | POST | Evaluates RAG. Retrieves context from ChromaDB, generates an answer, and returns retrieved docs. |
| `/phase4/agent` | POST | (Placeholder) Full ReAct agent evaluation. |

All endpoints accept `{"prompt": "..."}` and return `{"response": "...", "trace": [...], "details": {...}}`.

---

## Environment Variables

Create a `.env` file in the project root with:

```env
TAVILY_API_KEY=tvly-xxxxxxxxxxxx        # Required for internet_search tool
ALPHAVANTAGE_API_KEY=xxxxxxxxxx          # Required for stock_price tool
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Load the RAG Knowledge Base

This downloads the WikiCAT_esv2 dataset and indexes the **Economía** articles into ChromaDB. **Must be run once** before using RAG:

```bash
python -m src.rag.load_dataset
```

### 3. Run the Unified Agent (CLI)

#### With the full HuggingFace model (requires GPU + LoRA weights):

```bash
python -m src.main
```

#### With a lightweight Ollama model (local testing, no GPU needed):

```bash
# Default model (llama3.2)
python -m src.main --ollama

# Specify a model
python -m src.main --ollama qwen2.5:3b
```

> **Note:** Ollama must be running locally (`ollama serve`).

### 4. Run the FastAPI Evaluation Server

```bash
python api/app.py
```

The server starts on `http://0.0.0.0:8045`. Use the auto-generated docs at `/docs` to test endpoints.

---

## Training Pipeline

These steps run inside the Docker container (GPU required):

### Phase 1a — Supervised Fine-Tuning (SFT)

```bash
python src/rlm/train_sft.py
```

Trains a LoRA adapter on **GSM8K** using the `<think>…</think>` format. Saves weights to `src/rlm/weights/sft_lora_gsm8k/`.

### Phase 1b — GRPO Reinforcement Learning

```bash
python src/rlm/train_grpo.py
```

Loads the SFT adapter and applies GRPO to improve reasoning accuracy. Saves to `src/rlm/weights/final_rlm_lora/`.

### Evaluate SFT vs GRPO

```bash
python src/rlm/eval_phase1_adapters.py --n 100 --split test
```

Compares both adapters on GSM8K accuracy and reports the delta.

---

## Docker

### Build & Run

```bash
docker compose build
docker compose up -d
```

### Enter the Container

```bash
docker exec -it openqa-apa bash
```

From inside the container, run any of the above commands (`python -m src.main`, `python api/app.py`, etc.).

The container exposes port **8045** and mounts `./src`, `./.env`, and `./weights` as volumes.

---

## Full Pipeline Summary

```
1.  pip install -r requirements.txt          # Install deps
2.  python src/rlm/train_sft.py              # Train SFT adapter (GPU)
3.  python src/rlm/train_grpo.py             # Train GRPO adapter (GPU)
4.  python src/rlm/eval_phase1_adapters.py   # Evaluate adapters
5.  python -m src.rag.load_dataset           # Index RAG knowledge base
6.  python -m src.main                       # Run unified agent (CLI)
7.  python api/app.py                        # Start evaluation API
```

---

## Tech Stack

- **Base Model:** Qwen/Qwen2.5-7B-Instruct
- **Fine-Tuning:** LoRA (PEFT) + SFTTrainer (TRL) + Custom GRPO
- **Vector Store:** ChromaDB with SentenceTransformer embeddings
- **Tools:** numexpr, Tavily, SEC EDGAR, Alpha Vantage
- **API:** FastAPI + Uvicorn
- **Infra:** Docker + NVIDIA CUDA 12.1
