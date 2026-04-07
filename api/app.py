from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os

# Añadir el directorio raíz al path para poder importar los módulos de las fases
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- IMPORTACIONES DE LOS MÓDULOS DE LOS ALUMNOS ---
from src.rlm.inference import load_rlm_model, generate_reasoning
from src.tool_use.tool_handler import parse_and_execute_tool_call, run_agent_loop
from src.rag.rag_engine import RAGEngine
from src.react.agent import ReActAgent

app = FastAPI(
    title="Práctica Master: Modelos Generativos Profundos",
    description="API para evaluar las 4 fases de la práctica."
)

# --- Variables Globales (Modelos) ---
# Se cargan al inicio para no recargarlos en cada petición
MODEL = None
TOKENIZER = None
AGENT = None
RAG_ENGINE = None

@app.on_event("startup")
async def startup_event():
    global MODEL, TOKENIZER, AGENT, RAG_ENGINE
    print("Inicializando API...")
    # TODO: Cargar el modelo de la Fase 1 aquí
    MODEL, TOKENIZER = load_rlm_model()
    # Inicializar motor RAG de la Fase 3
    RAG_ENGINE = RAGEngine()
    if MODEL:
        AGENT = ReActAgent(MODEL, TOKENIZER)
    print("Modelos cargados correctamente.")

# --- Modelos de Pydantic para Request/Response ---
class QueryRequest(BaseModel):
    prompt: str

class GenericResponse(BaseModel):
    response: str
    trace: list[dict] = []
    details: dict = {}


# ================= ENDPOINTS DE EVALUACIÓN =================

# --- FASE 1: Razonamiento (RLM) ---
@app.post("/phase1/reasoning", response_model=GenericResponse, tags=["Fase 1"])
async def phase1_endpoint(request: QueryRequest):
    """
    Evalúa el modelo RLM. Debe devolver la respuesta con el razonamiento (CoT) visible.
    """
    if not MODEL or not TOKENIZER:
        return {"response": "ERROR: Modelo de Fase 1 no cargado.", "details": {"status": "todo"}}
    
    # TODO: Usar la función de inferencia de Fase 1
    response_text = generate_reasoning(request.prompt, MODEL, TOKENIZER)
    print("Response Text:", response_text)
    try:
        reasoning, response = response_text.split("ASSISTANT:")[1].split("Final answer:")
    except:
        reasoning, response = response_text, response_text
    return {
        "response": response, "trace": [{"step": 0, "content": reasoning}], "details": {"stage": "sft_grpo"}
    }


# --- FASE 2: Tool Use ---
@app.post("/phase2/tools", response_model=GenericResponse, tags=["Fase 2"])
async def phase2_endpoint(request: QueryRequest):
    """
    Evalúa la capacidad de llamar herramientas.
    Si el prompt requiere una herramienta, debe devolver la ejecución simulada.
    """
    # 1. Simular generación del modelo (o usar el real si ya sabe usar tools)
    # model_output_simulated = '''... Thought: Necesito la calculadora. Action: '''
    tool_result = run_agent_loop(MODEL, request.prompt, TOKENIZER)

    # 2. Usar el handler de Fase 2
    # TODO: Descomentar
    # tool_result = parse_and_execute_tool_call(model_output_simulated)

    # tool_result = "Placeholder: Resultado de herramienta (Fase 2) no implementado."
    
    if tool_result:
        return {"response": f"Tool execution result: {tool_result[-1]['content'].replace('</s>', '')}", "details": {"tool_called": True}, "trace": tool_result[1:]}
    else:
        return {"response": "No tool call detected or needed.", "details": {"tool_called": False}}


# --- FASE 3: RAG ---
@app.post("/phase3/rag", response_model=GenericResponse, tags=["Fase 3"])
async def phase3_endpoint(request: QueryRequest):
    """
    Evalúa el RAG. Recupera contexto de ChromaDB y genera una respuesta.
    """
    if not RAG_ENGINE:
        return {"response": "ERROR: Motor RAG no inicializado.", "details": {"status": "error"}}

    # 1. Recuperar contexto
    context_list = RAG_ENGINE.retrieve_context(request.prompt, top_k=5, similarity_threshold=0.75)

    # 2. Formatear prompt con el contexto recuperado
    rag_prompt = RAG_ENGINE.format_rag_prompt(request.prompt, context_list)

    # 3. Generar respuesta con el modelo
    if MODEL and TOKENIZER:
        response_text = generate_reasoning(rag_prompt, MODEL, TOKENIZER)
        try:
            response_text = response_text.split("ASSISTANT:")[-1].strip()
        except:
            pass
    else:
        response_text = "Modelo no disponible. Contexto recuperado correctamente."

    retrieved_docs = [{"text": ctx["text"][:200] + "...", "label": ctx["label"], "distance": ctx["distance"]} for ctx in context_list]

    return {
        "response": response_text,
        "trace": [{"step": i, "content": f"[{ctx['label']}] {ctx['text'][:150]}..."} for i, ctx in enumerate(context_list)],
        "details": {"retrieved_docs": retrieved_docs, "num_results": len(context_list)}
    }


# --- FASE 4: Agente ReAct ---
@app.post("/phase4/agent", tags=["Fase 4"])
async def phase4_endpoint(request: QueryRequest):
    """
    Evalúa el agente completo. Devuelve la respuesta final y la traza de ejecución.
    """
    if not AGENT:
        return {"final_answer": "ERROR: Agente no inicializado.", "trace": []}

    # TODO: Ejecutar agente
    result = AGENT.run(request.prompt)
    # result = {"final_answer": "Placeholder Fase 4 Agent", "trace": [{"step": 0, "content": "..."}]} # TODO remove

    return result

if __name__ == "__main__":
    # Para correr localmente: python api/app.py
    uvicorn.run(app, host="0.0.0.0", port=8045)