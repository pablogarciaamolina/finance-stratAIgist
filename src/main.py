"""
Main execution loop: Unified agent that combines:
- Reasoning (RLM) for chain-of-thought
- Tool Use for external actions (calculator, search, finance APIs)
- RAG for context retrieval from the Economia knowledge base

Usage:
    python -m src.main
    python -m src.main --ollama <model tag (e.g. qwen2.5:3b) or nothing (llama3.2)>
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rlm.inference import load_rlm_model, load_ollama_model, generate_reasoning
from src.tool_use.tool_handler import (
    parse_and_execute_tool_call,
    get_tools_description,
)
from src.rag.rag_engine import RAGEngine

# Token to strip from model output
_END_TOKEN = "<|" + "endoftext" + "|>"

# ---------------------------------------------------------------
# System prompt that combines tool use + RAG capabilities
# ---------------------------------------------------------------
UNIFIED_SYSTEM_PROMPT = """Eres un asistente inteligente con acceso a herramientas externas y a una base de conocimiento sobre Economia.

Herramientas disponibles:
{tools_description}

Base de conocimiento (RAG):
Tambien tienes acceso a una base de conocimiento con articulos sobre economia. Si recibes contexto de la base de conocimiento, usalo para fundamentar tu respuesta siempre que sea relevante para la pregunta.

INSTRUCCIONES IMPORTANTES:
1. Si la pregunta requiere calculos, informacion de mercado, busquedas en internet o datos financieros, usa la herramienta apropiada.
2. Si recibes contexto de la base de conocimiento, utilizalo solo si es relevante para la pregunta. Si no lo es, ignoralo.
3. Si decides usar una herramienta, formatea tu llamada ESTRICTAMENTE en JSON:
   {{"nombre": "nombre_de_la_herramienta", "argumentos": {{"parametro": "valor"}}}}
4. Tras la ejecucion de la herramienta, recibiras el resultado y deberas proporcionar una respuesta final.
5. Si no necesitas herramientas ni contexto adicional, responde directamente con tu conocimiento.
"""


def format_conversation_to_prompt(conversation_history):
    """Convierte la conversation_history (lista de dicts) en un string formateado."""
    text_parts = []
    for message in conversation_history:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text_parts.append(f"SYSTEM: {content}")
        elif role == "user":
            text_parts.append(f"USER: {content}")
        elif role == "assistant":
            text_parts.append(f"ASSISTANT: {content}")
    return "\n".join(text_parts)


def run_unified_agent(
    model,
    tokenizer,
    user_question,
    rag_engine=None,
    max_iterations=5,
    rag_top_k=3,
    verbose=True,
):
    """
    Bucle principal del agente unificado.

    En cada iteracion el agente puede:
      a) Llamar a una herramienta -> se ejecuta y el resultado se inyecta.
      b) Dar una respuesta final -> se devuelve la conversacion completa.

    Antes de la primera iteracion se hace una consulta RAG para inyectar contexto
    relevante de la base de datos de Economia (si esta disponible).

    Args:
        model: Modelo de lenguaje cargado.
        tokenizer: Tokenizer asociado al modelo.
        user_question: Pregunta del usuario.
        rag_engine: Instancia de RAGEngine (opcional). Si None, se omite RAG.
        max_iterations: Maximo de iteraciones herramienta -> modelo.
        rag_top_k: Numero de documentos a recuperar del RAG.
        verbose: Si True, imprime el proceso paso a paso.

    Returns:
        Lista de dicts con el historial completo de la conversacion.
    """

    # -- 1. Construir system prompt --
    system_prompt = UNIFIED_SYSTEM_PROMPT.format(
        tools_description=get_tools_description()
    )

    conversation_history = [
        {"role": "system", "content": system_prompt},
    ]

    # -- 2. Inyectar contexto RAG (si procede) --
    rag_context_text = ""
    if rag_engine is not None:
        context_list = rag_engine.retrieve_context(user_question, top_k=rag_top_k)
        if context_list:
            rag_context_text = "\n\n".join(
                [f"[Documento {i+1} - {ctx['label']}]\n{ctx['text']}"
                 for i, ctx in enumerate(context_list)]
            )
            if verbose:
                print(f"\nContexto RAG recuperado ({len(context_list)} docs):")
                for i, ctx in enumerate(context_list):
                    print(f"   [{i+1}] ({ctx['label']}) {ctx['text'][:120]}...")

    # Construir el mensaje de usuario, opcionalmente con contexto RAG
    if rag_context_text:
        user_message = (
            f"Se ha recuperado el siguiente contexto que podria ser relevante. "
            f"--- CONTEXTO ---\n{rag_context_text}\n--- FIN CONTEXTO ---\n\n"
            f"Pregunta del usuario: {user_question}"
        )
    else:
        user_message = user_question

    conversation_history.append({"role": "user", "content": user_message})

    # -- 3. Bucle agente: Modelo -> Herramienta -> Modelo --
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteracion {iteration + 1}")
            print(f"{'='*60}")

        prompt = format_conversation_to_prompt(conversation_history)
        raw_output = generate_reasoning(prompt, model, tokenizer)
        model_output = raw_output.split("ASSISTANT:")[-1].strip().replace(_END_TOKEN, "")

        if verbose:
            print(f"\nModelo dice:\n{model_output}")

        # Intentar ejecutar herramienta
        tool_result = parse_and_execute_tool_call(model_output)

        if tool_result is None:
            # No se detecto herramienta -> respuesta final
            if verbose:
                print(f"\nRespuesta final (sin herramienta)")
            conversation_history.append({"role": "assistant", "content": model_output})
            return conversation_history

        # Herramienta ejecutada -> inyectar resultado y continuar
        if verbose:
            print(f"\nResultado de herramienta:\n{tool_result}")

        conversation_history.append({"role": "assistant", "content": model_output})
        conversation_history.append({
            "role": "user",
            "content": (
                f"Resultado de la herramienta: {tool_result}\n\n"
                f"Ahora proporciona una respuesta final en lenguaje natural "
                f"basandote en este resultado."
            )
        })

    # Max iteraciones alcanzado -> forzar respuesta final
    if verbose:
        print(f"\nAlcanzado el maximo de iteraciones ({max_iterations})")

    prompt = format_conversation_to_prompt(conversation_history)
    final_response = generate_reasoning(prompt, model, tokenizer)
    conversation_history.append({"role": "assistant", "content": final_response})
    return conversation_history


def main():
    """Punto de entrada principal. Carga modelo, RAG y ejecuta el agente."""
    import sys

    use_ollama = "--ollama" in sys.argv

    print("=" * 60)
    print("AGENTE UNIFICADO: Reasoning + Tools + RAG")
    if use_ollama:
        print("(Modo Ollama - modelo ligero para testing)")
    print("=" * 60)

    # Cargar modelo
    if use_ollama:
        model_name = "llama3.2"
        idx = sys.argv.index("--ollama")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            model_name = sys.argv[idx + 1]
        print(f"\nCargando modelo Ollama ({model_name})...")
        model, tokenizer = load_ollama_model(model_name)
    else:
        print("\nCargando modelo RLM...")
        model, tokenizer = load_rlm_model()

    # Inicializar motor RAG (puede fallar si no se ha cargado el dataset)
    rag_engine = None
    try:
        rag_engine = RAGEngine()
        print(f"Motor RAG inicializado.")
    except Exception as e:
        print(f"RAG no disponible: {e}")
        print("Ejecuta 'python -m src.rag.load_dataset' primero para cargar los datos.")

    # Preguntas de prueba
    questions = [
        "Cuanto es 15 * 3 + 50?",
        # "Cuando saldra el proximo Call of Duty?",
        # "Dame un resumen de la situacion financiera basica de Microsoft.",
        # "Que es el PIB y como se calcula?",
        # "Como se esta comportando la accion de NVIDIA hoy en el mercado?",
        # "Explica qué es la LIBOR y compárala con el Euribor y la Federal Funds Rate. Además, relaciona el concepto de progreso técnico con la función de producción (Y = F(K, L)) y analiza cómo un aumento del progreso técnico podría afectar al pleno empleo y a una política de empleo garantizado."
    ]

    for q in questions:
        print(f"\n{'#'*60}")
        print(f"Pregunta: {q}")
        print(f"{'#'*60}")

        history = run_unified_agent(
            model, tokenizer, q,
            rag_engine=rag_engine,
            max_iterations=5,
            verbose=True
        )

        # Mostrar respuesta final
        final = history[-1]["content"] if history else "Sin respuesta"
        print(f"\nRESPUESTA FINAL:\n{final}")
        print("=" * 60)


if __name__ == "__main__":
    main()