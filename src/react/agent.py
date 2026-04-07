import json
import re

# Importar componentes de fases anteriores
# from phase1_training.inference import generate_reasoning, load_rlm_model
# from phase2_tool_use.tools import get_tools_prompt
# from phase2_tool_use.tool_handler import parse_and_execute_tool_call
# from phase3_rag.rag_engine import retrieve_context
from src.rlm.inference import generate_reasoning
from src.tool_use.tool_handler import (
    TOOL_SCHEMAS,
    get_tools_description,
    parse_and_execute_tool_call,
)
from src.rag.rag_engine import RAGEngine


class ReActAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # TODO: Obtener prompt de herramientas (Fase 2)
        # Implementado usando la descripción real de herramientas de Fase 2
        self.tools_prompt = self._build_tools_prompt()

        # Inicializamos RAG para poder usar recuperación de contexto dentro del agente
        self.rag_engine = RAGEngine()

        # MEJORAR EL PROMPT DEL SISTEMA. Mirar langsmith, por ejemplo: https://smith.langchain.com/hub/hwchase17
        self.system_prompt = f"""Eres un agente autónomo útil.
Usa el siguiente formato de Pensamiento-Acción-Observación para resolver tareas complejas.

{self.tools_prompt}

Formato a seguir:
Thought: Debo pensar qué hacer a continuación.
Action: 
Observation: El resultado de la herramienta.
... (repetir hasta tener la respuesta final)
Thought: Ya tengo suficiente información.
Final Answer: La respuesta final al usuario.

Reglas importantes:
- Si necesitas usar una herramienta, debes escribir Action seguido de un JSON válido.
- El formato de Action debe ser:
  Action: {{"nombre": "nombre_herramienta", "argumentos": {{"arg1": "valor1"}}}}
- Si necesitas consultar la base de conocimiento interna, usa:
  Action: {{"nombre": "retrieve_context", "argumentos": {{"query": "tu consulta", "top_k": 3}}}}
- No inventes observaciones.
- Solo da Final Answer cuando ya tengas suficiente información.

"""

    def _build_tools_prompt(self):
        """
        Construye el prompt con la descripción de herramientas de Fase 2
        y añade la pseudo-herramienta de RAG para Fase 3.
        """
        tools_description = get_tools_description()

        rag_tool_description = """
Herramienta adicional:
- retrieve_context(query, top_k): Recupera contexto relevante desde la base vectorial interna.
  Formato:
  {"nombre": "retrieve_context", "argumentos": {"query": "pregunta", "top_k": 3}}
"""
        return f"{tools_description}\n{rag_tool_description}"

    def _format_history_as_prompt(self, history):
        """
        Convierte el historial en un prompt plano para el modelo.
        """
        prompt_parts = []
        for msg in history:
            role = msg["role"].upper()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)

    def _extract_final_answer(self, model_output):
        """
        Extrae la respuesta final si existe.
        """
        # TODO: Implementar la detección de "Final Answer" esto debe ser modificado según las
        # necesitadades de cada equipo, en el caso más sencillo se puede buscar el string "Final Answer:" en el modelo output.
        # En casos más complejos se puede usar un regex o un modelo de clasificación (con LLM incluso) para detectar si la respuesta final satisface la pregunta del usuario.
        if "Final Answer:" in model_output:
            return model_output.split("Final Answer:")[-1].strip()
        return None

    def _extract_action_json(self, model_output):
        """
        Extrae el JSON de Action del output del modelo.
        Intenta capturar el bloque:
        Action: {"nombre": "...", "argumentos": {...}}
        """
        match = re.search(r'Action:\s*(\{.*\})', model_output, re.DOTALL)
        if not match:
            return None

        action_text = match.group(1).strip()
        action_text = action_text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(action_text)
        except json.JSONDecodeError:
            return None

    def _execute_rag_action(self, action_json):
        """
        Ejecuta la pseudo-herramienta retrieve_context para integrar Fase 3 dentro del agente.
        """
        argumentos = action_json.get("argumentos", {})
        query = argumentos.get("query")
        top_k = argumentos.get("top_k", 3)

        if not query:
            return "Error: Falta el argumento 'query' en retrieve_context."

        try:
            contexts = self.rag_engine.retrieve_context(query=query, top_k=top_k)
        except Exception as e:
            return f"Error al ejecutar retrieve_context: {str(e)}"

        if not contexts:
            return "No se encontró contexto relevante en la base de conocimiento."

        formatted_contexts = []
        for i, ctx in enumerate(contexts, start=1):
            label = ctx.get("label", "unknown")
            distance = ctx.get("distance", "N/A")
            text = ctx.get("text", "")
            formatted_contexts.append(
                f"[Documento {i}] label={label}, distance={distance}\n{text}"
            )

        return "\n\n".join(formatted_contexts)

    def _execute_action(self, model_output):
        """
        Ejecuta la acción detectada en el output del modelo.
        Primero intenta usar el handler de Fase 2.
        Si no aplica, intenta la pseudo-herramienta de RAG.
        """
        # TODO: Usar el handler de Fase 2 y RAG de Fase 3 para ver si hay JSON de herramienta
        # tool_result = parse_and_execute_tool_call(model_output)

        # Primero intentamos herramientas reales de Fase 2
        tool_result = parse_and_execute_tool_call(model_output)
        if tool_result is not None:
            return tool_result

        # Si no era una herramienta de Fase 2, probamos retrieve_context
        action_json = self._extract_action_json(model_output)
        if not action_json:
            return None

        tool_name = action_json.get("nombre")
        if tool_name == "retrieve_context":
            return self._execute_rag_action(action_json)

        return None

    def run(self, user_query, max_steps=5):
        """
        Ejecuta el bucle ReAct para resolver la query.
        """
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        trace = []  # Para guardar los pasos dados y mostrarlos en la API

        step = 0
        while step < max_steps:
            # 1. Generar pensamiento (Thought) y posible Acción
            # TODO: Llamar al modelo de Fase 1 con el historial actual
            # model_output = generate_reasoning(current_prompt, self.model, self.tokenizer)

            current_prompt = self._format_history_as_prompt(history)
            model_output = generate_reasoning(current_prompt, self.model, self.tokenizer)

            print(f"--- Step {step} ---")

            # Limpiamos un poco la salida si el modelo devuelve prefijos tipo ASSISTANT:
            if isinstance(model_output, str) and "ASSISTANT:" in model_output:
                model_output = model_output.split("ASSISTANT:")[-1].strip()

            if isinstance(model_output, str):
                model_output = model_output.replace("<|endoftext|>", "").strip()

            print(f"Model Output: {model_output}")

            # Añadir output al historial y al trace
            history.append({"role": "assistant", "content": model_output})
            trace.append({"step": step, "type": "model_output", "content": model_output})

            # 2. Detectar si hay "Final Answer"
            final_answer = self._extract_final_answer(model_output)
            if final_answer:
                return {"final_answer": final_answer, "trace": trace}

            # 3. Intentar ejecutar Acción (Herramienta)
            tool_result = self._execute_action(model_output)

            if tool_result:
                print(f"Observation: {tool_result}")
                observation_msg = f"Observation: {tool_result}"
                history.append({"role": "user", "content": observation_msg})  # Se suele añadir como rol user o system
                trace.append({"step": step, "type": "observation", "content": tool_result})
            else:
                # Si no hubo herramienta ni respuesta final, forzar al modelo a continuar o parar.
                if "Action:" in model_output and not tool_result:
                    history.append({
                        "role": "user",
                        "content": "Observation: Error: No se pudo ejecutar la acción. Revisa el formato JSON."
                    })
                    trace.append({
                        "step": step,
                        "type": "observation",
                        "content": "Error: No se pudo ejecutar la acción. Revisa el formato JSON."
                    })
                else:
                    # El modelo solo pensó, dejar que siga en el siguiente loop
                    pass

            step += 1

        return {"final_answer": "Error: Se excedió el número máximo de pasos.", "trace": trace}


# Ejemplo de uso (si se ejecuta directamente)
if __name__ == "__main__":
    model, tokenizer = None, None
    agent = ReActAgent(model, tokenizer)
    # response = agent.run("¿Cuál es la raíz cuadrada de la edad del presidente de Francia?")
    # print(response)