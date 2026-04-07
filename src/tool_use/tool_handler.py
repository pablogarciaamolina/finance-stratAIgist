import json
import re
from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOllama
from .tools import calculator, internet_search, company_fundamentals, company_events, stock_price
from src.rlm.inference import generate_reasoning

load_dotenv()

TOOL_SCHEMAS = {
    "calculator": {
        "name": "calculator",
        "description": "Evalúa una expresión matemática simple. Útil para realizar cálculos aritméticos.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "La expresión matemática a evaluar (ej: '25 * 4 + 100')"
                }
            },
            "required": ["expression"]
        }
    },

    "internet_search": {
        "name": "internet_search",
        "description": "Busca información en internet usando Tavily API. Usa esta herramienta para información actualizada o no disponible en la base de datos.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La consulta de búsqueda en internet"
                }
            },
            "required": ["query"]
        }
    },

    "company_fundamentals": {
        "name": "company_fundamentals",
        "description": "Obtiene datos económicos y financieros básicos de una empresa pública (ingresos, beneficios, activos, pasivos) usando información oficial.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker bursátil de la empresa (ej: AAPL, TSLA, MSFT)"
                }
            },
            "required": ["ticker"]
        }
    },

    "company_events": {
        "name": "company_events",
        "description": "Obtiene eventos recientes y noticias materiales de una empresa (earnings, adquisiciones, cambios relevantes) a partir de filings 8-K de SEC EDGAR.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker bursátil de la empresa (ej: AAPL, TSLA, MSFT)"
                }
            },
            "required": ["ticker"]
        }
    },

    "stock_price": {
        "name": "stock_price",
        "description": "Obtiene el precio actual de una acción y su variación reciente en el mercado.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker bursátil de la acción (ej: AAPL, TSLA, MSFT)"
                }
            },
            "required": ["ticker"]
        }
    }
}

SYSTEM_PROMPT = """Eres un asistente inteligente que puede usar herramientas para responder preguntas.

Herramientas disponibles:
{tools_description}

INSTRUCCIONES IMPORTANTES:
Revisa las herramientas disponibles y busca la más adecuada para responder a la pregunta del usuario. Si decides usar una herramienta, hazlo siguiendo el formato JSON especificado. No respondas directamente a la pregunta sin usar herramientas si crees que alguna puede ayudarte a obtener información relevante o realizar cálculos necesarios.
Debes responder únicamente las preguntas propuestas por el usuario. Deberás decidir si quieres llamar a una herramienta. En tal caso, tu llamada deberá estar formateada de la siguiente forma:
{{"nombre": "nombre_de_la_herramienta", "argumentos": {{"parametro": "valor"}}}}
El formato es estrictamente JSON, con los campos "nombre" y "argumentos". No uses otro formato. Para los "argumentos" cada parametro debe tener el nombre descrito en el esquema de la herramienta.

Tras la ejecución de la herramienta, recibirás el resultado de la llamada a la herramienta.
"""

def get_tools_description():
    """Genera una descripción legible de las herramientas disponibles."""
    descriptions = []
    for tool_name, schema in TOOL_SCHEMAS.items():
        params = ", ".join(schema["parameters"]["properties"].keys())
        descriptions.append(f"- {tool_name}({params}): {schema['description']}")
    return "\n".join(descriptions)

def parse_and_execute_tool_call(model_output):
    """
    Intenta detectar y ejecutar múltiples llamadas a herramientas en el output del modelo.
    Busca el formato JSON: {"nombre": "...", "argumentos": {...}}
    
    Retorna:
    - Un diccionario con los resultados de todas las herramientas ejecutadas si hubo llamadas exitosas.
      Formato: {"tool_calls": [{"tool": "nombre", "result": "..."}], "combined_result": "..."}
    - None si no se detectó ninguna llamada válida.
    """
    
    available_tools = {
        "calculator": calculator,
        "internet_search": internet_search,
        "company_fundamentals": company_fundamentals,
        "company_events": company_events,
        "stock_price": stock_price
    }
    
    tool_results = []
    
    try:
        # Buscar todas las ocurrencias de JSON con campos "nombre" y "argumentos"
        # Patrón más robusto que maneje JSON anidado
        pattern = r'\{[^{}]*"nombre"[^{}]*"argumentos"[^{}]*\{[^{}]*\}[^{}]*\}'
        json_matches = list(re.finditer(pattern, model_output, re.DOTALL))
        
        # Si no encuentra con argumentos anidados, buscar formato simple
        if not json_matches:
            pattern = r'\{[^{}]*"nombre"[^{}]*"argumentos"[^{}]*\}'
            json_matches = list(re.finditer(pattern, model_output, re.DOTALL))
        
        if not json_matches:
            return None
        
        # Procesar cada llamada a herramienta encontrada
        for match in json_matches:
            json_str = match.group(0).strip()
            
            try:
                tool_call = json.loads(json_str)
                
                # Usar los nombres en español: "nombre" y "argumentos"
                tool_name = tool_call.get("nombre")
                arguments = tool_call.get("argumentos", {})
                
                if not tool_name:
                    tool_results.append({
                        "tool": "unknown",
                        "result": "Error: No se especificó el nombre de la herramienta.",
                        "success": False
                    })
                    continue
                
                if tool_name not in available_tools:
                    tool_results.append({
                        "tool": tool_name,
                        "result": f"Error: Herramienta '{tool_name}' no encontrada.",
                        "success": False
                    })
                    continue
                
                # Ejecutar la herramienta
                tool_function = available_tools[tool_name]
                result = tool_function.invoke(arguments)
                
                tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "success": True
                })
                
            except json.JSONDecodeError:
                tool_results.append({
                    "tool": "unknown",
                    "result": f"Error: JSON inválido en la llamada a herramienta.",
                    "success": False
                })
            except Exception as e:
                tool_results.append({
                    "tool": tool_name if 'tool_name' in locals() else "unknown",
                    "result": f"Error ejecutando herramienta: {str(e)}",
                    "success": False
                })
        
        # Si no se ejecutó ninguna herramienta exitosamente, retornar None
        if not tool_results:
            return None
        
        # Crear un resultado combinado
        combined_parts = []
        for i, tr in enumerate(tool_results, 1):
            if len(tool_results) > 1:
                combined_parts.append(f"Herramienta {i} ({tr['tool']}): {tr['result']}")
            else:
                combined_parts.append(str(tr['result']))
        
        combined_result = "\n\n".join(combined_parts)
        
        # Retornar formato compatible con el código existente (string)
        # pero también incluir información estructurada
        return combined_result
        
    except Exception as e:
        return f"Error general procesando herramientas: {str(e)}"

# def get_model():
#     """Crea y retorna el modelo Ollama local."""
#     return ChatOllama(
#         model="llama3",
#         temperature=0
#     )

def run_agent_loop(model, user_question, tokenizer, max_iterations=5, verbose=True):
    """
    Ejecuta el loop ReAct: Model → Tool → Model → Answer
    
    Args:
        user_question: La pregunta del usuario
        max_iterations: Número máximo de iteraciones para evitar loops infinitos
        verbose: Si True, imprime el proceso paso a paso
    
    Returns:
        La respuesta final del modelo
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
    
    system_prompt = SYSTEM_PROMPT.format(tools_description=get_tools_description())
    
    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteración {iteration + 1}")
            print(f"{'='*60}")
        
        # Convertir conversation_history a string antes de pasarlo a generate_reasoning
        prompt = format_conversation_to_prompt(conversation_history)
        model_output = generate_reasoning(prompt, model, tokenizer).split("ASSISTANT:")[-1].strip().replace("<|endoftext|>", "")
        
        if verbose:
            print(f"\n🤖 Modelo dice:\n{model_output}")
        
        tool_result = parse_and_execute_tool_call(model_output)
        
        if tool_result is None:
            if verbose:
                print(f"\n✅ Respuesta final (sin herramienta)")
            conversation_history.append({"role": "assistant", "content": model_output})
            return conversation_history
        
        if verbose:
            print(f"\n🔧 Resultado de herramienta:\n{tool_result}")
        
        conversation_history.append({"role": "assistant", "content": model_output})
        conversation_history.append({
            "role": "user", 
            "content": f"Resultado de la herramienta: {tool_result}\n\nAhora proporciona una respuesta final en lenguaje natural basándote en este resultado."
        })
    
    if verbose:
        print(f"\n⚠️ Alcanzado el máximo de iteraciones ({max_iterations})")
    
    # Convertir conversation_history a string antes de pasarlo a generate_reasoning
    prompt = format_conversation_to_prompt(conversation_history)
    final_response = generate_reasoning(prompt, model, tokenizer)
    conversation_history.append({"role": "assistant", "content": final_response})
    return conversation_history

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRUEBA DEL AGENTE ReAct")
    print("="*60)
    
    # Prueba simple con cálculo
    # question = "¿Cuánto es 15 * 3 + 50?"
    question = "Cuando saldrá el próximo Call of Duty?"
    print(f"\n❓ Pregunta: {question}\n")
    
    answer = run_agent_loop(question, verbose=True)
    
    print("\n" + "="*60)
    print("📝 RESPUESTA FINAL:")
    print("="*60)
    print(answer)
    print("\n")
