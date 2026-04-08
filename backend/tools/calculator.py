"""Calculator tool — evaluates simple math expressions."""

import re
import numexpr as ne
from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión matemática simple. Útil para realizar cálculos aritméticos."""
    expression = re.sub(r"[^0-9+\-*/().%\s]", "", expression)
    try:
        result = ne.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error calculando: {e}"
