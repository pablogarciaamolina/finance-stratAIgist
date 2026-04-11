"""
Wrapper del modelo general del sistema.

Este archivo reutiliza la lógica ya existente en inference.py para:
- cargar el modelo general con LoRA
- generar respuestas de razonamiento

La idea es separar conceptualmente:
- modelo general
- modelo financiero
sin tener que tocar inference.py.
"""

from backend.models.inference import load_rlm_model, generate_reasoning


def load_general_model():
    """
    Carga el modelo general del proyecto.

    Returns:
        tuple: (model, tokenizer)
    """
    return load_rlm_model()


def generate_general_reasoning(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
):
    """
    Wrapper para mantener una interfaz homogénea con fin_model.py.

    Args:
        prompt: Prompt de entrada.
        model: Modelo general cargado.
        tokenizer: Tokenizer asociado.
        max_new_tokens: Se mantiene por compatibilidad de interfaz,
                        aunque inference.generate_reasoning ya controla
                        internamente la generación.

    Returns:
        str: Texto generado.
    """
    return generate_reasoning(prompt, model, tokenizer)