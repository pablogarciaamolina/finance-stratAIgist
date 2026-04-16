"""
Wrapper del modelo financiero Fin-R1.

Este archivo carga Fin-R1 como modelo especializado para el
Recommendation Agent.

Por simplicidad, se usa transformers directamente.
Si más adelante queréis desplegar Fin-R1 con vLLM, este archivo
sería el lugar natural para sustituir la implementación.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FIN_MODEL_NAME = "SUFE-AIFLM-Lab/Fin-R1"


def load_fin_model(model_name: str = FIN_MODEL_NAME):
    """
    Carga el modelo financiero Fin-R1 y su tokenizer.

    Args:
        model_name: nombre del modelo en Hugging Face o ruta local.

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Cargando modelo financiero desde {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


def _build_prompt_with_chat_template(prompt: str, tokenizer):
    """
    Intenta construir el prompt usando la chat template del tokenizer.
    Si no existe, devuelve None.
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return None


def _build_fallback_prompt(prompt: str) -> str:
    """
    Fallback simple si el tokenizer no soporta chat template.
    """
    return f"USER: {prompt}\nASSISTANT:"


def generate_financial_reasoning(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> (str, dict):
    """
    Genera una respuesta usando Fin-R1.

    Args:
        prompt: texto de entrada
        model: modelo financiero cargado
        tokenizer: tokenizer cargado
        max_new_tokens: máximo de tokens a generar
        do_sample: activa sampling si es True
        temperature: temperatura de generación

    Returns:
        str: texto generado
    """
    text = _build_prompt_with_chat_template(prompt, tokenizer)
    if text is None:
        text = _build_fallback_prompt(prompt)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    input_tokens = input_length
    output_tokens = generated_tokens.shape[0]
    total_tokens = input_tokens + output_tokens

    token_info = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    if not generated_text.startswith("ASSISTANT:"):
        return f"ASSISTANT: {generated_text}"

    return generated_text


if __name__ == "__main__":
    model, tokenizer = load_fin_model()

    test_prompt = (
        "Analiza si Nvidia podría ser interesante para un inversor moderado "
        "a 12 meses, teniendo en cuenta crecimiento, valoración y riesgos."
    )

    result, token_info = generate_financial_reasoning(
        test_prompt,
        model,
        tokenizer,
        max_new_tokens=300,
    )
    print(result)