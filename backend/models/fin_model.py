"""
Wrapper del modelo financiero Fin-R1.
 
Soporta dos backends:
- Hugging Face (cuantizado en 4-bit si hay CUDA)
- Ollama (para pruebas locales ligeras)
"""
 
from __future__ import annotations
 
import time
from typing import Any
 
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
 
 
FIN_MODEL_NAME = "SUFE-AIFLM-Lab/Fin-R1"
FIN_OLLAMA_MODEL_NAME = "mychen76/Fin-R1:Q5"
 
 
class OllamaWrapper:
    _is_ollama = True
 
    def __init__(self, model_name: str = FIN_OLLAMA_MODEL_NAME):
        from langchain_community.chat_models import ChatOllama
 
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.model_name = model_name
        print(f"[Ollama] Modelo financiero '{model_name}' cargado.", flush=True)
 
 
def _get_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def _build_quant_config() -> BitsAndBytesConfig | None:
    """
    Configuración de cuantización agresiva en 4-bit.
    Solo se aplica si hay CUDA disponible.
    """
    if not torch.cuda.is_available():
        return None
 
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
 
 
def load_fin_model(
    model_name: str = FIN_MODEL_NAME,
    backend: str = "huggingface",
):
    """
    Carga el modelo financiero usando Hugging Face o Ollama.
 
    backend:
    - "huggingface"
    - "ollama"
    """
    backend = backend.lower().strip()
 
    if backend == "ollama":
        print(f"Cargando modelo financiero Ollama desde {model_name}...", flush=True)
        model = OllamaWrapper()
        return model, True
 
    if backend != "huggingface":
        raise ValueError(
            f"Backend no soportado: {backend}. Usa 'huggingface' o 'ollama'."
        )
 
    print(f"Cargando modelo financiero desde {model_name}...", flush=True)
 
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    quant_config = _build_quant_config()
 
    if quant_config is not None:
        print("[Fin-R1] Cargando en 4-bit NF4...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        print("[Fin-R1] CUDA no disponible. Cargando sin cuantización...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
 
    model.eval()
    return model, tokenizer
 
 
def _build_prompt_with_chat_template(prompt: str, tokenizer) -> str | None:
    if tokenizer is None:
        return None
 
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
    return f"USER: {prompt}\nASSISTANT:"
 
 
def generate_financial_reasoning(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> tuple[str, dict]:
    start_total = time.perf_counter()
 
    # Rama Ollama
    if hasattr(model, "_is_ollama") and model._is_ollama:
        text = _build_fallback_prompt(prompt)
 
        invoke_start = time.perf_counter()
        response = model.llm.invoke(text)
        generation_latency = time.perf_counter() - invoke_start
 
        generated_text = response.content.strip()
 
        input_tokens = len(text.split())
        output_tokens = len(generated_text.split())
        total_tokens = input_tokens + output_tokens
 
        token_info = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "generation_latency": generation_latency,
            "total_latency": time.perf_counter() - start_total,
            "prompt_build_path": "fallback_prompt",
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "model_name": getattr(model, "model_name", FIN_OLLAMA_MODEL_NAME),
            "backend": "ollama",
            "quantized": None,
            "quantization_mode": None,
        }
 
        if not generated_text.startswith("ASSISTANT:"):
            return f"ASSISTANT: {generated_text}", token_info
        return generated_text, token_info
 
    # Rama Hugging Face
    text = _build_prompt_with_chat_template(prompt, tokenizer)
    prompt_build_path = "chat_template"
    if text is None:
        text = _build_fallback_prompt(prompt)
        prompt_build_path = "fallback_prompt"
 
    device = _get_model_device(model)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = int(inputs["input_ids"].shape[1])
 
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
 
    generation_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)
    generation_latency = time.perf_counter() - generation_start
 
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()
 
    input_tokens = input_length
    output_tokens = int(generated_tokens.shape[0])
    total_tokens = input_tokens + output_tokens
 
    token_info = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "generation_latency": generation_latency,
        "total_latency": time.perf_counter() - start_total,
        "prompt_build_path": prompt_build_path,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "model_name": (
            getattr(getattr(model, "config", None), "_name_or_path", None)
            or FIN_MODEL_NAME
        ),
        "backend": "huggingface",
        "quantized": bool(torch.cuda.is_available()),
        "quantization_mode": "4bit_nf4" if torch.cuda.is_available() else None,
    }
 
    if not generated_text.startswith("ASSISTANT:"):
        return f"ASSISTANT: {generated_text}", token_info
    return generated_text, token_info