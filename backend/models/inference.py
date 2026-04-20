"""
Model loading and inference utilities.

Supports both HuggingFace models (with LoRA adapters) and
lightweight Ollama models for local testing.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "./backend/models/weights/sft_lora_gsm8k"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_GENERAL_BACKEND = "auto"
DEFAULT_GENERAL_OLLAMA_MODEL = "llama3.2"


class OllamaWrapper:
    _is_ollama = True

    def __init__(self, model_name: str = DEFAULT_GENERAL_OLLAMA_MODEL):
        from langchain_community.chat_models import ChatOllama

        self.llm = ChatOllama(model=model_name, temperature=0)
        self.model_name = model_name
        print(f"[Ollama] Modelo '{model_name}' cargado.", flush=True)


def load_ollama_model(model_name: str = DEFAULT_GENERAL_OLLAMA_MODEL):
    wrapper = OllamaWrapper(model_name)
    return wrapper, None


def _get_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_4bit_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _load_hf_quantized_model(base_model: str, sft_path: str):
    tokenizer = _load_tokenizer(base_model)
    quant_config = _build_4bit_quant_config()

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base, sft_path)
    model.eval()
    return model, tokenizer


def _load_hf_cpu_model(base_model: str, sft_path: str):
    tokenizer = _load_tokenizer(base_model)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base, sft_path)
    model.eval()
    return model, tokenizer


def _normalized_backend_name(backend: str | None) -> str:
    if not backend:
        backend = os.getenv("GENERAL_MODEL_BACKEND", DEFAULT_GENERAL_BACKEND)
    return backend.strip().lower()


def load_rlm_model(
    base_model: str = BASE_MODEL,
    sft_path: str = MODEL_PATH,
    backend: str | None = None,
):
    backend_name = _normalized_backend_name(backend)
    ollama_model_name = os.getenv("GENERAL_OLLAMA_MODEL", DEFAULT_GENERAL_OLLAMA_MODEL)

    print(
        f"Cargando modelo RLM desde {sft_path} "
        f"(backend={backend_name}, cuda={torch.cuda.is_available()})...",
        flush=True,
    )

    if backend_name == "ollama":
        return load_ollama_model(ollama_model_name)

    if backend_name not in {"auto", "huggingface", "hf"}:
        raise ValueError(
            f"Backend no soportado: {backend_name}. Usa 'auto', 'huggingface' o 'ollama'."
        )

    hf_error: Exception | None = None

    if torch.cuda.is_available():
        try:
            print("[GeneralModel] Intentando cargar HuggingFace en 4-bit sobre GPU...", flush=True)
            return _load_hf_quantized_model(base_model, sft_path)
        except Exception as exc:
            hf_error = exc
            print(
                "[GeneralModel] Fallo cargando el modelo general en GPU 4-bit. "
                f"Se intentara un fallback. Error: {exc}",
                flush=True,
            )
            if backend_name in {"huggingface", "hf"}:
                raise

    if backend_name == "auto":
        try:
            print(
                f"[GeneralModel] Probando fallback a Ollama con modelo '{ollama_model_name}'...",
                flush=True,
            )
            return load_ollama_model(ollama_model_name)
        except Exception as exc:
            print(
                "[GeneralModel] Fallo el fallback a Ollama. "
                f"Se intentara CPU. Error: {exc}",
                flush=True,
            )

    print("[GeneralModel] Intentando cargar HuggingFace sobre CPU...", flush=True)
    try:
        return _load_hf_cpu_model(base_model, sft_path)
    except Exception as cpu_exc:
        if hf_error is not None:
            raise RuntimeError(
                "No se pudo cargar el modelo general ni en GPU ni en CPU. "
                "Si tienes Ollama disponible, exporta GENERAL_MODEL_BACKEND=ollama."
            ) from cpu_exc
        raise


def generate_reasoning(prompt, model, tokenizer, max_new_tokens: int = 512) -> tuple[str, dict]:
    start_total = time.perf_counter()

    if hasattr(model, "_is_ollama") and model._is_ollama:
        text = f"USER: {prompt}\nASSISTANT:"
        invoke_start = time.perf_counter()
        response = model.llm.invoke(text)
        generation_latency = time.perf_counter() - invoke_start

        response_text = response.content.strip()
        input_tokens = len(text.split())
        output_tokens = len(response_text.split())

        token_info = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens + output_tokens),
            "generation_latency": generation_latency,
            "total_latency": time.perf_counter() - start_total,
            "backend": "ollama",
            "model_name": getattr(model, "model_name", "unknown_ollama_model"),
            "max_new_tokens": max_new_tokens,
        }
        return f"ASSISTANT: {response_text}", token_info

    text = f"USER: {prompt}\nASSISTANT:"
    device = _get_model_device(model)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_tokens = int(inputs["input_ids"].shape[1])

    generation_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generation_latency = time.perf_counter() - generation_start

    generated_tokens = outputs[0][input_tokens:]
    output_tokens = int(generated_tokens.shape[0])
    response_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    token_info = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens + output_tokens),
        "generation_latency": generation_latency,
        "total_latency": time.perf_counter() - start_total,
        "backend": "huggingface",
        "model_name": (
            getattr(getattr(model, "config", None), "_name_or_path", None)
            or getattr(model, "name_or_path", None)
            or BASE_MODEL
        ),
        "max_new_tokens": max_new_tokens,
    }

    if not response_text.startswith("ASSISTANT:"):
        return f"ASSISTANT: {response_text}", token_info
    return response_text, token_info
