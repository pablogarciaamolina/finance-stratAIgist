"""
Model loading and inference utilities.

Supports both HuggingFace models (with LoRA adapters) and
lightweight Ollama models for local testing.
"""

from __future__ import annotations

import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./backend/models/weights/sft_lora_gsm8k"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class OllamaWrapper:
    _is_ollama = True

    def __init__(self, model_name="llama3.2"):
        from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.model_name = model_name
        print(f"[Ollama] Modelo '{model_name}' cargado.", flush=True)


def load_ollama_model(model_name="llama3.2"):
    wrapper = OllamaWrapper(model_name)
    return wrapper, None


def load_rlm_model(base_model: str = BASE_MODEL, sft_path: str = MODEL_PATH):
    print(f"Cargando modelo RLM desde {sft_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, sft_path)
    return model, tokenizer


def generate_reasoning(prompt, model, tokenizer, max_new_tokens: int = 512):
    start_total = time.perf_counter()

    if hasattr(model, "_is_ollama") and model._is_ollama:
        text = f"USER: {prompt}\nASSISTANT:"
        invoke_start = time.perf_counter()
        response = model.llm.invoke(text)
        generation_latency = time.perf_counter() - invoke_start

        input_tokens = len(text.split())
        output_tokens = len(response.content.split())
        token_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "generation_latency": generation_latency,
            "total_latency": time.perf_counter() - start_total,
            "backend": "ollama",
            "model_name": getattr(model, "model_name", "unknown_ollama_model"),
            "max_new_tokens": max_new_tokens,
        }
        return f"ASSISTANT: {response.content}", token_info

    text = f"USER: {prompt}\nASSISTANT:"
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    input_tokens = int(input_ids["input_ids"].shape[1])

    generation_start = time.perf_counter()
    outputs = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generation_latency = time.perf_counter() - generation_start

    output_tokens = int(outputs.shape[1] - input_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    token_info = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "generation_latency": generation_latency,
        "total_latency": time.perf_counter() - start_total,
        "backend": "huggingface",
        "model_name": getattr(model, "name_or_path", None) or getattr(getattr(model, "base_model", None), "name_or_path", BASE_MODEL),
        "max_new_tokens": max_new_tokens,
    }
    return response, token_info
