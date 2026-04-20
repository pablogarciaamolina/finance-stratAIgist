"""
Wrapper del modelo financiero Fin-R1.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FIN_MODEL_NAME = "SUFE-AIFLM-Lab/Fin-R1"


def _get_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fin_model(model_name: str = FIN_MODEL_NAME):
    print(f"Cargando modelo financiero desde {model_name}...", flush=True)

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


def _build_prompt_with_chat_template(prompt: str, tokenizer) -> str | None:
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

    text = _build_prompt_with_chat_template(prompt, tokenizer)
    prompt_build_path = "chat_template"
    if text is None:
        text = _build_fallback_prompt(prompt)
        prompt_build_path = "fallback_prompt"

    device = _get_model_device(model)
    inputs = tokenizer(text, return_tensors="pt").to(device)
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
        "model_name": getattr(getattr(model, "config", None), "_name_or_path", None) or FIN_MODEL_NAME,
    }

    if not generated_text.startswith("ASSISTANT:"):
        return f"ASSISTANT: {generated_text}", token_info
    return generated_text, token_info
