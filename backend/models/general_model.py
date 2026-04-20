"""
Wrapper del modelo general del sistema.
"""

from __future__ import annotations

from backend.models.inference import load_rlm_model, generate_reasoning


def load_general_model():
    return load_rlm_model()


def generate_general_reasoning(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> tuple[str, dict]:
    return generate_reasoning(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
