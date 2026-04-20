"""Helpers for sanitizing model outputs while preserving internal reasoning."""

from __future__ import annotations

import re
from typing import Any


THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
FINAL_LABEL_RE = re.compile(
    r"^\s*(?:final answer|respuesta final|answer|respuesta)\s*:\s*",
    re.IGNORECASE,
)
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTIBREAK_RE = re.compile(r"\n{3,}")


def normalize_model_output(text: Any) -> str:
    if text is None:
        return ""

    cleaned = str(text)
    if "ASSISTANT:" in cleaned:
        cleaned = cleaned.split("ASSISTANT:", 1)[-1]
    return cleaned.replace("<|endoftext|>", "").strip()


def extract_internal_reasoning(text: Any) -> str:
    cleaned = normalize_model_output(text)
    blocks = [
        match.strip()
        for match in THINK_BLOCK_RE.findall(cleaned)
        if isinstance(match, str) and match.strip()
    ]
    return "\n\n".join(blocks)


def strip_internal_reasoning(text: Any) -> str:
    cleaned = normalize_model_output(text)
    return THINK_BLOCK_RE.sub("", cleaned).strip()


def sanitize_visible_answer(text: Any) -> str:
    cleaned = strip_internal_reasoning(text)
    cleaned = FINAL_LABEL_RE.sub("", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    cleaned = MULTIBREAK_RE.sub("\n\n", cleaned)
    return cleaned.strip()
