"""
Model loading and inference utilities.

Supports both HuggingFace models (with LoRA adapters) and
lightweight Ollama models for local testing.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths — adjust based on deployment
MODEL_PATH = "./backend/models/weights/sft_lora_gsm8k"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ─────────────────────────────────────────────────────────────
# Ollama wrapper for lightweight local testing
# ─────────────────────────────────────────────────────────────

class OllamaWrapper:
    """
    Wrapper around ChatOllama that makes it compatible with generate_reasoning().
    """
    _is_ollama = True

    def __init__(self, model_name="llama3.2"):
        from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.model_name = model_name
        print(f"[Ollama] Modelo '{model_name}' cargado.")


def load_ollama_model(model_name="llama3.2"):
    """Load a lightweight Ollama model for local testing."""
    wrapper = OllamaWrapper(model_name)
    return wrapper, None


# ─────────────────────────────────────────────────────────────
# HuggingFace model loading
# ─────────────────────────────────────────────────────────────

def load_rlm_model(base_model: str = BASE_MODEL, sft_path: str = MODEL_PATH):
    """Load the base model with LoRA adapter for inference."""
    print(f"Cargando modelo RLM desde {sft_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, sft_path)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Unified generation
# ─────────────────────────────────────────────────────────────

def generate_reasoning(prompt, model, tokenizer):
    """
    Generate response + token counting
    """

    # Ollama path
    if hasattr(model, '_is_ollama') and model._is_ollama:
        text = f"USER: {prompt}\nASSISTANT:"
        response = model.llm.invoke(text)

        # aproximación simple
        input_tokens = len(text.split())
        output_tokens = len(response.content.split())

        return f"ASSISTANT: {response.content}"

    # HuggingFace path
    text = f"""
    USER: {prompt}
    ASSISTANT:"""

    input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    input_tokens = input_ids["input_ids"].shape[1]

    outputs = model.generate(
        **input_ids,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_tokens = outputs.shape[1] - input_tokens

    response = tokenizer.decode(outputs[0])
    
    return response