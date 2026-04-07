import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ruta a tu modelo final de fase 1
MODEL_PATH = "./src/rlm/weights/sft_lora_gsm8k"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ─────────────────────────────────────────────────────────────
# Ollama wrapper for lightweight local testing
# ─────────────────────────────────────────────────────────────

class OllamaWrapper:
    """
    Wrapper around ChatOllama that makes it compatible with generate_reasoning().
    This allows using a lightweight Ollama model (e.g. llama3.2, qwen2.5:3b)
    in place of the full HuggingFace RLM model for local testing.
    """
    _is_ollama = True

    def __init__(self, model_name="llama3.2"):
        from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.model_name = model_name
        print(f"[Ollama] Modelo '{model_name}' cargado.")


def load_ollama_model(model_name="llama3.2"):
    """
    Carga un modelo Ollama ligero para testing local.
    Requiere que Ollama esté corriendo localmente (ollama serve).

    Args:
        model_name: Nombre del modelo Ollama (ej: 'llama3.2', 'qwen2.5:3b', 'mistral').

    Returns:
        Tuple (OllamaWrapper, None) — el None sustituye al tokenizer.
    """
    wrapper = OllamaWrapper(model_name)
    return wrapper, None


# ─────────────────────────────────────────────────────────────
# HuggingFace model loading (original)
# ─────────────────────────────────────────────────────────────

def load_rlm_model(base_model: str = BASE_MODEL, sft_path: str = MODEL_PATH):
    # TODO: Cargar el modelo base y el adaptador LoRA
    print(f"Cargando modelo RLM desde {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Unified generation (works with both HuggingFace and Ollama)
# ─────────────────────────────────────────────────────────────

def generate_reasoning(prompt, model, tokenizer):
    """
    Genera una respuesta que incluye el razonamiento (CoT).
    Funciona tanto con modelos HuggingFace (model+tokenizer) como
    con OllamaWrapper (model=wrapper, tokenizer=None).
    """
    # ── Ollama path ──
    if hasattr(model, '_is_ollama') and model._is_ollama:
        text = f"USER: {prompt}\nASSISTANT:"
        response = model.llm.invoke(text)
        return f"ASSISTANT: {response.content}"

    # ── HuggingFace path (original) ──
    text = f"""
    USER: {prompt}
    ASSISTANT:"""
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    im_end = "<|" + "im_end" + "|>"
    outputs = model.generate(**input_ids, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0]).replace(im_end, "")
    return response


if __name__ == "__main__":
    import sys
    # Si se pasa --ollama, usar Ollama para test rapido
    if "--ollama" in sys.argv:
        model_name = "llama3.2"
        # Allow custom model name: --ollama qwen2.5:3b
        idx = sys.argv.index("--ollama")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            model_name = sys.argv[idx + 1]
        model, tokenizer = load_ollama_model(model_name)
    else:
        model, tokenizer = load_rlm_model()

    test_prompt = "It's Ava's birthday party. Her parents bought a unicorn pinata for $13 and filled it with all of her favorite treats. They bought 4 bags of Reese's for $9 per bag, 3 bags of Snickers for $5 per bag, and 5 bags of Skittles for $7 per bag. How much did the unicorn pinata and the treats cost altogether?"
    print(generate_reasoning(test_prompt, model, tokenizer))

