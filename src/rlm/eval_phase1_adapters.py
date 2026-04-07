# eval_phase1_adapters.py
# Compara SFT vs GRPO en GSM8K (accuracy sobre N ejemplos) evitando "stacking" de adapters.
# Uso (desde el contenedor):
#   python eval_phase1_adapters.py --n 100
#   python eval_phase1_adapters.py --n 200 --split test
#   python eval_phase1_adapters.py --sft_path /app/rlm/weights/sft_lora_gsm8k --grpo_path /app/weights/final_rlm_lora

import os
import re
import argparse
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def norm_num(s: str) -> str:
    return s.strip().replace(",", "").replace("$", "").replace("€", "")


def extract_final_number(text: str) -> Optional[str]:
    # Prefer "Final answer:"
    m = re.search(r"Final answer:\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return norm_num(m.group(1))
    # Fallback: last number in text
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return norm_num(nums[-1]) if nums else None


def parse_gsm8k_gt(answer_field: str) -> str:
    # GSM8K format: "... #### 42"
    if "####" in answer_field:
        return norm_num(answer_field.split("####")[-1])
    return norm_num(answer_field)


def adapter_exists(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def pick_existing_path(primary: str, fallback: str) -> str:
    if adapter_exists(primary):
        return primary
    if adapter_exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"No encuentro adapter_config.json en:\n  - {primary}\n  - {fallback}"
    )


@torch.inference_mode()
def eval_adapter(
    base_model_name: str,
    adapter_path: str,
    tokenizer: AutoTokenizer,
    dataset,
    max_new_tokens: int = 256,
    device_map: str = "auto",
) -> Tuple[int, int]:
    """
    Devuelve (aciertos, total).
    Carga un base model "limpio" por adapter para evitar stacking.
    """
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, adapter_path).eval()

    correct = 0
    total = len(dataset)

    for ex in dataset:
        # Chat template correcto para Qwen instruct
        messages = [{"role": "user", "content": ex["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decodifica SOLO lo generado (sin el prompt)
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        pred = extract_final_number(gen)
        gt = parse_gsm8k_gt(ex["answer"])

        if pred is None:
            continue

        try:
            correct += int(abs(float(pred) - float(gt)) < 1e-6)
        except Exception:
            correct += int(pred == gt)

    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=100, help="Número de ejemplos a evaluar")
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Paths por defecto (docker-friendly)
    parser.add_argument("--sft_path", type=str, default="/app/rlm/weights/sft_lora_gsm8k")
    parser.add_argument("--grpo_path", type=str, default="/app/rlm/weights/final_rlm_lora")

    # Fallbacks típicos si entrenaste/guardaste en /app/weights
    parser.add_argument(
        "--sft_fallback", type=str, default="/app/weights/sft_lora_gsm8k"
    )
    parser.add_argument(
        "--grpo_fallback", type=str, default="/app/weights/final_rlm_lora"
    )

    args = parser.parse_args()

    # Resolver paths reales
    sft_path = pick_existing_path(args.sft_path, args.sft_fallback)
    grpo_path = pick_existing_path(args.grpo_path, args.grpo_fallback)

    print("SFT path :", sft_path)
    print("GRPO path:", grpo_path)

    # Dataset
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.n is not None and args.n > 0:
        ds = ds.select(range(min(args.n, len(ds))))

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    # Eval
    sft_ok, total = eval_adapter(
        args.base, sft_path, tok, ds, max_new_tokens=args.max_new_tokens
    )
    grpo_ok, _ = eval_adapter(
        args.base, grpo_path, tok, ds, max_new_tokens=args.max_new_tokens
    )

    def pct(x: int, n: int) -> float:
        return 100.0 * x / max(n, 1)

    print("\n=== RESULTS ===")
    print(f"SFT : {sft_ok}/{total} ({pct(sft_ok,total):.1f}%)")
    print(f"GRPO: {grpo_ok}/{total} ({pct(grpo_ok,total):.1f}%)")
    print(f"Δ    : {(grpo_ok - sft_ok):+d} ({pct(grpo_ok,total)-pct(sft_ok,total):+.1f} pp)")


if __name__ == "__main__":
    main()
