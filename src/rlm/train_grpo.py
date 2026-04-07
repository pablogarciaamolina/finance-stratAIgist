import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
import re
from dataclasses import dataclass
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader

# TODO: Configuración
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
"""
SFT_ADAPTER_PATH = "./weights/sft_lora_gsm8k" # Ruta al modelo de la Fase 1, Parte 1
OUTPUT_DIR = "./weights/final_rlm_lora"
"""
def resolve_adapter_path(primary, fallback):
    if os.path.exists(os.path.join(primary, "adapter_config.json")):
        return primary
    if os.path.exists(os.path.join(fallback, "adapter_config.json")):
        return fallback
    raise FileNotFoundError(f"No adapter_config.json in {primary} nor {fallback}")

SFT_ADAPTER_PATH = resolve_adapter_path("/app/rlm/weights/sft_lora_gsm8k",
                                       "/app/weights/sft_lora_gsm8k")
OUTPUT_DIR = "/app/rlm/weights/final_rlm_lora"

GRPO_GROUP_SIZE = 4 # N respuestas por pregunta


"""
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
"""
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"


# GRPO
GRPO_GROUP_SIZE = 4          # N respuestas por pregunta
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 256
EPS = 1e-8

# Entrenamiento
NUM_EPOCHS = 1
BATCH_SIZE = 1               # IMPORTANTE: GRPO suele ir B=1 y grupo N
LR = 1e-5
GRAD_CLIP = 1.0
GRAD_ACCUM_STEPS = 1
LOG_EVERY = 25 # 25
MAX_STEPS = 500 #500              # para acotar; pon None si quieres recorrer todo
SEED = 42

SYSTEM_PROMPT = ""  # puedes dejarlo vacío o meter un system prompt fijo

torch.manual_seed(SEED)

# -----------------------------
# Utilidades
# -----------------------------
def build_prompt(question: str) -> str:
    # Estilo consistente con tu SFT (USER / ASSISTANT + <think>)
    # OJO: en GRPO NO damos la respuesta al prompt, solo la pregunta.
    if SYSTEM_PROMPT:
        return f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {question}\nASSISTANT:"
    return f"USER: {question}\nASSISTANT:"

def normalize_number_str(s: str) -> str:
    # Quita separadores comunes: comas, espacios, $
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("€", "")
    return s

def extract_final_number(text: str) -> str | None:
    """
    Intenta extraer la "respuesta final" como número.
    Priorizamos patrones del estilo: "Final answer: X"
    y si no, el último número del texto.
    """
    # 1) "Final answer: ..."
    m = re.search(r"Final answer:\s*([-+]?\d+(\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return normalize_number_str(m.group(1))

    # 2) "#### X" (a veces aparece si el modelo imita GSM8K)
    m = re.search(r"####\s*([-+]?\d+(\.\d+)?)", text)
    if m:
        return normalize_number_str(m.group(1))

    # 3) último número del texto
    nums = re.findall(r"[-+]?\d+(\.\d+)?", text)
    if not nums:
        return None
    # re.findall con grupo puede devolver cosas raras; usamos una versión segura:
    nums2 = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not nums2:
        return None
    return normalize_number_str(nums2[-1])

def parse_gsm8k_answer(answer_field: str) -> str:
    """
    GSM8K: answer viene como ".... #### 42"
    """
    if "####" in answer_field:
        gt = answer_field.split("####")[-1].strip()
    else:
        gt = answer_field.strip()
    return normalize_number_str(gt)

# TODO: Implementar función de recompensa
def reward_function(generated_text, ground_truth_answer):
    """
    Analiza el texto generado, extrae el número final y lo compara con la respuesta real.
    Devuelve 1.0 si es correcto, 0.0 si no.
    """
    # 1. Extraer la respuesta numérica del generated_text (regex suele ser útil)
    # 2. Comparar con ground_truth_answer
    pred = extract_final_number(generated_text)
    if pred is None:
        return 0.0
    gt = normalize_number_str(ground_truth_answer)

    # Igualdad exacta numérica (int/float) con tolerancia mínima
    try:
        pred_f = float(pred)
        gt_f = float(gt)
        return 1.0 if abs(pred_f - gt_f) < 1e-6 else 0.0
    except Exception:
        # fallback string match
        return 1.0 if pred == gt else 0.0

def compute_logprob_of_generated_tokens(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Calcula sum(log p(token_t | context)) SOLO para tokens generados (no prompt).
    input_ids: [1, seq_len] (prompt + generated)
    prompt_len: longitud del prompt (n_tokens)
    Devuelve escalar tensor (sum logprobs de tokens generados).
    """
    # Teacher forcing: el modelo predice el token i dado tokens < i
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [1, seq_len, vocab]

    # Para token t (posición t), la prob viene de logits en t-1.
    # Así que para los tokens generados que empiezan en prompt_len,
    # sus logits están en posiciones [prompt_len-1 .. seq_len-2]
    seq_len = input_ids.size(1)
    if prompt_len >= seq_len:
        return torch.tensor(0.0, device=input_ids.device)

    # target tokens = input_ids[:, prompt_len:seq_len]
    target = input_ids[:, prompt_len:]  # [1, gen_len]

    # logits que predicen esos targets:
    pred_logits = logits[:, prompt_len - 1 : seq_len - 1, :]  # [1, gen_len, vocab]

    logprobs = torch.log_softmax(pred_logits, dim=-1)          # [1, gen_len, vocab]
    token_logprobs = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, gen_len]

    # suma sobre tokens generados
    return token_logprobs.sum()

# Dataset & collation
@dataclass
class Batch:
    questions: List[str]
    answers: List[str]

def collate_fn(examples) -> Batch:
    qs = [ex["question"] for ex in examples]
    ans = [parse_gsm8k_answer(ex["answer"]) for ex in examples]
    return Batch(questions=qs, answers=ans)


# TODO: Implementar el bucle de entrenamiento GRPO
def train_grpo():
    """
    # 1. Cargar modelo base y aplicarle el adaptador SFT
    # base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, ...)
    # model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    # tokenizer = ...
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Bucle principal de entrenamiento (pseudocódigo)
    # for epoch in range(epochs):
    #     for batch_questions, batch_answers in dataloader:
    #         
    #         # --- Paso 1: Sampling (Generación) ---
    #         # Para cada pregunta en el batch, generar GRPO_GROUP_SIZE respuestas usando sampling (do_sample=True, temperature>0)
    #         # generated_sequences = model.generate(..., num_return_sequences=GRPO_GROUP_SIZE)
    #
    #         # --- Paso 2: Scoring (Recompensa) ---
    #         # rewards = []
    #         # for seq in generated_sequences:
    #         #     r = reward_function(decode(seq), ground_truth)
    #         #     rewards.append(r)
    #         # rewards_tensor = torch.tensor(rewards)
    #
    #         # --- Paso 3: Cálculo de Ventaja GRPO ---
    #         # Calcular media y desviación estándar del grupo de recompensas
    #         # mean_reward = rewards_tensor.mean()
    #         # std_reward = rewards_tensor.std()
    #         # advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
    #
    #         # --- Paso 4: Actualización (Backprop) ---
    #         # Calcular log-probs de las secuencias generadas.
    #         # Loss es típicamente -log_prob * advantage (similar a REINFORCE/PPO)
    #         # loss.backward()
    #         # optimizer.step()
            
    # model.save_pretrained(OUTPUT_DIR)
    print("GRPO Training finished (TODO: Implement)")
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Modelo base + adaptador SFT (trainable)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    model.train()

    # Optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Dataset GSM8K (solo train)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    step = 0

    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            if MAX_STEPS is not None and step >= MAX_STEPS:
                break

            # GRPO se hace por pregunta (batch_size=1 recomendado)
            question = batch.questions[0]
            gt = batch.answers[0]
            prompt = build_prompt(question)

            # Tokeniza prompt una vez
            prompt_inputs = tokenizer(prompt, return_tensors="pt")
            prompt_input_ids = prompt_inputs["input_ids"].to(model.device)
            prompt_attn = prompt_inputs["attention_mask"].to(model.device)
            prompt_len = prompt_input_ids.size(1)

            # -------------------------------------------------
            # Paso 1: Sampling (N respuestas)
            # -------------------------------------------------
            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attn,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    num_return_sequences=GRPO_GROUP_SIZE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # gen_out: [N, prompt_len + gen_len] (cada una puede variar; HF devuelve padded en batch)
            # Para cada secuencia, decodifica y recompensa
            rewards = []
            decoded = []
            seq_tensors = []
            for i in range(gen_out.size(0)):
                seq = gen_out[i].unsqueeze(0).to(model.device)  # [1, seq_len]
                text = tokenizer.decode(seq[0], skip_special_tokens=False)
                r = reward_function(text, gt)
                rewards.append(r)
                decoded.append(text)
                seq_tensors.append(seq)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=model.device)  # [N]

            # -------------------------------------------------
            # Paso 2: Ventajas GRPO (normalización por grupo)
            # -------------------------------------------------
            mean_r = rewards_t.mean()
            std_r = rewards_t.std(unbiased=False)
            advantages = (rewards_t - mean_r) / (std_r + EPS)  # [N]

            # Si todas recompensas iguales (std ~ 0), ventajas ~ 0: no update. Esto es correcto.
            # Pero a veces quieres forzar un pequeño learning signal:
            # (lo dejamos "puro" y estable)

            # -------------------------------------------------
            # Paso 3: Update (REINFORCE con logprobs)
            # -------------------------------------------------
            # Loss = - (1/N) * sum_i [ adv_i * sum_logprob_i ]
            # NOTA: sum_logprob es negativo (log probs), así que si adv>0 => baja loss empujando a aumentar prob.
            total_loss = 0.0

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for i in range(GRPO_GROUP_SIZE):
                    seq = seq_tensors[i]

                    # attention_mask: 1 donde hay token (no padding)
                    attn_mask = (seq != tokenizer.pad_token_id).long()

                    # logprob SOLO de tokens generados
                    logp = compute_logprob_of_generated_tokens(
                        model=model,
                        input_ids=seq,
                        attention_mask=attn_mask,
                        prompt_len=prompt_len,
                    )

                    # REINFORCE loss (negativo)
                    total_loss = total_loss + (-advantages[i] * logp)

                loss = total_loss / GRPO_GROUP_SIZE
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # clip grad
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            if step % LOG_EVERY == 0:
                # imprime un resumen ligero
                best_idx = int(torch.argmax(rewards_t).item())
                print(
                    f"[epoch {epoch} | step {step}] "
                    f"meanR={mean_r.item():.3f} stdR={std_r.item():.3f} "
                    f"rewards={rewards} "
                    f"loss={loss.item():.4f}"
                )
                # opcional: ver la pred final del mejor sample
                pred_best = extract_final_number(decoded[best_idx])
                print(f"  GT={gt} | best_pred={pred_best} | best_reward={rewards[best_idx]}")

            step += 1

        if MAX_STEPS is not None and step >= MAX_STEPS:
            break

    # Guardado final (solo LoRA)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ GRPO Training finished. Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train_grpo()
