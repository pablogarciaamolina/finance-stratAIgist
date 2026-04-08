"""
Supervised Fine-Tuning (SFT) on GSM8K dataset.
Trains a LoRA adapter on Qwen2.5-7B-Instruct using <think>...</think> format.

Usage:
    python -m backend.models.training.train_sft
"""

import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from ..config import LORA_PARAMETERS

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "gsm8k"
OUTPUT_DIR = "./backend/models/weights/sft_lora_gsm8k"


def formatting_prompts_func_gsm8k(example):
    reasoning, answer = example["answer"].split("####")
    reasoning = reasoning.strip()
    answer = answer.strip()
    text = f"""
    USER:{example["question"]}
    ASSISTANT: <think> {reasoning} </think>
    Final answer: {answer}
    """
    return text


def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )

    peft_config = LoraConfig(**LORA_PARAMETERS)
    dataset = load_dataset(DATASET_NAME, "main", split="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func_gsm8k,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("SFT Training finished.")


if __name__ == "__main__":
    train()
