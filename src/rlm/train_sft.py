import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from config import LORA_PARAMETERS

# TODO: Configuración del modelo y dataset
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # O un modelo más pequeño si es necesario
DATASET_NAME = "gsm8k"
# DATASET_NAME = "nvidia/OpenMathInstruct-1"
OUTPUT_DIR = "./weights/sft_lora_gsm8k"

def formatting_prompts_func_gsm8k(example):
    # Format the dataset for SFT training
    # The objective is to create a string that contains:
    # User prompt + Start of thinking (<think>) + Reasoning + End of thinking (</think>) + Final answer
    
    reasoning, answer = example["answer"].split("####")
    reasoning = reasoning.strip()
    answer = answer.strip()

    text = f"""
    USER:{example["question"]}
    ASSISTANT: <think> {reasoning} </think>
    Final answer: {answer}
    """
    
    return text

def formatting_prompts_func_openmathinstruct(example):
    # output_texts = []
    # TODO: Implementar la lógica para formatear el dataset.
    # El objetivo es crear un string que contenga:
    # Prompt del usuario + Inicio de pensamiento (<think>) + Razonamiento + Fin de pensamiento (</think>) + Respuesta final.
    # Ejemplo conceptual:
    # text = f"User: {example['question']}\nAssistant: <think>{example['reasoning']}</think> La respuesta es {example['answer']}"

    answer = example["expected_answer"]
    reasoning = example["generated_solution"]
    reasoning = reasoning.strip()
    answer = answer.strip()

    text = f"""
    USER: {example["question"]}
    ASSISTANT: <think> {reasoning} </think>
    Final answer: {answer}
    """
    return text

def train():
    # 1. Cargar Modelo y Tokenizer (usar cuantización 4bit/8bit si es necesario)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto", # Cambiar para cuantización
        device_map="auto"
    )

    # 2. Configurar LoRA
    peft_config = LoraConfig(**LORA_PARAMETERS)

    # 3. Cargar Dataset
    dataset = load_dataset(DATASET_NAME, "main", split="train")
    # dataset = load_dataset(DATASET_NAME, split="train")
    # dataset = dataset.filter(lambda x: x["is_correct"] is True)
    # dataset = dataset.select(range(10000))  

    # 4. Configurar Entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        # ... otros argumentos
    )

    # 5. Inicializar SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func_gsm8k,
    )
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     peft_config=peft_config,
    #     processing_class=tokenizer,
    #     args=training_args,
    #     formatting_func=formatting_prompts_func_openmathinstruct,
    # )

    # 6. Entrenar y guardar
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("SFT Training finished (TODO: Implement)")

if __name__ == "__main__":
    train()
