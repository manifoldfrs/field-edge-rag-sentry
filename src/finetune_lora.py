"""
Very simple finetuning of a LoRA adapter on a small base model (TinyLlama-1.1B)
so the full pipeline can prove PEFT works on-device.

----------------
Create 50 synthetic Q-A training samples from existing doctrine snippets and
fine-tune a LoRA adapter on a small base model (TinyLlama-1.1B) so the full
pipeline can prove PEFT works on-device.

Outputs:
    models/lora_out/adapter_config.json  (and related files)

This is a *demo*; swap BASE_MODEL to your 7B HF checkpoint if you have RAM/GPU.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.retriever import SNIPPET_PATH, retrieve

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
OUT_DIR = Path("models/lora_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(0)


def make_synthetic_pairs(snippets: list[str], n: int = 50):
    pairs = []
    for _ in range(n):
        context = random.choice(snippets)
        question = f'What does the following passage mean? "{context[:80]}..."'
        # TODO: create actual answers and better questions
        answer = context  # identity answer for demo
        pairs.append({"instruction": question, "output": answer})
    return pairs


def tokenize(example, tokenizer):
    prompt = f"<s>[INST] {example['instruction']} [/INST] {example['output']} </s>"
    tok = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    tok["labels"] = tok["input_ids"].clone()
    return tok


def main() -> None:
    snippets = SNIPPET_PATH.read_bytes() and None  # ensure index built

    with open(SNIPPET_PATH, "rb") as f:
        snippets = pickle.load(f)

    data = make_synthetic_pairs(snippets)
    ds = Dataset.from_list(data)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = ds.map(
        tokenize, remove_columns=ds.column_names, fn_kwargs={"tokenizer": tokenizer}
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"[DONE] LoRA adapter saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
