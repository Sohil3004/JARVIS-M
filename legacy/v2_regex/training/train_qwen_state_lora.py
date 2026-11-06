"""
JARVIS-M v2: Fine-Tuning Qwen-2.5-0.5B with LoRA on MultiWOZ
============================================================
Trains a lightweight, neural Epistemic State & Overwrite Reasoner using
Hugging Face PEFT + LoRA on Cambridge MultiWOZ Dialogue State Tracking.

Model: Qwen/Qwen2.5-0.5B-Instruct
Dataset: pietrolesci/multiwoz_all_versions (MultiWOZ 2.2 ground-truth states)
Target: Direct JSON state & belief extraction with sub-millisecond execution.
"""

import sys
import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "pietrolesci/multiwoz_all_versions"
CACHE_DIR = "./cache"
OUTPUT_DIR = "./models/jarvis-qwen-state-lora"


def format_state_target(raw_state: dict) -> str:
    """Filters None values and formats active state slots into compact JSON."""
    active_slots = {}
    if not raw_state:
        return "{}"
    for k, v in raw_state.items():
        if v is not None and len(v) > 0:
            val_str = v[0] if isinstance(v, list) else str(v)
            if val_str.lower() != "none" and val_str.strip():
                active_slots[k] = val_str
    return json.dumps(active_slots)


def prepare_multiwoz_dataset(tokenizer, max_samples: int = 3000):
    print(f"\n[*] Loading MultiWOZ 2.2 dataset from {DATASET_NAME} (max {max_samples} samples)...")
    raw_dataset = load_dataset(DATASET_NAME, "default", split="train", cache_dir=CACHE_DIR)

    # Filter turns that have meaningful active state slots
    filtered_examples = []
    print("[*] Extracting and filtering turns with active belief states...")
    for ex in raw_dataset:
        state_json = format_state_target(ex.get("states_22", {}))
        if state_json != "{}":
            usr_utt = ex.get("usr_utt", "").strip()
            sys_utt = ex.get("sys_utt", "").strip()
            if len(usr_utt) > 5:
                filtered_examples.append({
                    "usr_utt": usr_utt,
                    "sys_utt": sys_utt,
                    "state_target": state_json,
                })
        if len(filtered_examples) >= max_samples:
            break

    print(f"[+] Prepared {len(filtered_examples)} high-value state-tracking training turns.")

    # Format using Qwen Chat Template
    formatted_texts = []
    for ex in filtered_examples:
        sys_context = f"System: {ex['sys_utt']}\n" if ex['sys_utt'] and ex['sys_utt'] != "none" else ""
        messages = [
            {
                "role": "system",
                "content": "You are a Neural Dialogue State Tracker. Extract all active entity slots and verified state values from the conversation turn into compact JSON."
            },
            {
                "role": "user",
                "content": f"{sys_context}User: {ex['usr_utt']}"
            },
            {
                "role": "assistant",
                "content": ex['state_target']
            }
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_texts.append(chat_text)

    # Tokenize
    print("[*] Tokenizing training samples...")
    encodings = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=256,
        padding=False,
    )

    class QwenStateDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            # For CausalLM training, labels are copies of input_ids
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.input_ids[idx],
            }

    return QwenStateDataset(encodings)


def train_qwen_state_tracker(num_epochs: int = 1, max_samples: int = 2500, batch_size: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("🚀 JARVIS-M v2: FINE-TUNING QWEN-2.5-0.5B WITH LoRA ON MULTIWOZ")
    print("=" * 70)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Tokenizer & Base Model
    print(f"\n[*] Loading Tokenizer & Model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # 2. Configure LoRA for Qwen 2.5
    print("\n[*] Applying LoRA configuration to Qwen 2.5 attention and MLP layers...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"  Trainable Parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # 3. Load & Process MultiWOZ Data
    train_dataset = prepare_multiwoz_dataset(tokenizer, max_samples=max_samples)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        fp16=(device == "cuda"),
        logging_steps=25,
        save_strategy="no",  # Save final model at end
        report_to="none",
        dataloader_num_workers=0,  # Avoid Windows multiprocessing worker locks
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 5. Execute Training
    print("\n[*] Starting LoRA Training Run on CUDA...")
    train_result = trainer.train()

    # 6. Save Adapter
    print(f"\n[+] Saving Fine-Tuned LoRA Adapter to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print(f"LoRA Adapter saved successfully at: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 on MultiWOZ")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=2000, help="Number of MultiWOZ training samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    args = parser.parse_args()

    train_qwen_state_tracker(num_epochs=args.epochs, max_samples=args.samples, batch_size=args.batch_size)
