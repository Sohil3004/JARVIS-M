"""
JARVIS-M v2: Multi-Task LoRA Fine-Tuning for Qwen-2.5-0.5B
==========================================================
Trains a single unified LoRA adapter on Qwen-2.5-0.5B that simultaneously masters:
1. Task 1: Neural Epistemic State & Overwrite Tracking (Cambridge MultiWOZ 2.2)
2. Task 2: Abstractive Narrative Dialogue Summarization (DialogSum)

Outcome:
- One single lightweight model (<1 GB VRAM)
- Dual output modalities: Structured JSON State Feed (for agents/databases)
  AND Narrative Executive Summary (for human users).
"""

import sys
import os
import json
import random
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
MULTIWOZ_DATASET = "pietrolesci/multiwoz_all_versions"
DIALOGSUM_DATASET = "knkarthick/dialogsum"
CACHE_DIR = "./cache"
OUTPUT_DIR = "./models/jarvis-qwen-multitask-lora"


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


def build_multitask_dataset(tokenizer, num_state_samples: int = 1500, num_summary_samples: int = 750):
    all_formatted_texts = []

    # ------------------------------------------------------------
    # Stream 1: MultiWOZ Epistemic State Tracking
    # ------------------------------------------------------------
    print(f"\n[*] Loading MultiWOZ 2.2 state samples (target: {num_state_samples})...")
    mwoz_raw = load_dataset(MULTIWOZ_DATASET, "default", split="train", cache_dir=CACHE_DIR)
    state_count = 0

    for ex in mwoz_raw:
        state_json = format_state_target(ex.get("states_22", {}))
        if state_json != "{}":
            usr_utt = ex.get("usr_utt", "").strip()
            sys_utt = ex.get("sys_utt", "").strip()
            if len(usr_utt) > 5:
                sys_context = f"System: {sys_utt}\n" if sys_utt and sys_utt != "none" else ""
                messages = [
                    {
                        "role": "system",
                        "content": "You are a Neural Dialogue State Tracker. Extract all active entity slots and verified state values from the conversation turn into compact JSON."
                    },
                    {
                        "role": "user",
                        "content": f"{sys_context}User: {usr_utt}"
                    },
                    {
                        "role": "assistant",
                        "content": state_json
                    }
                ]
                all_formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
                state_count += 1
        if state_count >= num_state_samples:
            break

    print(f"[+] Ingested {state_count} state-tracking training turns.")

    # ------------------------------------------------------------
    # Stream 2: DialogSum Narrative Dialogue Summarization
    # ------------------------------------------------------------
    print(f"\n[*] Loading DialogSum summarization samples (target: {num_summary_samples})...")
    dsum_raw = load_dataset(DIALOGSUM_DATASET, split="train", cache_dir=CACHE_DIR)
    summary_count = 0

    for ex in dsum_raw:
        dialogue = ex.get("dialogue", "").strip()
        summary = ex.get("summary", "").strip()
        if len(dialogue) > 20 and len(summary) > 10:
            messages = [
                {
                    "role": "system",
                    "content": "You are an Abstractive Dialogue Summarizer. Provide a concise, coherent narrative summary capturing all key action items, speaker decisions, and deadlines."
                },
                {
                    "role": "user",
                    "content": f"Dialogue Transcript:\n{dialogue}\n\nConcise Summary:"
                },
                {
                    "role": "assistant",
                    "content": summary
                }
            ]
            all_formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
            summary_count += 1
        if summary_count >= num_summary_samples:
            break

    print(f"[+] Ingested {summary_count} narrative summarization pairs.")

    # Shuffle the multi-task streams to prevent catastrophic forgetting
    random.seed(42)
    random.shuffle(all_formatted_texts)
    print(f"[+] Total Blended Multi-Task Training Samples: {len(all_formatted_texts)}")

    # Tokenize
    print("[*] Tokenizing blended multi-task dataset...")
    encodings = tokenizer(
        all_formatted_texts,
        truncation=True,
        max_length=384,
        padding=False,
    )

    class MultiTaskDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.input_ids[idx],
            }

    return MultiTaskDataset(encodings)


def train_multitask_qwen(epochs: int = 1, state_samples: int = 1500, summary_samples: int = 750, batch_size: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("🚀 JARVIS-M v2: MULTI-TASK LoRA TRAINING ON QWEN-2.5-0.5B")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Tokenizer & Base Qwen Model
    print(f"\n[*] Loading Tokenizer & Model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # 2. Configure Unified LoRA
    print("\n[*] Initializing LoRA configuration for unified multi-task adaptation...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable Parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # 3. Build Blended Dataset
    train_dataset = build_multitask_dataset(tokenizer, num_state_samples=state_samples, num_summary_samples=summary_samples)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        num_train_epochs=epochs,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        fp16=(device == "cuda"),
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
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
    print("\n[*] Starting Unified Multi-Task LoRA Training Run on CUDA...")
    train_result = trainer.train()

    # 6. Save Model
    print(f"\n[+] Saving Unified Multi-Task LoRA Adapter to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\n" + "=" * 80)
    print("✅ MULTI-TASK TRAINING COMPLETE!")
    print(f"Unified Adapter saved at: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task LoRA Training on Qwen 2.5")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--state_samples", type=int, default=1000, help="Number of MultiWOZ samples")
    parser.add_argument("--summary_samples", type=int, default=500, help="Number of DialogSum samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    args = parser.parse_args()

    train_multitask_qwen(
        epochs=args.epochs,
        state_samples=args.state_samples,
        summary_samples=args.summary_samples,
        batch_size=args.batch_size
    )
