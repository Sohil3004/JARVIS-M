"""
JARVIS-M: Fine-Tuning Script for BART-large-CNN on DialogSum
=============================================================
This script fine-tunes facebook/bart-large-cnn on the knkarthick/dialogsum 
dataset using LoRA (Low-Rank Adaptation) for efficient single-GPU training.

Goal: Improve ROUGE scores by adapting the model to dialogue summarization.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "facebook/bart-large-cnn"
DATASET_NAME = "knkarthick/dialogsum"
OUTPUT_DIR = "./models/jarvis-bart-lora"
CACHE_DIR = "./cache"

# Training hyperparameters (optimized for single GPU)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128

# LoRA configuration
LORA_R = 16  # Rank
LORA_ALPHA = 32  # Scaling factor
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]  # Attention layers


def setup_device():
    """Setup and return the appropriate device for training."""
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        cc = torch.cuda.get_device_capability()
        print(f"  Compute Capability: sm_{cc[0]}{cc[1]}")
        return True
    else:
        print("⚠ No GPU available, using CPU (training will be slow)")
        return False


def load_and_prepare_dataset():
    """Load the entire DialogSum dataset from Hugging Face."""
    print("\n📥 Loading DialogSum dataset...")
    
    # Load the complete dataset
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    return dataset


def preprocess_function(examples, tokenizer):
    """Preprocess dialogues and summaries for training."""
    # Prefix for dialogue summarization task
    prefix = "summarize: "
    
    # Prepare inputs (dialogues)
    inputs = [prefix + dialogue for dialogue in examples["dialogue"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    # Tokenize targets (summaries)
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    # Replace padding token id with -100 for loss computation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def setup_lora_model(model):
    """Apply LoRA configuration to the model."""
    print("\n🔧 Applying LoRA configuration...")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    
    return model


def compute_metrics(eval_pred, tokenizer, rouge_metric):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    predictions, labels = eval_pred
    
    # Decode predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode to text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Add newlines for ROUGE (sentence-level)
    decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split()) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    
    # Extract mid-F1 scores
    result = {
        "rouge1": result["rouge1"] * 100,
        "rouge2": result["rouge2"] * 100,
        "rougeL": result["rougeL"] * 100,
    }
    
    # Calculate average
    result["rouge_avg"] = (result["rouge1"] + result["rouge2"] + result["rougeL"]) / 3
    
    return {k: round(v, 4) for k, v in result.items()}


def main():
    """Main training function."""
    print("=" * 60)
    print("JARVIS-M: Fine-Tuning BART-large-CNN on DialogSum with LoRA")
    print("=" * 60)
    
    # Setup device
    use_cuda = setup_device()
    
    # Load dataset
    dataset = load_and_prepare_dataset()
    
    # Load tokenizer and model
    print("\n📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    # Apply LoRA
    model = setup_lora_model(model)
    
    # Preprocess dataset
    print("\n🔄 Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    
    # Load ROUGE metric
    print("\n📊 Loading ROUGE metric...")
    rouge_metric = evaluate.load("rouge")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=False,  # Disabled for CPU/unsupported GPU training
        no_cuda=(not use_cuda),  # Force CPU mode if GPU not supported
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_avg",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard for simplicity
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge_metric),
    )
    
    # Train
    print("\n🚀 Starting training...")
    print("-" * 60)
    train_result = trainer.train()
    
    # Save the final model
    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation on test set
    print("\n📈 Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📁 Model saved to: {OUTPUT_DIR}")
    print("\nFinal Test Results:")
    print(f"  ROUGE-1: {test_results.get('eval_rouge1', 'N/A'):.2f}")
    print(f"  ROUGE-2: {test_results.get('eval_rouge2', 'N/A'):.2f}")
    print(f"  ROUGE-L: {test_results.get('eval_rougeL', 'N/A'):.2f}")
    print(f"  Average: {test_results.get('eval_rouge_avg', 'N/A'):.2f}")
    
    return trainer, test_results


if __name__ == "__main__":
    main()
