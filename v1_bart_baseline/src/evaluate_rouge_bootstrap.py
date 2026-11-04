import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rouge_score import rouge_scorer

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "facebook/bart-large-cnn"
DATASET_NAME = "knkarthick/dialogsum"
LORA_ADAPTER_PATH = "./jarvis-bart-lora"
CACHE_DIR = "./cache"
N_ITERATIONS = 10000  

def bootstrap_rouge_evaluation(predictions, references, n_iterations=N_ITERATIONS, confidence_level=0.95):
    """
    Performs bootstrap resampling to compute confidence intervals for ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    boot_rouge1, boot_rouge2, boot_rougeL = [], [], []
    n = len(predictions)
    
    print(f"Running bootstrap resampling with {n_iterations} iterations on {n} samples...")
    for _ in range(n_iterations):
        # Sample indices with replacement
        indices = np.random.randint(0, n, size=n)
        sample_preds = [predictions[i] for i in indices]
        samples_refs = [references[i] for i in indices]
        
        r1, r2, rL = [], [], []
        for pred, ref in zip(sample_preds, samples_refs):
            scores = scorer.score(ref, pred)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rL.append(scores['rougeL'].fmeasure)
            
        boot_rouge1.append(np.mean(r1) * 100)
        boot_rouge2.append(np.mean(r2) * 100)
        boot_rougeL.append(np.mean(rL) * 100)
        
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    results = {
        "ROUGE-1": (np.mean(boot_rouge1), np.percentile(boot_rouge1, lower_percentile), np.percentile(boot_rouge1, upper_percentile)),
        "ROUGE-2": (np.mean(boot_rouge2), np.percentile(boot_rouge2, lower_percentile), np.percentile(boot_rouge2, upper_percentile)),
        "ROUGE-L": (np.mean(boot_rougeL), np.percentile(boot_rougeL, lower_percentile), np.percentile(boot_rougeL, upper_percentile)),
    }
    return results

def evaluate_factuality(sources, summaries):
    """
    Uses a lightweight NLI model to check if summaries are entailed by the source texts.
    Acts as a proxy for hallucination detection.
    """
    print("Loading NLI model for factuality check...")
    nli_pipeline = pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli")
    
    entailment_scores = []
    for source, summary in zip(sources, summaries):
        truncated_source = source[:1000]
        result = nli_pipeline(f"Premise: {truncated_source} Hypothesis: {summary}", truncation=True)
        score = next((item['score'] for item in result if 'entailment' in item['label'].lower()), 0.0)
        entailment_scores.append(score)
        
    return np.mean(entailment_scores)

if __name__ == "__main__":
    print("=" * 60)
    print("JARVIS-M: Bootstrap ROUGE & Factuality Evaluation on DialogSum")
    print("=" * 60)
    
    # 1. Load the real DialogSum test split from Hugging Face
    print("\n📥 Loading DialogSum test split...")
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    test_data = dataset["test"]  # Can slice e.g., test_dataSelect[:100] for rapid debugging
    
    sources = test_data["dialogue"]
    references = test_data["summary"]
    
    # 2. Load Tokenizer & Model (Base or Fine-tuned LoRA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📦 Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    if os.path.exists(LORA_ADAPTER_PATH):
        print(f"🔄 Loading fine-tuned LoRA adapter from {LORA_ADAPTER_PATH}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
    
    model.to(device)
    model.eval()
    
    # 3. Generate predictions on the DialogSum dataset
    print("\n🔄 Generating predictions across test set...")
    predictions = []
    
    # Batch generation for efficiency
    batch_size = 16
    for i in range(0, len(sources), batch_size):
        batch_dialogues = sources[i:i + batch_size]
        inputs = ["summarize: " + d for d in batch_dialogues]
        
        encoded = tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=128,
                min_length=20,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                num_beams=4,
                early_stopping=True
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded])
        
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(sources):
            print(f"  Processed {min(i + batch_size, len(sources))}/{len(sources)} samples...")

    # 4. Run Bootstrap Evaluation
    print("\n")
    ci_results = bootstrap_rouge_evaluation(predictions, references)
    print("\n--- Bootstrap Results (Mean ± CI bounds) ---")
    for metric, (mean, lower, upper) in ci_results.items():
        ci_range = (upper - lower) / 2
        print(f"{metric}: {mean:.2f} ± {ci_range:.2f} (95% CI: [{lower:.2f}, {upper:.2f}])")
        
    # 5. Run Factuality Check (Optional / Subset for speed)
    print("\nRunning NLI factuality check on a sample subset...")
    subset_size = min(50, len(sources))  # Adjust size as needed
    mean_entailment = evaluate_factuality(sources[:subset_size], predictions[:subset_size])
    print(f"\nMean NLI Entailment Score (Factuality Proxy over {subset_size} samples): {mean_entailment:.4f}")