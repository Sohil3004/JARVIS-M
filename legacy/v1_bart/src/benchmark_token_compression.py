"""
JARVIS-M v2: Comprehensive Token & Context Optimization Benchmark
=================================================================
Benchmarks multiple compression strategies across token budgets:
1. Naive Head/Tail Truncation
2. Extractive Saliency Pruning
3. Jarvis-M Abstractive LoRA Distillation

Metrics:
- Token Compression Ratio & Savings (%)
- Semantic Retention (Sentence-BERT Cosine Similarity)
- ROUGE-1, ROUGE-2, ROUGE-L against human reference
- Latency (ms/sample)
"""

import os
import sys
import time
import argparse
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from context_optimizer import ContextOptimizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATASET_NAME = "knkarthick/dialogsum"
CACHE_DIR = "./cache"

def evaluate_compression_methods(
    optimizer: ContextOptimizer,
    dialogues: list,
    references: list,
    budget_ratios: list = [0.3, 0.5, 0.7],
    num_samples: int = 50,
):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    sample_dialogues = dialogues[:num_samples]
    sample_references = references[:num_samples]

    print(f"\n[*] Running Token Compression Benchmark on {len(sample_dialogues)} dialogues...")
    print(f"[*] Token Budgets Tested: {[f'{int(b*100)}%' for b in budget_ratios]}\n")

    results = {}

    for budget in budget_ratios:
        budget_label = f"{int(budget * 100)}%"
        results[budget_label] = {}

        # ------------------------------------------------------------
        # 1. Baseline: Naive Truncation (First N tokens)
        # ------------------------------------------------------------
        trunc_r1, trunc_r2, trunc_rL, trunc_sim, trunc_comp, trunc_lat = [], [], [], [], [], []
        for d, ref in zip(sample_dialogues, sample_references):
            t0 = time.perf_counter()
            tokens = optimizer.tokenizer.encode(d, add_special_tokens=False)
            target_len = max(15, int(len(tokens) * budget))
            truncated_tokens = tokens[:target_len]
            truncated_text = optimizer.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            lat = (time.perf_counter() - t0) * 1000

            scores = scorer.score(ref, truncated_text)
            trunc_r1.append(scores['rouge1'].fmeasure * 100)
            trunc_r2.append(scores['rouge2'].fmeasure * 100)
            trunc_rL.append(scores['rougeL'].fmeasure * 100)
            trunc_comp.append(len(truncated_tokens) / max(1, len(tokens)))
            trunc_lat.append(lat)

            # Cosine similarity with original
            embs = optimizer.embedder.encode([d, truncated_text], convert_to_numpy=True, normalize_embeddings=True)
            trunc_sim.append(float(np.dot(embs[0], embs[1])))

        results[budget_label]["Naive Truncation"] = {
            "ROUGE-1": np.mean(trunc_r1),
            "ROUGE-2": np.mean(trunc_r2),
            "ROUGE-L": np.mean(trunc_rL),
            "Semantic Sim": np.mean(trunc_sim),
            "Compression": np.mean(trunc_comp),
            "Latency (ms)": np.mean(trunc_lat),
        }

        # ------------------------------------------------------------
        # 2. Extractive Saliency Pruning
        # ------------------------------------------------------------
        ext_r1, ext_r2, ext_rL, ext_sim, ext_comp, ext_lat = [], [], [], [], [], []
        for d, ref in zip(sample_dialogues, sample_references):
            compressed_text, stats = optimizer.compress_extractive(d, budget_ratio=budget)
            scores = scorer.score(ref, compressed_text)

            ext_r1.append(scores['rouge1'].fmeasure * 100)
            ext_r2.append(scores['rouge2'].fmeasure * 100)
            ext_rL.append(scores['rougeL'].fmeasure * 100)
            ext_comp.append(stats["compression_ratio"])
            ext_lat.append(stats["latency_ms"])

            embs = optimizer.embedder.encode([d, compressed_text], convert_to_numpy=True, normalize_embeddings=True)
            ext_sim.append(float(np.dot(embs[0], embs[1])))

        results[budget_label]["Extractive Saliency"] = {
            "ROUGE-1": np.mean(ext_r1),
            "ROUGE-2": np.mean(ext_r2),
            "ROUGE-L": np.mean(ext_rL),
            "Semantic Sim": np.mean(ext_sim),
            "Compression": np.mean(ext_comp),
            "Latency (ms)": np.mean(ext_lat),
        }

        # ------------------------------------------------------------
        # 3. Jarvis-M Abstractive LoRA Distillation
        # ------------------------------------------------------------
        abs_r1, abs_r2, abs_rL, abs_sim, abs_comp, abs_lat = [], [], [], [], [], []
        for d, ref in zip(sample_dialogues, sample_references):
            compressed_text, stats = optimizer.compress_abstractive(d, budget_ratio=budget)
            scores = scorer.score(ref, compressed_text)

            abs_r1.append(scores['rouge1'].fmeasure * 100)
            abs_r2.append(scores['rouge2'].fmeasure * 100)
            abs_rL.append(scores['rougeL'].fmeasure * 100)
            abs_comp.append(stats["compression_ratio"])
            abs_lat.append(stats["latency_ms"])

            embs = optimizer.embedder.encode([d, compressed_text], convert_to_numpy=True, normalize_embeddings=True)
            abs_sim.append(float(np.dot(embs[0], embs[1])))

        results[budget_label]["Jarvis-M LoRA Distill"] = {
            "ROUGE-1": np.mean(abs_r1),
            "ROUGE-2": np.mean(abs_r2),
            "ROUGE-L": np.mean(abs_rL),
            "Semantic Sim": np.mean(abs_sim),
            "Compression": np.mean(abs_comp),
            "Latency (ms)": np.mean(abs_lat),
        }

    # ------------------------------------------------------------
    # Display Formatted Benchmark Summary
    # ------------------------------------------------------------
    print("=" * 80)
    print("JARVIS-M v2: TOKEN OPTIMIZATION BENCHMARK SUMMARY")
    print("=" * 80)

    for budget_label, methods in results.items():
        print(f"\n[+] [Target Token Budget: {budget_label}]")
        print(f"{'Method':<24} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8} | {'Sem Sim':<8} | {'Comp Ratio':<10} | {'Latency':<10}")
        print("-" * 88)
        for method_name, metrics in methods.items():
            print(
                f"{method_name:<24} | "
                f"{metrics['ROUGE-1']:<8.2f} | "
                f"{metrics['ROUGE-2']:<8.2f} | "
                f"{metrics['ROUGE-L']:<8.2f} | "
                f"{metrics['Semantic Sim']:<8.3f} | "
                f"{metrics['Compression']:<10.2f} | "
                f"{metrics['Latency (ms)']:<7.1f} ms"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Jarvis-M v2 Token Compression Benchmark")
    parser.add_argument("--samples", type=int, default=15, help="Number of test dialogues to evaluate")
    args = parser.parse_args()

    # Load dataset
    print("[*] Loading DialogSum test set...")
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    dialogues = dataset["test"]["dialogue"]
    references = dataset["test"]["summary"]

    optimizer = ContextOptimizer(load_generator=True)
    evaluate_compression_methods(optimizer, dialogues, references, budget_ratios=[0.3, 0.5, 0.7], num_samples=args.samples)


if __name__ == "__main__":
    main()
