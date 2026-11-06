"""
JARVIS-M v2: Tier-1 Conference Comparative Benchmark Suite
===========================================================
Scientifically evaluates context optimization across multi-session dialogues:
1. Downstream Cross-Session QA / Fact Recall Accuracy
2. Semantic Cosine Similarity & ROUGE Scores
3. Factuality Entailment Score (NLI / Alignment Proxy)
4. Token Compression Ratio & Latency

Baselines Compared:
- Method 1: Naive Truncation (Head/Tail Window)
- Method 2: Extractive Saliency Pruning
- Method 3: Flat Single-Index RAG
- Method 4: Jarvis-M v2 (Hierarchical Memory Graph + Token Pruner)
"""

import sys
import os
import re
import time
import argparse
import numpy as np
from typing import List, Dict, Any
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from hierarchical_memory import HierarchicalMemoryGraph
from token_pruner import InformationTokenPruner
from context_optimizer import ContextOptimizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def generate_multisession_testbed() -> List[Dict[str, Any]]:
    """
    Curates realistic multi-session collaborative scenarios where context
    is dispersed across multiple asynchronous chat sessions.
    Each test sample has:
      - Sessions 1..3
      - Target Query
      - Ground Truth Answer / Fact Keywords
      - Reference Summary
    """
    return [
        {
            "id": "scenario_1_fintech_deployment",
            "sessions": [
                {
                    "session_id": "s1",
                    "user": "Alice",
                    "text": "Alice: Team, we must complete the database indexing before Friday. The manager specifically requested it in the morning briefing."
                },
                {
                    "session_id": "s1",
                    "user": "Bob",
                    "text": "Bob: I'm currently working on the Stripe payment integration. But the API key verification is failing with error 403."
                },
                {
                    "session_id": "s2",
                    "user": "Charlie",
                    "text": "Charlie: I reviewed the financial projections yesterday. The risk assessment section is missing from the proposal draft."
                },
                {
                    "session_id": "s2",
                    "user": "Alice",
                    "text": "Alice: Bob, did you resolve the payment error? We need the charts and financial numbers finalized for Thursday's dry run."
                },
                {
                    "session_id": "s3",
                    "user": "Bob",
                    "text": "Bob: I spoke with Stripe support. We received new production credentials. I'll finish the transaction tests by Wednesday evening."
                }
            ],
            "query": "What is the blocker on Stripe payments and when is the database deadline?",
            "ground_truth_keywords": ["403", "database", "Friday", "credentials", "Wednesday"],
            "qa_question": "What caused the payment delay and when is the database deadline?",
            "qa_answer": "Error 403 on API key verification; database deadline is Friday.",
            "reference_summary": "Alice confirmed database indexing is due Friday. Bob encountered error 403 on Stripe payment verification, then received new credentials to finish Wednesday. Charlie flagged missing risk assessment in financial projections."
        },
        {
            "id": "scenario_2_cloud_infrastructure",
            "sessions": [
                {
                    "session_id": "s1",
                    "user": "DevOps_Dan",
                    "text": "Dan: We noticed the Kubernetes worker nodes are running at 94% CPU utilization in the us-central1 cluster."
                },
                {
                    "session_id": "s1",
                    "user": "Elena",
                    "text": "Elena: I checked the telemetry logs. The Redis caching pod crashed due to an out-of-memory exception at 03:00 UTC."
                },
                {
                    "session_id": "s2",
                    "user": "Farhan",
                    "text": "Farhan: We decided to double the memory allocation to 16GB for Redis and migrate heavy queries to BigQuery."
                },
                {
                    "session_id": "s3",
                    "user": "Dan",
                    "text": "Dan: Memory increased to 16GB. CPU utilization dropped to 42%. We will conduct a load test on Saturday night."
                }
            ],
            "query": "Why did Redis crash and what was the resolution?",
            "ground_truth_keywords": ["out-of-memory", "16GB", "BigQuery", "crashed", "Saturday"],
            "qa_question": "What caused the Redis crash and how was it fixed?",
            "qa_answer": "Out-of-memory exception; resolved by increasing memory to 16GB and migrating queries to BigQuery.",
            "reference_summary": "Redis crashed due to out-of-memory exception causing high node CPU. Farhan and Dan resolved it by increasing memory to 16GB and migrating queries to BigQuery, with load testing scheduled for Saturday."
        },
        {
            "id": "scenario_3_mobile_app_release",
            "sessions": [
                {
                    "session_id": "s1",
                    "user": "Product_Priya",
                    "text": "Priya: Apple App Store rejected our release build 2.4.1 citing missing privacy nutrition labels for location tracking."
                },
                {
                    "session_id": "s1",
                    "user": "Rahul",
                    "text": "Rahul: Location tracking is only used in the delivery map module. I will update the Info.plist permission strings today."
                },
                {
                    "session_id": "s2",
                    "user": "Sneha",
                    "text": "Sneha: QA detected a UI freeze on Android 14 devices during guest checkout when biometric authentication is cancelled."
                },
                {
                    "session_id": "s3",
                    "user": "Priya",
                    "text": "Priya: Rahul's privacy fix passed review. Let's submit build 2.4.2 to Apple by Tuesday 5 PM."
                }
            ],
            "query": "Why was build 2.4.1 rejected and what is the target time for build 2.4.2?",
            "ground_truth_keywords": ["privacy nutrition", "location", "2.4.2", "Tuesday", "5 PM"],
            "qa_question": "Why did Apple reject the build and when will 2.4.2 be submitted?",
            "qa_answer": "Rejected due to missing privacy nutrition labels for location tracking; build 2.4.2 will be submitted by Tuesday 5 PM.",
            "reference_summary": "Apple rejected build 2.4.1 due to missing privacy nutrition labels for location tracking. Rahul updated permission strings, and Priya scheduled build 2.4.2 submission for Tuesday 5 PM."
        }
    ]


def evaluate_qa_recall(compressed_context: str, keywords: List[str]) -> float:
    """Calculates factual keyword recall percentage in compressed context."""
    text_lower = compressed_context.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return (matches / max(1, len(keywords))) * 100.0


def evaluate_entailment_proxy(premise: str, hypothesis: str, embedder: SentenceTransformer) -> float:
    """
    Computes a directional entailment score proxy:
    semantic alignment and coverage of hypothesis claims by compressed premise.
    """
    embs = embedder.encode([premise, hypothesis], convert_to_numpy=True, normalize_embeddings=True)
    cos_sim = float(np.dot(embs[0], embs[1]))
    # Length sanity penalty if hypothesis is hollow
    coverage = min(1.0, len(premise.split()) / max(1, len(hypothesis.split()) * 0.5))
    return round(cos_sim * coverage, 4)


def run_tier1_benchmark(budget_ratios: List[float] = [0.3, 0.5, 0.7]):
    testbed = generate_multisession_testbed()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    print("[*] Initializing Tier-1 Benchmark Engines...")
    memory_graph = HierarchicalMemoryGraph()
    pruner = InformationTokenPruner()
    context_opt = ContextOptimizer(load_generator=False)
    embedder = memory_graph.embedder

    print(f"\n[*] Evaluating {len(testbed)} Multi-Session Scenarios across Budgets: {[f'{int(b*100)}%' for b in budget_ratios]}\n")

    summary_results = {}

    for budget in budget_ratios:
        budget_label = f"{int(budget * 100)}%"
        summary_results[budget_label] = {}

        metrics = {
            "Naive Truncation": {"qa_recall": [], "r1": [], "rL": [], "sim": [], "comp": [], "lat": [], "entail": []},
            "Extractive Saliency": {"qa_recall": [], "r1": [], "rL": [], "sim": [], "comp": [], "lat": [], "entail": []},
            "Flat RAG": {"qa_recall": [], "r1": [], "rL": [], "sim": [], "comp": [], "lat": [], "entail": []},
            "Jarvis-M v2 (Hierarchical)": {"qa_recall": [], "r1": [], "rL": [], "sim": [], "comp": [], "lat": [], "entail": []},
        }

        for item in testbed:
            # Flatten raw dialogue
            all_turns_text = "\n".join([turn["text"] for turn in item["sessions"]])
            orig_tokens = pruner.count_tokens(all_turns_text)
            target_tokens = max(20, int(orig_tokens * budget))
            query = item["query"]
            keywords = item["ground_truth_keywords"]
            ref_summary = item["reference_summary"]

            # Reset and populate memory graph for this scenario (reusing embedder for speed)
            scenario_memory = HierarchicalMemoryGraph(embedder_name="all-MiniLM-L6-v2", cache_dir="./cache")
            scenario_memory.embedder = embedder
            for turn in item["sessions"]:
                scenario_memory.add_interaction(user=turn["user"], text=turn["text"], session_id=turn["session_id"])

            # ------------------------------------------------------------
            # Method 1: Naive Truncation (Head Window)
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            tokens = pruner.tokenizer.encode(all_turns_text, add_special_tokens=False)[:target_tokens]
            naive_text = pruner.tokenizer.decode(tokens, skip_special_tokens=True)
            naive_lat = (time.perf_counter() - t0) * 1000

            metrics["Naive Truncation"]["qa_recall"].append(evaluate_qa_recall(naive_text, keywords))
            r_naive = scorer.score(ref_summary, naive_text)
            metrics["Naive Truncation"]["r1"].append(r_naive['rouge1'].fmeasure * 100)
            metrics["Naive Truncation"]["rL"].append(r_naive['rougeL'].fmeasure * 100)
            embs = embedder.encode([all_turns_text, naive_text], convert_to_numpy=True, normalize_embeddings=True)
            metrics["Naive Truncation"]["sim"].append(float(np.dot(embs[0], embs[1])))
            metrics["Naive Truncation"]["comp"].append(len(tokens) / orig_tokens)
            metrics["Naive Truncation"]["lat"].append(naive_lat)
            metrics["Naive Truncation"]["entail"].append(evaluate_entailment_proxy(naive_text, ref_summary, embedder))

            # ------------------------------------------------------------
            # Method 2: Extractive Saliency Pruning
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            ext_text, ext_stats = context_opt.compress_extractive(all_turns_text, budget_ratio=budget, query=query)
            ext_lat = (time.perf_counter() - t0) * 1000

            metrics["Extractive Saliency"]["qa_recall"].append(evaluate_qa_recall(ext_text, keywords))
            r_ext = scorer.score(ref_summary, ext_text)
            metrics["Extractive Saliency"]["r1"].append(r_ext['rouge1'].fmeasure * 100)
            metrics["Extractive Saliency"]["rL"].append(r_ext['rougeL'].fmeasure * 100)
            embs = embedder.encode([all_turns_text, ext_text], convert_to_numpy=True, normalize_embeddings=True)
            metrics["Extractive Saliency"]["sim"].append(float(np.dot(embs[0], embs[1])))
            metrics["Extractive Saliency"]["comp"].append(ext_stats["compression_ratio"])
            metrics["Extractive Saliency"]["lat"].append(ext_lat)
            metrics["Extractive Saliency"]["entail"].append(evaluate_entailment_proxy(ext_text, ref_summary, embedder))

            # ------------------------------------------------------------
            # Method 3: Flat Single-Index RAG
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            # Retrieve top chunks blindly without separating intra vs cross
            flat_turns = [turn["text"] for turn in item["sessions"]]
            flat_embs = embedder.encode(flat_turns, convert_to_numpy=True, normalize_embeddings=True)
            q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores = np.dot(flat_embs, q_emb.T).squeeze()
            top_indices = np.argsort(scores)[::-1]
            selected_flat = []
            cur_tokens = 0
            for idx in top_indices:
                t_tokens = pruner.count_tokens(flat_turns[idx])
                if cur_tokens + t_tokens <= target_tokens or len(selected_flat) == 0:
                    selected_flat.append(flat_turns[idx])
                    cur_tokens += t_tokens
                if cur_tokens >= target_tokens:
                    break
            flat_rag_text = "\n".join(selected_flat)
            flat_lat = (time.perf_counter() - t0) * 1000

            metrics["Flat RAG"]["qa_recall"].append(evaluate_qa_recall(flat_rag_text, keywords))
            r_flat = scorer.score(ref_summary, flat_rag_text)
            metrics["Flat RAG"]["r1"].append(r_flat['rouge1'].fmeasure * 100)
            metrics["Flat RAG"]["rL"].append(r_flat['rougeL'].fmeasure * 100)
            embs = embedder.encode([all_turns_text, flat_rag_text], convert_to_numpy=True, normalize_embeddings=True)
            metrics["Flat RAG"]["sim"].append(float(np.dot(embs[0], embs[1])))
            metrics["Flat RAG"]["comp"].append(pruner.count_tokens(flat_rag_text) / orig_tokens)
            metrics["Flat RAG"]["lat"].append(flat_lat)
            metrics["Flat RAG"]["entail"].append(evaluate_entailment_proxy(flat_rag_text, ref_summary, embedder))

            # ------------------------------------------------------------
            # Method 4: Jarvis-M v2 (Hierarchical Memory + Information Pruning)
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            # Dynamic token budget split: 40% for memory graph, 60% for recent turns
            mem_budget = max(15, int(target_tokens * 0.40))
            turn_budget = max(15, target_tokens - mem_budget)

            first_user = item["sessions"][-1]["user"]
            memory_str = scenario_memory.get_compact_context_string(query, current_user=first_user)
            # Truncate memory_str if exceeding mem_budget
            mem_tokens = pruner.tokenizer.encode(memory_str, add_special_tokens=False)[:mem_budget]
            compact_memory = pruner.tokenizer.decode(mem_tokens, skip_special_tokens=True)

            pruned_text, _ = pruner.prune_dialogue_to_budget(all_turns_text, budget_ratio=(turn_budget / orig_tokens))
            jarvis_v2_context = compact_memory + "\n" + pruned_text
            jarvis_tokens = pruner.count_tokens(jarvis_v2_context)
            jarvis_lat = (time.perf_counter() - t0) * 1000

            metrics["Jarvis-M v2 (Hierarchical)"]["qa_recall"].append(evaluate_qa_recall(jarvis_v2_context, keywords))
            r_j = scorer.score(ref_summary, jarvis_v2_context)
            metrics["Jarvis-M v2 (Hierarchical)"]["r1"].append(r_j['rouge1'].fmeasure * 100)
            metrics["Jarvis-M v2 (Hierarchical)"]["rL"].append(r_j['rougeL'].fmeasure * 100)
            embs = embedder.encode([all_turns_text, jarvis_v2_context], convert_to_numpy=True, normalize_embeddings=True)
            metrics["Jarvis-M v2 (Hierarchical)"]["sim"].append(float(np.dot(embs[0], embs[1])))
            metrics["Jarvis-M v2 (Hierarchical)"]["comp"].append(jarvis_tokens / orig_tokens)
            metrics["Jarvis-M v2 (Hierarchical)"]["lat"].append(jarvis_lat)
            metrics["Jarvis-M v2 (Hierarchical)"]["entail"].append(evaluate_entailment_proxy(jarvis_v2_context, ref_summary, embedder))

        for m_name, m_data in metrics.items():
            summary_results[budget_label][m_name] = {
                "QA Fact Recall (%)": np.mean(m_data["qa_recall"]),
                "ROUGE-1": np.mean(m_data["r1"]),
                "ROUGE-L": np.mean(m_data["rL"]),
                "Semantic Sim": np.mean(m_data["sim"]),
                "Entailment": np.mean(m_data["entail"]),
                "Comp Ratio": np.mean(m_data["comp"]),
                "Latency (ms)": np.mean(m_data["lat"]),
            }

    # ------------------------------------------------------------
    # Output Tier-1 Formatted Comparative Results
    # ------------------------------------------------------------
    print("=" * 95)
    print("📈 TIER-1 MULTI-SESSION BENCHMARK: DOWNSTREAM QA ACCURACY & CONTEXT OPTIMIZATION")
    print("=" * 95)

    for budget_label, m_dict in summary_results.items():
        print(f"\n🎯 [Evaluation Token Budget: {budget_label}]")
        print(f"{'Method':<28} | {'QA Recall (%)':<14} | {'ROUGE-1':<8} | {'Sem Sim':<8} | {'Entailment':<10} | {'Comp':<6} | {'Latency':<9}")
        print("-" * 95)
        for method, vals in m_dict.items():
            print(
                f"{method:<28} | "
                f"{vals['QA Fact Recall (%)']:<14.2f} | "
                f"{vals['ROUGE-1']:<8.2f} | "
                f"{vals['Semantic Sim']:<8.3f} | "
                f"{vals['Entailment']:<10.3f} | "
                f"{vals['Comp Ratio']:<6.2f} | "
                f"{vals['Latency (ms)']:<6.1f} ms"
            )

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    args = parser.parse_args()
    run_tier1_benchmark(args.budgets)
