"""
JARVIS-M v2: Downstream Reasoning with Qwen-2.5-0.5B-Instruct
============================================================
Evaluates real LLM generation accuracy, contradiction hallucinations,
and prompt token efficiency using Qwen-2.5-0.5B-Instruct on CUDA.

Methods compared:
1. Full Raw Context (100% tokens)
2. Naive Truncation (Sliding Window)
3. Flat Vector RAG
4. State-Blind Pruner (LLMLingua-style)
5. Jarvis-M v2 (Temporal State Reconciled)
"""

import sys
import os
import time
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from memory_conditioned_optimizer import MemoryConditionedContextOptimizer
from sentence_transformers import SentenceTransformer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_DIR = "./cache"


def get_qwen_test_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Cloud Deployment Override",
            "chat_history": [
                {"session_id": "s1", "speaker": "Alice", "text": "Team, we officially decided to use AWS EC2 for our backend deployment."},
                {"session_id": "s1", "speaker": "Alice", "text": "The release deadline is Friday 5 PM."},
                {"session_id": "s1", "speaker": "Bob", "text": "I will begin writing Terraform scripts for EC2 today."},
                {"session_id": "s2", "speaker": "Charlie", "text": "Wait, our startup grant only covers Google Cloud credits."},
                {"session_id": "s2", "speaker": "Alice", "text": "Good catch! We officially switched to Google Cloud Run to utilize the credits."},
                {"session_id": "s2", "speaker": "Bob", "text": "Understood, abandoning EC2 scripts, preparing Cloud Run Dockerfile."},
                {"session_id": "s3", "speaker": "Bob", "text": "Cloud Run deployment is failing with Error 403 on container registry permissions."},
                {"session_id": "s3", "speaker": "Charlie", "text": "I will grant Artifact Registry Reader IAM role to Bob's service account today."},
                {"session_id": "s3", "speaker": "Alice", "text": "The updated deadline is Monday 10 AM."},
            ],
            "test_questions": [
                {
                    "q": "What hosting platform should we deploy to?",
                    "target_truth": "Google Cloud Run",
                    "stale_trap": "AWS EC2",
                    "truth_keywords": ["cloud run", "google cloud"],
                    "stale_keywords": ["aws", "ec2"]
                },
                {
                    "q": "What is the final confirmed submission deadline?",
                    "target_truth": "Monday 10 AM",
                    "stale_trap": "Friday 5 PM",
                    "truth_keywords": ["monday 10 am", "monday"],
                    "stale_keywords": ["friday 5 pm", "friday"]
                }
            ]
        },
        {
            "name": "Fintech Database Migration",
            "chat_history": [
                {"session_id": "s1", "speaker": "Dan", "text": "We decided to deploy MongoDB as the primary database for user financial records."},
                {"session_id": "s1", "speaker": "Elena", "text": "I will begin designing the MongoDB collections and indexing."},
                {"session_id": "s2", "speaker": "Farhan", "text": "Compliance review flagged MongoDB: financial audits require strict relational ACID transactions."},
                {"session_id": "s2", "speaker": "Dan", "text": "Understood. We officially migrated to PostgreSQL for ACID compliance."},
                {"session_id": "s2", "speaker": "Elena", "text": "Rewriting schemas for PostgreSQL tables and foreign keys."},
                {"session_id": "s3", "speaker": "Elena", "text": "PostgreSQL connection pool is exhausted with Error 503 during stress testing."},
                {"session_id": "s3", "speaker": "Farhan", "text": "I will implement PgBouncer connection pooling to resolve the 503 errors."},
            ],
            "test_questions": [
                {
                    "q": "Which database was ultimately chosen for user financial records?",
                    "target_truth": "PostgreSQL",
                    "stale_trap": "MongoDB",
                    "truth_keywords": ["postgresql", "postgres"],
                    "stale_keywords": ["mongodb"]
                },
                {
                    "q": "Who is resolving the Error 503 connection pool issue, and with what tool?",
                    "target_truth": "Farhan with PgBouncer",
                    "stale_trap": "None",
                    "truth_keywords": ["farhan", "pgbouncer"],
                    "stale_keywords": []
                }
            ]
        }
    ]


class QwenDownstreamReasoner:
    def __init__(self, model_name: str = QWEN_MODEL_NAME, cache_dir: str = CACHE_DIR):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        print("[+] Qwen 2.5 loaded successfully.")

    def generate_answer(self, context_str: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an accurate enterprise assistant. Answer the user's question concisely based ONLY on the verified current state in the provided conversation context. If a decision was changed or rescheduled, state only the latest finalized decision."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}\nConcise Answer:"
            }
        ]

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.1,  # Low temperature for factual fidelity
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Slice generated response tokens only
        input_len = inputs.input_ids.shape[1]
        response_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        return response_text


def run_qwen_evaluation():
    reasoner = QwenDownstreamReasoner()
    optimizer = MemoryConditionedContextOptimizer()
    embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)
    scenarios = get_qwen_test_scenarios()

    target_budget = 140
    print("\n" + "=" * 90)
    print("🤖 BENCHMARKING QWEN-2.5-0.5B-INSTRUCT ACROSS CONTEXT OPTIMIZATION STRATEGIES")
    print("=" * 90)

    methods = [
        "Full Raw Context",
        "Naive Truncation (Sliding Window)",
        "Flat Vector RAG",
        "State-Blind Pruner (LLMLingua)",
        "Jarvis-M v2 (State Reconciled)",
    ]

    method_stats = {m: {"correct": 0, "hallucinated_stale": 0, "total": 0, "tokens": []} for m in methods}

    for sc in scenarios:
        turns = sc["chat_history"]
        raw_full_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])
        total_tokens = optimizer.count_tokens(raw_full_text)

        print(f"\n📂 SCENARIO: {sc['name']} (Raw Tokens: {total_tokens})")

        # Prepare context for each method
        contexts = {}

        # 1. Full Raw
        contexts["Full Raw Context"] = raw_full_text

        # 2. Naive Truncation (Sliding Window: last turns)
        tokens_list = optimizer.pruner.tokenizer.encode(raw_full_text, add_special_tokens=False)
        contexts["Naive Truncation (Sliding Window)"] = optimizer.pruner.tokenizer.decode(tokens_list[-target_budget:], skip_special_tokens=True)

        # 3. Flat Vector RAG
        turn_texts = [f"{t['speaker']}: {t['text']}" for t in turns]
        turn_embs = embedder.encode(turn_texts, convert_to_numpy=True, normalize_embeddings=True)
        q_comb = " ".join([q["q"] for q in sc["test_questions"]])
        q_emb = embedder.encode([q_comb], convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(turn_embs, q_emb.T).squeeze()
        top_idx = np.argsort(scores)[::-1]
        rag_turns = []
        cur_t = 0
        for i in top_idx:
            c = optimizer.count_tokens(turn_texts[i])
            if cur_t + c <= target_budget or len(rag_turns) == 0:
                rag_turns.append(turn_texts[i])
                cur_t += c
            if cur_t >= target_budget:
                break
        contexts["Flat Vector RAG"] = "\n".join(rag_turns)

        # 4. State-Blind Pruner (LLMLingua-style)
        blind_prompt, _ = optimizer.pruner.prune_dialogue_to_budget(raw_full_text, budget_ratio=(target_budget / total_tokens), preserve_recent_turns=1)
        contexts["State-Blind Pruner (LLMLingua)"] = blind_prompt

        # 5. Jarvis-M v2
        jarvis_prompt, _ = optimizer.optimize_multisession_context(turns, target_token_budget=target_budget)
        contexts["Jarvis-M v2 (State Reconciled)"] = jarvis_prompt

        # Evaluate each question
        for q_item in sc["test_questions"]:
            q_text = q_item["q"]
            print(f"\n❓ Question: {q_text}")
            print(f"   [Ground Truth Target]: '{q_item['target_truth']}' | [Stale Trap]: '{q_item.get('stale_trap', 'None')}'")

            for m in methods:
                ctx = contexts[m]
                ctx_tokens = optimizer.count_tokens(ctx)
                method_stats[m]["tokens"].append(ctx_tokens)
                method_stats[m]["total"] += 1

                # Generate Answer with Qwen 2.5
                gen_ans = reasoner.generate_answer(ctx, q_text)
                gen_lower = gen_ans.lower()

                # Score correctness
                is_correct = any(kw.lower() in gen_lower for kw in q_item["truth_keywords"])
                is_stale = False
                if q_item.get("stale_keywords"):
                    is_stale = any(kw.lower() in gen_lower for kw in q_item["stale_keywords"]) and not is_correct

                if is_correct:
                    method_stats[m]["correct"] += 1
                if is_stale:
                    method_stats[m]["hallucinated_stale"] += 1

                status_flag = "✓ CORRECT" if is_correct else ("⚠ STALE CONTRADICTION" if is_stale else "✗ INCORRECT")
                print(f"   [{m:<32}] ({ctx_tokens} tok) -> {status_flag}: \"{gen_ans}\"")

    # Final Summary Table
    print("\n" + "=" * 90)
    print("🏆 FINAL BENCHMARK SUMMARY: REAL QWEN-2.5-0.5B-INSTRUCT REASONING RESULTS")
    print("=" * 90)
    print(f"{'Method':<34} | {'LLM Accuracy (%)':<18} | {'Stale Hallucination (%)':<24} | {'Prompt Tokens':<12}")
    print("-" * 94)

    for m in methods:
        total = method_stats[m]["total"]
        acc = (method_stats[m]["correct"] / max(1, total)) * 100.0
        halluc = (method_stats[m]["hallucinated_stale"] / max(1, total)) * 100.0
        avg_tokens = int(np.mean(method_stats[m]["tokens"]))
        print(f"{m:<34} | {acc:<18.2f} | {halluc:<24.2f} | {avg_tokens:<12}")

    print("=" * 90)


if __name__ == "__main__":
    run_qwen_evaluation()
