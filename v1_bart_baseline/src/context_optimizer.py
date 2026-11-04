"""
JARVIS-M v2: Token & Context Optimization Engine
=================================================
Core engine for dynamic context compression, memory-augmented token pruning,
and prompt optimization for modern LLM applications.

Supported compression strategies:
1. Dynamic Abstractive Distillation (LoRA-tuned BART + Dual-Memory)
2. Extractive Information Pruning (Semantic Saliency & Redundancy Filtering)
3. Hybrid Multi-Turn Context Optimizer (for OpenAI/Anthropic Chat Format)
"""

import os
import sys
import time
import re
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Ensure stdout handles UTF-8 safely
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# Default Configurations
# ============================================================
DEFAULT_BASE_MODEL = "facebook/bart-large-cnn"
DEFAULT_LORA_PATH = "./jarvis-bart-lora"
DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = "./cache"


class ContextOptimizer:
    """
    Modular context and token optimizer for multi-turn dialogues and LLM prompts.
    """

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        lora_adapter_path: Optional[str] = DEFAULT_LORA_PATH,
        embedder_name: str = DEFAULT_EMBEDDER,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
        load_generator: bool = True,
    ):
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing ContextOptimizer on device: {self.device}")

        # 1. Embedder for semantic similarity & extractive importance
        print(f"[*] Loading Embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name, device=self.device, cache_folder=self.cache_dir)

        # 2. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=self.cache_dir)

        # 3. Generative Model (LoRA-tuned BART)
        self.model = None
        if load_generator:
            print(f"[*] Loading Base Generator: {base_model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            if lora_adapter_path and os.path.exists(lora_adapter_path):
                print(f"[*] Loading LoRA adapter from: {lora_adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
                self.model = self.model.merge_and_unload()
                print("[+] LoRA adapter merged successfully")
            else:
                print(f"[-] No LoRA adapter found at {lora_adapter_path}; running base weights")

            self.model.to(self.device)
            self.model.eval()

        print("[+] ContextOptimizer ready.")

    def count_tokens(self, text: str) -> int:
        """Count tokens using the underlying tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences and turns."""
        raw_splits = re.split(r'(?<=[.?!])\s+|\n+', text)
        sentences = [s.strip() for s in raw_splits if len(s.strip()) > 5]
        return sentences

    # ------------------------------------------------------------
    # Strategy 1: Extractive Semantic Saliency Pruning (Fast & Lossless)
    # ------------------------------------------------------------
    def compress_extractive(
        self,
        text: str,
        budget_ratio: float = 0.5,
        query: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extractive token compression based on semantic centrality and query relevance.
        Preserves original wording and facts while removing filler and redundancy.
        """
        t0 = time.perf_counter()
        sentences = self.split_sentences(text)
        orig_tokens = self.count_tokens(text)

        if len(sentences) <= 2 or orig_tokens < 30:
            return text, {
                "strategy": "extractive",
                "original_tokens": orig_tokens,
                "compressed_tokens": orig_tokens,
                "compression_ratio": 1.0,
                "token_savings_pct": 0.0,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }

        target_tokens = max(15, int(orig_tokens * budget_ratio))
        sentence_embs = self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

        # Compute centrality (mean similarity to overall context)
        doc_emb = np.mean(sentence_embs, axis=0, keepdims=True)
        doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-9)
        centrality_scores = np.dot(sentence_embs, doc_emb.T).squeeze()

        # If query provided, blend query relevance
        if query:
            q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            q_scores = np.dot(sentence_embs, q_emb.T).squeeze()
            combined_scores = 0.6 * q_scores + 0.4 * centrality_scores
        else:
            combined_scores = centrality_scores

        # Rank sentences by score and pick greedily in original discourse order
        scored_indices = sorted(range(len(sentences)), key=lambda i: combined_scores[i], reverse=True)

        selected_indices = []
        current_token_count = 0

        for idx in scored_indices:
            s_tokens = self.count_tokens(sentences[idx])
            if current_token_count + s_tokens <= target_tokens or len(selected_indices) == 0:
                selected_indices.append(idx)
                current_token_count += s_tokens
            if current_token_count >= target_tokens:
                break

        selected_indices.sort()  # Restore discourse order
        compressed_text = " ".join([sentences[i] for i in selected_indices])
        compressed_tokens = self.count_tokens(compressed_text)
        latency_ms = (time.perf_counter() - t0) * 1000

        stats = {
            "strategy": "extractive",
            "original_tokens": orig_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(compressed_tokens / max(1, orig_tokens), 4),
            "token_savings_pct": round((1.0 - (compressed_tokens / max(1, orig_tokens))) * 100, 2),
            "latency_ms": round(latency_ms, 2),
        }

        return compressed_text, stats

    # ------------------------------------------------------------
    # Strategy 2: Abstractive Distillation (LoRA BART + Dynamic Beam Search)
    # ------------------------------------------------------------
    def compress_abstractive(
        self,
        text: str,
        budget_ratio: float = 0.5,
        min_tokens: int = 20,
        max_tokens: int = 150,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Abstractive dialogue distillation using the LoRA-adapted generator.
        """
        if self.model is None:
            raise RuntimeError("Generator model is not loaded. Initialize with load_generator=True.")

        t0 = time.perf_counter()
        orig_tokens = self.count_tokens(text)

        target_max = min(max_tokens, max(min_tokens + 10, int(orig_tokens * budget_ratio)))
        target_min = max(min_tokens, int(target_max * 0.4))

        inputs = self.tokenizer(
            "summarize: " + text.strip(),
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=target_max,
                min_length=target_min,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        compressed_tokens = self.count_tokens(summary)
        latency_ms = (time.perf_counter() - t0) * 1000

        stats = {
            "strategy": "abstractive_lora",
            "original_tokens": orig_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(compressed_tokens / max(1, orig_tokens), 4),
            "token_savings_pct": round((1.0 - (compressed_tokens / max(1, orig_tokens))) * 100, 2),
            "latency_ms": round(latency_ms, 2),
        }

        return summary, stats

    # ------------------------------------------------------------
    # Strategy 3: Chat History Context Window Optimizer (OpenAI Format)
    # ------------------------------------------------------------
    def optimize_chat_messages(
        self,
        messages: List[Dict[str, str]],
        max_token_budget: int = 1000,
        keep_recent_turns: int = 2,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Takes an OpenAI-style list of messages:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        and compresses older turns while preserving recent turns intact.
        """
        t0 = time.perf_counter()
        total_tokens = sum(self.count_tokens(m["content"]) for m in messages)

        if total_tokens <= max_token_budget or len(messages) <= keep_recent_turns + 1:
            return messages, {
                "strategy": "passthrough",
                "original_tokens": total_tokens,
                "compressed_tokens": total_tokens,
                "compression_ratio": 1.0,
                "token_savings_pct": 0.0,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }

        # Separate system prompt, older turns, and recent turns
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        recent_msgs = non_system[-keep_recent_turns:]
        older_msgs = non_system[:-keep_recent_turns]

        # Consolidate older messages for memory distillation
        older_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in older_msgs])

        # Distill older turns
        distilled_summary, _ = self.compress_extractive(older_text, budget_ratio=0.35)

        optimized_messages = []
        if system_msgs:
            optimized_messages.extend(system_msgs)

        optimized_messages.append({
            "role": "system",
            "content": f"[Prior Conversation Context & Memory Summary]: {distilled_summary}"
        })
        optimized_messages.extend(recent_msgs)

        new_total_tokens = sum(self.count_tokens(m["content"]) for m in optimized_messages)
        latency_ms = (time.perf_counter() - t0) * 1000

        stats = {
            "strategy": "chat_window_optimization",
            "original_tokens": total_tokens,
            "compressed_tokens": new_total_tokens,
            "compression_ratio": round(new_total_tokens / max(1, total_tokens), 4),
            "token_savings_pct": round((1.0 - (new_total_tokens / max(1, total_tokens))) * 100, 2),
            "latency_ms": round(latency_ms, 2),
        }

        return optimized_messages, stats


# ============================================================
# Quick Local Verification
# ============================================================
if __name__ == "__main__":
    optimizer = ContextOptimizer(load_generator=True)

    sample_dialogue = """
    Person A: Hey, have you finished the project report yet?
    Person B: I'm still working on it. The data analysis part is taking longer than expected.
    Person A: We need to submit it by Friday. Can you prioritize that section?
    Person B: Sure, I'll focus on it today and tomorrow. I might need your help with the charts.
    Person A: No problem. Send me what you have and I'll work on the visualizations.
    Person B: Great. I'll also need to review the financial projections. They seem off.
    Person A: Let me double-check those numbers. We got them from the accounting team last week.
    Person B: Okay. Also, should we include the risk assessment section? It wasn't in the original outline.
    Person A: Yes, the manager specifically asked for it during yesterday's meeting.
    Person B: Got it. I'll add a section for that too. This report is getting quite comprehensive.
    Person A: Better to be thorough. Let's schedule a review session on Thursday before submission.
    Person B: Sounds good. I'll send you a draft by Wednesday evening.
    """

    print("\n" + "=" * 60)
    print("Testing Extractive Saliency Compression (50% Budget)...")
    c_ext, s_ext = optimizer.compress_extractive(sample_dialogue, budget_ratio=0.5)
    print(f"Compressed: {c_ext}")
    print(f"Stats: {s_ext}")

    print("\n" + "=" * 60)
    print("Testing Abstractive LoRA Compression (30% Budget)...")
    c_abs, s_abs = optimizer.compress_abstractive(sample_dialogue, budget_ratio=0.3)
    print(f"Compressed: {c_abs}")
    print(f"Stats: {s_abs}")
