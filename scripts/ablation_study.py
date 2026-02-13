"""
JARVIS-M: Inference-Only Ablation Study
========================================
Isolates the contribution of the dual-memory retrieval module from the
LoRA-fine-tuned BART backbone.  The summarization model is NEVER retrained;
all four configurations use the identical LoRA-merged weights and identical
decoding hyper-parameters.

Architecture
------------
Retrieval operates at inference time in TWO stages:

  1. **Candidate generation** — the LoRA-merged BART model generates
     multiple beam candidates (num_return_sequences > 1) from the raw
     dialogue.  All configs share the *same* candidate pool.

  2. **Memory-guided reranking** — retrieved summaries from the memory
     stores are used to rerank the candidates.  A candidate whose content
     aligns more closely with similar past summaries is preferred, because
     the memory stores capture entities, phrasing, and facts that recur
     across similar dialogues.  The reranking score blends the model's own
     log-probability with cosine similarity to retrieved context:

         score(c) = alpha * norm_log_prob(c)
                  + (1 - alpha) * max_sim(c, retrieved)

This is a standard retrieval-augmented reranking approach that does NOT
modify the generator's input (avoiding out-of-distribution degradation)
and does NOT require retraining.

Configurations
--------------
    A.  LoRA-only        – take the top beam (no reranking)
    B.  + Intra-user     – rerank using same-user memory
    C.  + Cross-user     – rerank using other-users' memory
    D.  Full Jarvis-M    – rerank using both memory stores

Usage:
    python ablation_study.py                    # full 1500-sample evaluation
    python ablation_study.py --max_samples 50   # quick sanity check
"""

import os
import sys
import json
import time
import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import faiss
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# ============================================================
# Configuration — shared across ALL four ablation runs
# ============================================================
BASE_MODEL_NAME = "facebook/bart-large-cnn"
LORA_ADAPTER_PATH = "./models/jarvis-bart-lora"
DATASET_NAME = "knkarthick/dialogsum"
CACHE_DIR = "./cache"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
RESULTS_DIR = "./results/ablation"

# Decoding hyper-parameters — identical for all configs.
# These use the inference-time settings from update_inference.py so
# the LoRA-only baseline reflects real deployment performance (~32 ROUGE-1).
GENERATION_KWARGS = {
    "num_beams": 4,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "do_sample": False,
    "repetition_penalty": 1.2,
}

# Compression bounds (from update_inference.py)
TARGET_COMPRESSION = 0.5
ABSOLUTE_MIN_LENGTH = 20
ABSOLUTE_MAX_LENGTH = 150
MAX_INPUT_TOKENS = 1024

# Reranking: generate multiple candidates per sample
NUM_RETURN_SEQUENCES = 6          # total candidates (1 beam + 5 sampled)
RERANK_ALPHA = 0.4                # weight on model log-prob vs retrieval sim

# Retrieval settings
TOP_K_INTRA = 3                   # summaries retrieved from same-user history
TOP_K_CROSS = 3                   # summaries retrieved from other-user history
NUM_SYNTHETIC_USERS = 50
EMBED_BATCH_SIZE = 128

SEED = 42


# ============================================================
# Data structures
# ============================================================
@dataclass
class AblationConfig:
    name: str
    enable_intra: bool = False
    enable_cross: bool = False


ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="LoRA only",      enable_intra=False, enable_cross=False),
    AblationConfig(name="+ Intra-user",   enable_intra=True,  enable_cross=False),
    AblationConfig(name="+ Cross-user",   enable_intra=False, enable_cross=True),
    AblationConfig(name="Full Jarvis-M",  enable_intra=True,  enable_cross=True),
]


@dataclass
class AblationResult:
    config_name: str
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    num_samples: int = 0
    wall_time_s: float = 0.0


# ============================================================
# Helpers
# ============================================================
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device : {name} ({mem:.2f} GB)")
        return "cuda"
    print("  Device : CPU (this will be slow)")
    return "cpu"


def dynamic_lengths(tokenizer, text: str) -> Tuple[int, int]:
    """Compute min/max generation lengths based on input length."""
    n_tok = len(tokenizer.encode(text, add_special_tokens=False))
    target = int(n_tok * TARGET_COMPRESSION)
    mx = min(ABSOLUTE_MAX_LENGTH, max(target, int(n_tok * 0.7)))
    mn = max(ABSOLUTE_MIN_LENGTH, int(n_tok * 0.3))
    mn = min(mn, mx - 10)
    mn = max(10, mn)
    return mn, mx


# ============================================================
# Model loading  (ONE-TIME, shared across all configs)
# ============================================================
def load_model_and_tokenizer(device: str):
    print("\n[1/5] Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=CACHE_DIR)

    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if os.path.exists(LORA_ADAPTER_PATH):
        print(f"  Merging LoRA adapter from {LORA_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
    else:
        print("  WARNING: LoRA adapter not found — using base BART-large-CNN")
        model = base

    model.to(device).eval()
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ============================================================
# Dataset loading + synthetic user partitioning
# ============================================================
def load_data():
    print("\n[2/5] Loading DialogSum …")
    ds = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    print(f"  Train : {len(ds['train'])}  |  Val : {len(ds['validation'])}  "
          f"|  Test : {len(ds['test'])}")
    return ds


def build_user_partitions(
    train_dialogues: List[str],
    train_summaries: List[str],
    n_users: int = NUM_SYNTHETIC_USERS,
) -> Dict[int, Dict[str, List[str]]]:
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(train_dialogues))
    rng.shuffle(indices)
    user_data: Dict[int, Dict[str, List[str]]] = {}
    splits = np.array_split(indices, n_users)
    for uid, idx_block in enumerate(splits):
        user_data[uid] = {
            "dialogues": [train_dialogues[i] for i in idx_block],
            "summaries": [train_summaries[i] for i in idx_block],
        }
    return user_data


# ============================================================
# Retrieval engine (FAISS)
# ============================================================
class MemoryRetriever:
    """Manages per-user and cross-user FAISS indices over training summaries."""

    def __init__(self, embedder: SentenceTransformer, user_data: Dict, device: str):
        self.embedder = embedder
        self.device = device
        self.n_users = len(user_data)

        print("\n[3/5] Building per-user FAISS indices …")
        self.user_indices: Dict[int, faiss.IndexFlatIP] = {}
        self.user_summaries: Dict[int, List[str]] = {}
        self.all_summaries: List[str] = []
        self.all_embeddings_list: List[np.ndarray] = []

        for uid in tqdm(range(self.n_users), desc="  Users"):
            sums = user_data[uid]["summaries"]
            self.user_summaries[uid] = sums
            self.all_summaries.extend(sums)
            embs = self._encode(sums)
            self.all_embeddings_list.append(embs)
            self.user_indices[uid] = self._make_index(embs)

        all_embs = np.vstack(self.all_embeddings_list)
        self.global_index = self._make_index(all_embs)

        self._global_uid_map: List[int] = []
        for uid in range(self.n_users):
            self._global_uid_map.extend(
                [uid] * len(user_data[uid]["summaries"])
            )

    def _encode(self, texts: List[str]) -> np.ndarray:
        embs = self.embedder.encode(
            texts, batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True, device=self.device,
            show_progress_bar=False,
        )
        faiss.normalize_L2(embs)
        return embs

    @staticmethod
    def _make_index(embs: np.ndarray) -> faiss.IndexFlatIP:
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        return idx

    def assign_user(self, test_idx: int) -> int:
        h = int(hashlib.md5(str(test_idx).encode()).hexdigest(), 16)
        return h % self.n_users

    def retrieve_intra(self, query: str, user_id: int,
                       top_k: int = TOP_K_INTRA) -> List[str]:
        if user_id not in self.user_indices:
            return []
        q = self._encode([query])
        k = min(top_k, self.user_indices[user_id].ntotal)
        if k == 0:
            return []
        _, I = self.user_indices[user_id].search(q, k)
        return [self.user_summaries[user_id][i]
                for i in I[0]
                if 0 <= i < len(self.user_summaries[user_id])]

    def retrieve_cross(self, query: str, user_id: int,
                       top_k: int = TOP_K_CROSS) -> List[str]:
        q = self._encode([query])
        fetch_k = min(top_k * 5, self.global_index.ntotal)
        _, I = self.global_index.search(q, fetch_k)
        results: List[str] = []
        for gi in I[0]:
            if gi < 0 or gi >= len(self.all_summaries):
                continue
            if self._global_uid_map[gi] == user_id:
                continue
            results.append(self.all_summaries[gi])
            if len(results) >= top_k:
                break
        return results

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Public helper — embed a list of texts (for reranking)."""
        return self._encode(texts)


# ============================================================
# Generation — produces multiple beam candidates
# ============================================================
@torch.inference_mode()
def generate_candidates(
    model, tokenizer, dialogue: str, device: str,
) -> Tuple[List[str], List[float]]:
    """
    Generate NUM_RETURN_SEQUENCES diverse candidates for a single dialogue.

    Strategy: generate one greedy/beam candidate (highest quality), then
    additional candidates via nucleus sampling with temperature to introduce
    diversity.  This gives the reranker meaningfully different options to
    choose from.

    Returns (candidates, sequence_scores).
    """
    mn, mx = dynamic_lengths(tokenizer, dialogue)
    input_text = f"summarize: {dialogue}"

    inputs = tokenizer(
        input_text, max_length=MAX_INPUT_TOKENS,
        truncation=True, return_tensors="pt",
    ).to(device)

    candidates: List[str] = []
    scores: List[float] = []

    # Candidate 0: standard beam search (highest quality baseline)
    beam_out = model.generate(
        **inputs,
        max_length=mx,
        min_length=mn,
        num_beams=GENERATION_KWARGS["num_beams"],
        length_penalty=GENERATION_KWARGS["length_penalty"],
        no_repeat_ngram_size=GENERATION_KWARGS["no_repeat_ngram_size"],
        early_stopping=GENERATION_KWARGS["early_stopping"],
        do_sample=False,
        repetition_penalty=GENERATION_KWARGS["repetition_penalty"],
        output_scores=True,
        return_dict_in_generate=True,
    )
    candidates.append(
        tokenizer.decode(beam_out.sequences[0], skip_special_tokens=True)
    )
    if hasattr(beam_out, "sequences_scores") and beam_out.sequences_scores is not None:
        scores.append(beam_out.sequences_scores[0].item())
    else:
        scores.append(0.0)

    # Candidates 1..N-1: nucleus sampling with varying temperatures
    temps = [0.6, 0.7, 0.85, 1.0, 1.2][:NUM_RETURN_SEQUENCES - 1]
    for temp in temps:
        sample_out = model.generate(
            **inputs,
            max_length=mx,
            min_length=mn,
            do_sample=True,
            temperature=temp,
            top_p=0.92,
            top_k=50,
            no_repeat_ngram_size=GENERATION_KWARGS["no_repeat_ngram_size"],
            repetition_penalty=GENERATION_KWARGS["repetition_penalty"],
            output_scores=True,
            return_dict_in_generate=True,
        )
        candidates.append(
            tokenizer.decode(sample_out.sequences[0], skip_special_tokens=True)
        )
        if hasattr(sample_out, "sequences_scores") and sample_out.sequences_scores is not None:
            scores.append(sample_out.sequences_scores[0].item())
        else:
            scores.append(-1.0)  # lower than beam to reflect lower confidence

    return candidates, scores


# ============================================================
# Reranking — memory-guided candidate selection
# ============================================================
def rerank_candidates(
    candidates: List[str],
    model_scores: List[float],
    retrieved_summaries: List[str],
    retriever: MemoryRetriever,
    alpha: float = RERANK_ALPHA,
) -> str:
    """
    Rerank beam candidates using retrieval similarity.

    score(c) = alpha * norm_model_score(c)
             + (1 - alpha) * max cosine_sim(c, retrieved_summaries)

    If no retrieved summaries are available, returns the top beam (index 0).
    """
    if not retrieved_summaries:
        return candidates[0]

    # Normalize model scores to [0, 1]
    s_arr = np.array(model_scores, dtype=np.float64)
    s_min, s_max = s_arr.min(), s_arr.max()
    if s_max - s_min > 1e-8:
        norm_model = (s_arr - s_min) / (s_max - s_min)
    else:
        norm_model = np.ones_like(s_arr) * 0.5

    # Embed candidates and retrieved summaries
    cand_embs = retriever.embed_texts(candidates)           # (C, d)
    ret_embs  = retriever.embed_texts(retrieved_summaries)  # (R, d)

    # Cosine similarity matrix (already L2-normalized)
    sim_matrix = cand_embs @ ret_embs.T                     # (C, R)
    max_sim = sim_matrix.max(axis=1)                        # (C,)

    # Normalize retrieval similarity to [0, 1]
    sim_min, sim_max = max_sim.min(), max_sim.max()
    if sim_max - sim_min > 1e-8:
        norm_sim = (max_sim - sim_min) / (sim_max - sim_min)
    else:
        norm_sim = np.ones_like(max_sim) * 0.5

    combined = alpha * norm_model + (1.0 - alpha) * norm_sim
    best_idx = int(np.argmax(combined))

    return candidates[best_idx]


# ============================================================
# Evaluation
# ============================================================
def compute_rouge(predictions: List[str],
                  references: List[str]) -> Dict[str, float]:
    """Compute corpus-level ROUGE scores (x100).

    Uses standard ROUGE without word-per-line inflation so scores
    reflect real inference performance.
    """
    predictions = [p.strip() for p in predictions]
    references  = [r.strip() for r in references]

    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {
        "rouge1": round(scores["rouge1"] * 100, 2),
        "rouge2": round(scores["rouge2"] * 100, 2),
        "rougeL": round(scores["rougeL"] * 100, 2),
    }


# ============================================================
# Main ablation loop
# ============================================================
def run_ablation(
    model, tokenizer,
    retriever: MemoryRetriever,
    test_dialogues: List[str],
    test_summaries: List[str],
    device: str,
    max_samples: Optional[int] = None,
) -> List[AblationResult]:

    if max_samples is not None:
        test_dialogues = test_dialogues[:max_samples]
        test_summaries = test_summaries[:max_samples]

    n = len(test_dialogues)

    # -- Phase 1: generate candidates ONCE for all configs --
    print(f"\n[4/5] Generating {NUM_RETURN_SEQUENCES} beam candidates "
          f"per sample ({n} samples) …")

    all_candidates: List[List[str]] = []
    all_scores: List[List[float]] = []

    for i in tqdm(range(n), desc="  Generating", ncols=80):
        cands, sc = generate_candidates(
            model, tokenizer, test_dialogues[i], device
        )
        all_candidates.append(cands)
        all_scores.append(sc)

    # -- Phase 2: rerank per config --
    print(f"\n  Reranking across {len(ABLATION_CONFIGS)} configs …\n")
    results: List[AblationResult] = []

    for cfg in ABLATION_CONFIGS:
        print(f"{'─'*60}")
        print(f"  Config : {cfg.name}")
        print(f"  Intra  : {'ON' if cfg.enable_intra else 'OFF'}   |   "
              f"Cross : {'ON' if cfg.enable_cross else 'OFF'}")
        print(f"{'─'*60}")

        predictions: List[str] = []
        t0 = time.time()

        for i in tqdm(range(n), desc=f"  {cfg.name}", ncols=80):
            dialogue = test_dialogues[i]
            uid = retriever.assign_user(i)

            retrieved: List[str] = []
            if cfg.enable_intra:
                retrieved.extend(
                    retriever.retrieve_intra(dialogue, uid, TOP_K_INTRA)
                )
            if cfg.enable_cross:
                retrieved.extend(
                    retriever.retrieve_cross(dialogue, uid, TOP_K_CROSS)
                )

            if not retrieved:
                pred = all_candidates[i][0]
            else:
                pred = rerank_candidates(
                    all_candidates[i], all_scores[i],
                    retrieved, retriever,
                    alpha=RERANK_ALPHA,
                )

            predictions.append(pred)

        elapsed = time.time() - t0
        scores = compute_rouge(predictions, test_summaries)

        res = AblationResult(
            config_name=cfg.name,
            rouge1=scores["rouge1"],
            rouge2=scores["rouge2"],
            rougeL=scores["rougeL"],
            num_samples=n,
            wall_time_s=round(elapsed, 1),
        )
        results.append(res)
        print(f"  R-1 {res.rouge1:.2f}  |  R-2 {res.rouge2:.2f}  |  "
              f"R-L {res.rougeL:.2f}  |  {elapsed:.0f}s\n")

    return results


# ============================================================
# Reporting
# ============================================================
def print_table(results: List[AblationResult]) -> None:
    hdr = (f"{'Configuration':<20} {'ROUGE-1':>8} {'ROUGE-2':>8} "
           f"{'ROUGE-L':>8} {'Time (s)':>9}")
    line = "─" * len(hdr)

    print(f"\n{'='*len(hdr)}")
    print("  ABLATION RESULTS — Dual-Memory Retrieval Contribution")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print(line)

    baseline = results[0]
    for r in results:
        d1 = r.rouge1 - baseline.rouge1
        d2 = r.rouge2 - baseline.rouge2
        dL = r.rougeL - baseline.rougeL
        delta = (f"  (Δ {d1:+.2f} / {d2:+.2f} / {dL:+.2f})"
                 if r is not baseline else "")
        print(f"{r.config_name:<20} {r.rouge1:>8.2f} {r.rouge2:>8.2f} "
              f"{r.rougeL:>8.2f} {r.wall_time_s:>9.1f}{delta}")

    print(line)
    print(f"  Samples per config : {results[0].num_samples}")
    print(f"  Decoding           : beam={GENERATION_KWARGS['num_beams']}, "
          f"lp={GENERATION_KWARGS['length_penalty']}, "
          f"nrng={GENERATION_KWARGS['no_repeat_ngram_size']}, "
          f"rp={GENERATION_KWARGS['repetition_penalty']}")
    print(f"  Reranking          : alpha={RERANK_ALPHA}, "
          f"candidates={NUM_RETURN_SEQUENCES}")
    print(f"  Retrieval k        : intra={TOP_K_INTRA}, cross={TOP_K_CROSS}")
    print(f"  Synthetic users    : {NUM_SYNTHETIC_USERS}")
    print()


def print_latex_table(results: List[AblationResult]) -> None:
    baseline = results[0]
    print("% ── LaTeX (booktabs) ─────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study: retrieval-augmented reranking contribution.")
    print(r"         All runs use identical LoRA-merged BART-large-CNN weights")
    print(r"         and decoding parameters. Retrieval reranks beam candidates")
    print(r"         without modifying the generator input.}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{l c c c}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{ROUGE-1} & "
          r"\textbf{ROUGE-2} & \textbf{ROUGE-L} \\")
    print(r"\midrule")

    for r in results:
        d1 = r.rouge1 - baseline.rouge1
        d2 = r.rouge2 - baseline.rouge2
        dL = r.rougeL - baseline.rougeL
        if r is baseline:
            print(f"{r.config_name:<20} & {r.rouge1:.2f} & "
                  f"{r.rouge2:.2f} & {r.rougeL:.2f} \\\\")
        else:
            print(
                f"{r.config_name:<20} & "
                f"{r.rouge1:.2f}\\textsuperscript{{{d1:+.2f}}} & "
                f"{r.rouge2:.2f}\\textsuperscript{{{d2:+.2f}}} & "
                f"{r.rougeL:.2f}\\textsuperscript{{{dL:+.2f}}} \\\\"
            )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% ─────────────────────────────────────────────────────────\n")


def save_results(results: List[AblationResult], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ablation_results.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": BASE_MODEL_NAME,
        "lora_adapter": LORA_ADAPTER_PATH,
        "generation_kwargs": GENERATION_KWARGS,
        "reranking": {
            "alpha": RERANK_ALPHA,
            "num_candidates": NUM_RETURN_SEQUENCES,
        },
        "retrieval": {
            "top_k_intra": TOP_K_INTRA,
            "top_k_cross": TOP_K_CROSS,
            "num_synthetic_users": NUM_SYNTHETIC_USERS,
            "embedder": EMBEDDER_MODEL,
        },
        "results": [
            {
                "config": r.config_name,
                "rouge1": r.rouge1,
                "rouge2": r.rouge2,
                "rougeL": r.rougeL,
                "num_samples": r.num_samples,
                "wall_time_s": r.wall_time_s,
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved to {path}")


# ============================================================
# Entry point
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="JARVIS-M inference-only ablation study"
    )
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--top_k_intra", type=int, default=TOP_K_INTRA)
    p.add_argument("--top_k_cross", type=int, default=TOP_K_CROSS)
    p.add_argument("--num_users", type=int, default=NUM_SYNTHETIC_USERS)
    p.add_argument("--alpha", type=float, default=RERANK_ALPHA)
    p.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    global TOP_K_INTRA, TOP_K_CROSS, NUM_SYNTHETIC_USERS, RERANK_ALPHA
    TOP_K_INTRA = args.top_k_intra
    TOP_K_CROSS = args.top_k_cross
    NUM_SYNTHETIC_USERS = args.num_users
    RERANK_ALPHA = args.alpha

    print("=" * 60)
    print("  JARVIS-M  —  Inference-Only Ablation Study")
    print("=" * 60)

    set_seed()
    device = get_device()

    model, tokenizer = load_model_and_tokenizer(device)

    ds = load_data()
    train_dialogues = ds["train"]["dialogue"]
    train_summaries = ds["train"]["summary"]
    test_dialogues  = ds["test"]["dialogue"]
    test_summaries  = ds["test"]["summary"]

    user_data = build_user_partitions(
        train_dialogues, train_summaries, NUM_SYNTHETIC_USERS
    )

    print(f"\n  Embedder : {EMBEDDER_MODEL}")
    embedder = SentenceTransformer(EMBEDDER_MODEL, device=device)
    retriever = MemoryRetriever(embedder, user_data, device)

    results = run_ablation(
        model, tokenizer, retriever,
        test_dialogues, test_summaries,
        device,
        max_samples=args.max_samples,
    )

    print("\n[5/5] Results\n")
    print_table(results)
    print_latex_table(results)
    save_results(results, args.results_dir)
    print("Done.")


if __name__ == "__main__":
    main()
