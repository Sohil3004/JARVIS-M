"""
JARVIS-M v2: Memory-Conditioned Context Optimizer
=================================================
Couples the Temporal State Reconciliation Engine with Information-Theoretic Token Pruning.

Core Logic:
1. Ingests multi-turn conversational stream into the State Reconciliation Engine.
2. Formats reconciled ground truth (verified active decisions, cleared blockers, active issues).
3. Evaluates older turns: if their underlying topic is already settled in state (or superseded),
   they are collapsed into the compact state header, eliminating duplicate tokens and Primacy Bias.
4. Channels surplus token budget to active and recent conversational discourse.
5. Emits an optimized prompt strictly adhering to target token budget B.
"""

import sys
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from reconciliation_engine import TemporalStateReconciliationEngine
from token_pruner import InformationTokenPruner

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MemoryConditionedContextOptimizer:
    """
    End-to-end prompt and context optimizer conditioned on temporal epistemic memory.
    """
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        cache_dir: str = os.environ.get("JARVIS_CACHE_DIR", "./cache"),
    ):
        self.pruner = InformationTokenPruner(tokenizer_name=tokenizer_name, cache_dir=cache_dir)
        self.reconciler = TemporalStateReconciliationEngine()

    def count_tokens(self, text: str) -> int:
        return self.pruner.count_tokens(text)

    def optimize_multisession_context(
        self,
        chat_turns: List[Dict[str, str]],
        target_token_budget: int = 120,
        preserve_recent_turns: int = 2,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Takes raw multi-session turns:
        [{'session_id': 's1', 'speaker': 'Alice', 'text': '...'}, ...]
        and returns an optimized, contradiction-free context string.
        """
        t0 = time.perf_counter()

        # Step 1: Reconcile temporal state across all turns
        reconciler = TemporalStateReconciliationEngine()
        reconciler.ingest_session_turns(chat_turns)
        reconciled_header = reconciler.format_reconciled_prefix()
        header_tokens = self.count_tokens(reconciled_header)
        ground_truth = reconciler.get_reconciled_ground_truth()

        # Step 2: Calculate remaining budget for recent discourse
        remaining_budget = max(15, target_token_budget - header_tokens)

        # Step 3: Extract recent discourse vs older turns
        raw_full_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in chat_turns])
        orig_tokens = self.count_tokens(raw_full_text)

        if len(chat_turns) <= preserve_recent_turns:
            recent_turns = chat_turns
            older_turns = []
        else:
            recent_turns = chat_turns[-preserve_recent_turns:]
            older_turns = chat_turns[:-preserve_recent_turns]

        recent_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in recent_turns])
        recent_tokens = self.count_tokens(recent_text)

        # Step 4: Conditioned Pruning of older discourse
        # Turns already captured in the epistemic graph (ACTIVE, SUPERSEDED, RESOLVED, BLOCKED)
        # are collapsed into the header so stale superseded anchors never reach the LLM.
        extracted_raw_texts = {n["raw_text"].strip() for n in ground_truth["all_nodes"]}
        unsettled_older_turns = [
            t for t in older_turns
            if t.get("text", "").strip() not in extracted_raw_texts
        ]

        if unsettled_older_turns and remaining_budget > recent_tokens + 15:
            older_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in unsettled_older_turns])
            budget_for_older = max(8, remaining_budget - recent_tokens)
            ratio_for_older = min(0.30, budget_for_older / max(1, self.count_tokens(older_text)))

            pruned_older, _ = self.pruner.prune_dialogue_to_budget(
                dialogue_text=older_text,
                budget_ratio=ratio_for_older,
                preserve_recent_turns=0
            )
            assembled_discourse = "[Recent Context]:\n" + pruned_older + "\n" + recent_text
        else:
            assembled_discourse = "[Recent Context]:\n" + recent_text

        # Step 5: Final Assembled Prompt
        final_prompt = reconciled_header + "\n\n" + assembled_discourse
        final_tokens = self.count_tokens(final_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        stats = {
            "original_tokens": orig_tokens,
            "compressed_tokens": final_tokens,
            "compression_ratio": round(final_tokens / max(1, orig_tokens), 4),
            "token_savings_pct": round((1.0 - (final_tokens / max(1, orig_tokens))) * 100, 2),
            "reconciled_header_tokens": header_tokens,
            "latency_ms": round(latency_ms, 2),
            "active_states_count": len(ground_truth["active_states"]),
            "superseded_states_count": ground_truth["superseded_count"],
        }

        return final_prompt, stats
