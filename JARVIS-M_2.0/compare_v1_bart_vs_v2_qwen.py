"""
JARVIS-M: Head-to-Head Model & Architecture Comparison (v1 BART vs. v2 Qwen-2.5)
================================================================================
Directly compares:
1. Old Architecture (`v1_bart_baseline/src/bart_legacy_model.py`):
   - Model: facebook/bart-large-cnn + jarvis-bart-lora
   - Behavior: Summarizes raw dialogue without temporal state reconciliation.
2. New Architecture (`v2_qwen_epistemic/core/qwen_unified_model.py`):
   - Model: Qwen/Qwen2.5-0.5B-Instruct + jarvis-qwen-multitask-lora
   - Behavior: Couples AGM Temporal State Reconciliation + Multi-Task Dual Modality
     (Modality 1: Structured JSON State + Modality 2: Reconciled Executive Summary + Modality 3: Downstream QA).
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "v1_bart_baseline", "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "v2_qwen_epistemic", "core"))

from bart_legacy_model import BartLegacySummarizer
from qwen_unified_model import QwenUnifiedMultiTaskModel
from memory_conditioned_optimizer import MemoryConditionedContextOptimizer


def run_comparison():
    print("=" * 85)
    print("JARVIS-M ARCHITECTURE COMPARISON: v1 (BART-Large-CNN) vs. v2 (Unified Qwen-2.5)")
    print("=" * 85)

    # Multi-session test scenario with contradiction override
    turns = [
        {"session_id": "s1", "speaker": "Alice", "text": "Team, we officially decided to use AWS EC2 for backend hosting."},
        {"session_id": "s1", "speaker": "Alice", "text": "The release deadline is Friday 5 PM."},
        {"session_id": "s2", "speaker": "Charlie", "text": "Wait, our startup grant only covers Google Cloud credits."},
        {"session_id": "s2", "speaker": "Alice", "text": "Good catch! We officially switched to Google Cloud Run to use the credits."},
        {"session_id": "s3", "speaker": "Bob", "text": "Cloud Run deployment is failing with Error 403 on gcr.io container registry."},
        {"session_id": "s3", "speaker": "Charlie", "text": "I granted the Artifact Registry Reader IAM role to Bob's service account."},
        {"session_id": "s3", "speaker": "Bob", "text": "Resolved Error 403 container registry permissions issue."},
        {"session_id": "s3", "speaker": "Alice", "text": "Great, the updated deadline is Monday 10 AM."},
    ]
    raw_transcript = "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])
    query = "Which cloud platform are we deploying on and what is the current release deadline?"

    # ------------------------------------------------------------------
    # 1. Run Old v1 Architecture (BART-Large-CNN + LoRA on Raw Text)
    # ------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[OLD ARCHITECTURE] v1_bart_baseline: facebook/bart-large-cnn + jarvis-bart-lora")
    print("-" * 85)
    v1_bart = BartLegacySummarizer()
    v1_out = v1_bart.summarize_dialogue(raw_transcript)
    print(f"Modality Supported : {v1_out['modality']}")
    print(f"Inference Latency  : {v1_out['latency_ms']} ms")
    print(f"Generated Summary  :\n  \"{v1_out['summary']}\"")
    print("Limitation         : Retains obsolete 'AWS EC2' / 'Friday 5 PM' anchors and cannot emit JSON state.")

    # ------------------------------------------------------------------
    # 2. Run New v2 Architecture (Epistemic Optimizer + Unified Qwen-2.5)
    # ------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[NEW ARCHITECTURE] v2_qwen_epistemic: Qwen-2.5-0.5B + jarvis-qwen-multitask-lora")
    print("-" * 85)
    optimizer = MemoryConditionedContextOptimizer(
        tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
        cache_dir=os.path.join(ROOT_DIR, "cache")
    )
    optimized_prompt, stats = optimizer.optimize_multisession_context(turns, target_token_budget=150)

    v2_qwen = QwenUnifiedMultiTaskModel()
    mod1_json = v2_qwen.extract_epistemic_state_json("User: I need a cheap hotel in the center with free wifi and parking for 3 people.")
    mod2_summary = v2_qwen.summarize_executive_narrative(optimized_prompt)
    mod3_qa = v2_qwen.answer_downstream_query(optimized_prompt, query)

    print(f"Context Compression: {stats['original_tokens']} -> {stats['compressed_tokens']} tokens ({stats['token_savings_pct']}% saved)")
    print(f"Superseded Purged  : {stats['superseded_states_count']} obsolete decisions invalidated")
    print(f"\n[Modality 1 - Neural JSON State Feed ({mod1_json['latency_ms']} ms)]:\n  {mod1_json['output']}")
    print(f"\n[Modality 2 - Reconciled Narrative Summary ({mod2_summary['latency_ms']} ms)]:\n  \"{mod2_summary['output']}\"")
    print(f"\n[Modality 3 - Downstream Grounded QA ({mod3_qa['latency_ms']} ms)]:\n  \"{mod3_qa['output']}\"")
    print("=" * 85)


if __name__ == "__main__":
    run_comparison()
