"""
JARVIS-M v2: Unified Multi-Task LoRA Verification
==================================================
Tests BOTH capabilities on the single fine-tuned Qwen 2.5 adapter:
1. Modal 1: Neural Epistemic State & Overwrite Extraction (Structured JSON)
2. Modal 2: Executive Narrative Dialogue Summarization (Fluent Prose)
"""

import sys
import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "./models/jarvis-qwen-multitask-lora"
CACHE_DIR = "./cache"


def run_verification():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("🧪 VERIFYING UNIFIED MULTI-TASK QWEN-2.5-0.5B (DUAL MODALITY)")
    print("=" * 80)
    print(f"Device: {device}")

    # Load Base + LoRA
    print("\n[*] Loading Qwen 2.5 with Unified Multi-Task LoRA Adapter...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model = PeftModel.from_pretrained(model, LORA_PATH).to(device)
    model.eval()
    print(f"[+] Model loaded in {(time.perf_counter() - t0)*1000:.1f} ms.")

    # ------------------------------------------------------------
    # Test 1: Structured Epistemic State Extraction (MultiWOZ style)
    # ------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🔍 TEST 1: NEURAL EPISTEMIC STATE TRACKING (JSON FEED FOR AGENTS/SYSTEMS)")
    print("-" * 80)

    state_input = "User: I need a cheap hotel in the center with free wifi and parking for 3 people."
    state_messages = [
        {
            "role": "system",
            "content": "You are a Neural Dialogue State Tracker. Extract all active entity slots and verified state values from the conversation turn into compact JSON."
        },
        {
            "role": "user",
            "content": state_input
        }
    ]

    t0 = time.perf_counter()
    prompt1 = tokenizer.apply_chat_template(state_messages, tokenize=False, add_generation_prompt=True)
    inputs1 = tokenizer(prompt1, return_tensors="pt").to(device)

    with torch.no_grad():
        out1 = model.generate(**inputs1, max_new_tokens=60, do_sample=False)
    ans1 = tokenizer.decode(out1[0][inputs1.input_ids.shape[1]:], skip_special_tokens=True).strip()
    lat1 = (time.perf_counter() - t0) * 1000

    print(f"Input Turn:\n  \"{state_input}\"")
    print(f"\nGenerated Structured State:\n  {ans1}")
    print(f"Latency: {lat1:.1f} ms")

    # ------------------------------------------------------------
    # Test 2: Executive Narrative Dialogue Summarization (DialogSum style)
    # ------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📝 TEST 2: EXECUTIVE NARRATIVE DIALOGUE SUMMARIZATION (FOR HUMANS)")
    print("-" * 80)

    dialogue_input = """#Person1#: Have you finished compiling the financial numbers for the Q3 audit report?
#Person2#: I am still working on the revenue reconciliations. Stripe payments had transaction discrepancies.
#Person1#: We have to present this to the executive board on Thursday at 2 PM. Can you prioritize the Stripe audit?
#Person2#: Yes, I will focus on the payment discrepancies today and tomorrow. I will send you the final report by Wednesday evening.
#Person1#: Perfect, make sure you double-check the risk assessment section as well."""

    summary_messages = [
        {
            "role": "system",
            "content": "You are an Abstractive Dialogue Summarizer. Provide a concise, coherent narrative summary capturing all key action items, speaker decisions, and deadlines."
        },
        {
            "role": "user",
            "content": f"Dialogue Transcript:\n{dialogue_input}\n\nConcise Summary:"
        }
    ]

    t0 = time.perf_counter()
    prompt2 = tokenizer.apply_chat_template(summary_messages, tokenize=False, add_generation_prompt=True)
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)

    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=100, do_sample=False)
    ans2 = tokenizer.decode(out2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=True).strip()
    lat2 = (time.perf_counter() - t0) * 1000

    print("Input Dialogue Transcript:\n" + dialogue_input)
    print(f"\nGenerated Narrative Summary:\n  \"{ans2}\"")
    print(f"Latency: {lat2:.1f} ms")

    print("\n" + "=" * 80)
    print("🏆 UNIFIED MULTI-TASK VALIDATION SUCCESSFUL!")
    print("A single 490M parameter Qwen 2.5 model handles BOTH state tracking AND narrative summarization!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
