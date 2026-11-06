"""
JARVIS-M v2 (New Unified Architecture): Qwen-2.5-0.5B Multi-Task LoRA Model Wrapper
===================================================================================
Dedicated model definition and inference interface for the v2 Tier-1 architecture.
- Base Model: Qwen/Qwen2.5-0.5B-Instruct (490M Causal Decoder with GQA)
- Adapters Supported:
    1. jarvis-qwen-multitask-lora (MultiWOZ 2.2 JSON State Tracking + DialogSum Narrative Summarization)
    2. jarvis-qwen-state-lora (Single-task MultiWOZ 2.2 Epistemic State Reasoner)
- Coupled with: AGM Temporal State Reconciliation Engine + Information-Theoretic Token Pruner
"""

import sys
import os
import time
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class QwenUnifiedMultiTaskModel:
    """
    Dedicated wrapper for the new v2 Unified Qwen-2.5-0.5B + Multi-Task LoRA Model.
    Supports:
      - Modality 1: Neural Epistemic State Tracking (JSON output)
      - Modality 2: Executive Narrative Dialogue Summarization (Prose output)
      - Modality 3: Memory-Conditioned Downstream QA
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lora_adapter_path: str = os.environ.get("JARVIS_V2_ADAPTER", "./models/jarvis-qwen-multitask-lora"),
        cache_dir: str = os.environ.get("JARVIS_CACHE_DIR", "./cache"),
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.cache_dir = cache_dir

        print(f"[*] Loading New v2 Model ({base_model_name}) on {self.device.upper()}...")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        if os.path.exists(lora_adapter_path):
            print(f"[*] Attaching v2 Multi-Task Qwen LoRA Adapter from: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_path).to(self.device)
            self.adapter_loaded = True
        else:
            print(f"[!] Warning: LoRA adapter not found at {lora_adapter_path}, using base Qwen.")
            self.model = base_model
            self.adapter_loaded = False

        self.model.eval()
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
        print(f"[+] New v2 Qwen-2.5 Multi-Task Model ready in {self.load_ms} ms.")

    def _generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 80, use_adapter: bool = True) -> Tuple[str, float]:
        t0 = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if not use_adapter and self.adapter_loaded and hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        response = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return response, latency_ms

    def extract_epistemic_state_json(self, dialogue_text: str) -> Dict[str, Any]:
        """Modality 1: Structured JSON Slot Extraction (MultiWOZ 2.2 trained)."""
        messages = [
            {
                "role": "system",
                "content": "You are a Neural Dialogue State Tracker. Extract all active entity slots and verified state values from the conversation turn into compact JSON."
            },
            {"role": "user", "content": dialogue_text}
        ]
        ans, lat = self._generate(messages, max_new_tokens=75, use_adapter=True)
        return {"architecture": "v2_Qwen2.5_MultiTask_LoRA", "modality": "structured_state_json", "output": ans, "latency_ms": lat}

    def summarize_executive_narrative(self, dialogue_text: str) -> Dict[str, Any]:
        """Modality 2: Fluent Executive Narrative Summary (DialogSum trained)."""
        messages = [
            {
                "role": "system",
                "content": "You are an Abstractive Dialogue Summarizer. Provide a concise, coherent narrative summary capturing all key action items, speaker decisions, and deadlines."
            },
            {"role": "user", "content": f"Dialogue Transcript:\n{dialogue_text}\n\nConcise Summary:"}
        ]
        ans, lat = self._generate(messages, max_new_tokens=100, use_adapter=True)
        return {"architecture": "v2_Qwen2.5_MultiTask_LoRA", "modality": "narrative_summary", "output": ans, "latency_ms": lat}

    def answer_downstream_query(self, optimized_context: str, query: str) -> Dict[str, Any]:
        """Modality 3: Downstream Question Answering over Reconciled Context."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise enterprise assistant. Answer the user's question concisely in 1 to 2 sentences based ONLY on the active current state in the provided context."
            },
            {"role": "user", "content": f"Context:\n{optimized_context}\n\nQuestion: {query}"}
        ]
        ans, lat = self._generate(messages, max_new_tokens=70, use_adapter=False)
        return {"architecture": "v2_Qwen2.5_Base_Instruct", "modality": "downstream_qa", "output": ans, "latency_ms": lat}


if __name__ == "__main__":
    v2_model = QwenUnifiedMultiTaskModel()
    sample_state = "User: I need a cheap hotel in the center with free wifi and parking for 3 people."
    print("\n--- v2 Qwen Modality 1 (JSON State) ---")
    print(v2_model.extract_epistemic_state_json(sample_state))

    sample_diag = (
        "Alice: Good catch, we officially switched to Google Cloud Run.\n"
        "Charlie: I granted Artifact Registry Reader IAM role to Bob's service account.\n"
        "Alice: Updated deadline is Monday 10 AM."
    )
    print("\n--- v2 Qwen Modality 2 (Narrative Summary) ---")
    print(v2_model.summarize_executive_narrative(sample_diag))
