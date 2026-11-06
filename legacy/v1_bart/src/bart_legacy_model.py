"""
JARVIS-M v1 (Legacy Architecture): BART-Large-CNN + LoRA Model Wrapper
======================================================================
Dedicated model definition and inference interface for the v1 ICCSIT 2026 baseline.
- Base Model: facebook/bart-large-cnn (406M Encoder-Decoder)
- Adapter: jarvis-bart-lora (Fine-tuned on DialogSum for abstractive summarization)
- Retrieval: Flat FAISS vector similarity (all-MiniLM-L6-v2)
"""

import sys
import os
import time
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class BartLegacySummarizer:
    """
    Dedicated wrapper for the v1 Legacy BART-Large-CNN + LoRA Summarizer.
    """
    def __init__(
        self,
        base_model_name: str = "facebook/bart-large-cnn",
        lora_adapter_path: str = os.environ.get("JARVIS_V1_ADAPTER", "./models/jarvis-bart-lora-checkpoints"),
        cache_dir: str = os.environ.get("JARVIS_CACHE_DIR", "./cache"),
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.cache_dir = cache_dir

        print(f"[*] Loading Legacy v1 Model ({base_model_name}) on {self.device.upper()}...")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, cache_dir=cache_dir).to(self.device)

        if os.path.exists(lora_adapter_path):
            print(f"[*] Attaching v1 BART LoRA Adapter from: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_path).to(self.device)
        else:
            print(f"[!] Warning: LoRA adapter not found at {lora_adapter_path}, using base BART.")
            self.model = base_model

        self.model.eval()
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
        print(f"[+] Legacy v1 BART-Large-CNN ready in {self.load_ms} ms.")

    def summarize_dialogue(self, dialogue_text: str, max_length: int = 128, min_length: int = 25) -> Dict[str, Any]:
        """
        Generates a single-modality abstractive dialogue summary using BART-Large-CNN.
        Note: Does NOT perform temporal state overwrite reconciliation or structured JSON slot tracking.
        """
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            dialogue_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                num_beams=4,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True,
            )

        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "architecture": "v1_BART_Large_CNN_LoRA",
            "modality": "abstractive_prose_only",
            "summary": summary_text,
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    v1_model = BartLegacySummarizer()
    sample = (
        "Alice: We decided to use AWS EC2 for backend hosting. Deadline is Friday 5 PM.\n"
        "Charlie: Our grant only covers Google Cloud credits.\n"
        "Alice: Good catch, we officially switched to Google Cloud Run. Updated deadline is Monday 10 AM."
    )
    out = v1_model.summarize_dialogue(sample)
    print("\n--- v1 Legacy BART Output ---")
    for k, v in out.items():
        print(f"{k}: {v}")
