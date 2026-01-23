"""
JARVIS-M: Updated Inference Module with Improved Compression
============================================================
This script provides an updated summarization module that enforces proper
compression ratios and prevents the model from simply rewriting input text.

Goal: Fix the "output too long" issue by enforcing compression constraints.
"""

import os
import torch
from typing import Optional, Union, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

# ============================================================
# Configuration
# ============================================================
BASE_MODEL_NAME = "facebook/bart-large-cnn"
LORA_ADAPTER_PATH = "./models/jarvis-bart-lora"  # Path to fine-tuned LoRA adapter
CACHE_DIR = "./cache"

# Compression settings
TARGET_COMPRESSION_RATIO = 0.5  # Output should be ~50% of input length
MIN_COMPRESSION_RATIO = 0.3    # Minimum compression (30% of input)
MAX_COMPRESSION_RATIO = 0.7    # Maximum compression (70% of input)
ABSOLUTE_MIN_LENGTH = 20       # Minimum output tokens
ABSOLUTE_MAX_LENGTH = 150      # Maximum output tokens

# Generation parameters for better compression
GENERATION_CONFIG = {
    "length_penalty": 2.0,          # Penalize longer outputs to encourage brevity
    "no_repeat_ngram_size": 3,      # Prevent repetition of 3-grams
    "num_beams": 4,                 # Beam search for better quality
    "early_stopping": True,         # Stop when all beams finish
    "do_sample": False,             # Deterministic generation
    "temperature": 1.0,             # Not used with do_sample=False, but set for clarity
    "repetition_penalty": 1.2,      # Additional repetition penalty
}


class JarvisSummarizer:
    """
    Enhanced summarizer with proper compression enforcement.
    
    Key improvements:
    1. Dynamic max_length based on input length (enforces compression ratio)
    2. length_penalty=2.0 to encourage shorter outputs
    3. no_repeat_ngram_size=3 to prevent repetitive text
    4. Optional LoRA adapter loading for fine-tuned model
    """
    
    def __init__(
        self,
        use_lora: bool = True,
        device: Optional[str] = None,
        compression_ratio: float = TARGET_COMPRESSION_RATIO,
    ):
        """
        Initialize the summarizer.
        
        Args:
            use_lora: Whether to load the fine-tuned LoRA adapter
            device: Device to use ('cuda', 'cpu', or None for auto)
            compression_ratio: Target output/input length ratio (0.0-1.0)
        """
        self.compression_ratio = compression_ratio
        
        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔧 Initializing Jarvis Summarizer on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME, 
            cache_dir=CACHE_DIR
        )
        
        # Load base model
        print(f"📦 Loading base model: {BASE_MODEL_NAME}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_NAME,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        # Load LoRA adapter if available and requested
        if use_lora and os.path.exists(LORA_ADAPTER_PATH):
            print(f"🔄 Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(self.model, LORA_ADAPTER_PATH)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
            print("✓ LoRA adapter loaded and merged")
        elif use_lora:
            print(f"⚠ LoRA adapter not found at {LORA_ADAPTER_PATH}, using base model")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Summarizer initialized successfully")
    
    def _calculate_dynamic_lengths(self, input_text: str) -> tuple:
        """
        Calculate dynamic min/max lengths based on input length.
        
        This ensures the model actually compresses the text rather than
        just rewriting it at similar length.
        """
        # Tokenize to get accurate input length
        input_tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        input_length = len(input_tokens)
        
        # Calculate target length based on compression ratio
        target_length = int(input_length * self.compression_ratio)
        
        # Calculate min/max with bounds
        min_length = max(
            ABSOLUTE_MIN_LENGTH,
            int(input_length * MIN_COMPRESSION_RATIO)
        )
        max_length = min(
            ABSOLUTE_MAX_LENGTH,
            max(target_length, int(input_length * MAX_COMPRESSION_RATIO))
        )
        
        # Ensure min <= max
        min_length = min(min_length, max_length - 10)
        min_length = max(10, min_length)  # Absolute minimum
        
        return min_length, max_length, input_length
    
    def summarize_text_safe(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        return_stats: bool = False,
    ) -> Union[str, tuple]:
        """
        Summarize text with proper compression enforcement.
        
        This is a drop-in replacement for the original summarize_text_safe
        function with improved compression behavior.
        
        Args:
            text: Input text to summarize
            max_length: Override maximum output length (uses dynamic if None)
            min_length: Override minimum output length (uses dynamic if None)
            return_stats: If True, return (summary, stats_dict)
            
        Returns:
            Summarized text, or (summary, stats) if return_stats=True
        """
        if not text or len(text.strip()) < 10:
            return ("", {"error": "Input too short"}) if return_stats else ""
        
        # Clean input
        text = text.strip()
        
        # Calculate dynamic lengths if not provided
        dyn_min, dyn_max, input_token_count = self._calculate_dynamic_lengths(text)
        
        if max_length is None:
            max_length = dyn_max
        if min_length is None:
            min_length = dyn_min
        
        # Prepare input with task prefix
        input_text = "summarize: " + text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate with compression-focused parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=GENERATION_CONFIG["length_penalty"],
                no_repeat_ngram_size=GENERATION_CONFIG["no_repeat_ngram_size"],
                num_beams=GENERATION_CONFIG["num_beams"],
                early_stopping=GENERATION_CONFIG["early_stopping"],
                do_sample=GENERATION_CONFIG["do_sample"],
                repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate compression statistics
        output_token_count = len(self.tokenizer.encode(summary, add_special_tokens=False))
        actual_compression = output_token_count / max(1, input_token_count)
        
        if return_stats:
            stats = {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "compression_ratio": actual_compression,
                "target_compression": self.compression_ratio,
                "max_length_used": max_length,
                "min_length_used": min_length,
            }
            return summary, stats
        
        return summary
    
    def summarize_chunks(
        self,
        text: str,
        chunk_size: int = 1500,
        final_summary: bool = True,
    ) -> str:
        """
        Summarize long text by chunking and combining summaries.
        
        For very long inputs, this splits the text into chunks,
        summarizes each, then optionally combines into a final summary.
        
        Args:
            text: Input text to summarize
            chunk_size: Maximum characters per chunk
            final_summary: Whether to summarize the combined chunk summaries
            
        Returns:
            Final summarized text
        """
        text = text.strip()
        
        # If short enough, summarize directly
        if len(text) <= chunk_size:
            return self.summarize_text_safe(text)
        
        # Split into chunks at sentence boundaries
        chunks = self._split_into_chunks(text, chunk_size)
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarize_text_safe(chunk)
            if summary:
                chunk_summaries.append(summary)
        
        if not chunk_summaries:
            return ""
        
        # Combine summaries
        combined = " ".join(chunk_summaries)
        
        # Optionally create final summary of summaries
        if final_summary and len(combined) > chunk_size // 2:
            return self.summarize_text_safe(combined)
        
        return combined
    
    def _split_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Find last sentence boundary before end
            slice_text = text[start:end]
            
            # Try to break at sentence end
            for sep in ['. ', '? ', '! ', '\n']:
                last_sep = slice_text.rfind(sep)
                if last_sep > max_chars // 2:
                    end = start + last_sep + 1
                    break
            
            chunks.append(text[start:end].strip())
            start = end
        
        return [c for c in chunks if c]


def create_summarize_function(use_lora: bool = True, compression_ratio: float = 0.5):
    """
    Factory function to create a summarize_text_safe compatible function.
    
    This can be used as a drop-in replacement in existing code:
    
    ```python
    from update_inference import create_summarize_function
    summarize_text_safe = create_summarize_function(use_lora=True)
    ```
    """
    summarizer = JarvisSummarizer(use_lora=use_lora, compression_ratio=compression_ratio)
    return summarizer.summarize_text_safe


# ============================================================
# Pipeline-based alternative (for simpler integration)
# ============================================================
def get_improved_pipeline(use_lora: bool = True):
    """
    Get an improved summarization pipeline with compression settings.
    
    This returns a pipeline object that can be used similarly to the
    original transformers pipeline but with better compression.
    """
    device = 0 if torch.cuda.is_available() else -1
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=CACHE_DIR)
    
    # Load LoRA if available
    if use_lora and os.path.exists(LORA_ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
    
    # Create pipeline with improved generation config
    pipe = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    
    return pipe


def summarize_with_compression(
    pipe,
    text: str,
    compression_ratio: float = TARGET_COMPRESSION_RATIO,
) -> str:
    """
    Use a pipeline with dynamic compression-based length limits.
    
    Args:
        pipe: Hugging Face summarization pipeline
        text: Input text
        compression_ratio: Target compression ratio
        
    Returns:
        Summarized text
    """
    # Calculate input length
    input_tokens = len(text.split())  # Approximate
    
    # Calculate target output length
    max_length = min(ABSOLUTE_MAX_LENGTH, max(ABSOLUTE_MIN_LENGTH, int(input_tokens * compression_ratio)))
    min_length = max(10, int(max_length * 0.5))
    
    result = pipe(
        text,
        max_length=max_length,
        min_length=min_length,
        length_penalty=GENERATION_CONFIG["length_penalty"],
        no_repeat_ngram_size=GENERATION_CONFIG["no_repeat_ngram_size"],
        num_beams=GENERATION_CONFIG["num_beams"],
        early_stopping=GENERATION_CONFIG["early_stopping"],
        do_sample=GENERATION_CONFIG["do_sample"],
    )
    
    return result[0]["summary_text"]


# ============================================================
# Demo / Test
# ============================================================
def main():
    """Demo the improved summarization with compression metrics."""
    print("=" * 60)
    print("JARVIS-M: Updated Inference with Compression Control")
    print("=" * 60)
    
    # Sample dialogue from DialogSum style
    test_dialogue = """
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
    
    # Initialize summarizer
    print("\n📦 Initializing summarizer...")
    summarizer = JarvisSummarizer(use_lora=False, compression_ratio=0.5)  # use_lora=False for demo without fine-tuned model
    
    # Test with stats
    print("\n" + "-" * 60)
    print("📝 Test Input:")
    print(test_dialogue.strip())
    
    print("\n" + "-" * 60)
    print("🔄 Generating summary with compression control...")
    
    summary, stats = summarizer.summarize_text_safe(test_dialogue, return_stats=True)
    
    print("\n📊 Compression Statistics:")
    print(f"  Input tokens:  {stats['input_tokens']}")
    print(f"  Output tokens: {stats['output_tokens']}")
    print(f"  Compression:   {stats['compression_ratio']:.2%}")
    print(f"  Target:        {stats['target_compression']:.2%}")
    print(f"  Max length:    {stats['max_length_used']}")
    print(f"  Min length:    {stats['min_length_used']}")
    
    print("\n" + "-" * 60)
    print("📄 Summary:")
    print(summary)
    
    # Compare with original settings (no compression enforcement)
    print("\n" + "=" * 60)
    print("📊 Comparison: Original vs Improved Settings")
    print("=" * 60)
    
    # Original settings (approximate)
    original_settings = {
        "max_length": 120,
        "min_length": 25,
        "do_sample": False,
    }
    
    # Improved settings
    improved_settings = {
        "max_length": stats['max_length_used'],
        "min_length": stats['min_length_used'],
        "length_penalty": GENERATION_CONFIG["length_penalty"],
        "no_repeat_ngram_size": GENERATION_CONFIG["no_repeat_ngram_size"],
        "repetition_penalty": GENERATION_CONFIG["repetition_penalty"],
    }
    
    print("\nOriginal settings:")
    for k, v in original_settings.items():
        print(f"  {k}: {v}")
    
    print("\nImproved settings:")
    for k, v in improved_settings.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Key improvements:")
    print("  1. Dynamic max_length based on input (enforces ~50% compression)")
    print("  2. length_penalty=2.0 (encourages shorter outputs)")
    print("  3. no_repeat_ngram_size=3 (prevents repetitive phrases)")
    print("  4. repetition_penalty=1.2 (additional repetition control)")
    
    print("\n" + "=" * 60)
    print("Integration Guide:")
    print("=" * 60)
    print("""
To use in your existing code, replace:

    from jarvis_m_plus_full import summarize_text_safe

With:

    from update_inference import create_summarize_function
    summarize_text_safe = create_summarize_function(use_lora=True)

Or for more control:

    from update_inference import JarvisSummarizer
    summarizer = JarvisSummarizer(use_lora=True, compression_ratio=0.5)
    summary = summarizer.summarize_text_safe(text)
    """)
    
    return summarizer


if __name__ == "__main__":
    main()
