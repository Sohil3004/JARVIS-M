"""Chat-model backends used by the reader and the LLM judge.

* ``hf``: local transformers model (Qwen3 / Qwen3.5 / Qwen2.5 ...), greedy decoding.
* ``openai``: any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, LM Studio),
  stdlib-only HTTP so no extra dependency.
* ``scripted``: deterministic fake for tests.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Callable, Dict, Optional, Protocol


class ChatModel(Protocol):
    name: str

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str: ...


class ScriptedChat:
    """Returns ``fn(system, user)``; default echoes a fixed string."""

    def __init__(self, fn: Optional[Callable[[str, str], str]] = None, name: str = "scripted") -> None:
        self.fn = fn or (lambda s, u: "Final: I don't know")
        self.name = name
        self.calls = 0

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        self.calls += 1
        return self.fn(system, user)


class OverlapChat:
    """No-LLM 'reader': answers with the retrieved line that best overlaps the
    question. Lets retrieval be smoke-tested end to end on CPU in seconds."""

    name = "overlap"

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        from .text import word_tokens

        q_part, _, ctx = user.partition("Retrieved passages:")
        q = set(word_tokens(q_part.split("Question:")[-1]))
        best, best_s = "", 0.0
        for line in ctx.splitlines():
            line = line.strip()
            if not line or line.startswith("["):
                continue
            toks = word_tokens(line)
            s = len(q & set(toks)) / (1 + 0.05 * len(toks))
            if s > best_s:
                best, best_s = line, s
        return f"Final: {best or 'Not mentioned in the conversation'}"


class HFChat:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: Optional[str] = None,
        dtype: str = "auto",
        enable_thinking: bool = False,
        adapter: Optional[str] = None,
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        max_input_tokens: Optional[int] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = cache_dir or os.environ.get("JARVIS_CACHE_DIR")
        self.name = model + (":4bit" if load_in_4bit else "")
        self.tok = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        kw = {}
        if load_in_4bit:  # NF4 via bitsandbytes: ~0.6 GB per 1B params (8B judge ~5.5 GB)
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model, cache_dir=cache_dir, torch_dtype=torch_dtype,
            device_map=device or ("cuda" if torch.cuda.is_available() else "cpu"), **kw,
        )
        self.max_input_tokens = max_input_tokens
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.enable_thinking = enable_thinking
        self._torch = torch

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
            )
        except TypeError:  # templates without the thinking switch
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt")
        if self.max_input_tokens and enc["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError(
                f"prompt is {enc['input_ids'].shape[1]} tokens > max_input_tokens={self.max_input_tokens}; "
                "lower retrieval.budget_tokens / max_items"
            )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with self._torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in text:  # drop any reasoning trace
            text = text.split("</think>", 1)[1]
        return text.strip()


class OpenAICompatChat:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        timeout: float = 120.0,
        retries: int = 4,
        extra: Optional[Dict] = None,
    ) -> None:
        self.name = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.temperature, self.timeout, self.retries = temperature, timeout, retries
        self.extra = extra or {}

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        body = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            **self.extra,
        }
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                return (data["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:  # noqa: BLE001 - network errors vary
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"chat request failed after {self.retries} tries: {last}")


_CACHE: Dict[str, ChatModel] = {}


def release_all() -> int:
    """Drop every cached chat model and free GPU memory (e.g. reader -> judge)."""
    n = len(_CACHE)
    _CACHE.clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return n


def build_chat(cfg: Optional[dict]) -> ChatModel:
    cfg = dict(cfg or {"type": "scripted"})
    key = json.dumps(cfg, sort_keys=True, default=str)
    if key in _CACHE:
        return _CACHE[key]
    kind = cfg.pop("type")
    if kind == "hf":
        m: ChatModel = HFChat(**cfg)
    elif kind == "openai":
        m = OpenAICompatChat(**cfg)
    elif kind == "scripted":
        m = ScriptedChat(**cfg)
    elif kind == "overlap":
        m = OverlapChat()
    else:
        raise ValueError(f"unknown chat backend {kind!r}")
    _CACHE[key] = m
    return m
