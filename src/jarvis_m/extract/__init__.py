"""State extraction: heuristic (inline, free) and LLM (offline, cached).

    jarvis-eval extract configs/exp/<exp>.yaml --extractor configs/models/extractor_qwen3_4b_4bit.yaml

writes ``$JARVIS_DATA_DIR/../extract/<dataset>/<extractor>.jsonl`` (resumable). Hypergraph
memories then use ``extractor: {type: cached, path: ...}``; messages missing from the
cache fall back to the heuristic extractor so a partial cache is still usable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from ..types import Message
from .heuristic import HeuristicExtractor, entities
from .llm_extractor import PROMPT_VERSION, LLMExtractor, parse_records
from .records import MODALITIES, STATEFUL, StateRecord


class ExtractionStore:
    """Append-only JSONL: one line per processed message ``{"msg": id, "records": [...]}``,
    plus one line per LLM batch ``{"batch": key, "records": [...]}`` (the batch cache: records
    carry the target's index ``i`` instead of a message id). The LLM extractor's slot
    registry is kept next to it (``<name>.registry.json``)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: Dict[str, List[StateRecord]] = {}
        self.batches: Dict[str, List[Dict[str, Any]]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    if "batch" in d:
                        self.batches[d["batch"]] = d["records"]
                    else:
                        self.records[d["msg"]] = [StateRecord.from_json(r) for r in d["records"]]

    @property
    def registry_path(self) -> Path:
        return self.path.with_suffix(".registry.json")

    def get_batch(self, key: str) -> Optional[List[Dict[str, Any]]]:
        return self.batches.get(key)

    def put_batch(self, key: str, records: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batches[key] = records
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"batch": key, "records": records}) + "\n")

    def done(self) -> Set[str]:
        return set(self.records)

    def append(self, recs: Sequence[StateRecord], done_ids: Sequence[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        by: Dict[str, List[StateRecord]] = {i: [] for i in done_ids}
        for r in recs:
            by.setdefault(r.msg_id, []).append(r)
        with self.path.open("a", encoding="utf-8") as f:
            for mid, rs in by.items():
                self.records[mid] = rs
                f.write(json.dumps({"msg": mid, "records": [r.to_json() for r in rs]}) + "\n")


_STORES: Dict[tuple, ExtractionStore] = {}


def _shared_store(path: str) -> ExtractionStore:
    p = Path(path)
    key = (str(p.resolve()), p.stat().st_mtime_ns if p.exists() else 0)
    if key not in _STORES:
        _STORES[key] = ExtractionStore(p)
    return _STORES[key]


class CachedExtractor:
    """Reads an ExtractionStore; heuristic fallback for messages not in it."""

    def __init__(self, path: str, fallback: Optional[str] = "heuristic", allow_missing: bool = False) -> None:
        if not Path(path).exists() and not allow_missing:
            raise FileNotFoundError(
                f"extraction cache {path} not found: run `jarvis-eval extract ...` first "
                "(or set allow_missing: true to use the fallback extractor only)")
        self.store = _shared_store(path)  # memories are rebuilt per conversation: parse the cache once
        self.fallback = HeuristicExtractor() if fallback == "heuristic" else None
        self.name = f"cached:{Path(path).stem}"
        self.misses = 0

    def extract(self, messages: Sequence[Message]) -> List[StateRecord]:
        out: List[StateRecord] = []
        for m in messages:
            if m.id in self.store.records:
                out.extend(self.store.records[m.id])
            elif self.fallback is not None:
                self.misses += 1
                out.extend(self.fallback.extract_one(m))
        return out


class NullExtractor:
    name = "none"

    def extract(self, messages: Sequence[Message]) -> List[StateRecord]:
        return []


def build_extractor(cfg: Optional[Dict[str, Any]]):
    cfg = dict(cfg or {"type": "heuristic"})
    kind = cfg.pop("type")
    if kind == "heuristic":
        return HeuristicExtractor()
    if kind == "llm":
        return LLMExtractor(**cfg)
    if kind == "cached":
        return CachedExtractor(**cfg)
    if kind == "none":
        return NullExtractor()
    raise ValueError(f"unknown extractor {kind!r}")


__all__ = [
    "MODALITIES", "STATEFUL", "StateRecord", "HeuristicExtractor", "LLMExtractor", "CachedExtractor",
    "NullExtractor", "ExtractionStore", "build_extractor", "entities", "parse_records", "PROMPT_VERSION",
]
