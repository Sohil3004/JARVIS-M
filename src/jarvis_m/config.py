"""YAML experiment configs: ``base`` includes, ``--set`` overrides, ablation arms.

An experiment file may define either

* ``arms: [{name: ..., set: {dotted.key: value}}, ...]`` - named ablation arms, or
* ``sweep: {dotted.key: [v1, v2], ...}`` - cartesian product,

each expanded against the rest of the file into concrete run configs.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not v.get("_replace"):
            out[k] = deep_merge(out[k], v)
        else:
            v = copy.deepcopy(v)
            if isinstance(v, dict):
                v.pop("_replace", None)
            out[k] = v
    return out


def set_dotted(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = cfg.pop("base", []) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        merged = deep_merge(merged, load_config((path.parent / b).resolve()))
    return deep_merge(merged, cfg)


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        set_dotted(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def expand(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(arm_name, run_cfg)]; a plain config yields one arm named ''."""
    cfg = copy.deepcopy(cfg)
    arms = cfg.pop("arms", None)
    sweep = cfg.pop("sweep", None)
    if arms:
        out = []
        for arm in arms:
            c = copy.deepcopy(cfg)
            for k, v in (arm.get("set") or {}).items():
                set_dotted(c, k, copy.deepcopy(v))
            out.append((arm["name"], c))
        return out
    if sweep:
        keys = list(sweep)
        out = []
        for combo in itertools.product(*(sweep[k] for k in keys)):
            c = copy.deepcopy(cfg)
            for k, v in zip(keys, combo):
                set_dotted(c, k, copy.deepcopy(v))
            name = ",".join(f"{k.split('.')[-1]}={_short(v)}" for k, v in zip(keys, combo))
            out.append((name, c))
        return out
    return [("", cfg)]


def _short(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("label") or v.get("type") or "cfg")
    return str(v)
