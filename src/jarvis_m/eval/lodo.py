"""Leave-one-domain-out (LODO) configs.

    jarvis-eval make-lodo configs/train/hgnn_v2_mix.yaml --out configs/train/lodo/ \
        [--eval-config configs/eval/rank_v2.yaml]

For every domain D in the train config's ``train_data`` this writes

* ``<name>_lodo_<D>.yaml``       the train config without any dataset of domain D
  (train and dev), saving to ``<output stem>_lodo_<D>.pt``;
* ``rank_<name>_lodo_<D>.yaml``  a rank-eval config over the held-out domain's datasets
  with one arm that globs ``<output stem>_lodo_<D>*.pt`` (so extra seeds saved as
  ``..._lodo_<D>_s1.pt`` join the arm).

The held-out datasets come from ``--eval-config`` when given (its entries of domain D),
otherwise from the train config's own entries of domain D, which that model never saw.
Configs are written fully resolved (no ``base:``), so they can live anywhere.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import load_config
from ..data.loaders import dataset_domain


def make_lodo(train_cfg_path: str | Path, out_dir: str | Path,
              eval_cfg_path: Optional[str | Path] = None) -> List[Path]:
    cfg = load_config(train_cfg_path)
    for key in ("train_data", "dev_data"):
        for e in cfg.get(key) or []:
            dataset_domain(e, required=True)
    domains = sorted({e["domain"] for e in cfg["train_data"]})
    if len(domains) < 2:
        raise ValueError(f"leave-one-domain-out needs at least two domains in train_data, got {domains}")
    eval_sets = load_config(eval_cfg_path)["datasets"] if eval_cfg_path else None
    name = cfg.get("name", Path(train_cfg_path).stem)
    base_out = Path(cfg.get("output", f"models/hgnn/{name}.pt"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for dom in domains:
        arm = f"{name}_lodo_{dom}"
        ckpt = base_out.with_name(f"{base_out.stem}_lodo_{dom}.pt")
        c: Dict[str, Any] = copy.deepcopy(cfg)
        c["name"] = arm
        c["output"] = ckpt.as_posix()
        c["train_data"] = [e for e in cfg["train_data"] if e["domain"] != dom]
        c["dev_data"] = [e for e in cfg.get("dev_data") or [] if e["domain"] != dom]
        c["lodo"] = {"held_out_domain": dom, "source": str(train_cfg_path)}
        if eval_sets is not None:
            held = [copy.deepcopy(e) for e in eval_sets if dataset_domain(e, required=True) == dom]
        else:
            held = [{k: v for k, v in e.items() if k != "weight"}
                    for e in [*cfg["train_data"], *(cfg.get("dev_data") or [])] if e["domain"] == dom]
        rank: Dict[str, Any] = {
            "memory": copy.deepcopy(cfg["memory"]), "k": [5, 10], "datasets": held,
            "arms": {arm: [ckpt.with_name(f"{ckpt.stem}*.pt").as_posix()]},
            "lodo": {"held_out_domain": dom, "source": str(train_cfg_path)},
        }
        if cfg.get("tokenizer"):
            rank["tokenizer"] = cfg["tokenizer"]
        for path, obj in ((out_dir / f"{arm}.yaml", c), (out_dir / f"rank_{arm}.yaml", rank)):
            path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
            written.append(path)
    return written
