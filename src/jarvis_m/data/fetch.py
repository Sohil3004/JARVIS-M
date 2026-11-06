"""Download public benchmarks into ``$JARVIS_DATA_DIR`` (default ``data/raw``)."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

from .loaders import data_root

SOURCES = {
    "groupmembench": ("git", "https://github.com/UCSB-NLP-Chang/GroupMemBench.git", "GroupMemBench"),
    "evermembench": ("git", "https://github.com/EverMind-AI/EverMemBench.git", "EverMemBench"),
    "locomo": ("files", {"locomo10.json": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"}, "locomo"),
    "longmemeval": ("files", {
        "longmemeval_s_cleaned.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        "longmemeval_oracle.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    }, "longmemeval"),
}

LICENSES = {
    "groupmembench": "MIT",
    "locomo": "CC BY-NC 4.0 (non-commercial: research/eval only, not for product training)",
    "longmemeval": "MIT",
    "evermembench": "see repository",
}


def fetch(name: str, force: bool = False) -> Path:
    kind, src, sub = SOURCES[name]
    dest = data_root() / sub
    if dest.exists() and not force:
        print(f"[fetch] {name}: already at {dest}")
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {name} -> {dest}  (licence: {LICENSES.get(name, '?')})")
    if kind == "git":
        subprocess.run(["git", "clone", "--depth", "1", src, str(dest)], check=True)
    else:
        dest.mkdir(parents=True)
        for fname, url in src.items():
            print(f"  downloading {fname}")
            urllib.request.urlretrieve(url, dest / fname)
    return dest
