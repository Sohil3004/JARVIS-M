"""Download public benchmarks into ``$JARVIS_DATA_DIR`` (default ``data/raw``).

Licences and what each dataset is used for: ``docs/DATASETS.md``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .loaders import data_root

_EMB = "https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic/resolve/main"
_SMB = "https://huggingface.co/datasets/anon4data/socialmembench/resolve/main"

SOURCES = {
    "groupmembench": ("git", "https://github.com/UCSB-NLP-Chang/GroupMemBench.git", "GroupMemBench"),
    "evermembench": ("files", {f"{t}/{f}": f"{_EMB}/{t}/{f}" for t in ("01", "02", "03", "04", "05")
                               for f in ("dialogue.json", f"qa_{t}.json")}, "EverMemBench-Dynamic"),
    "socialmembench": ("files", {f"{n}.parquet": f"{_SMB}/{n}.parquet" for n in ("conversations", "qa", "networks")},
                       "SocialMemBench"),
    # ConvoMem: the evidence questions only (the pre-mixed test cases and filler are not used)
    "convomem": ("hf-listing", ("Salesforce/ConvoMem", "core_benchmark/evidence_questions/"), "ConvoMem"),
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
    "evermembench": "Apache-2.0",
    "socialmembench": "CC BY 4.0 (attribution)",
    "convomem": "CC BY-NC 4.0 (non-commercial: research/eval only, not for product training)",
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
        return dest
    if kind == "hf-listing":  # every file of a Hugging Face dataset under a prefix
        repo, prefix = src
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}") as r:
            files = [s["rfilename"] for s in json.load(r)["siblings"] if s["rfilename"].startswith(prefix)]
        src = {f: f"https://huggingface.co/datasets/{repo}/resolve/main/{f}" for f in files}
    dest.mkdir(parents=True)
    for i, (fname, url) in enumerate(src.items()):
        if i < 5 or i % 100 == 0:
            print(f"  downloading {fname} ({i + 1}/{len(src)})")
        (dest / fname).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest / fname)
    return dest
