# Legacy code

Kept for reproducing earlier results; not part of the `jarvis_m` package.

| Folder | What | Run from |
| --- | --- | --- |
| `v1_bart/` | ICCSIT 2026 system: BART-large-CNN + LoRA, intra/cross-user FAISS retrieval, ablation | `cd legacy/v1_bart; python src/<script>.py` |
| `v2_regex/` | Nov 2026 prototype: regex state reconciliation, heuristic token pruner, Qwen2.5-0.5B LoRA | `$env:PYTHONPATH="legacy/v2_regex/core;legacy/v1_bart/src"` then `python legacy/v2_regex/evaluation/<script>.py` |

Dependencies: `pip install -r legacy/requirements-legacy.txt`.
Model/cache paths come from `JARVIS_CACHE_DIR`, `JARVIS_V1_ADAPTER`, `JARVIS_V2_ADAPTER`
(previously hard-coded to one machine).

Known issues are documented in the audit doc and pinned by
`tests/test_legacy_v2_probes.py` (xfail cases). The v2 evaluation scripts use
hand-written scenarios scored by keyword match; do not report their numbers.
The v2 regex reconciler is available in the new harness as `memory: {type: legacy_v2}`.
