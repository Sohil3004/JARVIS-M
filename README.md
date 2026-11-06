# Jarvis-M

Memory for multi-party, multi-session group chat: who decided what, what has been
superseded, and what each participant needs to know. This repo is the research
harness: every memory system (ours and baselines) sits behind one interface and is
evaluated with the same reader LLM, the same judge and the same statistics.

> The ICCSIT 2026 system (BART + LoRA + dual FAISS retrieval) lives in
> [`legacy/v1_bart`](legacy/v1_bart/README.md). The v2 regex state-reconciliation
> prototype lives in [`legacy/v2_regex`](legacy/v2_regex) and is wrapped as the
> `legacy_v2` baseline.

## Layout

```
src/jarvis_m/
  types.py          Message / Conversation / Question / Dataset (no answer leakage to systems)
  interfaces.py     MemorySystem: reset() -> ingest(messages) -> retrieve(query, asker, budget)
  memory/           baselines (full_context, recency, bm25, dense, hybrid), legacy_v2 adapter, registry
  data/             loaders (GroupMemBench, LoCoMo, LongMemEval, native JSON) + `fetch`
  eval/             reader + judges (official benchmark prompts), metrics, runner, report
  llm.py            chat backends: local HF (Qwen3...), OpenAI-compatible server, test fakes
  embeddings.py     hashing (tests) and sentence-transformers (e.g. Qwen3-Embedding)
  config.py, cli.py YAML experiments with `base`, `--set`, `arms` (ablations) and `sweep`
configs/
  defaults.yaml     shared defaults (no-LLM reader/judge so anything runs on CPU)
  models/           reader / judge presets (8 GB: *_4bit_hf + judge_deferred)
  exp/              experiments: smoke_toy, gmb_baselines, gmb_scope_ablation, locomo, longmemeval
tests/              pytest suite (CPU-only, seconds) + fixtures
legacy/             v1 (ICCSIT) and v2 code, kept runnable for reproduction
```

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"                 # core + tests (numpy, pyyaml, pytest)
pytest                                   # should finish in a few seconds
# GPU work: install the CUDA build of torch first, then the extras
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[retrieval,llm]"
```

Environment variables: `JARVIS_DATA_DIR` (default `data/raw`), `JARVIS_CACHE_DIR`
(HF cache), `OPENAI_API_KEY` (only for API judges).

## Running experiments

```powershell
jarvis-eval run configs/exp/smoke_toy.yaml                 # CPU, seconds, all baselines
jarvis-eval report runs/smoke_toy --baseline bm25          # table + CIs + paired tests

jarvis-eval data fetch groupmembench                       # also: locomo, longmemeval, evermembench
jarvis-eval data info configs/exp/gmb_baselines.yaml
jarvis-eval run configs/exp/gmb_baselines.yaml --arm bm25 `
  --set dataset.domains=[Finance] --set dataset.limit=50   # quick slice first
```

Use a real reader/judge by adding presets to an experiment's `base:` list, e.g.
`base: [../defaults.yaml, ../models/reader_qwen3_4b_4bit_hf.yaml, ../models/judge_deferred.yaml]`.
Runs are resumable: re-running skips questions already in `results.jsonl`.

### Small GPU (8 GB): two passes

The reader and the judge never need to be in VRAM together. Generate every answer
with the judge deferred, then load the judge once and grade all saved runs:

```powershell
pip install -e ".[retrieval,llm,quant,dev]"
jarvis-eval run   configs/exp/gmb_baselines.yaml            # Qwen3-4B NF4 reader, judge: deferred
jarvis-eval judge runs/gmb_baselines --judge configs/models/judge_qwen3_8b_4bit_hf.yaml
jarvis-eval report runs/gmb_baselines --baseline bm25
# or in one go (frees the reader before loading the judge):
jarvis-eval run configs/exp/gmb_baselines.yaml --then-judge configs/models/judge_qwen3_8b_4bit_hf.yaml
```

Judging is resumable (`results.judged.jsonl`) and `--force` re-grades with a different
judge, so the same predictions can be scored by a local judge now and an API judge later.
Before reporting local-judge numbers, grade a ~200-question subset with an API judge and
by hand and report agreement (Cohen's κ): a 4-bit 8B judge is weaker than the gpt-5 judge
GroupMemBench used, and a Qwen judge grading a Qwen reader can be biased.

Each run writes `runs/<name>/<arm>/{config.yaml,results.jsonl,summary.json}`. Rows carry
the prediction, judge verdict, retrieved message ids, evidence recall (where gold
evidence exists), stale-answer hits, abstention flags, context tokens and latencies.

## Adding a memory system

Subclass `jarvis_m.interfaces.MemorySystem`, register it in `jarvis_m/memory/__init__.py`,
add it to `SYSTEMS` in `tests/test_memory_conformance.py`, then reference it from an
experiment as `memory: {type: <name>, ...}`. Ablations are `arms` that flip its config.

## Benchmarks and licences

| Dataset | Use | Licence |
| --- | --- | --- |
| GroupMemBench | primary multi-party eval (no gold evidence ids) | MIT |
| LoCoMo | standard long-term memory eval, has evidence ids | CC BY-NC 4.0: eval only, not product training |
| LongMemEval | knowledge updates, abstention, has evidence turns | MIT |
| EverMemBench | multi-group eval (loader TODO) | see repo |

## Status

- [x] Repo restructure, harness, baselines, loaders, tests
- [ ] Learned state extractor (replaces v2 regexes)
- [ ] Hypergraph memory + HGNN retriever
- [ ] Jarvis-StateBench stress generator, EverMemBench loader
- [ ] External baselines (Mem0, Zep/Graphiti, HippoRAG 2, LLMLingua-2)

Licence: MIT (see `LICENSE`).
