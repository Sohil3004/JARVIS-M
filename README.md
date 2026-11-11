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
  memory/hypergraph hypergraph memory: graph builder, subgraphs, fixed propagation, HGNN + trainer
  extract/          state extraction: heuristic (inline) and Qwen LLM (offline, cached, resumable)
  data/             loaders (GroupMemBench, LoCoMo, LongMemEval, native JSON), `fetch`, StateBench generator
  eval/             reader + judges (official benchmark prompts), metrics, runner, report
  llm.py            chat backends: local HF (Qwen3...), OpenAI-compatible server, test fakes
  embeddings.py     hashing (tests) and sentence-transformers (e.g. Qwen3-Embedding)
  config.py, cli.py YAML experiments with `base`, `--set`, `arms` (ablations) and `sweep`
configs/
  defaults.yaml     shared defaults (no-LLM reader/judge so anything runs on CPU)
  models/           reader / judge presets (8 GB: *_4bit_hf + judge_deferred)
  hypergraph_base.yaml  shared hypergraph memory settings (embedder must match the HGNN checkpoint)
  exp/              experiments: smoke_*, gmb_baselines, gmb_scope_ablation, locomo, longmemeval,
                    statebench_hypergraph, gmb_hypergraph (full ablation grids)
  train/            HGNN training configs
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

## Hypergraph memory (Jarvis-M v3)

Messages are nodes; raw text is never replaced. Typed hyperedges add structure:
`reply` threads, time-gap `episode`s, `channel` and `speaker` scope, `entity` mentions,
and `version` chains (successive values of one topic slot; older values are marked
superseded, rejected alternatives never become current). Retrieval:

1. hybrid BM25 + dense seeds (top `n_seeds`);
2. expand along small hyperedges; always add the current value of any touched slot, and
   the asker's own commitments for first-person questions ("what am I doing?");
3. score the candidate subgraph with `scorer: fixed` (HyperMem-style propagation +
   version logic), `seeds` (no propagation) or `hgnn` (learned, query-gated message passing);
4. pack to the budget; optional `state_header: true` prepends current slot values + history.

Full pipeline on an 8 GB GPU (each step is resumable):

```powershell
# data
jarvis-eval data make-stress --split train --n 300
jarvis-eval data make-stress --split dev   --n 60
jarvis-eval data make-stress --split test  --n 300      # held-out phrasing families
jarvis-eval data fetch groupmembench ; jarvis-eval data fetch longmemeval
# optional: Qwen state extraction (hours on GroupMemBench; minutes on StateBench)
jarvis-eval extract configs/exp/statebench_hypergraph.yaml --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml
jarvis-eval extract configs/exp/gmb_hypergraph.yaml        --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml
# HGNN (+ the no-message-passing ablation)
jarvis-eval train-hgnn configs/train/hgnn_statebench_lme.yaml
jarvis-eval train-hgnn configs/train/hgnn_statebench_lme.yaml --set model.layers=0 --set output=models/hgnn/hgnn_v1_mlp.pt
# evaluate every arm (reader pass), then judge, then report
jarvis-eval run   configs/exp/statebench_hypergraph.yaml
jarvis-eval judge runs/statebench_hypergraph --judge configs/models/judge_qwen3_8b_4bit_hf.yaml
jarvis-eval report runs/statebench_hypergraph --baseline hybrid
# ranking tables (no reader/judge): seeds, fixed and each checkpoint on StateBench test,
# LoCoMo and the held-out stratified LongMemEval split -> results/rank/rank_real.{md,json}
jarvis-eval data fetch locomo
jarvis-eval rank-eval configs/eval/rank_real.yaml --checkpoint models/hgnn/hgnn_v1.pt --checkpoint models/hgnn/hgnn_v1_mlp.pt
```

The HGNN trains on `longmemeval` with `split: train` (a stratified, hash-based 80% per
question type) and `rank-eval` reports on `split: test`, so the two never overlap.

Arms that need an artefact you have not built yet (HGNN checkpoint, extraction cache)
fail fast; run the rest with `--arm`. See [docs/HYPERGRAPH.md](docs/HYPERGRAPH.md) for the
design rationale and first CPU-only findings.

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
| EverMemBench-Dynamic | multi-group eval, has evidence ranges | Apache-2.0 |
| ConvoMem | assistant memory, evidence quoted by text | CC BY-NC 4.0 |
| SocialMemBench | social group chats, evidence anchors | CC BY 4.0 |

Details and what was found in each release: `docs/DATASETS.md`.

## Status

- [x] Repo restructure, harness, baselines, loaders, tests
- [x] State extractors: heuristic + Qwen LLM (cached)
- [x] Hypergraph memory, fixed propagation, learned HGNN + trainer
- [x] Jarvis-StateBench stress generator (held-out phrasing test split)
- [ ] GPU runs: Qwen extraction, HGNN with Qwen3-Embedding, full ablation grids + judge pass
- [x] EverMemBench, ConvoMem and SocialMemBench loaders
- [ ] LoCoMo held-out-conversation HGNN eval
- [ ] External baselines (Mem0, Zep/Graphiti, HippoRAG 2, LLMLingua-2)

Licence: MIT (see `LICENSE`).
