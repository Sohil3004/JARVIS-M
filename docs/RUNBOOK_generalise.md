# Runbook: GPU runs after the generalisation work (phases A–F)

For one RTX 5060 8 GB. Run the steps in order; each says what it needs, roughly how much
VRAM and time it takes, and what to look at. Everything not on the GPU was run on CPU
during phases A–F (`docs/CHANGES_generalise.md`), so the commands and configs are known to
work; the GPU times are **estimates** (how they were made is stated at each step) and the
first run of each step prints the real rate.

Setup once:

```bash
pip install -e '.[all]'            # torch, transformers, bitsandbytes, sentence-transformers, pyarrow
jarvis-eval data fetch locomo
jarvis-eval data fetch longmemeval
# optional external sets (docs/DATASETS.md): evermembench, convomem, socialmembench
```

## 0. Data (CPU, about 2 minutes)

`data/` is not tracked; these regenerate everything the configs read. StateBench v1 uses
the original sizes and seed, so its conversations are byte-identical to earlier files; the
files now also carry `stale_evidence_ids` and supersession labels.

```bash
# StateBench v1 (v1-compatible), with labels in data/labels/supersession/
jarvis-eval data make-stress --split train --n 300 --seed 0
jarvis-eval data make-stress --split dev   --n 60  --seed 0
jarvis-eval data make-stress --split test  --n 300 --seed 0

# StateBench v2, every domain
for d in team assistant personal enterprise; do
  jarvis-eval data make-stress --domain $d --split train --n 300 --seed 0
  jarvis-eval data make-stress --domain $d --split dev   --n 60  --seed 0
  jarvis-eval data make-stress --domain $d --split test  --n 300 --seed 0
done

# v2 with real background turns (evaluation only; needs locomo / longmemeval fetched)
jarvis-eval data make-stress --domain personal  --split test --n 100 --seed 0 --background locomo
jarvis-eval data make-stress --domain assistant --split test --n 100 --seed 0 --background longmemeval
```

Sizes: v2 train 300 conversations per domain (4,900–10,700 messages), dev 60, test 300;
background sets about 1,600 messages per conversation (the assistant one is a 190 MB file).
Optional, GPU: `--paraphrase configs/models/paraphrase_qwen3_4b_4bit_hf.yaml` on any v2
command rewrites the generated messages (cached next to the output; writes `*_para.json`).

## 1. LLM extraction (GPU, ~4 GB VRAM; the long step)

Qwen3-4B-Instruct NF4 through transformers (`configs/models/extractor_qwen3_4b_4bit_hf.yaml`:
batch 8, context 6, `prefilter: light`, slot registry 40). Each dataset gets its own cache
`data/extract/<dataset>/qwen3_4b.jsonl` (resumable: rerun the same command after a stop).

```bash
# a) StateBench v1 + v2 test and dev (the supersession baseline sets)
jarvis-eval extract configs/eval/supersession_v2.yaml --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml \
  --only "StateBench v1 test" --only "SB2 team test" --only "SB2 assistant test" \
  --only "SB2 personal test" --only "SB2 enterprise test"

# b) LongMemEval-S, test split first (evaluation), then the train split if training should use LLM records
jarvis-eval extract configs/eval/rank_v2.yaml --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml \
  --only "LongMemEval-S test (stratified 20%)"
jarvis-eval extract configs/train/hgnn_v2_mix.yaml --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml \
  --only "LongMemEval-S train"            # same cache file as the test split

# c) LoCoMo (small: 10 conversations, 5,882 messages)
jarvis-eval extract configs/eval/rank_v2.yaml --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml --only LoCoMo
```

`extract` prints `msg/s` and the hours left every 400 messages, and at the end the batch
cache hit rate. Check the first two numbers after a few minutes and adjust the plan.

**Workload, measured** (dry run of the same extractor with a stub model, the defaults above):

| Set | messages | LLM calls | batch-cache hit rate | target messages sent | prompt characters |
| --- | --- | --- | --- | --- | --- |
| StateBench v1 test | 8,894 | 2,606 | 1.0% | 7,564 | 7.8 M |
| SB2 test, 4 domains | 30,110 | 7,562 | 0.4–29% | 23,534 | 22.9 M |
| LongMemEval-S, all 500 questions | 246,750 | 32,314 | **22.7%** | 189,600 | 278.5 M |
| LongMemEval-S, test split (102) | 50,485 | 8,066 | 5.5% | 47,231 | 69.6 M |
| SB2 personal + LoCoMo background | 159,940 | 12,277 | 39.4% | 93,986 | 65.7 M |
| LoCoMo | 5,882 | 846 | 0% | 5,809 | 4.1 M |

The LongMemEval-S hit rate matches the measured session repetition: 23,867 session
occurrences, 18,464 distinct contents, so 22.6% of sessions repeat across questions. The
batch cache extracts each repeated session once.

**Time, estimated.** Prompt tokens ≈ characters / 4. Output ≈ 30 tokens per target message
plus 5 per call (about half the messages give one ~60-token record, the rest `[]`).
Assumed rates for Qwen3-4B NF4 in transformers + bitsandbytes, batch 1, on an RTX 5060:
prefill ~3,000 tokens/s, decode ~20 tokens/s (bitsandbytes 4-bit decode is several times
slower than the GPU's memory bandwidth allows). Decode dominates.

| Set | input tokens | output tokens | transformers NF4 | llama.cpp / Ollama Q4 server |
| --- | --- | --- | --- | --- |
| StateBench v1 + SB2 test (a) | 7.7 M | 0.98 M | ~14 h (11–18 h) | ~4 h |
| LongMemEval-S test split | 17.4 M | 1.46 M | ~22 h (17–30 h) | ~6 h |
| LongMemEval-S, all | 70 M | 5.9 M | ~88 h (70–118 h) | ~25 h |
| LoCoMo | 1.0 M | 0.18 M | ~2.5 h | ~0.7 h |

The server column assumes ~80 tokens/s decode and ~4,000 tokens/s prefill for a Q4_K_M GGUF
of the same model on this card (`configs/models/extractor_local_server.yaml`, set `model:`).
Recommendation: calibrate on the first 400 messages of (a); if decode is near 20 tokens/s,
switch to the server config. Extract the LongMemEval train split only if you want an
LLM-record training run; the default training uses the heuristic extractor.

Afterwards, use the caches with `memory.extractor: {type: cached, path:
data/extract/{dataset}/qwen3_4b.jsonl, allow_missing: true}`. `{dataset}` becomes each
dataset's name; `allow_missing: true` lets sets without a cache (the background sets)
fall back to the heuristic extractor; the cache's own misses fall back too.

## 2. Baseline rank-eval (GPU, ~3 GB VRAM for Qwen3-Embedding-0.6B)

```bash
# heuristic extractor (the config's default)
jarvis-eval rank-eval configs/eval/rank_v2.yaml --out results/rank/v2_baseline.md
# LLM records
jarvis-eval rank-eval configs/eval/rank_v2.yaml --out results/rank/v2_baseline_qwen.md \
  --set memory.extractor='{type: cached, path: "data/extract/{dataset}/qwen3_4b.jsonl", allow_missing: true}'
```

Time: embedding is the cost and is cached on disk (`cache/embeddings/`). The plain sets
are ~45,000 short messages; the two background sets ~318,000 messages, many long. Estimate
from an assumed ~150 short / ~50 long texts per second: 1–2 h the first time, minutes
afterwards (graph building measured on CPU: ~4 minutes per background set). Look at:
per-domain MRR, `leak_rate` (must be 0.000 on enterprise), stale-rank.

## 3. Train HGNN v2, three seeds (GPU, < 4 GB VRAM)

```bash
for s in 0 1 2; do
  jarvis-eval train-hgnn configs/train/hgnn_v2_mix.yaml --set train.seed=$s --set output=models/hgnn/hgnn_v2_mix_s$s.pt
done
```

Mixed training (SB1 at weight 0.5, SB2 four domains, LongMemEval-S train), temperature 0.5
sampling, early stopping on the dev macro over the four domains (printed per epoch with the
per-domain MRR). Time: the first seed embeds ~230,000 messages (mostly LongMemEval turns),
estimated ~1 h; later seeds reuse the cache and should take 10–20 minutes each (graph
building on CPU, 15 epochs of a 128-unit model). The old `hgnn_v1*.pt` checkpoints are
rejected (feature set changed).

## 4. Leave-one-domain-out (GPU, < 4 GB)

```bash
for d in assistant enterprise personal team; do
  jarvis-eval train-hgnn configs/train/lodo/hgnn_v2_mix_lodo_$d.yaml
  jarvis-eval rank-eval  configs/train/lodo/rank_hgnn_v2_mix_lodo_$d.yaml --out results/rank/lodo_$d.md
done
```

Each LODO model never sees its held-out domain (train or dev); its rank config evaluates on
that domain's `rank_v2.yaml` sets. Extra seeds: `--set train.seed=1 --set
output=models/hgnn/hgnn_v2_mix_lodo_<d>_s1.pt` joins the same arm. Time: 4 × (training
~15 min + evaluation 5–30 min), embeddings cached from step 3.

## 5. rank-eval with significance (GPU, ~3 GB)

```bash
jarvis-eval rank-eval configs/eval/rank_v2.yaml \
  --arm hgnn_v2_mix=models/hgnn/hgnn_v2_mix_s{0,1,2}.pt --reference seeds --n-boot 10000 \
  --out results/rank/v2_hgnn.md
# with LLM records (add the same --set memory.extractor=... as step 2)
# external sets with evidence ids (after fetching them):
jarvis-eval rank-eval configs/eval/rank_external.yaml --arm hgnn_v2_mix=models/hgnn/hgnn_v2_mix_s{0,1,2}.pt \
  --out results/rank/external_hgnn.md
```

The significance table resamples conversations (and seeds within the arm), Holm-corrected
over every (domain, arm) row plus the macro rows. Time: as step 2, from cache: minutes.

## 6. Supersession eval (CPU is enough; a few minutes)

```bash
# heuristic (the recorded pre-Qwen baseline)
jarvis-eval supersession-eval configs/eval/supersession_v2.yaml --out results/supersession/v2_heuristic.md
# LLM records
jarvis-eval supersession-eval configs/eval/supersession_v2.yaml --out results/supersession/v2_qwen.md \
  --set memory.extractor='{type: cached, path: "data/extract/{dataset}/qwen3_4b.jsonl", allow_missing: true}'
# B4 ablation with LLM records: global instead of windowed linking evidence
jarvis-eval supersession-eval configs/eval/supersession_v2.yaml --out results/supersession/v2_qwen_globaldf.md \
  --set memory.extractor='{type: cached, path: "data/extract/{dataset}/qwen3_4b.jsonl", allow_missing: true}' \
  --set memory.link_df_window=1000000000
```

Compare with the heuristic table in `docs/CHANGES_generalise.md` (phase E). Hand-labelled
slots, once you have some in `data/labels/slots/` (`docs/LABELING.md`):
`jarvis-eval extract-eval configs/eval/extract.yaml` and the same with
`--set extractor.type=cached --set extractor.path=data/extract/{dataset}/qwen3_4b.jsonl`.

## 7. Small end-to-end smoke run (GPU, ~4 GB then ~6 GB; ~15 minutes)

Retrieval → Qwen3-4B reader → Qwen3-8B judge on 20 StateBench questions, to check that the
whole pipeline still runs with the new memory code before any long QA run:

```bash
jarvis-eval run configs/exp/statebench_hypergraph.yaml --arm hybrid --arm hg_fixed \
  --set dataset.limit=20 --set name=smoke_e2e --then-judge configs/models/judge_qwen3_8b_4bit_hf.yaml
jarvis-eval report runs/smoke_e2e --baseline hybrid
```

The reader (NF4, ~3 GB weights) is unloaded before the judge (NF4, ~5.5 GB) loads.
`results.jsonl` rows carry the access audit (`mode: auto, active: false` on this data).
