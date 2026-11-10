# Generalisation work: change log

Task: `docs/prompts/generalise_v1.md`. Phases A–C are done; D–F are not started.
All numbers on this page are CPU numbers (hashing embedder, heuristic extractor). They
are for before/after comparison only, not for the paper.

Tests: 75 passed + 4 xfailed before phase A.

## Phase A: measurement infrastructure

### What changed

* `eval/significance.py`: `cluster_paired_bootstrap` (resamples conversations, and seeds
  when an arm has several score vectors), `holm`, `non_inferior`, `compare_arms`.
* `rank-eval`:
  * arms with several checkpoints: `--arm hgnn_v2=models/hgnn/hgnn_v2_s{0,1,2}.pt` (braces,
    comma lists and globs; also `arms:` in the config). `--checkpoint` still works and makes
    a one-checkpoint arm named `hgnn:<file stem>`.
  * new metrics: nDCG@10, all@k, stale-rank.
  * tables per dataset, per domain and the macro-average over domains.
  * significance table against a reference arm (`--reference`, default `seeds`).
  * per-question rows in `<out>.questions.json`.
* Domain registry: `domain:` on every dataset entry (`team`, `assistant`, `personal`,
  `enterprise`). `Dataset.domain` is set by `load_dataset`.
* K-fold splits: `split: {n_folds: 5, fold: i, seed: 0, part: train|test}` for LoCoMo (by
  conversation) and LongMemEval (stratified by question type). `split: train|test` on
  LongMemEval is unchanged.
* `jarvis-eval make-lodo <train_cfg> --out <dir> [--eval-config <rank_cfg>]`.
* `Question.stale_evidence_ids`; the StateBench generator fills it.
* `jarvis-eval supersession-eval <cfg>` and the label format in `docs/LABELING.md`. The
  StateBench generator emits labels (`annotations["supersession"]`, and
  `data/labels/supersession/statebench_<split>.jsonl` from `make-stress`).
* Configs: `domain:` added everywhere; new `configs/eval/supersession.yaml`.

### Tests added

`tests/test_significance.py` (6), `tests/test_supersession_lodo.py` (4),
`tests/test_rank_eval.py` (rewritten, 5), three new loader tests in `tests/test_loaders.py`.

### Decisions

* **Domains of existing data.** LoCoMo = `personal`, LongMemEval = `assistant`, StateBench,
  GroupMemBench and the toy fixture = `team`. `rank-eval` and `make-lodo` refuse a dataset
  entry without `domain:`; `run`, `train-hgnn` and `extract` accept one.
* **Domain and macro numbers.** A domain value pools the questions of all its datasets. The
  macro value is the unweighted mean of the domain values.
* **Multi-seed arms.** The reported value is the mean over seeds of the per-seed mean. The
  bootstrap draws conversations with replacement and, for each arm separately, seeds with
  replacement.
* **Significance table.** One row per (domain, arm) plus one row per arm for the macro
  average (conversations drawn inside each domain). Metric: MRR (`bootstrap.metric` changes
  it). Holm runs over the whole table. p uses `(count + 1) / (n + 1)`, so it is never 0.
* **K-fold `part`.** The spec gives `{n_folds, fold, seed}` but not which side to load. I
  added `part`, default `test` (fold `fold`); `part: train` is the other folds.
* **LODO held-out data.** Without `--eval-config`, the rank-eval config uses the held-out
  domain's own train and dev entries (that model never saw them). The arm globs
  `<output stem>_lodo_<domain>*.pt` so seed checkpoints join it. Generated configs are
  written resolved (no `base:`).
* **Stale-rank.** A stale message that is not in the candidate set cannot outrank anything.
  Questions with no stale evidence are left out of the rate.
* **Supersession precision.** Counted only over predicted pairs that have a label, because
  hand labels are never complete. Unlabelled predictions are reported as a count.
* **StateBench labels.** In a hand-off, the pair is (first owner's claim, new owner's
  acceptance); the request message is not labelled. Up to 30 `unrelated` pairs per
  conversation, sampled with a separate random generator, so conversations are byte-identical
  to the files generated before this change.
* **Label location.** `make-stress` writes labels to `data/labels/supersession/` when the
  output is under `data/stress`, otherwise to `<out dir>/labels/supersession/`.
  `statebench_*.jsonl` label files are git-ignored.
* **Leak rate (A8)** needs the access model, so it is implemented in the phase C commit.

### Known limitations

* The StateBench files already in `data/stress/` do not have `stale_evidence_ids` or
  labels. Regenerate them with the same `--n` and `--seed`; the conversations do not change.
* K-fold is implemented for LoCoMo and LongMemEval only.

### First measurement (before phase B)

Supersession on StateBench (heuristic extractor):

| Set | precision | recall | F1 | false-supersession rate |
| --- | --- | --- | --- | --- |
| StateBench dev (seen phrasing) | 0.965 | 0.269 | 0.421 | 0.002 |
| StateBench test (held-out phrasing) | 0.869 | 0.073 | 0.135 | 0.002 |

The heuristic chains are precise but miss most updates, and nearly all of them on held-out
phrasing.
