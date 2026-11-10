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

## Phase B: graph and feature fixes

### What changed

* **Two layers.** `Hypergraph` stores messages, records and labels and builds version
  chains. `GraphView` is what one viewer sees: a mask over nodes plus everything derived
  from the visible nodes (all hyperedges, slots, current / superseded flags, participants).
  `Hypergraph.view()` with no mask is the omniscient view. Retrieval always goes through a
  view.
* **Holders (B1).** `StateRecord.holder`; slot key = `holder|subject|attribute`. The
  heuristic extractor sets `speaker:<name>` or `group:<channel>` and lists names used as
  possessor or subject; the graph turns such a name into the holder only when it is a known
  participant.
* **Scope-aware chains (B2).** A chain holds records of one visibility class (channel,
  level, compartments). A view merges the visible chains that share a slot, in time order.
  The viewer's current value is the latest version the viewer can see.
* **Timestamp ordering (B3).** Chains are a function of the messages in time order. An
  out-of-order arrival triggers one replay on next use.
* **Windowed linking evidence (B4).** Document frequency is counted over the last
  `link_df_window` messages of the channel (default 500); `max_link_df` is the ratio inside
  that window.
* **Edge-size consistency (B5).** `build_subgraph` keeps an edge by its membership inside
  the candidate set. More than `max_edge_size` members inside: keep the top ones by base
  score. Version edges: current + first + last `version_keep_k`. Expansion follows the same
  rule, so a long chain is no longer skipped.
* **Mention edges (B6).** New hyperedge type `mention`. `EDGE_TYPES` has 7 entries.
* **Features (B7).** `FEATURE_NAMES` changed (24 features): `base_rank` is a percentile
  inside the subgraph, `recency` became `age_frac` and `log_age_days`, `log_degree` counts
  edges inside the subgraph, and `is_assistant`, `query_mentions_speaker`,
  `node_mentions_query_person`, `log_n_participants` are new. Old checkpoints are rejected.
  Code refers to features by name (`FI["is_current"]`), not by position.
* **Own commitments (B8).** `GraphView.own_chain_nodes(speaker)` replaces the scan.
* The speaker-scope ablation (`scope: asker|others`) is now a filter on the view. The
  `base = -1` hack is gone.
* Seeds are ranked and normalised over the visible messages only (`_raw_scores(query, mask)`).

### Tests added

`tests/test_graph_generalise.py` (24 tests). `tests/test_hypergraph.py` and
`tests/test_extract.py` were updated for the new key format and feature names.

### Decisions

* **Holder rules.** Work state (`owner`, `status`) always belongs to the group, even in the
  first person. Otherwise "I'll take load testing" and the next owner's "I'll own load
  testing from now on" would have different holders and the hand-off could never supersede.
  For other attributes: "my / mine" gives the speaker, then "we / our / let's" gives the
  group, then "I / me" gives the speaker, and no pronoun gives the group.
* **Group scope** is the channel: `group:<channel>`.
* **`speaker:` and `person:` share one key space** (`speaker:Priya` and `person:Priya` are
  the same slot holder). The heuristic extractor never emits `person:`; only an LLM
  extractor can (phase D).
* **Named holders need a known participant.** "Kafka is not happening" would otherwise make
  Kafka a holder. A name counts once that person has spoken in a class the record's readers
  can read.
* **Replay instead of bisect insertion.** Bisect insertion only makes key-linked records
  order-independent. Entity linking, the frequency window and the confirmation window also
  depend on order, so the graph replays all messages in time order when one arrives late.
  In-order ingest stays incremental (no replay).
* **Tie-breaks.** Equal timestamps are ordered by ingest order. A message with no timestamp
  takes the timestamp of the message ingested before it.
* **All hyperedges are derived per view.** Episodes, threads, entity and mention edges are
  built from the visible nodes, so they cannot depend on hidden messages. Views are cached
  (16 per graph) and dropped on every ingest.
* **Read down, write own class.** A record may link to a slot in a class that every reader
  of its own class can read; it then starts a chain in its own class with the same slot
  instead of extending the other chain.
* **Cross-channel reading** is allowed only from channels with no membership restriction
  (all channels, when there is no channel directory). This keeps the old behaviour on
  public multi-channel data.
* **Assistant confirmation** ("sounds good") only applies to a suggestion of the same
  visibility class.
* **Reply threads live inside one channel.** A cross-channel `reply_to` neither links nor
  inherits a label.
* **Frequency window** is per channel and counts messages.
* **Expansion over large edges.** A seed in an edge larger than `max_edge_size` now pulls in
  the edge's top `max_edge_size` members by base score (before: nothing).
* **Topicless statements** (subject is just the channel) each get their own slot, so merging
  chains by slot in a view cannot bring back the channel-wide chain.
* **Superseded flag.** A node that is the latest value of any slot is never superseded.
* **Mentions.** A message mentions a participant when it contains the full name or any word
  of the name with 3 or more characters, case-insensitive. Role labels (`user`, `assistant`,
  `human`, `system`, `bot`, `ai`) are not names. Mention edges have expansion and
  fixed-propagation weight 0; the HGNN sees them.
* **Feature names.** The spec keeps the name `recency` for age / span. I renamed it to
  `age_frac` because it grows with age.

### Measurements (CPU)

MRR before (phase A commit) and after phase B, same data:

| Set | seeds before | seeds after | fixed before | fixed after |
| --- | --- | --- | --- | --- |
| StateBench dev | 0.751 | 0.751 | 0.821 | 0.821 |
| StateBench test | 0.752 | 0.752 | 0.787 | 0.787 |
| LoCoMo | 0.287 | 0.287 | 0.275 | 0.275 |
| LongMemEval-S test | 0.588 | 0.588 | 0.536 | 0.536 |

Supersession on StateBench is unchanged (dev P 0.965 / R 0.269, test P 0.869 / R 0.073).
Mean superseded fraction: LoCoMo 0.0045 to 0.0044, LongMemEval 0.0104 to 0.0103.

### Not done / known limitations

* **B4 measurement on StateBench v2-personal** is not possible yet: that set is part of
  phase E. The mechanism is covered by a unit test
  (`test_recurring_topic_links_again_after_the_window`).
* A view is rebuilt from scratch after every ingest. That is fine for "ingest, then ask
  many questions"; a chat that alternates single messages and questions pays O(messages)
  per question.
* First-name matching ("Priya" for the participant "Priya Sharma") uses the participants
  seen up to that point in time.
* The top-by-base rule can drop a low-scoring seed from a very large edge.
