# Generalisation work: change log

Task: `docs/prompts/generalise_v1.md` (phases A–C) and its addendum
`docs/prompts/generalise_v1_DF.md` (phases D–F).
All numbers on this page are CPU numbers (hashing embedder, heuristic extractor). They
are for before/after comparison only, not for the paper.

Tests: 75 passed + 4 xfailed before phase A; 142 passed + 4 xfailed after phase C.

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

## Phase C: clearance-aware access control

### What changed

* `jarvis_m/access.py`: `Label`, `join`, `Principal`, `meet`, `ChannelDirectory`,
  `AccessPolicy`, `ThreadLabeler`, `AccessController` (`resolve`, `effective`, `can_read`,
  `mask`), `AccessViolation`, and the `Redactor` interface stub.
* `configs/access/default.yaml`: levels, roles, channel defaults, markers and the policy
  knobs.
* Every message-level memory and the hypergraph memory take `access:` and enforce it:
  candidate mask, seeds over visible messages, view-based expansion, features, header and
  packing, final check, and `Retrieval.audit`.
* `retrieve(..., audience=None)` on every memory system. `MemorySystem.configure_access`
  loads the dataset's channel and principal directory; the runner and `build_examples`
  call it, and pass `question.meta["audience"]`.
* BM25 `idf_scope: global | view` (also on `hybrid` and `hypergraph`).
* `rank-eval` reports `leak_rate` for datasets with ACL metadata (Phase A8).
* Measure-only mode (`enforce: false`) for computing what an unprotected system would leak.
* Viewer-relative state from phase B; optional neutral note for hidden updates.
* `tests/fixtures/toy_enterprise.json`: three channels, five principals, a confidential
  deadline update, a manager-only hosting update, a compartmented decision, audience
  questions.
* `docs/ACCESS_CONTROL.md`.

### Tests added

`tests/test_access.py` (27 tests): policy and labels, audience meet, membership at query
time, viewer-relative current values, hidden-update signalling, the debug / prod final
check, baseline enforcement, measure-only mode, non-interference on 6 random enterprise
fixtures (graph state and retrieval), the BM25 residual, property tests over random
principals and audiences, leak rate, and runner audit rows. `tests/conftest.py` sets
`JARVIS_ACCESS_MODE=debug`.

### Decisions

* **Final check placement.** Ranked nodes and the nodes quoted by the state header are
  checked before packing, so a violation can never reach the context. The check reads labels
  and the directory directly, not the view mask.
* **Audience includes the asker.** The meet is over the asker and every audience member.
  An unknown or unlisted closed channel as audience gives an inactive principal (nothing).
  An open channel gives "anyone" (level 0, no compartments, member of open channels only).
* **Inactive audience members** block the answer by default (fail closed).
* **Unknown askers** are anonymous principals at the lowest level with no compartments.
* **Unlisted channels are closed** when access control is on. Without a channel directory
  in the dataset, configure one or set `unlisted_channels: open`.
* **Levels by name.** `acl_level` may be a level name, a role name or an index. Without a
  policy, names from the default level list are accepted, so labels exist (and split
  visibility classes) even when nothing is enforced.
* **`idf_scope` default stays `global`**, so numbers on non-ACL data do not move. The
  non-interference test uses `view`; the residual is documented and tested.
* **Extractor suggestions** are read from `StateRecord.meta["sensitivity"]` (a level, or
  `{level, compartments}`); unknown levels are ignored. The LLM extractor emits them in
  phase D.
* **Hidden-update note.** It is attached to the header line of the affected slot, or added
  as a separate `notice` item when there is no state header. It is the only place that
  reads the omniscient view, and only when the option is on.
* **Audit.** `filtered` counts items the final check removed. In measure-only mode it is 0
  and `unreadable_returned` holds the count instead.

### Measurements (CPU)

Unchanged on non-ACL data (same MRR as phases A and B on StateBench, LoCoMo and
LongMemEval; same supersession numbers). On the toy enterprise fixture, `leak_rate` is 0
with enforcement and above 0 in measure-only mode (asserted in the tests).

### Known limitations

* See "Residual leaks" in `docs/ACCESS_CONTROL.md`: global BM25 statistics, learned
  weights, extraction context, timing and audit counts, and membership changes after
  ingest.
* `legacy_v2` has no access control.
* Labels are fixed at ingest from the policy in force then; changing a channel's default
  level requires rebuilding the memory.
* There is no enterprise benchmark yet (StateBench enterprise domain is phase E), so there
  is no `leak_rate` number on real-sized data.

## Phase D: extractor generality, access mode and group chats

### What changed

* **Access mode** (`access.mode: off | auto | on`, addendum D1). `auto` is the new default:
  every memory builds an access controller, and enforcement is in force as soon as any
  access-control metadata exists (a principal or channel directory, a policy channel
  default or text marker, or one message with a non-public label); with none it is a no-op.
  `on` raises `AccessConfigError` at the first question if there is no metadata. `off`
  builds nothing (`access: off`). The old `mode: prod|debug` (the final check) is now
  `final_check`; `mode: prod|debug` is still read as `final_check`. The environment
  variable is now `JARVIS_ACCESS_CHECK`. Enforcement is never switched off by participant
  count.
* **One group-chat constant**, `types.GROUP_MIN_PARTICIPANTS = 3`, used by the default
  audience, the holder rule and the fixed scorer's own-state bonus.
* **Default audience** (D2). `retrieve(..., channel=...)`: with no explicit audience, an
  answer in a group chat (3 or more members, counting the asker) or an open channel is for
  everyone in it (the meet); in a DM, or with no channel, for the asker. Members come from
  the directory, or for an unlisted channel from who wrote there. The runner and
  `build_examples` pass `question.meta["channel"]`.
* **Holder rule** (D2). First-person-singular work state (owner, status, deadline,
  decision) is group-held only in group chats and speaker-held in DMs. The heuristic marks
  such records `meta.group_in_group_chat`; the graph decides from the channel's directory
  size or its distinct non-assistant speakers that readers of the record's class can see,
  counted over all ingested messages (a new speaker triggers a replay, so ingest order does
  not matter). `my / mine` and personal statements are always the speaker's.
* **Personal statements** (D5). New modality `state` (stateful) and the attribute set
  `decision, deadline, owner, status, plan, requirement, preference, location,
  relationship, possession, role, schedule, info`. The heuristic recognises cheap
  first-person patterns ("I moved to Pune", "i'm vegetarian now", "we got a puppy", "my
  sister", "I work at", "I have an appointment") and gives them a canonical subject
  (`home`, `diet`, the object noun). `FEATURE_NAMES` gained `mod_state` (checkpoints are
  rejected again).
* **Lowercase-robust entities** (D5): head nouns of `my / our <noun phrase>` ("my new
  laptop" → `laptop`). `my` phrases also count as topic phrases.
* **LLM extractor** (D1–D4): domain-neutral prompt (`PROMPT_VERSION = "2"`) with holder
  guidance by chat type, the new attribute set, `state`, and an optional `sensitivity`
  suggestion (raise-only, used by access control). Each call says whether the chat is a
  group chat or one-to-one. Batches are per (channel, session), so context never crosses a
  session.
* **Slot registry** (D2): the last 40 subjects per (channel, holder) are listed in the
  prompt; saved as `<cache>.registry.json` next to the extraction cache.
* **Prefilter** (D3): `prefilter: none | light | heuristic`, default `light` (skip only
  messages with fewer than 3 content words and no digit or entity).
* **Batch cache** (D4): key = SHA-1 of (PROMPT_VERSION, model, context and target texts with
  speaker and channel), stored in the extraction JSONL. `jarvis-eval extract` prints the
  batch-cache hit rate.
* **`jarvis-eval extract-eval`** (D6) against `data/labels/slots/*.jsonl`: record recall,
  holder / attribute / modality accuracy, subject purity and completeness, spurious rate.
  Label guide in `docs/LABELING.md`; config `configs/eval/extract.yaml`.

### LongMemEval-S session repetition (measured before the cache)

Over the 500 questions there are 23,867 haystack session occurrences, 19,195 distinct
session ids and 18,464 distinct session contents (role + text of every turn). No session id
has two different contents. So 22.6% of session occurrences (22.5% of turns: 246,750
occurrences, 191,332 distinct) repeat an earlier one and are batch-cache hits when batches
are per session. 3.8% of distinct ids are byte-identical copies under another id, which
only the content key catches.

### Tests added

`tests/test_access_modes.py` (13): `auto` gives the same rankings and contexts as `off` on
a StateBench sample and a LoCoMo sample (hypergraph, BM25, hybrid); `auto` with only a
message label still keeps a #leadership message out of an intern's DM; `on` without
metadata raises; default audience in a group chat, a DM, an open channel, an unlisted
channel, and an explicit override; compartment property tests over random principals and
audiences, checked against compartments computed from the raw labels without
`can_read`. `tests/test_graph_generalise.py`: hand-off in a 4-person channel, the same
messages in a DM, a late third participant (in time and in ingest order), directory size,
personal statements in a group chat. `tests/test_extract.py` (5): prompt content and chat
type line, holder / attribute / sensitivity parsing, light prefilter, slot registry in the
prompt and after a resume, batch cache across identical sessions, on disk, and per model.
`tests/test_supersession_lodo.py`: `extract-eval`.

### Decisions

* **Name clash.** The addendum's `access.mode` takes the key the final check used, so the
  final check is now `final_check` (and `JARVIS_ACCESS_CHECK`).
* **Default policy in `auto`.** With no `access:` setting the built-in policy has no channel
  defaults and no markers, so only message metadata and a dataset's directory turn
  enforcement on. Pass `configs/access/default.yaml` to get its markers and channel defaults.
* **An empty channel directory restricts nothing.** `unlisted_channels: closed` applies only
  when the directory lists at least one channel; otherwise `auto` triggered by a label alone
  would make every channel unreadable.
* **A question with no channel** is answered for the asker alone (like a DM).
* **Group-chat size** counts distinct non-assistant speakers (or directory members) and
  includes the asker for the audience rule. LongMemEval (user + assistant) and LoCoMo (two
  people) are DMs; StateBench and GroupMemBench channels are group chats.
* **The graph, not the extractor, decides group vs DM** for heuristic records, because only
  the graph sees the whole channel in a viewer-safe way. The LLM extractor is told the chat
  type in the prompt and its holder is used as given.
* **`state` modality.** Without it a personal fact ("I live in Pune") is `info`, which never
  enters a version chain.
* **NP heads only after `my / our`.** Heads after `the` were tried: on StateBench they
  lowered supersession precision (test 0.869 → 0.843) and fixed-scorer MRR (0.787 → 0.784)
  by linking unrelated statements on generic nouns.
* **The registry is not in the cache key**, as the spec says. A hit can carry subjects
  chosen under another registry state; they are still that model's subjects for that text.
* **Batching per (channel, session)** instead of per channel, so identical sessions give
  identical batches and context does not cross a session boundary.

### Measurements (CPU, hashing embedder, heuristic extractor)

| Set | fixed MRR before (phase C) | after | seeds |
| --- | --- | --- | --- |
| StateBench dev | 0.821 | 0.820 | 0.751 (same) |
| StateBench test | 0.787 | 0.787 | 0.752 (same) |
| LoCoMo | 0.275 | 0.273 | 0.287 (same) |
| LongMemEval-S test | 0.536 | 0.561 | 0.588 (same) |

Supersession on StateBench is unchanged (dev P 0.965 / R 0.269, test P 0.869 / R 0.073):
StateBench has no personal statements and all its channels are group chats.

### Known limitations

* No hand-labelled slots exist yet, so `extract-eval` has no number on real data.
* The heuristic's personal patterns are English, first-person and shallow ("I started
  eating fish again" is missed).
* The default audience for an unlisted channel counts every speaker who wrote there,
  including in messages the asker cannot read (only the count and ids of speakers, never
  text).

## Phase E: training data diversity

### What changed

* **StateBench v2** (`data/stress_v2.py`; `make-stress --domain team|assistant|personal|enterprise|mix`).
  v1 (`stress.py`) is unchanged, so v1 files regenerate byte-identically.
  * Slot names, deadlines, tasks, systems, personal subjects and suggestion topics are split
    60 / 20 / 20 into train / dev / test pools, on top of the phrasing families.
  * Updates by ellipsis: a message brings the slot up, the update does not name it ("too
    expensive, let's move it to Render"). Measured share of update messages that do not
    contain the slot name: team 60% (train) / 62% (test), assistant 100% / 81%, personal
    92% / 66%, enterprise 60% / 57%.
  * Per-person slots with `speaker:` holders (home, diet, coffee order, car, job, ...) and
    conflicting values across people (labelled `unrelated`).
  * Assistant domain: suggestions accepted explicitly ("sounds good"), implicitly ("booked
    X") or after a rejection, user corrections, and the user's own facts with corrections.
  * Enterprise domain: a per-conversation channel directory and principals
    (`annotations["access"][conversation_id]`, read with
    `access.conversation_directory`), a confidential update (in #leadership or with
    `[CONFIDENTIAL]` + `acl_level: L2`) to a slot with a public value, viewer-relative
    questions (`viewer_relative`: the same question has a different gold answer for a
    cleared and an uncleared asker), audience questions (`audience`: answered into #eng or
    for a named colleague), an exec-only compartmented fact, and 10% of messages swapped out
    of time order in the message list.
  * `--background locomo|longmemeval` splices the slot threads into real turns (300–3000
    messages per conversation, `--background-len`); questions are only about the inserted
    slots; in the personal domain the background speakers are renamed to the chat's people.
  * `--paraphrase <yaml with chat:>` rewrites generated messages with a local model; a
    rewrite that drops a value is discarded; cached next to the output.
  * Labels and `stale_evidence_ids` as in v1, plus `annotations["stats"]` and
    `annotations["protect"]` (the values each message states).
* **Training** (`train-hgnn`): per-dataset `weight:` and temperature sampling
  (`train.sampling_temperature`, default 0.5; P(dataset) ∝ (weight × size)^T); dev metrics
  per domain and early stopping on their unweighted mean. `configs/train/hgnn_v2_mix.yaml`,
  LODO configs in `configs/train/lodo/` (from `make-lodo ... --eval-config
  configs/eval/rank_v2.yaml`), `configs/eval/rank_v2.yaml` (all four domains),
  `configs/eval/supersession_v2.yaml`, `configs/eval/rank_external.yaml`.
* **Loaders** for EverMemBench-Dynamic, ConvoMem and SocialMemBench, `fetch` for all three,
  `docs/DATASETS.md` with licences and what the releases contain. GroupMemBench has no
  per-question evidence trace (checked in all 715 questions), so it stays QA-only.
* The runner and `build_examples` load each conversation's own directory
  (`conversation_directory`), so datasets whose conversations have different people work.

### Tests added

`tests/test_stress_v2.py` (9): disjoint vocabularies, labels and evidence ids, the ellipsis
share in every domain, determinism, personal holders and conflicts, assistant suggestion
kinds, enterprise gold answers checked under access control (a viewer can read their gold
evidence; an uncleared viewer or audience cannot read the hidden value), out-of-order lists,
background splicing, paraphrase (values kept, bad rewrites dropped, cache), the CLI, and
first-person-singular templates. `tests/test_loaders_real.py` (4) on excerpts of the real
releases (`tests/fixtures/real_samples/`). `tests/test_stress_and_hgnn.py`: temperature
sampling and the domain macro.

### Data regenerated (not committed)

`data/stress/statebench_{train,dev,test}.json` with the original sizes and seed (300 / 60 /
300, seed 0), now with `stale_evidence_ids` and supersession labels; StateBench v2 for every
domain at 300 / 60 / 300; and the two background test sets (100 conversations each, about
1,600 messages per conversation). Commands: `docs/RUNBOOK_generalise.md`, step 0. Every set
generates in seconds; CPU rank-eval with the hashing embedder takes under 10 seconds per
plain set and about 4 minutes per background set.

### Decisions

* **v2 is a new module** rather than flags on v1, so v1 stays byte-identical for
  comparisons with earlier results.
* **Per-conversation directories.** Names repeat across generated conversations with
  different clearances, so one dataset-wide principal list cannot work.
* **Enterprise channels.** Every enterprise conversation uses `general` (open), `eng` (the
  whole team, level intern) and `leadership` (the managers, level manager, compartment
  `leadership`); the team-domain threads are moved into `eng`.
* **Viewer-relative gold.** An uncleared asker's gold is the public value and the hidden
  value is a `stale_answer` (mentioning it is an error); a cleared asker's gold is the
  confidential value.
* **Questions in enterprise conversations** carry `meta.channel`: `dm-<asker>` for private
  questions (the asker alone), `eng` for answers into the channel (its members' meet).
* **LoCoMo and the background sets are evaluation only.** The personal domain is trained on
  synthetic data and tested on real chats; background turns come from the evaluation sets.
* **ConvoMem test fixture is synthetic** (CC BY-NC data stays out of the MIT repository).
* **EverMemBench evidence** is the release's context slice (`R`), used as given.
* **`idf_scope: view`** in `rank_v2.yaml` and `hgnn_v2_mix.yaml` (they include enterprise
  data); measured identical to `global` on LoCoMo and LongMemEval (fixed MRR 0.273 and
  0.561 both ways).

### Measurements (CPU: hashing embedder, heuristic extractor)

**B4: windowed linking evidence** (`link_df_window: 500`) against the old global document
frequency (`link_df_window: 1000000000`, the same `max_link_df` ratio), supersession-eval:

| Set | P window / global | R window / global | chained pairs window / global |
| --- | --- | --- | --- |
| SB2 personal test | – / – (1 chain) | 0.000 / 0.000 | 1 / 1 |
| SB2 personal test + LoCoMo background | – / – | 0.000 / 0.000 | 8,295 / 8,694 |
| SB2 assistant test + LongMemEval background | – / – | 0.000 / 0.000 | 7,289 / 7,716 |
| every other v1 / v2 set | identical | identical | identical |

The window changes no labelled pair. In the long background chats it removes 4.6–5.5% of
the (unlabelled) chains, all between background turns. On plain StateBench conversations
(about 30 messages) the window never fills, so the two are the same by construction. The
heuristic finds no labelled personal pair to begin with, so B4's effect on personal recall
cannot be measured until an LLM extraction cache exists; rerun the same two commands then.

**Supersession, heuristic extractor: the pre-Qwen baseline.**

| Set | precision | recall | F1 | false-supersession rate |
| --- | --- | --- | --- | --- |
| StateBench v1 test | 0.869 | 0.073 | 0.135 | 0.002 |
| SB2 team test | 1.000 | 0.050 | 0.096 | 0.000 |
| SB2 assistant test | – | 0.000 | – | 0.000 |
| SB2 personal test | – | 0.000 | – | 0.000 |
| SB2 enterprise test | 1.000 | 0.035 | 0.067 | 0.000 |
| SB2 personal test + LoCoMo bg | – | 0.000 | – | 0.000 |
| SB2 assistant test + LongMemEval bg | – | 0.000 | – | 0.000 |
| (dev, seen phrasing) SB2 team / personal / enterprise | 0.966 / 1.000 / 0.926 | 0.113 / 0.019 / 0.127 | | |

The heuristic covers v2's held-out phrasing, ellipsis and personal subjects almost not at
all: the gap the LLM extractor has to close.

**Ranking (MRR), `configs/eval/rank_v2.yaml` with the hashing embedder:**

| Domain | seeds | fixed |
| --- | --- | --- |
| team (v1 test + SB2 team) | 0.688 | 0.724 |
| assistant (SB2 + background + LongMemEval-S test) | 0.231 | 0.230 |
| personal (SB2 + background + LoCoMo) | 0.342 | 0.336 |
| enterprise (SB2) | 0.657 | 0.672 |
| macro | 0.480 | 0.490 |

SB2 per set (fixed): team 0.668, assistant 0.225, assistant + background 0.077, personal
0.381, personal + background 0.405, enterprise 0.672. v2 is harder than v1 (team 0.668 vs
0.787). Enterprise `leak_rate` is 0.000 with enforcement; in measure-only mode 45.4%
(seeds) and 47.1% (fixed) of questions would have a restricted message in their top 10,
and MRR drops to 0.608 / 0.625 because hidden versions compete.

A 2-epoch CPU smoke run of `hgnn_v2_mix` (60 questions per set, hashing embedder) trained
and evaluated end to end; its numbers are a pipeline check only (it gains on SB2 and loses
on LoCoMo and LongMemEval, the expected sign of a tiny template-only training run).

### Known limitations

* Paraphrasing needs a GPU model and was not run; the v2 numbers above are on templates.
* The heuristic extractor scores 0 recall on v2 assistant and personal; the LLM extractor
  has not been run.
* EverMemBench, ConvoMem and SocialMemBench are loaded and tested on excerpts; they are not
  in the training mix (licence and size decisions for the user, see the final notes).
