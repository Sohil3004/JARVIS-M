# Jarvis-M v3: hypergraph memory

Group-chat facts are n-ary (who said what, about which thing, in which group,
replacing what) and they change. The hypergraph keeps every raw message as a node
and adds typed hyperedges, so retrieval can follow threads, shared entities and the
version history of a topic instead of matching isolated messages.

## Design

| Hyperedge | Members | Built from | Purpose |
| --- | --- | --- | --- |
| `reply` | a thread root + all replies | `reply_to` | hand-offs, Q→A, multi-hop |
| `episode` | consecutive messages in a channel/session, gap ≤ 60 min, ≤ 25 msgs | timestamps | local context |
| `channel` | everyone in a group/DM | channel | scope, access control (product) |
| `speaker` | everything one person said | author | personal trajectory, "my …" questions |
| `entity` | messages mentioning one entity | extractor | joins across threads/channels |
| `version` | successive values of one (topic, attribute) slot | extractor | knowledge updates |

Version chains: a new value marks earlier ones `superseded`; a negation ("we are
*not* moving to X") joins the slot but never becomes current; a completion resolves
the blocker it shares a topic/entity with, a promise ("I will fix it") does not.
Slots link on the same (topic, attribute) key, else on a shared entity within the
same channel/phase; a replacement cue alone never links (it chained chit-chat into
real slots during development).

Retrieval: hybrid (BM25 + dense, RRF) seeds → expand over small hyperedges (+ current
value of every touched slot, + the asker's own commitments for first-person
questions) → score → pack. Scorers:

* `seeds`: candidate set only (≡ hybrid; sanity check).
* `fixed`: additive neighbour-evidence boost over reply/entity/version edges
  (β = 0.2) + version logic (pull current values up to their best chain-mate, demote
  superseded ones unless the question is historical: "originally", "before" …).
* `hgnn`: learned. Edge-type-aware node→hyperedge→node message passing with
  query-conditioned hyperedge gating, a residual on the seed score, trained with a
  listwise loss on gold evidence inside each question's candidate subgraph.

Closest prior work, HyperMem (ACL 2026), uses fixed propagation over a
topic→episode→fact hypergraph for a single user and does not retire superseded
facts. The `fixed` scorer is our stand-in for that family; the claim to test is that
learned, version- and speaker-aware propagation beats it on multi-party data.

## First findings (CPU only — not paper numbers)

Setup: hashing embedder (not Qwen3-Embedding), heuristic extractor, no reader LLM;
retrieval ranking metrics over each question's candidate subgraph. StateBench
**test** uses phrasing families never seen in training. Train 1,837 / dev 459 /
test 900 questions with evidence; candidate recall ≥ 0.996. One seed.

MRR on StateBench test (held-out phrasing):

| Scorer | R@5 | MRR | knowledge update | multi-hop | single-hop | temporal | user-implicit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seeds (= hybrid) | 0.930 | 0.757 | 0.768 | 0.565 | 0.941 | 0.873 | 0.131 |
| fixed: version only | 0.923 | 0.730 | 0.766 | 0.593 | 0.850 | 0.786 | 0.203 |
| fixed: default | 0.925 | 0.758 | 0.775 | 0.948 | 0.816 | 0.805 | 0.228 |
| HGNN, no message passing | 0.998 | 0.839 | 0.846 | 0.745 | 0.849 | 0.860 | 0.844 |
| HGNN, gated | 0.997 | 0.851 | 0.850 | 0.825 | 0.843 | 0.849 | 0.914 |
| HGNN, no gating | 0.997 | 0.863 | 0.825 | 0.909 | 0.898 | 0.844 | 0.914 |

What this says so far:

1. **The learned scorer is the gain.** +0.09–0.11 MRR over hybrid on unseen phrasing;
   fixed propagation alone is a trade (multi-hop 0.57→0.95, single-hop 0.94→0.82).
2. **Most of the HGNN's gain is learned feature weighting** (version flags, asker,
   modality): the no-message-passing model already gets 0.839. Message passing adds
   +0.08–0.16 on multi-hop and +0.07 on user-implicit — where structure should matter.
3. **Gating is not yet justified**: no-gating ≥ gated on test (one seed, Δ ≈ 0.01).
4. **Phrasing shift costs ~0.10 MRR** (dev 0.96–0.98 → test 0.85–0.86): the heuristic
   extractor misses casual phrasings, which is the case for Qwen extraction.
5. **Weak spot**: HGNN single-hop (0.84–0.90) is below hybrid (0.94) on test.
6. Harness end-to-end (2,834 questions, overlap reader): version-aware arms beat
   hybrid by +2.3 accuracy points (McNemar p ≈ 1e-13); stale-answer rate 0.353 → 0.320;
   legacy v2 retrieves gold evidence for 31% of questions vs 92% for hybrid.

## Open questions for the GPU runs

* Do the gains hold with Qwen3-Embedding seeds and a real reader + judge?
* Does Qwen extraction close the phrasing gap (test vs dev) and fix single-hop?
* Zero-shot transfer to GroupMemBench (no evidence labels to train on there).
* ≥ 3 seeds for every learned arm before claiming gated vs ungated.
* Keep single-hop safe: e.g. a learned mixing weight with the seed score per query type.
