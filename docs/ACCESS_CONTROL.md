# Access control

Enterprise chats mix people with different clearance in one memory. This page describes
how Jarvis-M decides what a question may see, where that decision is enforced, what can
still leak, and which settings change the behaviour.

Code: `src/jarvis_m/access.py` (model), `memory/baselines.py` (enforcement for every
message-level system), `memory/hypergraph/` (views, enforcement in the hypergraph memory).
Policy file: `configs/access/default.yaml`. Tests: `tests/test_access.py`.

## Model

**Levels.** An ordered list, lowest first, e.g. `public < intern < L1 < L2 < manager < exec`.
A role map turns roles into levels (`manager: manager`, `engineer: L1`).

**Message label.** A level plus a set of compartments (`{deal}`, `{hr}`). The effective
label is the join (highest level, union of compartments) of:

1. the channel default (policy `channels:`, raised by a dataset's channel directory);
2. the message's own metadata: `meta.acl_level` (a level name, role name or index) and
   `meta.acl_compartments` (a list);
3. explicit markers in the text, such as `[CONFIDENTIAL]` (policy `markers:`);
4. an extractor's `sensitivity` suggestion, if `use_extractor_suggestions` is on;
5. the label of the message it replies to, in the same channel.

Labels only go up: no source can lower what another raised. A reply into a restricted
thread is restricted too, even when the reply arrives before its parent.

**Principal.** `{id, level, compartments, active}`. Principals and channel membership come
from `Dataset.annotations["principals"]` and `["channels"]`, or from
`memory.configure_access(principals=..., channels=...)`.

**Channel membership** is a list of principal ids, or `"*"` for an open channel. It is
read when the question is asked, so removing someone from a channel takes effect on their
next question. With `unlisted_channels: closed` (the default), a channel the directory
does not list can be read by nobody.

**Read rule.** `can_read(p, m)` holds when `p` is active, `p` is a member of `m`'s channel,
`p.level >= m.level`, and `m.compartments` is a subset of `p.compartments`.

**Audience.** `retrieve(query, asker=..., audience=...)`. With no audience, the answer is
for the asker alone. With a channel name or a list of ids, the effective principal is the
*meet* of the asker and every audience member: the lowest level, the shared compartments,
and membership is checked for each of them. An open channel as audience means anyone may
read the answer, so only public content in open channels qualifies. An inactive audience
member makes the answer empty (`inactive_audience_members: block`) or is left out
(`ignore`). Questions in a dataset carry their audience in `meta.audience`; the runner and
rank-eval pass it on.

## Viewer-relative state

Retrieval in the hypergraph memory works on the effective principal's **view** of the
graph:

* Version chains are built per visibility class (channel, level, compartments). A record
  may *link* to a slot in a class that every reader of its own class can also read, but it
  only ever *writes* to a chain of its own class. So the chains a viewer can see never
  depend on messages that viewer cannot see.
* The view merges the visible chains of the same slot in time order. The viewer's current
  value is the latest version **the viewer can see**. Superseded and current flags,
  hyperedges (threads, episodes, entities, mentions, versions), participants and every
  HGNN feature are computed from visible messages only.
* Example (`tests/fixtures/toy_enterprise.json`): the beta launch was set to March 14 in
  `#eng` and moved to April 2 in a `[CONFIDENTIAL]` message. Bob (L1) is told March 14;
  Dana (exec) is told April 2, with March 14 as the earlier value.

By default a viewer gets **no signal** that a hidden update exists. With
`signal_hidden_updates: true`, a value that has a newer version the viewer cannot read
gets the neutral note "this may have been updated in a restricted context". Nothing else
is added: not the time, the author, the channel or the number of updates.

## Enforcement points (defence in depth)

| Step | What happens |
| --- | --- |
| Labelling | Every message is labelled at ingest (`ThreadLabeler`). |
| Candidate mask | `_candidates(asker, principal)`: the messages the principal may read, narrowed by the speaker-scope ablation. |
| Seeds | BM25 and dense scores are computed for visible messages only; rank fusion ranks among visible messages only. |
| Expansion | The subgraph is built from the view, so expansion never passes through a hidden node. |
| Features, state header, packing | Computed from the view only. |
| Final check | Every message about to be returned, and every message the state header quotes, is re-checked against the read rule from its label. In `debug` mode a violation raises `AccessViolation`; in `prod` mode it is withheld and logged at error level. |
| Audit | `Retrieval.audit`: the effective principal (id, level, compartments, audience members), visible / hidden counts, how many items the final check filtered, and the mode. |

The final check exists for bugs: if everything above works, it never fires. The test
suite sets `JARVIS_ACCESS_MODE=debug`, so any such bug fails a test.

Message-level baselines (`bm25`, `dense`, `hybrid`, `recency`, `full_context`) take the same
`access:` setting and enforce the same mask and final check. `legacy_v2` has no access
control: do not use it on restricted data.

## Guarantees and how they are tested

* **Non-interference** (`test_non_interference`, random fixtures): for a viewer, the graph
  state (chains, current, superseded, all hyperedges) is identical whether or not the
  messages hidden from that viewer were ingested, in any ingest order. With the hashing
  embedder and `idf_scope: view`, retrieved ids, the context text and the features of every
  candidate are identical too.
* **Nothing invisible is returned** (`test_nothing_returned_is_invisible`, random principals
  and audiences): every returned message, every candidate and every message quoted in the
  state header is readable by the effective principal, checked directly from labels and the
  directory. No hidden message text appears in the context.
* **Leak rate.** `rank-eval` reports `leak_rate` for datasets with ACL metadata: the share of
  questions whose top-k ranking holds a message the effective principal may not read. With
  enforcement on this is 0 by construction. `access: {policy: ..., enforce: false}`
  measures what an unprotected system would leak.

## Residual leaks

These are known, and accepted or configurable:

1. **BM25 statistics (default `idf_scope: global`).** IDF and average document length are
   computed over all stored messages, hidden ones included. A hidden message that uses a
   rare word lowers that word's IDF, which can shift the order of visible results. Set
   `idf_scope: view` to compute the statistics over visible messages only (exact
   non-interference, at the cost of recomputing them per query). The default is `global`
   so that numbers on datasets without ACL metadata do not change.
2. **Learned weights.** An HGNN trained on data that includes restricted messages carries
   whatever it learned from them. Train on data the deployment's lowest reader may see, or
   per clearance tier.
3. **Extraction.** The LLM extractor sees neighbouring messages as context. A record about
   a public message can therefore carry a subject or entity phrased with knowledge from a
   restricted neighbour. Extract per channel and clearance tier (the batching is already per
   channel), or turn context off for restricted channels.
4. **Timing and size.** Retrieval time and `audit.hidden` reveal that hidden messages exist.
   Do not show the audit to end users.
5. **Membership changes.** Membership is checked at query time, but the visibility classes
   that decide which chains a message may read were fixed at ingest, using whether a
   channel was open. Closing an open channel later does not rebuild those links. Rebuild the
   memory after such a change.
6. **Signal of hidden updates.** With `signal_hidden_updates: true` the note itself tells
   the viewer that a restricted update exists. It is off by default.

## Policy knobs

| Key | Default | Meaning |
| --- | --- | --- |
| `levels` | `[public, intern, L1, L2, manager, exec]` | Ordered levels, lowest first. |
| `roles` | see file | Role name to level name. |
| `default_level` | `public` | Label of a channel with no entry. |
| `channels` | see file | Per-channel default `level` and `compartments`. |
| `markers` | `[CONFIDENTIAL]` etc. | Text patterns that raise the label. |
| `unlisted_channels` | `closed` | A channel missing from the directory: `closed` (nobody) or `open`. |
| `signal_hidden_updates` | `false` | Add the neutral "may have been updated" note. |
| `inactive_audience_members` | `block` | `block` (empty answer) or `ignore`. |
| `use_extractor_suggestions` | `true` | Let an extractor's `sensitivity` raise labels. |
| `mode` | `prod` | Final check: `prod` (filter and log) or `debug` (raise). The environment variable `JARVIS_ACCESS_MODE` overrides it. |
| `enforce` | `true` | `false` = measure only (nothing is filtered). Never in production. |

Memory settings: `access:` (a path, a mapping, or `{policy: <path>, <key>: <override>}`),
`idf_scope: global | view`.

## Redaction

Out of scope. Messages are withheld whole. `access.Redactor` is an interface stub for a
later implementation that would return a redacted text, or None to withhold.
