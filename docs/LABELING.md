# Labelling supersession pairs

`jarvis-eval supersession-eval` checks the graph's version chains against pairs of
messages that a person (or the StateBench generator) has labelled. This page explains how
to label real chats: LoCoMo, LongMemEval and your own WhatsApp export.

## File format

One file per dataset in `data/labels/supersession/`, any name ending in `.jsonl`, one JSON
object per line:

```json
{"dataset": "locomo", "conversation_id": "conv-26", "older_id": "D3:4", "newer_id": "D9:12", "label": "supersedes", "holder": "speaker:Caroline"}
```

| Field | Meaning |
| --- | --- |
| `dataset` | The dataset name the loader reports: `locomo`, `longmemeval_s_cleaned`, `statebench_test`, or the `name` field of a native-format file. `jarvis-eval data info <exp.yaml>` prints it. |
| `conversation_id` | LoCoMo: `sample_id`. LongMemEval: the `question_id` (each question has its own haystack). Native format: the conversation `id`. |
| `older_id`, `newer_id` | Message ids as the loader builds them. LoCoMo: `dia_id` (`D3:4`). LongMemEval: `<session id>#<turn index>`, turns counted from 0. Native format: the message `id`. `older_id` must be the earlier message. |
| `label` | `supersedes`, `same_slot_no_change` or `unrelated` (below). |
| `holder` | Optional. Whose fact it is: `speaker:<name>`, `group:<channel>` or `person:<name>`. Not scored yet; it is kept for the extractor evaluation. |

Files named `statebench_*.jsonl` are written by `jarvis-eval data make-stress` and are not
tracked in git. Hand-labelled files are tracked.

## The three labels

**`supersedes`**: both messages state a value for the same fact about the same holder, and
after the newer message the older value is no longer true.

* "We're going with AWS EC2 for hosting." then "Scrap EC2, Cloud Run it is."
* "I'm vegetarian." then "I started eating fish again."
* "The staging deploy is failing with a 403." then "Fixed the staging deploy."

**`same_slot_no_change`**: both messages are about the same fact, but the newer one does
not change it. Use this for the cases a memory system is most likely to get wrong:

* a proposal that was not adopted ("Maybe we should try Kafka?");
* an explicit rejection ("No, we are not moving to Azure.");
* a promise ("I'll fix it tomorrow") after a problem report: the problem is still open;
* a restatement of the same value, or a question about it;
* an assistant suggestion the user did not accept.

**`unrelated`**: the two messages state values for different facts, or for the same kind
of fact about different holders. Pick pairs that look confusable: the same entity in two
different decisions, the same date for two different events, two people's preferences
about the same thing ("I prefer window seats" from two different speakers).

## Rules

1. **Label values, not mentions.** Only label a pair when both messages state (or try to
   change) a value. A message that merely mentions the topic is not part of a pair.
2. **Chains are transitive.** If a fact changes twice (A, then B, then C), label all three
   pairs as `supersedes`: A to B, B to C and A to C. The graph is scored on every pair of
   values it puts into one chain.
3. **Direction is time.** `older_id` is always the earlier message. If two people say
   conflicting things and nobody settles it, do not label the pair.
4. **Different holders never supersede.** "My laptop is a ThinkPad" from Anna and "My
   laptop is a MacBook" from Ben is `unrelated`.
5. **A confirmation carries no value.** If the assistant suggests something and the user
   answers "sounds good", the pair is (earlier value, assistant suggestion) with label
   `supersedes`. Do not use the "sounds good" message in a pair.
6. **Skip what you cannot decide in ten seconds.** An unlabelled pair costs nothing. A
   wrong label makes the numbers wrong.

## How much to label

Negatives matter as much as positives: the false-supersession rate is computed only from
`same_slot_no_change` and `unrelated` pairs. Aim for about one negative per positive.

| Source | Suggested amount | How to find candidates |
| --- | --- | --- |
| LongMemEval | 40 questions of type `knowledge-update`, 10 of each other type | The evidence turns (`has_answer`) of a `knowledge-update` question usually contain the old and the new value. |
| LoCoMo | 3 conversations, about 30 pairs each | Read the `temporal` and `multi_hop` questions first; they point at facts that change (jobs, plans, pets, moves). |
| WhatsApp | 2 group chats and 2 one-to-one chats, about 30 pairs each | Search for "actually", "instead", "changed", "moved to", "not anymore", "cancel", "new", and for dates. |

## What the evaluation reports

For each dataset, over the conversations that have at least one label:

* **recall**: labelled `supersedes` pairs that the graph also chains, divided by all
  labelled `supersedes` pairs.
* **precision**: of the pairs the graph chains *that have a label*, the share labelled
  `supersedes`. Chained pairs with no label are reported as "unlabelled predictions" and do
  not count either way, because hand labels are never complete.
* **false-supersession rate**: labelled negatives that the graph chains anyway, divided by
  all labelled negatives.
* **unknown message ids**: label rows whose ids are not in the conversation. This should be
  0; anything else means a typo or a different dataset variant.

```
jarvis-eval supersession-eval configs/eval/supersession.yaml
jarvis-eval supersession-eval configs/eval/supersession.yaml --labels data/labels/supersession/locomo.jsonl
```

## WhatsApp data

Convert the export to the native format first (`{"name", "conversations": [{"id",
"messages": [{"id", "speaker", "text", "timestamp", "channel"}]}], "questions": []}`), with
stable message ids such as `<chat>-<line number>`. Use the file's `name` as `dataset` in
the labels. Keep both the chat file and the label file out of git if the chat is private:
add them to `.gitignore` or store them outside the repository and pass `--labels <path>`.
