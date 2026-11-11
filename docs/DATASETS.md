# Datasets

What each dataset is, how Jarvis-M loads it, whether it can train the HGNN (it needs gold
evidence message ids), and its licence. Download with `jarvis-eval data fetch <name>` into
`$JARVIS_DATA_DIR` (default `data/raw`); `data/` is not tracked.

| Dataset | `name:` / fetch | Domain | Evidence ids (trainable) | Licence | Use here |
| --- | --- | --- | --- | --- | --- |
| Jarvis-StateBench v1 | `jarvis` (`data make-stress`) | team | yes | this repository (MIT) | train / dev / test |
| Jarvis-StateBench v2 | `jarvis` (`data make-stress --domain ...`) | team, assistant, personal, enterprise | yes | this repository (MIT) | train / dev / test; background variants are eval only |
| GroupMemBench | `groupmembench` | team | **no** | MIT | QA eval only |
| LoCoMo | `locomo` | personal | yes | CC BY-NC 4.0 | eval only (not product training) |
| LongMemEval-S | `longmemeval` | assistant | yes | MIT | train split (80%) + test split (20%) |
| EverMemBench-Dynamic | `evermembench` | team | yes (`R`, a context slice) | Apache-2.0 | eval; trainable |
| ConvoMem | `convomem` | assistant | yes (quoted text, matched) | CC BY-NC 4.0 | eval; research training only |
| SocialMemBench | `socialmembench` | personal | yes (`evidence_anchors`) | CC BY 4.0 | eval; trainable with attribution |

Notes per dataset, from inspecting the released files (November 2026):

* **GroupMemBench** (github.com/UCSB-NLP-Chang/GroupMemBench). Questions are
  `{id, question, answer, asking_user_id}` in all 715 rows: there is no per-question
  evidence trace. Messages carry decision labels (`is_decision_point`, `decision_*`), but
  nothing links a question to them, so the dataset is not trainable and stays a QA
  benchmark. Hidden message fields go to `Dataset.annotations`.
* **EverMemBench-Dynamic** (huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic).
  Five topics (`01`–`05`), each about 10,000 messages over three groups and about 250 days,
  with 488–500 questions. `R` lists `{date, group, message_index: "1, 4-6"}`; all 15,509
  references of topic 01 resolve. `R` is the contextual slice around the answer, not only
  the answer message, so recall@k is lower than for single-message evidence. Multiple-choice
  answers are loaded as the option text (the letter is in `meta.correct_option`). Message
  id: `<date>/<group>#<message_index>`. The GitHub repository holds the evaluation code
  only; the older `dataset/<nnn>/` files on the Hub use a different format and are not
  loaded.
* **ConvoMem** (huggingface.co/datasets/Salesforce/ConvoMem). Loaded from
  `core_benchmark/evidence_questions/<type>/<n>_evidence/<person>.json`; one conversation per
  evidence item (user and assistant, 1–6 sessions of about 20–40 messages). Evidence is
  quoted by text: about two thirds of the quotes equal a message, the rest are excerpts or
  near-copies. The loader matches exact text, then normalised containment, then similarity
  ≥ 0.9; on three sample files 498 of 499 quotes resolve (the count of unresolved quotes is
  in `annotations["unresolved_evidence"]`). The pre-mixed test cases and the filler
  conversations are not used. `fetch convomem` downloads the evidence questions only.
* **SocialMemBench** (huggingface.co/datasets/anon4data/socialmembench). 43 group chats,
  7,355 turns, 1,031 questions in nine types (Q1–Q9, used as categories by name). The
  parquet files store lists and dicts as JSON strings (`evidence_anchors_json`,
  `options_json`), unlike the dataset card; all 1,979 anchors resolve to turn ids. A few
  timestamps have minute 60; the loader rolls them over. Reading parquet needs `pyarrow`
  (`pip install 'jarvis-m[data]'`); JSONL exports of the same tables also load.

Test fixtures with small excerpts are in `tests/fixtures/real_samples/` (licences in its
README). The ConvoMem fixture is written for the tests in ConvoMem's layout, because
ConvoMem is non-commercial and this repository is MIT.
