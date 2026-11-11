# Loader fixtures from the real releases

Small excerpts of public benchmark releases, so the loader tests run against the real file
layouts without downloading anything. Each excerpt keeps its source licence.

| Folder | Source | Licence | What is in it |
| --- | --- | --- | --- |
| `groupmembench/` | GroupMemBench, github.com/UCSB-NLP-Chang/GroupMemBench | MIT, (c) 2026 The GroupMemBench Authors | the first 8 messages of one Finance channel and one `knowledge_update` and one `abstention` question |
| `evermembench/` | EverMemBench-Dynamic, huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic (topic 01) | Apache-2.0 | two questions (one open, one multiple choice) and the messages around their evidence |
| `socialmembench/` | SocialMemBench, huggingface.co/datasets/anon4data/socialmembench | CC BY 4.0 | the first two sessions of one network and the question anchored in them, as JSONL rows of the parquet tables |
| `convomem_schema/` | **not real data**: written for these tests in the exact file layout of ConvoMem (huggingface.co/datasets/Salesforce/ConvoMem) | (this repository) | ConvoMem is CC BY-NC 4.0, so no excerpt is copied into this MIT repository |
