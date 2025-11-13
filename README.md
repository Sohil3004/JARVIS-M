#  Jarvis-M: Cross-Session Memory-Aware Summarizer

**Jarvis-M** is an NLP-based system that performs cross-session summarization of WhatsApp-style group chats.  
It combines transformer-based summarization (BART/T5) with a memory retrieval module (Sentence-BERT; FAISS optional for scaling).

---
##  Features
- Summarizes each chat session individually.
- Retrieves relevant past session summaries (vector memory).
- Intra-user memory: recalls the most relevant summaries from the same user across sessions.
- Cross-user memory: retrieves top-k relevant summaries from other users to provide social/contextual cues.
- Generates a hybrid, memory-aware “cross-session + cross-user” summary.
- Exports retrieved contexts and final summary to a report file.
- Works on the DialogSum dataset or can be adapted to custom WhatsApp exports.


---



##  Installation

Prerequisites:
- Python 3.9–3.11
- pip

Option A: Using requirements.txt
```bash
git clone https://github.com/Sohil3004/Jarvis-M.git
cd Jarvis-M
pip install -r requirements.txt
```

Option B: Install key dependencies manually
```bash
# Core
pip install datasets transformers sentence-transformers

# PyTorch (Windows)
# CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# (Optional) GPU with CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional: FAISS for large-scale retrieval (demo uses in-memory cosine sim)
pip install faiss-cpu

# Notebook tooling
pip install jupyter ipykernel
```

Notes:
- The demo notebook uses Sentence-BERT embeddings with cosine similarity in-memory. Install FAISS if you plan to index many sessions.
- If you already have PyTorch installed, skip its line above.

---

##  Quickstart

VS Code (Windows):
1) Open the repository folder in VS Code.
2) Create/activate a Python environment and select it as the interpreter.
3) Open `jarvis_m_demo.ipynb`.
4) Run all cells. The demo:
   - Loads DialogSum (auto-download via `datasets`).
   - Summarizes three sessions.
   - Retrieves intra-user and cross-user memories.
   - Produces a memory-aware summary and writes:
     - `jarvisM_dialogsum_demo.txt`

Jupyter:
```bash
jupyter notebook
# open jarvis_m_demo.ipynb and Run All
```

---

##  How cross-user retrieval works

- For the current session, Jarvis-M encodes the raw dialogue with Sentence-BERT (all-MiniLM-L6-v2).
- It searches:
  - Intra-user: most relevant past summaries by the same user.
  - Cross-user: top-k most relevant summaries by other users.
- It concatenates retrieved contexts with the current dialogue and summarizes with BART (`facebook/bart-large-cnn`).

Key parameters (editable in the notebook):
- `top_k`: number of cross-user summaries to retrieve (default: 1).
- `summarizer` model: `facebook/bart-large-cnn` by default.
- `embedder` model: `sentence-transformers/all-MiniLM-L6-v2`.

Output artifacts:
- Retrieved intra-user context (if any).
- Retrieved cross-user context (if any).
- Final cross-user memory-aware summary.
- Text report: `jarvisM_dialogsum_demo.txt`.

---

##  Custom data

- DialogSum is used in the demo for convenience.
- For WhatsApp exports, adapt the ingestion step to:
  - Parse per-session conversations.
  - Associate messages with a `user_id`.
  - Produce per-session summaries and store them with `user_id` for retrieval.

---

##  Tips and troubleshooting

- Long inputs: Hybrid inputs (retrieved contexts + dialogue) can exceed model limits.
  - Reduce `top_k`, shorten contexts, or chunk the input before summarization.
  - Adjust `max_length`/`min_length` in the summarizer pipeline.
- GPU memory: If using GPU and you see OOM errors, switch to CPU or reduce batch sizes.
- First run downloads models from Hugging Face; ensure internet access.
- If FAISS install fails on Windows, keep using in-memory cosine similarity or try `faiss-cpu` prebuilt wheels.

---

##  Changelog

- 0.2: Added cross-user retrieval and hybrid memory-aware summarization; report file now includes retrieved cross-user context.
- 0.1: Intra-user cross-session summarization.
