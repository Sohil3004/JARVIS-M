# JARVIS-M — System Architecture

> Mermaid diagrams below render natively on GitHub. All colours use a dark-mode palette.

---

## High-Level Overview

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#1f6feb', 'primaryTextColor': '#e6edf3', 'primaryBorderColor': '#388bfd', 'secondaryColor': '#238636', 'tertiaryColor': '#161b22', 'lineColor': '#8b949e', 'textColor': '#e6edf3', 'mainBkg': '#0d1117', 'nodeBorder': '#30363d', 'clusterBkg': '#161b22', 'clusterBorder': '#30363d'}}}%%
graph TB
    subgraph DATA ["Data Layer"]
        direction LR
        D1["DialogSum Dataset<br/><i>12,460 train · 500 val · 1,500 test</i>"]
        D2["WhatsApp / API JSON<br/><i>user-, group-, session-tagged</i>"]
    end

    subgraph EMBED ["Embedding & Memory"]
        direction LR
        E1["Sentence-BERT<br/><code>all-MiniLM-L6-v2</code>"]
        E2["FAISS Index<br/><i>per-user / per-group</i>"]
        E3["Dual Memory<br/><i>intra-user + cross-user</i>"]
    end

    subgraph SUMM ["Summarization Layer"]
        direction LR
        S1["BART-Large-CNN<br/><i>LoRA r=16, α=32</i>"]
        S2["Compression Control<br/><i>50% ratio · length penalty</i>"]
        S3["Beam Reranking<br/><i>memory-guided scoring</i>"]
    end

    subgraph API ["Interface Layer"]
        direction LR
        A1["CLI / API<br/><code>summarize_text_safe()</code>"]
        A2["Evaluation<br/><i>ROUGE · Recall@K · MRR</i>"]
    end

    DATA --> EMBED --> SUMM --> API

    style DATA fill:#161b22,stroke:#1f6feb,color:#e6edf3
    style EMBED fill:#161b22,stroke:#238636,color:#e6edf3
    style SUMM fill:#161b22,stroke:#da3633,color:#e6edf3
    style API fill:#161b22,stroke:#a371f7,color:#e6edf3
```

---

## Training Pipeline

> **Entry point:** [`scripts/train_summarizer.py`](../scripts/train_summarizer.py)

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#1f6feb', 'primaryTextColor': '#e6edf3', 'primaryBorderColor': '#388bfd', 'lineColor': '#8b949e', 'textColor': '#e6edf3', 'mainBkg': '#0d1117'}}}%%
flowchart LR
    A["DialogSum<br/><code>cache/</code>"] -->|load_and_prepare_dataset| B["Tokenize<br/><code>preprocess_function()</code>"]
    B -->|LoRA injection| C["BART + LoRA<br/><code>setup_lora_model()</code>"]
    C -->|Seq2SeqTrainer| D["Fine-Tune<br/><i>3 epochs · lr 2e-4</i>"]
    D -->|save adapter| E["LoRA Weights<br/><code>models/jarvis-bart-lora/</code>"]
    D -->|compute_metrics| F["ROUGE Scores<br/><i>R1 36.1 · R2 14.4 · RL 27.6</i>"]

    style A fill:#1f6feb,stroke:#388bfd,color:#e6edf3
    style B fill:#238636,stroke:#2ea043,color:#e6edf3
    style C fill:#238636,stroke:#2ea043,color:#e6edf3
    style D fill:#da3633,stroke:#f85149,color:#e6edf3
    style E fill:#a371f7,stroke:#bc8cff,color:#e6edf3
    style F fill:#a371f7,stroke:#bc8cff,color:#e6edf3
```

### Key functions

| Function | File | Purpose |
|----------|------|---------|
| `setup_device()` | [`train_summarizer.py`](../scripts/train_summarizer.py) | GPU / CPU detection with sm_120 fallback |
| `load_and_prepare_dataset()` | [`train_summarizer.py`](../scripts/train_summarizer.py) | DialogSum loading & preprocessing |
| `setup_lora_model()` | [`train_summarizer.py`](../scripts/train_summarizer.py) | LoRA config (r=16, α=32, q/k/v/out_proj) |
| `compute_metrics()` | [`train_summarizer.py`](../scripts/train_summarizer.py) | ROUGE-1/2/L evaluation |

---

## Retrieval Evaluation Pipeline

> **Entry point:** [`scripts/eval_retrieval.py`](../scripts/eval_retrieval.py)  
> **Metrics explained:** [`docs/RETRIEVAL_EVALUATION.md`](RETRIEVAL_EVALUATION.md)

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#1f6feb', 'primaryTextColor': '#e6edf3', 'primaryBorderColor': '#388bfd', 'lineColor': '#8b949e', 'textColor': '#e6edf3', 'mainBkg': '#0d1117'}}}%%
flowchart TD
    A["DialogSum Test Set<br/><i>1,500 samples</i>"] --> B["Sentence-BERT<br/><code>encode()</code>"]
    B --> C["FAISS Index<br/><code>build_faiss_index()</code>"]

    C --> D1["Dialogue → Summary<br/><code>evaluate_dialogue_to_summary()</code>"]
    C --> D2["Summary → Dialogue<br/><code>evaluate_summary_to_dialogue()</code>"]
    C --> D3["Cross-Modal MRR<br/><code>evaluate_cross_retrieval()</code>"]

    D1 --> E["Recall@1 29% · @5 84% · @10 90%"]
    D2 --> E
    D3 --> E

    style A fill:#1f6feb,stroke:#388bfd,color:#e6edf3
    style B fill:#238636,stroke:#2ea043,color:#e6edf3
    style C fill:#238636,stroke:#2ea043,color:#e6edf3
    style D1 fill:#da3633,stroke:#f85149,color:#e6edf3
    style D2 fill:#da3633,stroke:#f85149,color:#e6edf3
    style D3 fill:#da3633,stroke:#f85149,color:#e6edf3
    style E fill:#a371f7,stroke:#bc8cff,color:#e6edf3
```

---

## Ablation Study — Dual-Memory Reranking

> **Entry point:** [`scripts/ablation_study.py`](../scripts/ablation_study.py)

### Ablation Configurations

| Config | Intra-User | Cross-User | Description |
|--------|:----------:|:----------:|-------------|
| `no_memory` | ✗ | ✗ | Baseline — beam search only |
| `intra_only` | ✓ | ✗ | Single-user session memory |
| `cross_only` | ✗ | ✓ | Cross-user collaborative memory |
| `full` | ✓ | ✓ | Full dual-memory pipeline |

---
