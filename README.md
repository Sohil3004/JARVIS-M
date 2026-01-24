# Jarvis-M: A Cross-Session and Cross-User Memory-Aware Dialogue Summarization Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A dialogue summarization system combining Retrieval-Augmented Generation (RAG) with fine-tuned BART-Large-CNN and dual-memory architecture for cross-session and cross-user context retention. This project addresses critical challenges in abstractive dialogue summarization through LoRA-based parameter-efficient fine-tuning, structured memory retrieval with FAISS, and collaborative intelligence mechanisms.

**Authors**: Sumit Santhosh Nair, Sohil Nadakeri Sudarshan  
**Institution**: PES University, Bengaluru, India  
**Department**: Computer Science & Engineering (AI & ML)

---

## Key Features

- **Fine-Tuned BART**: LoRA-adapted `facebook/bart-large-cnn` on DialogSum dataset
- **Validated Retrieval**: FAISS-based semantic search with Recall@K metrics
- **Compression Control**: Dynamic length constraints ensuring 50% compression ratio
- **Research-Backed**: Implements techniques from recent NLP literature on dialogue summarization

---

## Research Motivation

Dialogue summarization presents unique challenges compared to document summarization [1]:

1. **Informal language**: Conversational text contains disfluencies, interruptions, and context-dependent references
2. **Multi-party dynamics**: Cross-speaker references require coreference resolution
3. **Implicit information**: Critical context often unstated, requiring world knowledge

Traditional encoder-decoder models struggle with dialogue-specific phenomena [2]. Our approach combines:

- **Domain adaptation**: Fine-tuning on dialogue-specific data (DialogSum [3])
- **Efficient training**: LoRA (Low-Rank Adaptation) [4] reduces trainable parameters by 98%
- **Retrieval-augmented generation**: RAG framework [5] with Sentence-BERT embeddings [6] for semantic context retrieval
- **Cross-session memory**: Long-term dialogue context retention [7] inspired by transactive memory theory [11]

---

## Methodology

### 1. Fine-Tuning with LoRA

We apply **Low-Rank Adaptation (LoRA)** [4] to BART-large-CNN:

```python
# LoRA configuration
LORA_R = 16              # Rank
LORA_ALPHA = 32          # Scaling factor  
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]
```

**Benefits**:
- **98.85% parameter reduction**: 4.7M trainable vs. 406M total parameters
- **Memory efficient**: Trains on single RTX 3060 (8GB VRAM)
- **Modular**: Adapter can be merged or swapped without retraining base model

### 2. Retrieval Module Evaluation

We evaluate semantic retrieval using **FAISS** [8] with cosine similarity on the DialogSum test set:

| Metric | Task | Purpose |
|--------|------|---------|
| **Recall@1** | Dialogue → Summary | Measures top-1 retrieval accuracy |
| **Recall@5** | Dialogue → Summary | Measures top-5 retrieval coverage |
| **Recall@10** | Dialogue → Summary | Assesses retrieval robustness |
| **MRR** | Cross-modal | Mean Reciprocal Rank for ranking quality |

This replaces the unscientific "100% on 3 queries" claim with rigorous benchmarking on **1,500 test samples**.

### 3. Compression Enforcement

Traditional seq2seq models often generate outputs nearly as long as inputs [9]. We enforce compression via:

```python
# Dynamic length calculation
max_length = int(input_tokens * 0.5)  # 50% compression target

# Generation parameters
length_penalty = 2.0                   # Penalize longer outputs
no_repeat_ngram_size = 3              # Prevent repetition
repetition_penalty = 1.2              # Additional repetition control
```

**Research basis**:
- Length penalty [10]: Encourages brevity in beam search
- N-gram blocking [12]: Prevents redundant phrase generation
- Dynamic constraints [9]: Adapt output length to input complexity

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Sohil3004/JARVIS-M.git
cd JARVIS-M
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fine-tune BART on DialogSum (2-3 hours on GPU)
python train_summarizer.py

# 4. Evaluate retrieval module (5 minutes)
python eval_retrieval.py

# 5. Test improved inference
python update_inference.py
```

---

## Repository Structure

```
JARVIS-M/
├── train_summarizer.py          # Fine-tune BART with LoRA on DialogSum
├── eval_retrieval.py            # Benchmark retrieval with Recall@K metrics
├── update_inference.py          # Inference with compression control
├── check_gpu.py                 # GPU compatibility checker
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── RETRIEVAL_EVALUATION.md      # Retrieval metrics explanation
│
├── models/
│   └── jarvis-bart-lora/        # Fine-tuned LoRA adapters & checkpoints
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── checkpoint-1558/     # Mid-training checkpoint
│       └── checkpoint-2337/     # Final checkpoint (epoch 3)
│
├── cache/                       # HuggingFace model cache (DialogSum, BART)
├── paper/
│   ├── main.tex                 # IEEE conference paper
│   └── architecture_JARVIS-M.png
│
├── old/                         # Archived legacy code
│   ├── jarvis_m_plus_full.py
│   ├── jarvis_m_demo.ipynb
│   └── jarvisM_dialogsum_demo.txt
│
└── .venv/                       # Python virtual environment (not tracked)
```

---

## Training Details

**Algorithm**: LoRA + Seq2Seq optimization [4] with ROUGE evaluation [3]

**Hardware**:
- **Recommended**: T4/A100 GPU (Google Colab free tier)
- **Minimum**: 8GB RAM (CPU training, ~12-18 hours)

**Dataset**: DialogSum [3]
- **Train**: 12,460 dialogue-summary pairs
- **Validation**: 500 pairs
- **Test**: 1,500 pairs

**Hyperparameters**:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch size | 4 | Memory constraint for single GPU |
| Gradient accumulation | 4 | Effective batch size = 16 |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning [4] |
| Epochs | 3 | Prevent overfitting on small dataset |
| LoRA rank (r) | 16 | Balance between capacity and efficiency |

---

## Evaluation Results

### Fine-Tuning Performance

Trained on DialogSum dataset (12,460 training samples, 3 epochs) with LoRA adaptation:

| Metric | Test Set Score |
|--------|----------------|
| ROUGE-1 | **36.13** |
| ROUGE-2 | **14.44** |
| ROUGE-L | **27.60** |
| Average | **26.05** |

**Training Details**:
- Hardware: NVIDIA RTX 5060 Laptop GPU (8.55 GB VRAM)
- Training time: 2 hours 3 minutes
- Trainable parameters: 4.7M (1.15% of total)
- Final training loss: 1.0633

### Retrieval Performance

Evaluated on DialogSum test set (1,500 dialogue-summary pairs) using Sentence-BERT (all-MiniLM-L6-v2) with FAISS indexing:

| Metric | Dialogue → Summary | Summary → Dialogue | Average |
|--------|-------------------|-------------------|----------|
| Recall@1 | 29.07% | 27.73% | **28.40%** |
| Recall@5 | 84.20% | 86.93% | **85.57%** |
| Recall@10 | 90.07% | 91.07% | **90.57%** |
| MRR | 52.28% | - | **52.28%** |

**Key Insights**:
- **Recall@1** (29%): Correct match retrieved at top-1 position
- **Recall@5** (86%): Correct match found within top-5 results
- **High Recall@5-10**: System reliably finds relevant context in top results
- **MRR** (52%): Mean reciprocal rank indicates good ranking quality

These metrics validate the retrieval module on the full test set (1,500 samples), replacing previous unscientific "100% on 3 queries" claims with rigorous benchmarking.

---

## References

### Core Technologies

1. **BART**: Lewis, M., et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL 2020*, pp. 7871-7880.

2. **LoRA**: Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

3. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP-IJCNLP 2019*, pp. 3982-3992.

4. **FAISS**: Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 7(3), 535-547.

5. **Transformers**: Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS 2017*, vol. 30, pp. 5998-6008.

### Datasets & Evaluation

6. **DialogSum**: Chen, Y., Liu, Y., Chen, L., & Zhang, Y. (2021). "DialogSum: A Real-Life Scenario Dialogue Summarization Dataset." *Findings of ACL-IJCNLP 2021*, pp. 5062-5074.

7. **SAMSum**: Gliwa, B., et al. (2019). "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization." *EMNLP 2019*.

8. **ROUGE**: Lin, C.-Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL Workshop*, Barcelona, Spain, pp. 74-81.

### Memory Systems & RAG

9. **RAG**: Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, vol. 33, pp. 9459-9474.

10. **Goldfish Memory**: Xu, J., Szlam, A., & Weston, J. (2022). "Beyond Goldfish Memory: Long-Term Open-Domain Conversation." *ACL 2022*, pp. 5180-5197.

11. **MemGPT**: Packer, C., et al. (2023). "MemGPT: Towards LLMs as Operating Systems." [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)

12. **MADial**: He, J., et al. (2025). "MADial-Bench: Towards Real-world Evaluation of Memory-Augmented Dialogue Generation." *NAACL 2025*.

13. **Collaborative Memory**: Rezazadeh, A., et al. (2025). "Collaborative Memory: A Framework for Asymmetric, Time-Varying Access in Multi-Agent Systems." [arXiv:2505.18279](https://arxiv.org/abs/2505.18279)

### Additional Methods

14. **PEGASUS**: Zhang, J., et al. (2020). "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization." *ICML*, vol. 119, pp. 11328-11339.

15. **Lost in the Middle**: Liu, N. F., et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL*, vol. 12, pp. 157-173.

16. **DBSCAN**: Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." *KDD-96*, pp. 226-231.

17. **ToxicChat**: Liu, Y., et al. (2023). "ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation." *Findings of EMNLP 2023*, pp. 4693-4710.

18. **Transactive Memory**: Wegner, D. M. (1987). "Transactive memory: A contemporary analysis of the group mind." In *Theories of Group Behavior*, Springer-Verlag, pp. 185-208.

19. **PyTorch**: Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*, vol. 32, pp. 8024-8035.

20. **Hugging Face Transformers**: Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." *EMNLP 2020: System Demonstrations*, pp. 38-45.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{jarvism2025,
  author = {Nair, Sumit Santhosh and Sudarshan, Sohil Nadakeri and Gorripati, Ravi},
  title = {Jarvis-M: A Cross-Session and Cross-User Memory-Aware Dialogue Summarization Framework},
  year = {2026},
  institution = {PES University},
  address = {Bengaluru, India},
  note = {Department of Computer Science \& Engineering (AI \& ML)},
  url = {https://github.com/Sohil3004/JARVIS-M}
}
```
2]
- **RAG framework**: Lewis et al. [5]
- **LoRA implementation**: Hu et al. [4] via Hugging Face PEFT library
- **Sentence embeddings**: Reimers & Gurevych [6] via Sentence-Transformers library
- **FAISS indexing**: Johnson et al. [8]
- **Memory-augmented dialogue**: Inspired by work on long-term conversation [7], collaborative memory [13], and MemGPT [14
## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

This work builds upon foundational research in dialogue summarization and retrieval-augmented generation:

- **DialogSum dataset**: Chen et al. (2021) for high-quality dialogue-summary pairs
- **BART model**: Lewis et al. (2020) for pre-trained sequence-to-sequence architecture  
- **LoRA**: Hu et al. (2022) for parameter-efficient fine-tuning via Hugging Face PEFT
- **Sentence-BERT**: Reimers & Gurevych (2019) for semantic embeddings via Sentence-Transformers
- **FAISS**: Johnson et al. (2019) for efficient vector similarity search
- **RAG framework**: Lewis et al. (2020) for retrieval-augmented generation paradigm
- **Memory systems**: Inspired by Goldfish Memory (Xu et al., 2022), MemGPT (Packer et al., 2023), and Collaborative Memory (Rezazadeh et al., 2025)
- **Transactive memory theory**: Wegner (1987) for multi-agent knowledge sharing concepts
- **PyTorch & Transformers**: Paszke et al. (2019) and Wolf et al. (2020) for deep learning infrastructure

