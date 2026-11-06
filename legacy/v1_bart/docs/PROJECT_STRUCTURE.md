# Project Structure

This document provides a detailed overview of the Jarvis-M repository organization.

---

## Directory Layout

```
JARVIS-M/
│
├── � scripts/                      # Core Python scripts
│   ├── train_summarizer.py          # Fine-tune BART-Large-CNN with LoRA on DialogSum
│   ├── eval_retrieval.py            # Benchmark retrieval system with Recall@K metrics
│   ├── update_inference.py          # Inference module with compression controls
│   ├── ablation_study.py            # Dual-memory ablation experiments
│   └── check_gpu.py                 # GPU compatibility checker for PyTorch
│
├── 📁 docs/                         # Documentation
│   ├── ARCHITECTURE.md              # System architecture (Mermaid diagrams)
│   ├── RETRIEVAL_EVALUATION.md      # Retrieval metrics explanation
│   └── PROJECT_STRUCTURE.md         # This file
│
├── 📄 Root Files
│   ├── README.md                    # Main project documentation
│   ├── requirements.txt             # Python dependencies
│   └── LICENSE                      # MIT License
│
├── 📁 models/                       # Trained models (git-ignored except configs)
│   └── jarvis-bart-lora/           # Fine-tuned LoRA adapters
│       ├── adapter_model.safetensors   (9.6 MB)
│       ├── adapter_config.json
│       ├── training_args.bin
│       ├── checkpoint-1558/        # Epoch 2 checkpoint
│       └── checkpoint-2337/        # Epoch 3 final checkpoint
│
├── 📁 cache/                        # HuggingFace cache (git-ignored)
│   ├── knkarthick___dialogsum/     # DialogSum dataset (1.6 GB)
│   ├── models--facebook--bart-large-cnn/  (1.6 GB)
│   └── models--sentence-transformers--all-MiniLM-L6-v2/
│
├── 📁 old/                          # Archived legacy code
│   ├── jarvis_m_plus_full.py       # Original monolithic implementation
│   ├── jarvis_m_demo.ipynb         # Jupyter notebook demo
│   ├── jarvisM_dialogsum_demo.txt  # Demo output
│   └── jarvisM_live_demo.txt       # Live demo output
│
├── 📁 .venv/                        # Python virtual environment (git-ignored, 5.1 GB)
│
└── 🔒 .gitignore                    # Git ignore rules

```

---

## File Descriptions

### Core Scripts

> All scripts live in [`scripts/`](../scripts/).

#### `scripts/train_summarizer.py`
**Purpose**: Fine-tune BART-Large-CNN on DialogSum using LoRA (Low-Rank Adaptation)

**Key Components**:
- `setup_device()`: GPU/CPU detection with sm_120 fallback
- `load_and_prepare_data()`: DialogSum preprocessing
- `setup_model_and_trainer()`: LoRA configuration, ROUGE metrics
- Training: 3 epochs, batch_size=4, lr=2e-4

**Output**: 
- `models/jarvis-bart-lora/adapter_model.safetensors`
- Checkpoints at epochs 2 and 3
- ROUGE scores on validation set

**Runtime**: ~2 hours on CPU (RTX 5060 used CPU fallback)

---

#### `scripts/eval_retrieval.py`
**Purpose**: Rigorous retrieval evaluation on full DialogSum test set (1,500 samples)

**Key Components**:
- `RetrieverEvaluator` class
- Three evaluation modes:
  1. Dialogue → Summary retrieval
  2. Summary → Dialogue retrieval  
  3. Cross-modal Mean Reciprocal Rank (MRR)
- FAISS cosine similarity search
- Recall@1/5/10 metrics

**Output**:
- Recall@1: 29.07%
- Recall@5: 84.20%
- Recall@10: 90.07%
- MRR: 52.28%

**Runtime**: ~5 minutes

---

#### `scripts/update_inference.py`
**Purpose**: Production inference module with compression enforcement

**Key Components**:
- `JarvisSummarizer` class
- Dynamic length calculation: `max_length = int(input_tokens * 0.5)`
- Generation config:
  - `length_penalty = 2.0` (brevity encouragement)
  - `no_repeat_ngram_size = 3` (repetition blocking)
  - `repetition_penalty = 1.2` (redundancy control)
- Compression statistics tracking

**Output**:
- 29.30% compression ratio (273 tokens → 80 tokens)
- 70.7% token reduction

**Runtime**: <1 second per dialogue

---

#### `scripts/check_gpu.py`
**Purpose**: Utility to check GPU compatibility with PyTorch

**Usage**: 
```bash
python scripts/check_gpu.py
```

**Detects**:
- CUDA availability
- GPU name and memory
- Compute capability (e.g., sm_120 for RTX 5060)
- PyTorch version compatibility

---

### Documentation

> All docs live in [`docs/`](../docs/).

#### `README.md`
Comprehensive project overview including:
- Research motivation and methodology
- Quick start guide
- Training details and hyperparameters
- Evaluation results (ROUGE, Recall@K)
- 20 academic references
- Citation format (BibTeX)

#### `docs/RETRIEVAL_EVALUATION.md`
Explains "failed retrievals" concept:
- 1,064 "failures" = rank>1 retrievals (not actual failures)
- 84% Recall@5 is excellent performance
- Clarifies evaluation methodology

#### `docs/PROJECT_STRUCTURE.md`
This document - detailed repository organization guide.

#### `requirements.txt`
Python dependencies:
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
evaluate>=0.4.0
rouge-score>=0.1.2
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
accelerate>=0.20.0
tqdm>=4.65.0
```

---

### Directories

#### `models/`
**Size**: ~140 MB (140.45 MB total)

**Contents**:
- LoRA adapters (9.6 MB safetensors)
- Training checkpoints (optimizer states: 65 MB each)
- Configuration files (adapter_config.json, training_args.bin)

**Git Policy**: 
- Tracked: Config files (JSON)
- Ignored: Large binary files (*.safetensors, *.pt, checkpoint-*)

#### `cache/`
**Size**: ~1.66 GB (1,661.81 MB total)

**Contents**:
- DialogSum dataset: 12,460 train + 500 val + 1,500 test samples
- BART-Large-CNN: 406M parameter pre-trained model
- Sentence-BERT: all-MiniLM-L6-v2 embeddings model

**Git Policy**: Fully ignored (regenerated on first run)

#### `old/`
**Size**: ~30 KB

**Contents**: Archived legacy implementations
- `jarvis_m_plus_full.py`: Original monolithic script
- `jarvis_m_demo.ipynb`: Jupyter demo notebook
- `jarvisM_dialogsum_demo.txt`: Demo outputs
- `jarvisM_live_demo.txt`: Live demo outputs

**Git Policy**: Tracked (for historical reference)

#### `.venv/`
**Size**: ~5.12 GB (5,120.12 MB)

**Contents**: Python virtual environment
- PyTorch with CUDA 12.6 support (2.5 GB)
- Transformers, FAISS, Sentence-Transformers
- All dependencies from requirements.txt

**Git Policy**: Fully ignored

---

## Data Flow

### 1. Training Pipeline
```
DialogSum (cache/)
    ↓
scripts/train_summarizer.py
    ↓ (LoRA fine-tuning)
models/jarvis-bart-lora/
    ↓ (adapter weights)
Evaluation on validation set
    ↓
ROUGE scores saved
```

### 2. Retrieval Evaluation Pipeline
```
DialogSum test set (1,500 samples)
    ↓
Sentence-BERT embeddings
    ↓
FAISS indexing
    ↓
scripts/eval_retrieval.py
    ↓
Recall@K metrics, MRR
```

### 3. Inference Pipeline
```
Input dialogue (user input)
    ↓
scripts/update_inference.py
    ↓ (LoRA adapter loaded)
Dynamic length calculation
    ↓ (compression controls)
Generated summary
    ↓
Compression statistics
```

---

## Size Management

### Large Files (>10 MB)
All files >10MB are automatically git-ignored:

**Location**: `.venv/`, `cache/`, `models/checkpoint-*/`

**Examples**:
- `torch_cpu.dll` (PyTorch): ~150 MB
- `dialogsum-train.arrow`: ~50 MB
- `model.safetensors` (BART): ~1.6 GB
- `adapter_model.safetensors`: ~9.6 MB
- `optimizer.pt` (checkpoints): ~65 MB each

**Management Strategy**:
1. Use `.gitignore` to exclude large binaries
2. Keep only final adapter (not checkpoints) in cloud
3. Document download instructions in README
4. Consider Git LFS for model sharing (optional)

### Total Repository Size

| Component | Size | Tracked |
|-----------|------|---------|
| Scripts + Docs | ~60 KB | ✓ |
| Old files | ~30 KB | ✓ |
| Models (final) | ~10 MB | ✗ (optional) |
| Models (checkpoints) | ~130 MB | ✗ |
| Cache (datasets + models) | ~1.66 GB | ✗ |
| Virtual environment | ~5.12 GB | ✗ |
| **Total on disk** | **~6.92 GB** | |
| **Git tracked** | **~90 KB** | |

---

## Development Workflow

### Initial Setup
```bash
# 1. Clone repository
git clone https://github.com/Sohil3004/JARVIS-M.git
cd JARVIS-M

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies (downloads ~1.6 GB to cache/)
pip install -r requirements.txt
```

### Training
```bash
# Fine-tune BART with LoRA (~2 hours on CPU)
python scripts/train_summarizer.py

# Output: models/jarvis-bart-lora/ (~10 MB)
```

### Evaluation
```bash
# Benchmark retrieval (5 minutes)
python scripts/eval_retrieval.py

# Output: Console metrics (Recall@K, MRR)
```

### Inference
```bash
# Test compression (instant)
python scripts/update_inference.py

# Output: Summary + compression stats
```

---

## Git Workflow

### What's Tracked
- Python scripts (scripts/*.py)
- Documentation (docs/*.md, README.md)
- Configuration (requirements.txt, .gitignore)
- Old reference code (old/)

### What's Ignored
- Virtual environment (.venv/)
- Model cache (cache/)
- Large model files (*.safetensors, *.bin, *.pt)
- Training checkpoints (checkpoint-*)
- Python artifacts (__pycache__/, *.pyc)
- IDE configs (.vscode/, .idea/)
- Logs and outputs (*.log)

### Recommended Commands
```bash
# Check git status (should show only code/docs)
git status

# Stage changes
git add scripts/train_summarizer.py README.md

# Commit
git commit -m "Update training script with new hyperparameters"

# Push
git push origin main
```

---

## Future Enhancements

### Planned Additions
1. **Evaluation Scripts**: `test_suite.py` for CI/CD
2. **Experiment Tracking**: Weights & Biases integration
3. **Docker Support**: Containerized deployment
4. **API Server**: Flask/FastAPI endpoint for inference
5. **Visualization Tools**: ROUGE score plots, retrieval heatmaps
6. **Multi-GPU Support**: Distributed training scripts

### Maintenance
- **Monthly**: Update dependencies in requirements.txt
- **Per-experiment**: Archive old checkpoints to external storage
- **Releases**: Tag versions (v1.0, v1.1) for major updates

---

## Contact

**Authors**: Sumit Santhosh Nair, Sohil Nadakeri Sudarshan
**Institution**: PES University, Bengaluru, India  
**Email**: pes1ug23am324@pesu.pes.edu, pes1ug23am310@pesu.pes.edu  
**Repository**: https://github.com/Sohil3004/JARVIS-M

---

*Last Updated: January 24, 2026*
