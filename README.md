# Jarvis-M: Cross-Session Memory-Aware Summarizer

**Jarvis-M** is an advanced NLP system for cross-session and cross-user conversation summarization. It combines transformer-based summarization (BART) with semantic memory retrieval (Sentence-BERT + FAISS) to generate context-aware summaries from multi-session chat data.

## Key Features

- **Cross-Session Summarization**: Summarizes conversations across multiple sessions per user
- **Cross-User Memory**: Retrieves and incorporates relevant context from other users
- **Semantic Memory**: Vector-based retrieval using Sentence-BERT embeddings and FAISS
- **Toxicity Filtering**: ML-based and rule-based content filtering
- **Outlier Removal**: DBSCAN clustering to remove irrelevant summaries
- **Interactive UI**: Clean Streamlit interface for real-time interaction
- **Research-Ready**: Comprehensive evaluation metrics and visualization tools

## Research Evaluation

```bash
python evaluate_results.py
```

This produces:
- **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore metrics** for semantic similarity
- **Compression ratios** (original vs summary length)
- **Timing benchmarks** (load, summarization, retrieval)
- **Visualizations**: graphs and charts in `results/` directory
- **JSON export**: machine-readable results for further analysis

## Architecture

```
┌─────────────────────────────────────────┐
│  Data Loaders (data_loaders.py)         │
│  • WhatsApp exports                     │
│  • DialogSum dataset                    │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Models (models.py)                     │
│  • BART summarizer                      │
│  • Sentence-BERT embedder               │
│  • Toxicity classifier                  │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Summarizer (summarizer.py)             │
│  • Session summarization                │
│  • FAISS memory building                │
│  • Cross-user retrieval                 │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  UI Layer (app.py)                      │
│  • Streamlit interface                  │
│  • Interactive query system             │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.9–3.11
- pip or conda

### Quick Setup

```bash
git clone https://github.com/Sohil3004/JARVIS-M.git
cd JARVIS-M
pip install -r requirements.txt
```

## Usage

### Interactive UI

```bash
streamlit run app.py
```

Then:
1. Select data source (DialogSum or WhatsApp)
2. Load and build memory
3. Query across users or view per-user summaries

### Command Line Evaluation

```bash
# Run full evaluation pipeline
python evaluate_results.py

# Results saved to results/ directory:
# - timing_breakdown.png
# - rouge_distributions.png
# - compression_ratios.png
# - retrieval_performance.png
# - results_report.txt
# - results.json
```

## Project Structure

```
JARVIS-M/
├── app.py                    # Streamlit UI
├── models.py                 # Model initialization
├── data_loaders.py           # Data ingestion
├── summarizer.py             # Core summarization logic
├── utils.py                  # Helper functions
├── evaluate_results.py       # Research evaluation script
├── requirements.txt          # Dependencies
├── arch.md                   # Architecture documentation
└── results/                  # Generated evaluation results
```

## Technical Details

### Models Used
- **Summarizer**: `facebook/bart-large-cnn`
- **Embedder**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Toxicity**: `martin-ha/toxic-comment-model`

### Key Parameters
- `top_k`: Number of cross-user contexts to retrieve (default: 5)
- `eps`: DBSCAN clustering epsilon for outlier removal (default: 0.35)
- `max_length`: Maximum summary length (default: 120 tokens)

### Memory Architecture
1. Messages → Individual summaries (BART)
2. Summaries → Embeddings (Sentence-BERT)
3. Embeddings → FAISS index (L2 distance)
4. Query → Retrieve top-k → Re-rank by similarity × reputation
5. Retrieved contexts → Meta-summary (BART)

## Performance Metrics

Based on DialogSum evaluation (50 samples):
- **ROUGE-1**: ~0.35-0.45
- **ROUGE-L**: ~0.30-0.40
- **Compression Ratio**: ~3-5x
- **Retrieval Time**: <0.1s per query

## Customization

### Using WhatsApp Exports

```python
from data_loaders import load_whatsapp

# Load from folder of .txt files
chats = load_whatsapp("path/to/whatsapp/exports")

# Or single file
chats = load_whatsapp("chat_export.txt")
```

### Adjusting Memory Retrieval

```python
# In summarizer.py
def cross_user_summary(..., k=5):  # Change k value
    # More k = more context, slower
    # Less k = faster, less context
```

## Troubleshooting

**Out of Memory (OOM)**
- Switch to CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Reduce batch sizes or k value

**Model Download Issues**
- Ensure internet connection
- Models auto-download on first run (~2GB total)

**FAISS Installation Fails**
- Use `faiss-cpu` instead of `faiss-gpu`
- On Windows: ensure compatible Python version (3.9-3.11)

**Empty Summaries**
- Check toxicity filters aren't too strict
- Verify input data format matches expected structure


## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or collaboration:
- GitHub: [@Sohil3004](https://github.com/Sohil3004)
- Issues: [GitHub Issues](https://github.com/Sohil3004/JARVIS-M/issues)

