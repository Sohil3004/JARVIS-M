# Jarvis-M Architecture Documentation

## System Overview

Jarvis-M is a modular conversation summarization system that combines multiple NLP techniques for cross-session and cross-user context retrieval and summarization.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                       │
│                      (app.py)                           │
│  • Streamlit web UI                                     │
│  • Data source selection (WhatsApp/DialogSum)           │
│  • Query interface for cross-user retrieval             │
│  • Per-user summary visualization                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   DATA LAYER                            │
│                (data_loaders.py)                        │
│  • WhatsApp export parser                               │
│  • DialogSum dataset loader                             │
│  • Message extraction and user tagging                  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   MODEL LAYER                           │
│                   (models.py)                           │
│  • BART summarizer (facebook/bart-large-cnn)            │
│  • Sentence-BERT embedder (all-MiniLM-L6-v2)            │
│  • Toxicity classifier (martin-ha/toxic-comment-model)  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              PROCESSING & MEMORY LAYER                  │
│                  (summarizer.py)                        │
│  • Text cleaning and preprocessing (utils.py)           │
│  • Toxicity filtering                                   │
│  • Session summarization                                │
│  • Outlier removal (DBSCAN clustering)                  │
│  • FAISS index construction                             │
│  • Cross-user semantic retrieval                        │
│  • Meta-summary generation                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                EVALUATION & METRICS                     │
│              (evaluate_results.py)                      │
│  • ROUGE score calculation                              │
│  • BERTScore evaluation                                 │
│  • Compression ratio analysis                           │
│  • Timing benchmarks                                    │
│  • Visualization generation                             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Module Breakdown

### 2.1 Data Layer (`data_loaders.py`)

**Purpose**: Ingest and parse conversation data from multiple sources

**Components**:
- `parse_whatsapp_file()`: Parses individual WhatsApp export files
- `load_whatsapp()`: Handles both single files and directories
- `load_dialogsum_samples()`: Loads DialogSum dataset samples

**Data Flow**:
```
Raw text files → Regex parsing → User-message mapping → Dict[user, List[messages]]
```

**Key Features**:
- Regex-based WhatsApp format detection
- Multi-file aggregation
- DialogSum dialogue splitting
- UTF-8 error handling

---

### 2.2 Model Layer (`models.py`)

**Purpose**: Initialize and cache ML models

**Models**:
1. **Summarizer**: BART (facebook/bart-large-cnn)
   - Task: Abstractive text summarization
   - Input: Raw text or concatenated contexts
   - Output: Concise summary (configurable length)

2. **Embedder**: Sentence-BERT (all-MiniLM-L6-v2)
   - Task: Semantic text embedding
   - Input: Text strings
   - Output: 384-dimensional dense vectors

3. **Toxicity Classifier**: Toxic comment model
   - Task: Content moderation
   - Input: Text strings
   - Output: Toxicity probabilities
   - Fallback: Rule-based blacklist if model unavailable

**Optimization**:
- `@st.cache_resource` prevents repeated loading
- Lazy initialization for faster startup

---

### 2.3 Utility Layer (`utils.py`)

**Purpose**: Helper functions for text processing

**Functions**:

1. **Text Cleaning**:
   - `clean_message()`: Remove non-printable chars, timestamps
   - `chunk_text()`: Break long text into summarizable chunks

2. **Toxicity Detection**:
   - `is_toxic_ml()`: ML-based toxicity scoring
   - `is_toxic_basic()`: Rule-based blacklist matching
   - `is_toxic()`: Unified interface (ML with fallback)

3. **Debugging**:
   - `safe_print()`: Exception-safe logging

**Design Philosophy**: Fail gracefully, prefer permissive filtering

---

### 2.4 Summarization & Memory Layer (`summarizer.py`)

**Purpose**: Core business logic for summarization and retrieval

#### Workflow Pipeline:

```
1. Session Summarization
   ├─ Clean messages
   ├─ Filter toxic content
   ├─ Chunk long texts
   └─ Generate BART summaries

2. Outlier Removal
   ├─ Embed all summaries (Sentence-BERT)
   ├─ DBSCAN clustering (cosine distance)
   └─ Filter outliers (label = -1)

3. Memory Construction
   ├─ Aggregate cleaned summaries
   ├─ Generate embeddings
   └─ Build FAISS index (L2)

4. Cross-User Retrieval
   ├─ Embed query
   ├─ FAISS top-k search
   ├─ Re-rank by (similarity × reputation)
   └─ Generate meta-summary from retrieved contexts
```

#### Key Functions:

- `summarize_sessions()`: Per-user session summarization
- `remove_outliers()`: DBSCAN-based filtering
- `build_memory()`: FAISS index construction
- `cross_user_summary()`: Semantic search + meta-summarization

**Algorithmic Details**:
- **DBSCAN Parameters**: `eps=0.35`, `min_samples=2`
- **Re-ranking Formula**: `score = similarity × reputation`
- **Similarity Metric**: `1 / (1 + L2_distance)`

---

### 2.5 UI Layer (`app.py`)

**Purpose**: Interactive Streamlit interface

**Features**:
- Data source selection (sidebar)
- Load & build memory button
- Per-user summary viewer
- Cross-user query interface
- Debug mode (raw data inspection)

**Session State Management**:
- `user_chats`: Parsed conversations
- `user_summaries`: Generated summaries
- `index`, `all_summaries`, `meta`: FAISS memory

**User Flow**:
```
Select data source → Load data → Build memory → Query/Browse summaries
```

---

### 2.6 Evaluation Layer (`evaluate_results.py`)

**Purpose**: Generate research metrics and visualizations

#### Metrics Computed:

1. **ROUGE Scores**:
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence

2. **Compression Ratios**:
   - Original length / Summary length
   - Statistics: mean, std, min, max

3. **Timing Benchmarks**:
   - Load time
   - Summarization time
   - Memory build time
   - Average retrieval time

4. **Retrieval Performance**:
   - Contexts retrieved per query
   - Query-specific analysis

#### Outputs:

**Visualizations**:
- `timing_breakdown.png`: Bar chart of pipeline phases
- `rouge_distributions.png`: Histogram of ROUGE scores
- `compression_ratios.png`: Compression ratio distribution
- `retrieval_performance.png`: Retrieved contexts per query

**Reports**:
- `results_report.txt`: Human-readable summary
- `results.json`: Machine-readable data export

---

## 3. Data Flow Diagram

```
┌─────────────┐
│  Raw Data   │  (WhatsApp exports / DialogSum)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Parser    │  → user_chats: Dict[user, List[messages]]
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Cleaner    │  → Remove toxic/junk messages
└──────┬──────┘
       │
       ↓
┌─────────────┐
│ Summarizer  │  → user_summaries: Dict[user, List[summaries]]
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   DBSCAN    │  → Remove outlier summaries
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Embedder   │  → Generate 384-dim vectors
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    FAISS    │  → Build searchable index
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Query    │  → Semantic search + re-ranking
└──────┬──────┘
       │
       ↓
┌─────────────┐
│Meta-Summary │  → Final context-aware summary
└─────────────┘
```

---

## 4. Key Algorithms

### 4.1 Outlier Removal (DBSCAN)

```python
DBSCAN(metric='cosine', eps=0.35, min_samples=2)

Purpose: Remove irrelevant/noisy summaries before indexing
Input: List of summary embeddings
Output: Filtered summaries (non-outliers only)

Rationale:
- Cosine distance captures semantic similarity
- eps=0.35 balances strictness vs retention
- min_samples=2 ensures clusters have multiple members
```

### 4.2 Cross-User Retrieval

```python
1. Query embedding: q = embedder.encode(query)
2. Candidate retrieval: top_k candidates from FAISS
3. Re-ranking: score = similarity(q, context) × reputation
4. Top-k selection: Sort by score, take top k
5. Meta-summarization: Summarize concatenated contexts
```

**Why Re-ranking?**
- Incorporates user reputation/trust
- Balances relevance with source credibility

---

## 5. Configuration Parameters

### Summarization
- `max_length`: 120 tokens (adjustable)
- `min_length`: 25 tokens
- `chunk_size`: 1500 characters

### Memory
- `eps` (DBSCAN): 0.35
- `min_samples`: 2
- `top_k`: 5 (retrieval count)

### Models
- Summarizer: `facebook/bart-large-cnn`
- Embedder: `all-MiniLM-L6-v2`
- Toxicity: `martin-ha/toxic-comment-model`

---

## 6. Performance Characteristics

### Time Complexity
- **Summarization**: O(n × m) where n = users, m = avg messages
- **Embedding**: O(s × d) where s = summaries, d = embedding dim
- **FAISS Build**: O(s × d)
- **Retrieval**: O(log s) per query (approximate)

### Space Complexity
- **Memory**: O(s × d) for embeddings
- **FAISS Index**: O(s × d)
- **Summaries**: O(s × avg_summary_length)

### Scalability
- **Small scale** (< 1000 summaries): In-memory works fine
- **Medium scale** (1K-100K): FAISS CPU sufficient
- **Large scale** (> 100K): Consider FAISS GPU or distributed systems

---

## 7. Extension Points

### Adding New Data Sources
1. Create parser in `data_loaders.py`
2. Return `Dict[user, List[messages]]` format
3. Register in UI dropdown (`app.py`)

### Custom Summarization Models
1. Modify `models.py` to load new model
2. Ensure input/output format compatibility
3. Update `summarize_text_safe()` if needed

### Additional Metrics
1. Add computation logic to `evaluate_results.py`
2. Create visualization function
3. Update report generation

### New Retrieval Strategies
1. Modify `cross_user_summary()` in `summarizer.py`
2. Implement custom re-ranking logic
3. Optionally add hyperparameters to UI

---

## 8. Design Decisions & Rationale

### Why BART for Summarization?
- Strong abstractive capabilities
- Balanced speed/quality tradeoff
- Wide community support

### Why Sentence-BERT?
- Fast inference (vs full BERT)
- Good semantic representation
- Lightweight (384 dims)

### Why FAISS?
- Industry-standard vector search
- Efficient L2 distance computation
- Scalable to millions of vectors

### Why DBSCAN for Outliers?
- No need to specify cluster count
- Handles arbitrary cluster shapes
- Effective noise filtering

### Why Streamlit?
- Rapid prototyping
- Minimal boilerplate
- Built-in state management

---

## 9. Future Improvements

### Short-term
- [ ] Add more evaluation metrics (BERTScore, perplexity)
- [ ] Support for more chat platforms (Telegram, Slack)
- [ ] Batch processing for large datasets
- [ ] Export results to CSV/Excel

### Medium-term
- [ ] Fine-tune summarizer on dialogue data
- [ ] Implement user reputation learning
- [ ] Add temporal decay for old summaries
- [ ] Multi-language support

### Long-term
- [ ] Real-time streaming summarization
- [ ] Graph-based user relationship modeling
- [ ] Federated learning for privacy
- [ ] Production API deployment

---

## 10. References

### Models
- BART: [Lewis et al., 2020](https://arxiv.org/abs/1910.13461)
- Sentence-BERT: [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)

### Libraries
- Transformers: [Hugging Face](https://huggingface.co/transformers/)
- FAISS: [Facebook AI Research](https://github.com/facebookresearch/faiss)
- Streamlit: [Streamlit Docs](https://docs.streamlit.io)

### Datasets
- DialogSum: [Chen et al., 2021](https://arxiv.org/abs/2105.06762)

---

**Document Version**: 2.0  
**Last Updated**: November 2025  
**Maintainer**: @Sohil3004
