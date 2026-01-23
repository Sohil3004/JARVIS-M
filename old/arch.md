┌──────────────────────────────────────────────┐
│ 1. Data Layer                                │
│    • WhatsApp export / API JSON              │
│    • User-, group-, session-tagged messages  │
├──────────────────────────────────────────────┤
│ 2. Embedding + Memory Layer                  │
│    • Sentence-BERT → vector embeddings       │
│    • FAISS index per user / group            │
├──────────────────────────────────────────────┤
│ 3. Summarization Layer                       │
│    • BART or T5 summarizer                   │
│    • Cross-session & cross-user modes        │
├──────────────────────────────────────────────┤
│ 4. API / CLI Interface                       │
│    • “summarize_user(user)”                  │
│    • “summarize_cross_user(query)”           │
└──────────────────────────────────────────────┘
