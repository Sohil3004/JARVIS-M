#  Jarvis-M: Cross-Session Memory-Aware Summarizer

**Jarvis-M** is an NLP-based system that performs cross-session summarization of WhatsApp-style group chats.  
It combines transformer-based summarization (BART/T5) with a memory retrieval module (Sentence-BERT + FAISS).

---
##  Features
- Summarizes each chat session individually.
- Retrieves relevant past session summaries (vector memory).
- Generates memory-aware "cross-session" summaries.
- Works on the DialogSum or custom WhatsApp export datasets.

---

##  Architecture
![Architecture](images/architecture.png)

---

##  Installation
```bash
git clone https://github.com/<yourusername>/Jarvis-M.git
cd Jarvis-M
pip install -r requirements.txt
