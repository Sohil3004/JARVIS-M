
# Fully patched Jarvis-M++ with evaluation (ROUGE + BERTScore + retrieval + stats)
import os, re, json, faiss, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import streamlit as st

# transformers / sentence-transformers / datasets / sklearn
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.cluster import DBSCAN

# evaluation imports
from rouge_score import rouge_scorer

# BERTScore (optional)
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False

# Light-weight BERTScore model
BERTSCORE_MODEL = "microsoft/deberta-base-mnli"   # good balance
# For slow machines use tiny model:     "distilroberta-base"


# -------------------------
# Config
# -------------------------
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
TOXIC_CLASS_MODEL = "martin-ha/toxic-comment-model"  # light-weight fallback
# BERTScore model choice (smaller to reduce GPU/memory pressure)
BERTSCORE_MODEL = "roberta-base"

st.set_page_config(page_title="Jarvis-M++ Eval", layout="wide")

# -------------------------
# Lazy init models (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def init_models():
    models = {}
    models["summarizer"] = pipeline("summarization", model=SUMMARIZER_MODEL)
    models["embedder"] = SentenceTransformer(EMBEDDER_MODEL)
    try:
        models["tox_clf"] = pipeline("text-classification", model=TOXIC_CLASS_MODEL, top_k=None)
    except Exception:
        models["tox_clf"] = None
    return models

models = init_models()
summarizer = models["summarizer"]
embedder = models["embedder"]
tox_clf = models["tox_clf"]

# -------------------------
# Sanitizer & Toxicity (lenient)
# -------------------------
BASIC_BLACKLIST = ["nigger", "nigga", "bomb", "terror", "kill", "attack", "drugs", "heroin"]

def clean_message(msg):
    if msg is None:
        return None
    try:
        msg = msg.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        return None
    msg = msg.strip()
    if len(msg) < 3:
        return None
    # metadata-only lines
    meta_re = r"^\[?\d{1,2}\/\d{1,2}\/\d{2,4}.*-\s*[^:]+:\s*$"
    if re.match(meta_re, msg):
        return None
    # mostly punctuation
    if len(re.sub(r"[A-Za-z0-9]", "", msg)) > 0.8 * len(msg):
        return None
    low = msg.lower()
    for b in BASIC_BLACKLIST:
        if b in low:
            return None
    return msg

def is_toxic(text, threshold=0.85):
    if not text:
        return False
    if tox_clf:
        try:
            preds = tox_clf(text[:512])
            toxic_score = 0.0
            for p in preds:
                lab = p.get("label", "").lower()
                sc = float(p.get("score", 0.0))
                if any(k in lab for k in ["toxic", "insult", "hate", "offensive", "severe"]):
                    toxic_score += sc
            return toxic_score > threshold
        except Exception:
            pass
    # fallback
    low = text.lower()
    return any(b in low for b in BASIC_BLACKLIST)

# -------------------------
# WhatsApp parser (robust)
# -------------------------
whatsapp_re = re.compile(r"^\[?\d{1,2}\/\d{1,2}\/\d{2,4}.*?-\s*(.*?):\s*(.*)$")

def parse_whatsapp_file(path):
    user_msgs = defaultdict(list)
    last_user = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = whatsapp_re.match(line)
            if m:
                user = m.group(1).strip()
                msg = m.group(2).strip()
                cleaned = clean_message(msg)
                if cleaned:
                    user_msgs[user].append(cleaned)
                    last_user = user
            else:
                # continuation
                if last_user and user_msgs[last_user]:
                    cont = clean_message(line)
                    if cont:
                        user_msgs[last_user][-1] += " " + cont
    return dict(user_msgs)

def load_whatsapp(path_in):
    p = Path(path_in)
    if not p.exists():
        raise ValueError(f"Invalid WhatsApp path: {path_in}")
    if p.is_file():
        return parse_whatsapp_file(p)
    elif p.is_dir():
        aggregated = {}
        for f in p.iterdir():
            if f.suffix.lower() == ".txt":
                parsed = parse_whatsapp_file(f)
                # if group chat contains many users, prefix with filename
                for u, msgs in parsed.items():
                    key = f"{f.stem}__{u}"
                    aggregated[key] = msgs
        return aggregated
    else:
        raise ValueError(f"Invalid WhatsApp path: {path_in}")

# -------------------------
# DialogSum loader (returns dataset too)
# -------------------------
def load_dialogsum_samples(n=3):
    dataset = load_dataset("knkarthick/dialogsum")
    samples = {}
    total = len(dataset["train"])
    n = min(n, total)
    for i in range(n):
        dialogue = dataset["train"][i]["dialogue"]
        turns = []
        for line in dialogue.split("\n"):
            line = line.strip()
            if not line:
                continue
            # remove speaker token if present
            if ":" in line:
                parts = line.split(":", 1)
                turns.append(parts[1].strip())
            else:
                turns.append(line)
        samples[f"dialogsum_{i}"] = turns
    return samples, dataset

# -------------------------
# Summarization (adaptive, safe chunking)
# -------------------------
def chunk_text(text, max_chars=1500):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        slice_ = text[start:end]
        sep = max(slice_.rfind("."), slice_.rfind("\n"), slice_.rfind("?"), slice_.rfind("!"))
        if sep <= 0:
            sep = end
        else:
            sep = start + sep + 1
        chunks.append(text[start:sep].strip())
        start = sep
    return [c for c in chunks if c]

def summarize_text_adaptive(text):
    text = clean_message(text)
    if not text:
        return None
    words = text.split()
    wc = len(words)
    # adapt lengths
    max_len = max(5, int(wc * 0.6))
    min_len = max(3, int(wc * 0.25))
    try:
        if len(text) < 1500:
            out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            return out[0]["summary_text"]
        else:
            chunks = chunk_text(text, max_chars=1500)
            parts = []
            for ch in chunks:
                try:
                    out = summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)
                    parts.append(out[0]["summary_text"])
                except Exception:
                    continue
            joined = " ".join(parts)
            # final compress
            out = summarizer(joined, max_length=max_len, min_length=min_len, do_sample=False)
            return out[0]["summary_text"]
    except Exception:
        return text

# -------------------------
# Summarize sessions
# -------------------------
def summarize_sessions(user_chats, debug=False):
    user_summaries = {}
    for user, msgs in user_chats.items():
        sums = []
        for m in msgs:
            mclean = clean_message(m)
            if not mclean:
                if debug: st.write(f"[SKIP clean] {user}: {m[:60]}")
                continue
            if is_toxic(mclean):
                if debug: st.write(f"[SKIP toxic] {user}: {mclean[:60]}")
                continue
            s = summarize_text_adaptive(mclean)
            if s and len(s.strip())>0:
                sums.append(s)
                if debug: st.write(f"[OK sum] {user}: {s[:100]}")
        user_summaries[user] = sums
    return user_summaries

# -------------------------
# Outlier removal
# -------------------------
def remove_outliers(summaries, eps=0.37):
    if not summaries:
        return []
    if len(summaries) < 3:
        return summaries
    try:
        embs = embedder.encode(summaries, convert_to_numpy=True, normalize_embeddings=True)
        cl = DBSCAN(metric="cosine", eps=eps, min_samples=2).fit(embs)
        labels = cl.labels_
        filtered = [s for s,l in zip(summaries, labels) if l != -1]
        return filtered if filtered else summaries
    except Exception:
        return summaries

# -------------------------
# Build FAISS memory safely
# -------------------------
def build_memory(user_summaries):
    all_summaries = []
    meta = []
    for user, lst in user_summaries.items():
        cleaned = remove_outliers(lst)
        for s in cleaned:
            s2 = clean_message(s)
            if s2:
                all_summaries.append(s2)
                meta.append({"user": user, "rep": 1.0})
    if not all_summaries:
        raise ValueError("No valid summaries to index.")
    embs = embedder.encode(all_summaries, convert_to_numpy=True)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index, all_summaries, meta

# -------------------------
# Cross-user retrieval
# -------------------------
def cross_user_summary(index, all_summaries, meta, query, k=5):
    q = clean_message(query)
    if not q:
        return "Invalid query", []
    q_emb = embedder.encode([q], convert_to_numpy=True)
    candidate_k = min(max(10, k*3), index.ntotal)
    D, I = index.search(q_emb, candidate_k)
    scored = []
    for dist, idx in zip(D[0], I[0]):
        sim = 1.0/(1.0 + float(dist))
        w = float(meta[idx].get("rep", 1.0))
        scored.append((sim * w, idx))
    scored = sorted(scored, reverse=True)[:k]
    retrieved = [all_summaries[i] for _, i in scored]
    details = [{"user": meta[i]["user"], "summary": all_summaries[i], "score": float(sc)} for sc,i in scored]
    joined = " ".join(retrieved)
    final = summarize_text_adaptive(joined) if joined else "No retrieved content"
    return final, details

# -------------------------
# Evaluation utilities
# -------------------------
def compute_rouge(dialog_samples, dialog_dataset):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    r1s, r2s, rls = [], [], []
    for key, turns in dialog_samples.items():
        idx = int(key.split("_")[1])
        reference = dialog_dataset["train"][idx]["summary"]
        system = summarize_text_adaptive(" ".join(turns)) or ""
        sc = scorer.score(reference, system)
        r1s.append(sc["rouge1"].fmeasure)
        r2s.append(sc["rouge2"].fmeasure)
        rls.append(sc["rougeL"].fmeasure)
    if not r1s:
        return None
    return {"ROUGE-1": float(np.mean(r1s)), "ROUGE-2": float(np.mean(r2s)), "ROUGE-L": float(np.mean(rls))}

def compute_bertscore(references, predictions, model_type=BERTSCORE_MODEL):
    if not BERTSCORE_AVAILABLE:
        return None
    try:
        P, R, F1 = bert_score(
            predictions, 
            references, 
            model_type=model_type,
            rescale_with_baseline=True
        )
        return {
            "Precision": float(P.mean()),
            "Recall": float(R.mean()),
            "F1": float(F1.mean())
        }
    except Exception:
        return None


def compute_retrieval_accuracy(index, all_summaries, queries):
    if index.ntotal == 0:
        return 0.0
    correct = 0
    total = len(queries)
    for q in queries:
        q_emb = embedder.encode([q], convert_to_numpy=True)
        D, I = index.search(q_emb, min(5, index.ntotal))
        # simplistic hit: any retrieved summary contains first query token
        qtok = q.split()[0].lower() if q.split() else ""
        if any(qtok in all_summaries[idx].lower() for idx in I[0]):
            correct += 1
    return round(correct / max(1, total) * 100, 2)

def compression_stats(original_texts, summaries):
    orig_lens = [len(t.split()) for t in original_texts]
    sum_lens = [len(s.split()) for s in summaries]
    if not orig_lens:
        return None
    ratios = [s/o if o>0 else 0 for s,o in zip(sum_lens, orig_lens)]
    return {"avg_original": float(np.mean(orig_lens)), "avg_summary": float(np.mean(sum_lens)), "avg_compression_ratio": float(np.mean(ratios))}

def toxicity_stats(raw_msgs, cleaned_msgs):
    before = len(raw_msgs)
    after = len(cleaned_msgs)
    removed = before - after
    pct = round((removed / before * 100), 2) if before>0 else 0.0
    return {"total": before, "kept": after, "removed": removed, "removed_pct": pct}

def parsing_stats(user_chats):
    stats = {u: len(ms) for u,ms in user_chats.items()}
    total = sum(stats.values())
    return stats, total

# -------------------------
# Streamlit UI
# -------------------------
st.title(" Jarvis-M")

st.sidebar.header("Data source & settings")
mode = st.sidebar.radio("Mode", ["DialogSum (eval)", "WhatsApp (real data)"])
debug = st.sidebar.checkbox("Debug prints", value=False)

# DialogSum loading
if mode == "DialogSum (eval)":
    n_samples = st.sidebar.slider("DialogSum samples", 1, 10, 3)
    if st.sidebar.button("Load DialogSum & Build"):
        try:
            samples, dataset = load_dialogsum_samples(n_samples)
            st.session_state["dialogsum_samples"] = samples
            st.session_state["dialogsum_dataset"] = dataset
            st.session_state["user_chats"] = samples
            st.session_state["user_summaries"] = summarize_sessions(samples, debug=debug)
            st.session_state["index"], st.session_state["all_summaries"], st.session_state["meta"] = build_memory(st.session_state["user_summaries"])
            st.success("DialogSum loaded & memory built.")
        except Exception as e:
            st.error(f"Failed: {e}")

# WhatsApp loading
else:
    path = st.sidebar.text_input("WhatsApp path", "data/whatsapp")
    if st.sidebar.button("Load WhatsApp & Build"):
        try:
            chats = load_whatsapp(path)
            st.session_state["user_chats"] = chats
            st.session_state["user_summaries"] = summarize_sessions(chats, debug=debug)
            st.session_state["index"], st.session_state["all_summaries"], st.session_state["meta"] = build_memory(st.session_state["user_summaries"])
            st.success("WhatsApp loaded & memory built.")
        except Exception as e:
            st.error(f"Failed: {e}")

# per-user cross-session summary
st.subheader("Per-user cross-session summary")
if "user_summaries" in st.session_state:
    users = list(st.session_state["user_summaries"].keys())
    sel = st.selectbox("Select user", ["--select--"] + users)
    if st.button("Show user summary"):
        if sel == "--select--":
            st.warning("Choose a user.")
        else:
            joined = " ".join(st.session_state["user_summaries"].get(sel, []))
            out = summarize_text_adaptive(joined) if joined else "No content"
            st.success(out)
else:
    st.info("Load data first.")

# cross-user meta-summary
st.subheader("Cross-user meta-summary")
query = st.text_input("Query", "project progress")
topk = st.slider("Top-k", 1, 10, 5)
if st.button("Run cross-user meta-summary"):
    if "index" not in st.session_state:
        st.warning("Build memory first.")
    else:
        meta_sum, details = cross_user_summary(st.session_state["index"], st.session_state["all_summaries"], st.session_state["meta"], query, topk)
        st.markdown("### Retrieved contexts")
        for d in details:
            st.write(f"**{d['user']}** (score {d.get('score',0):.3f})")
            st.write("> " + d["summary"])
            st.markdown("---")
        st.markdown("### Meta-summary")
        st.success(meta_sum)

# evaluation section
st.markdown("---")
st.subheader("📊 Evaluation & Metrics")

if st.button("Compute evaluation metrics"):
    if "user_chats" not in st.session_state:
        st.warning("Load data first.")
    else:
        # DialogSum metrics
        if mode == "DialogSum (eval)":
            samples = st.session_state.get("dialogsum_samples")
            dataset = st.session_state.get("dialogsum_dataset")
            if samples and dataset:
                st.write("### ROUGE (DialogSum)")
                rouge_res = compute_rouge(samples, dataset)
                st.write(rouge_res if rouge_res else "No ROUGE results.")
                # BERTScore: compute using gold summaries and system summaries
                if st.checkbox("Enable BERTScore (slow)", value=False) and BERTSCORE_AVAILABLE:
                    st.write("### BERTScore (using lightweight DeBERTa-base)")
                    refs = []
                    preds = []
                    for key, turns in samples.items():
                        idx = int(key.split("_")[1])
                        refs.append(dataset["train"][idx]["summary"])
                        preds.append(summarize_text_adaptive(" ".join(turns)) or "")
                    bert_res = compute_bertscore(refs, preds, model_type=BERTSCORE_MODEL)
                    st.write(bert_res if bert_res else "BERTScore failed.")
                else:
                    st.info("BERTScore disabled (recommended for demo). Check box to run.")

        # WhatsApp metrics / general metrics
        # parsing stats
        chats = st.session_state["user_chats"]
        stats, total = parsing_stats(chats)
        st.write("### Parsing stats")
        st.write(f"Total parsed messages: {total}")
        st.write(stats)
        # toxicity
        raw_msgs = []
        cleaned = []
        for u, msgs in chats.items():
            for m in msgs:
                raw_msgs.append(m)
                if clean_message(m):
                    cleaned.append(m)
        tox = toxicity_stats(raw_msgs, cleaned)
        st.write("### Toxicity / Filtering stats")
        st.write(tox)
        # retrieval accuracy (example queries)
        if "index" in st.session_state:
            queries = st.text_input("Example queries (comma separated)", value="project,meeting,plan").split(",")
            queries = [q.strip() for q in queries if q.strip()]
            acc = compute_retrieval_accuracy(st.session_state["index"], st.session_state["all_summaries"], queries)
            st.write("### FAISS retrieval accuracy (hit%)")
            st.write(f"{acc} %")
        # compression ratio (per-user)
        all_originals = []
        all_summaries = []
        for u, msgs in chats.items():
            for m in msgs:
                all_originals.append(m)
        for u, sums in st.session_state.get("user_summaries", {}).items():
            for s in sums:
                all_summaries.append(s)
        comp = compression_stats(all_originals, all_summaries) if all_originals and all_summaries else None
        st.write("### Compression stats")
        st.write(comp if comp else "Not enough data to compute compression stats.")

st.markdown("---")
st.write("Notes: First run downloads models. BERTScore uses a transformer model; it may be slow on CPU.")
