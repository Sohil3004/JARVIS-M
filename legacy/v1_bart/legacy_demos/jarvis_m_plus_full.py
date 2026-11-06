
# ============================================================
# JARVIS-M++ FINAL: Cross-Session + Cross-User Summarizer
# Robust loaders, defenses, FAISS memory, Streamlit GUI
# ============================================================

import os
import re
import json
import math
import faiss
import numpy as np
import streamlit as st
from pathlib import Path
from collections import defaultdict

# transformers and sentence-transformers
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.cluster import DBSCAN

# -------------------------
# Configuration / Globals
# -------------------------
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
# small classifier attempt (optional). If it fails to load, fallback to regex.
TOXIC_CLASS_MODEL = "martin-ha/toxic-comment-model"  # small; replace if you prefer

# initialize basic models (some may download if first run)
st.set_page_config(page_title="Jarvis-M++ Final", layout="wide")

# We delay heavy initializations to when needed to keep Streamlit responsive
@st.cache_resource(show_spinner=False)
def init_models():
    models = {}
    # summarizer pipeline
    try:
        models["summarizer"] = pipeline("summarization", model=SUMMARIZER_MODEL)
    except Exception as e:
        st.error(f"Failed to load summarizer model: {e}")
        raise

    # embedder
    try:
        models["embedder"] = SentenceTransformer(EMBEDDER_MODEL)
    except Exception as e:
        st.error(f"Failed to load embedder model: {e}")
        raise

    # toxicity classifier (try small HF model; if fails, None -> fallback to regex)
    try:
        models["tox_clf"] = pipeline("text-classification", model=TOXIC_CLASS_MODEL, top_k=None)
    except Exception:
        models["tox_clf"] = None  # fallback to rule-based sanitizer

    return models

models = init_models()
summarizer = models["summarizer"]
embedder = models["embedder"]
tox_clf = models["tox_clf"]

# -------------------------
# Utilities
# -------------------------
def safe_print(msg):
    # helper to print to console for debugging when running locally
    try:
        print(msg)
    except Exception:
        pass

# -------------------------
# Toxicity / sanitizer
# -------------------------
# fallback blacklist words (lowercase)
BASIC_BLACKLIST = [
    "kill", "terror", "bomb", "attack", "drugs", "sex", "porn", "racist", "idiot", "nigga", "nigger"
]
# keep blacklist trimmed for demo; tune as needed

def is_toxic_ml(text, threshold=0.6):
    """Use small HF classifier if available; accumulate toxic-like scores."""
    if tox_clf is None:
        return False
    try:
        preds = tox_clf(text[:512])
        # preds is list of dicts (if top_k=None) or list of label dicts
        # Different models have different labels; check for commonly bad labels
        toxic_score = 0.0
        for p in preds:
            label = p.get("label", "").lower()
            score = float(p.get("score", 0.0))
            if any(x in label for x in ["toxic", "offensive", "severe_toxic", "insult", "threat", "hate"]):
                toxic_score += score
        return toxic_score >= threshold
    except Exception as e:
        safe_print("Toxic ML check failed: " + str(e))
        return False

def is_toxic_basic(text, threshold=0.0):
    """Simple rule-based checker (case-insensitive substrings)."""
    t = text.lower()
    for bad in BASIC_BLACKLIST:
        if bad in t:
            return True
    return False

def is_toxic(text):
    """Unified toxicity predicate: try ML classifier first, fallback to basic rules.
       Intentionally lenient to avoid removing benign messages in demo."""
    # fast checks
    if not text or len(text.strip()) < 3:
        return False
    # try ML classifier
    if tox_clf is not None:
        try:
            return is_toxic_ml(text, threshold=0.7)
        except Exception:
            return is_toxic_basic(text)
    else:
        return is_toxic_basic(text)

def clean_message(msg):
    if msg is None:
        return None

    # remove non-printable / weird unicode
    msg = msg.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # remove empty or placeholder messages
    if len(msg.strip()) < 3:
        return None

    # WhatsApp continuation lines or timestamps with no actual message
    ts_patterns = [
        r"^\d+\/\d+\/\d+.*-.*$",
        r"^\[?\d{1,2}\/\d{1,2}\/\d{2,4},"
    ]
    for pat in ts_patterns:
        if re.match(pat, msg) and ":" not in msg[15:]:
            return None

    return msg.strip()

# -------------------------
# Loaders
# -------------------------
whatsapp_line_re = re.compile(r"^\[?\d{1,2}\/\d{1,2}\/\d{2,4},?\s*\d{1,2}:\d{2}(?:[:APMapm\s]+)*\]?\s*-\s*(.*?):\s*(.*)$")
# common export formats vary; keep flexible

def parse_whatsapp_file(path):
    """Parse a WhatsApp-style .txt export into per-user messages (list of strings)."""
    user_messages = defaultdict(list)
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = whatsapp_line_re.match(line)
            if m:
                user = m.group(1).strip()
                msg = m.group(2).strip()
                msg = clean_message(msg)
                if msg:
                    user_messages[user].append(msg)

            else:
                # continuation lines (append to last message) - optional handling
                # If line doesn't match pattern, append to last user's last message if exists
                if user_messages:
                    last_user = list(user_messages.keys())[-1]
                    if user_messages[last_user]:
                        user_messages[last_user][-1] += " " + line
    return dict(user_messages)

def load_whatsapp(path_in):
    """Load a folder of .txt WhatsApp exports or a single .txt file."""
    p = Path(path_in)
    if p.is_file():
        return parse_whatsapp_file(p)
    elif p.is_dir():
        aggregated = {}
        for f in p.iterdir():
            if f.suffix.lower() == ".txt":
                user_base = f.stem
                parsed = parse_whatsapp_file(f)
                # If parsed contains multiple users (group export), include them under file stem prefix
                if len(parsed) == 1:
                    k = list(parsed.keys())[0]
                    aggregated[f"{user_base}__{k}"] = parsed[k]
                else:
                    for k, v in parsed.items():
                        aggregated[f"{user_base}__{k}"] = v
        return aggregated
    else:
        raise ValueError("Invalid WhatsApp path: " + str(path_in))

def load_dialogsum_samples(n=3):
    """Load a few sample dialogues from DialogSum, split into turns."""
    dataset = load_dataset("knkarthick/dialogsum")
    samples = {}
    total = len(dataset["train"])
    n = min(n, total)
    for i in range(n):
        d = dataset["train"][i]["dialogue"]
        # some dialogues are one string with '\n' separators, some not
        # split by newline then by speaker-colon if present
        turns = []
        for line in d.split("\n"):
            line = line.strip()
            if not line:
                continue
            # if "Person: utterance" present split, else use whole line
            if ":" in line:
                parts = line.split(":", 1)
                turns.append(parts[1].strip())
            else:
                turns.append(line)
        samples[f"dialogsum_{i}"] = turns
    return samples

# -------------------------
# Summarization (safe chunking)
# -------------------------
MAX_INPUT_TOKENS = 1000  # heuristic; BART handles ~1024 tokens; huggingface pipeline handles truncation but we chunk for safety

def chunk_text(text, max_chars=1500):
    """Naive chunker by characters that avoids cutting sentences badly."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # try to break at last period or newline before end
        slice_ = text[start:end]
        sep = max(slice_.rfind("."), slice_.rfind("\n"), slice_.rfind("?"), slice_.rfind("!"))
        if sep <= 0:
            sep = end
        else:
            sep = start + sep + 1
        chunks.append(text[start:sep].strip())
        start = sep
    return [c for c in chunks if c]

def summarize_text_safe(text, max_len=120, min_len=25):
    """Summarize by chunking and combining summaries if needed."""
    chunks = chunk_text(text, max_chars=1500)
    if not chunks:
        return ""
    summaries = []
    for ch in chunks:
        try:
            out = summarizer(ch, max_length=max_len, min_length=max(10, min_len), do_sample=False)
            if isinstance(out, list):
                summaries.append(out[0]['summary_text'])
            else:
                # older pipeline formats
                summaries.append(out['summary_text'])
        except Exception as e:
            safe_print("Summarizer chunk error: " + str(e))
            continue
    # if multiple chunk summaries, combine and summarize once more if long
    combined = " ".join(summaries)
    if len(combined.split()) > max_len * 2:
        try:
            out = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
            return out[0]['summary_text']
        except Exception:
            return combined
    return combined

# -------------------------
# Summarize sessions for each user (with debug)
# -------------------------
def summarize_sessions(user_chats, debug=False):
    user_summaries = {}
    for user, msgs in user_chats.items():
        summaries = []
        for m in msgs:
            m = clean_message(m)
            if not m:
                if debug: safe_print("[SKIP-cleaned] empty/invalid")
                continue

            if not m or len(m.strip()) < 3:
                if debug: safe_print(f"[SKIP-short] {user}: {m[:40]}")
                continue
            # toxicity check (lenient)
            if is_toxic(m):
                if debug: safe_print(f"[SKIP-toxic] {user}: {m[:40]}")
                continue
            # summarize
            try:
                s = summarize_text_safe(m)
                if s and len(s.strip()) > 3:
                    summaries.append(s)
                    if debug: safe_print(f"[OK] Summarized for {user}: {s[:80]}")
                else:
                    if debug: safe_print(f"[SKIP-emptySummary] {user}")
            except Exception as e:
                if debug: safe_print(f"[ERROR-sum] {user}: {e}")
        user_summaries[user] = summaries
    return user_summaries

# -------------------------
# Outlier removal (DBSCAN)
# -------------------------
def remove_outliers(summaries, eps=0.35, min_samples=2):
    if not summaries:
        return []
    if len(summaries) < 3:
        return summaries
    try:
        embs = embedder.encode(summaries, convert_to_numpy=True, normalize_embeddings=True)
        cl = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(embs)
        labels = cl.labels_
        filtered = [s for s, l in zip(summaries, labels) if l != -1]
        return filtered if filtered else summaries
    except Exception as e:
        safe_print("Outlier detection failed: " + str(e))
        return summaries

# -------------------------
# Build FAISS memory safely
# -------------------------
def build_memory(user_summaries):
    all_summaries = []
    meta = []
    # reputation default
    reputation = defaultdict(lambda: 1.0)

    for user, lst in user_summaries.items():
        if not lst:
            continue
        cleaned = remove_outliers(lst)
        for s in cleaned:
            if s and s.strip():
                all_summaries.append(s)
                meta.append({"user": user, "rep": float(reputation[user])})
                all_summaries = [clean_message(s) for s in all_summaries if clean_message(s)]


    if len(all_summaries) == 0:
        raise ValueError("No summaries to index. Check if messages were parsed correctly or filters too strict.")

    embs = embedder.encode(all_summaries, convert_to_numpy=True)
    # ensure 2D
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index, all_summaries, meta

# -------------------------
# Cross-user retrieval with re-ranking
# -------------------------
def cross_user_summary(index, all_summaries, meta, query, k=5):
    if index.ntotal == 0:
        return "No memory built.", []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    # request larger candidate set and re-rank with rep*sim
    candidate_k = min(max(10, k*3), index.ntotal)
    D, I = index.search(q_emb, candidate_k)
    scored = []
    for dist, idx in zip(D[0], I[0]):
        sim = 1.0 / (1.0 + float(dist))
        weight = float(meta[idx].get("rep", 1.0))
        score = sim * weight
        scored.append((score, idx))
    scored = sorted(scored, reverse=True)[:k]
    retrieved = [all_summaries[i] for _, i in scored]
    details = [{"user": meta[i]["user"], "summary": all_summaries[i], "score": float(sc)} for sc, i in scored]
    # produce meta summary
    combined = " ".join(retrieved)
    meta_summary = summarize_text_safe(combined, max_len=140, min_len=30)
    return meta_summary, details

# -------------------------
# Streamlit GUI
# -------------------------
st.title("🧠 Jarvis-M++ Final — Demo (DialogSum + WhatsApp)")

st.sidebar.header("Load Data / Build Memory")
mode = st.sidebar.radio("Data Mode", ["DialogSum samples", "WhatsApp export (folder/file)"])
debug_mode = st.sidebar.checkbox("Debug logging (console)", value=False)

if mode == "DialogSum samples":
    n_samples = st.sidebar.slider("Number of DialogSum samples to load", 1, 10, 3)
    if st.sidebar.button("Load DialogSum & Build Memory"):
        with st.spinner("Loading DialogSum samples and building memory..."):
            user_chats = load_dialogsum_samples(n=n_samples)
            st.session_state["user_chats"] = user_chats
            st.session_state["user_summaries"] = summarize_sessions(user_chats, debug=debug_mode)
            try:
                st.session_state["index"], st.session_state["all_summaries"], st.session_state["meta"] = build_memory(st.session_state["user_summaries"])
                st.success("Memory built successfully (DialogSum).")
            except Exception as e:
                st.error(f"Failed to build memory: {e}")
                safe_print(str(e))

elif mode == "WhatsApp export (folder/file)":
    path_input = st.sidebar.text_input("WhatsApp folder or file path", value="data/whatsapp")
    if st.sidebar.button("Load WhatsApp & Build Memory"):
        try:
            with st.spinner("Parsing WhatsApp files and building memory..."):
                user_chats = load_whatsapp(path_input)
                st.session_state["user_chats"] = user_chats
                st.session_state["user_summaries"] = summarize_sessions(user_chats, debug=debug_mode)
                st.session_state["index"], st.session_state["all_summaries"], st.session_state["meta"] = build_memory(st.session_state["user_summaries"])
                st.success("Memory built successfully (WhatsApp).")
        except Exception as e:
            st.error(f"Failed to load/build WhatsApp memory: {e}")
            safe_print(str(e))

# Show available users
st.subheader("Per-user Cross-Session Summary")
if "user_summaries" in st.session_state:
    users = list(st.session_state["user_summaries"].keys())
    sel_user = st.selectbox("Choose user", ["--select--"] + users)
    if st.button("Show user cross-session summary"):
        if sel_user == "--select--":
            st.warning("Select a user first.")
        else:
            s = st.session_state["user_summaries"].get(sel_user, [])
            if not s:
                st.info("No summaries available for this user (maybe filtered).")
            else:
                cs = " ".join(s)
                res = summarize_text_safe(cs, max_len=120, min_len=25)
                st.write("**Cross-session summary:**")
                st.success(res)
else:
    st.info("Load DialogSum or WhatsApp and build memory first (left sidebar).")

# Cross-user query UI
st.subheader("Cross-user Meta-Summary (Query)")
query_input = st.text_input("Query (topic to search across memory):", "project progress")
topk = st.slider("Top-k retrieved contexts", 1, 10, 5)
if st.button("Run cross-user meta-summary"):
    if "index" not in st.session_state:
        st.error("No memory built. Load data first.")
    else:
        meta_sum, details = cross_user_summary(st.session_state["index"], st.session_state["all_summaries"], st.session_state["meta"], query_input, k=topk)
        st.markdown("### Retrieved contexts (top-k):")
        if not details:
            st.write("No contexts retrieved.")
        else:
            for d in details:
                st.write(f"**{d['user']}** (score {d['score']:.3f})")
                st.write("> " + d['summary'])
                st.markdown("---")
        st.markdown("### Meta-summary:")
        st.success(meta_sum)

# Optional: show raw parsed chats (for debugging)
if st.sidebar.checkbox("Show parsed chats and summaries (debug)"):
    if "user_chats" in st.session_state:
        st.sidebar.write("Parsed chats (first 5 per user):")
        for u, msgs in st.session_state["user_chats"].items():
            st.sidebar.write(u + ":")
            for m in msgs[:5]:
                st.sidebar.write(" - " + (m[:140] + ("..." if len(m) > 140 else "")))
    if "user_summaries" in st.session_state:
        st.sidebar.write("User summaries (first 3 per user):")
        for u, sums in st.session_state["user_summaries"].items():
            st.sidebar.write(u + ":")
            for s in sums[:3]:
                st.sidebar.write(" • " + (s[:160] + ("..." if len(s) > 160 else "")))

# Footer
st.markdown("---")

