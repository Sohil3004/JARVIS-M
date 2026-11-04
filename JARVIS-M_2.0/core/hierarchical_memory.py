"""
JARVIS-M v2: Hierarchical Collaborative State Graph (HCSG)
==========================================================
Upgrades the flat dual-memory into a structured, hierarchical graph
with separated intra-user personal trajectories and cross-user collaborative states.

Core Components:
1. Semantic Triple Extractor: (Subject, Predicate, Object, Status, SessionID)
2. Partitioned Memory Indices:
   - M_intra: Personal user goals, queries, commitments, styles
   - M_cross: Shared project facts, task assignments, deadlines, decisions
3. Dual FAISS Vector Storage with Temporal Recency Decay
"""

import os
import sys
import re
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

# Ensure Windows terminal safe encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ConversationalTriple:
    """Represents an atomic structured state unit extracted from dialogue."""
    def __init__(
        self,
        user: str,
        predicate: str,
        obj: str,
        session_id: str = "session_0",
        timestamp: float = 0.0,
        raw_text: str = "",
        is_cross_user: bool = False,
    ):
        self.user = user
        self.predicate = predicate
        self.obj = obj
        self.session_id = session_id
        self.timestamp = timestamp or time.time()
        self.raw_text = raw_text or f"{user} {predicate} {obj}"
        self.is_cross_user = is_cross_user

    def to_string(self) -> str:
        return f"[{self.user}] -> {self.predicate}: {self.obj}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user": self.user,
            "predicate": self.predicate,
            "object": self.obj,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "raw_text": self.raw_text,
            "is_cross_user": self.is_cross_user,
        }


class HierarchicalMemoryGraph:
    """
    Hierarchical memory manager maintaining dual vector-indexed subgraphs.
    """
    def __init__(
        self,
        embedder_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "./cache",
        recency_decay_half_life_hours: float = 24.0,
    ):
        self.cache_dir = cache_dir
        self.recency_decay = recency_decay_half_life_hours * 3600

        print(f"[*] Initializing HierarchicalMemoryGraph with embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name, cache_folder=self.cache_dir)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # Stores: Separate intra-user list vs cross-user list
        self.intra_triples: Dict[str, List[ConversationalTriple]] = {}  # user -> list
        self.cross_triples: List[ConversationalTriple] = []

        # FAISS Indices
        self.cross_index = faiss.IndexFlatIP(self.dim)
        self.cross_vectors: List[np.ndarray] = []

        self.intra_indices: Dict[str, faiss.IndexFlatIP] = {}
        self.intra_vectors: Dict[str, List[np.ndarray]] = {}

        print("[+] HierarchicalMemoryGraph initialized.")

    def extract_triples_rule_based(self, text: str, user: str, session_id: str = "s0") -> List[ConversationalTriple]:
        """
        Fast, lightweight semantic triple extractor that identifies commitments,
        decisions, deadlines, and questions from conversational turns.
        """
        triples = []
        lines = [line.strip() for line in re.split(r'[.\n?!]', text) if len(line.strip()) > 8]

        intent_patterns = [
            (r"\b(will|going to|plan to|aim to|need to|must|should|have to)\s+([^,.]+)", "committed_to", True),
            (r"\b(deadline|by Friday|by Monday|by tomorrow|before|due on)\s+([^,.]+)", "has_deadline", True),
            (r"\b(decided|agreed|concluded|confirmed)\s+([^,.]+)", "decided", True),
            (r"\b(working on|focus on|analyzing|writing|testing)\s+([^,.]+)", "currently_executing", False),
            (r"\b(problem|error|bug|issue|delayed|taking longer)\s+([^,.]+)", "flagged_blocker", True),
            (r"\b(prefer|like|dislike|want|ask)\s+([^,.]+)", "preference", False),
        ]

        now = time.time()
        for line in lines:
            extracted = False
            for pattern, predicate, is_cross in intent_patterns:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    obj = m.group(0).strip()
                    triple = ConversationalTriple(
                        user=user,
                        predicate=predicate,
                        obj=obj,
                        session_id=session_id,
                        timestamp=now,
                        raw_text=line,
                        is_cross_user=is_cross,
                    )
                    triples.append(triple)
                    extracted = True
                    break

            if not extracted and len(line) > 15:
                # Default generic observational fact
                triples.append(
                    ConversationalTriple(
                        user=user,
                        predicate="stated",
                        obj=line[:120],
                        session_id=session_id,
                        timestamp=now,
                        raw_text=line,
                        is_cross_user=False,
                    )
                )

        return triples

    def add_interaction(self, user: str, text: str, session_id: str = "session_0"):
        """Extract and index triples for both personal and collaborative memory."""
        triples = self.extract_triples_rule_based(text, user=user, session_id=session_id)
        if not triples:
            return

        texts_to_embed = [t.raw_text for t in triples]
        embeddings = self.embedder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

        for triple, emb in zip(triples, embeddings):
            emb_2d = emb.reshape(1, -1)

            # 1. Intra-user memory (User's personal trajectory)
            if user not in self.intra_triples:
                self.intra_triples[user] = []
                self.intra_indices[user] = faiss.IndexFlatIP(self.dim)
                self.intra_vectors[user] = []

            self.intra_triples[user].append(triple)
            self.intra_vectors[user].append(emb)
            self.intra_indices[user].add(emb_2d)

            # 2. Cross-user collaborative memory (Shared team state)
            if triple.is_cross_user:
                self.cross_triples.append(triple)
                self.cross_vectors.append(emb)
                self.cross_index.add(emb_2d)

    def retrieve(
        self,
        query: str,
        current_user: str,
        top_k_intra: int = 3,
        top_k_cross: int = 3,
        apply_recency_weighting: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Dual-channel retrieval:
        surfaces relevant personal history + team collaborative state with temporal weighting.
        """
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        results = {"intra_user": [], "cross_user": []}
        now = time.time()

        # Channel 1: Intra-User Personal Memory
        if current_user in self.intra_indices and self.intra_indices[current_user].ntotal > 0:
            k = min(top_k_intra * 2, self.intra_indices[current_user].ntotal)
            D, I = self.intra_indices[current_user].search(q_emb, k)
            candidates = []
            for dist, idx in zip(D[0], I[0]):
                triple = self.intra_triples[current_user][idx]
                time_delta = max(0.0, now - triple.timestamp)
                recency_factor = np.exp(-time_delta / self.recency_decay) if apply_recency_weighting else 1.0
                final_score = float(dist) * float(recency_factor)
                candidates.append((final_score, triple))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for sc, tr in candidates[:top_k_intra]:
                results["intra_user"].append({"score": sc, "triple": tr.to_string(), "raw": tr.raw_text})

        # Channel 2: Cross-User Collaborative Memory (exclude current user to avoid redundancy)
        if self.cross_index.ntotal > 0:
            k = min(top_k_cross * 3, self.cross_index.ntotal)
            D, I = self.cross_index.search(q_emb, k)
            candidates = []
            for dist, idx in zip(D[0], I[0]):
                triple = self.cross_triples[idx]
                if triple.user == current_user:
                    continue  # Cross-user specifically surfaces what others committed/decided
                time_delta = max(0.0, now - triple.timestamp)
                recency_factor = np.exp(-time_delta / self.recency_decay) if apply_recency_weighting else 1.0
                final_score = float(dist) * float(recency_factor)
                candidates.append((final_score, triple))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for sc, tr in candidates[:top_k_cross]:
                results["cross_user"].append({"score": sc, "triple": tr.to_string(), "raw": tr.raw_text})

        return results

    def get_compact_context_string(self, query: str, current_user: str, max_tokens: int = 150) -> str:
        """Serializes top retrieved memory into a dense, token-minimal context prefix."""
        retrieved = self.retrieve(query, current_user, top_k_intra=2, top_k_cross=3)
        parts = []

        if retrieved["cross_user"]:
            parts.append("[Collaborative State & Commitments]:")
            for item in retrieved["cross_user"]:
                parts.append(f" - {item['triple']}")

        if retrieved["intra_user"]:
            parts.append(f"[{current_user}'s Prior History]:")
            for item in retrieved["intra_user"]:
                parts.append(f" - {item['triple']}")

        return "\n".join(parts)


# ============================================================
# Quick Local Verification
# ============================================================
if __name__ == "__main__":
    memory = HierarchicalMemoryGraph()

    # Simulate multi-session conversation
    print("\n[*] Populating Multi-User Graph...")
    session_1_alice = "Alice: I am working on the database indexing. We must deliver the finalized schema by Friday."
    session_1_bob = "Bob: I decided to handle the API endpoints. There is a blocker with authentication tokens taking longer."
    session_2_alice = "Alice: I finished the database indexing and now need to review the API latency."

    memory.add_interaction(user="Alice", text=session_1_alice, session_id="s1")
    memory.add_interaction(user="Bob", text=session_1_bob, session_id="s1")
    memory.add_interaction(user="Alice", text=session_2_alice, session_id="s2")

    print("\n[*] Querying Cross-User memory for Alice on topic 'blocker & API'...")
    results = memory.retrieve(query="API blocker deadline", current_user="Alice", top_k_cross=2, top_k_intra=2)

    print("\n[+] Retrieved Cross-User State (from Bob):")
    for r in results["cross_user"]:
        print(f"  Score: {r['score']:.3f} | {r['triple']}")

    print("\n[+] Retrieved Intra-User Memory (Alice's past):")
    for r in results["intra_user"]:
        print(f"  Score: {r['score']:.3f} | {r['triple']}")

    print("\n[+] Compact Assembled Context String:")
    compact_str = memory.get_compact_context_string("API blocker deadline", current_user="Alice")
    print(compact_str)
