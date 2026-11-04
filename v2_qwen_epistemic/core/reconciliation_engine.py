"""
JARVIS-Gateway: Temporal State Reconciliation Engine
====================================================
Implements Belief Revision and Epistemic State Overwrite Tracking for Multi-Party Dialogues.

Theoretical Basis:
- Truth Maintenance Systems (TMS) & AGM Belief Revision Theory
- Models dialogue not as static text chunks, but as dynamic state transitions.
- Every state node tracks: Entity Key, Value, Owner, Temporal Timestamp, and Status.
- Status values:
    ACTIVE: Current ground truth
    SUPERSEDED: Overridden by a later decision
    RESOLVED: Blocker that has been cleared
    BLOCKED: Active issue impeding progress
"""

import sys
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class EpistemicStateNode:
    """Represents a single verified state claim in the dialogue."""
    def __init__(
        self,
        node_id: str,
        entity_key: str,
        value: str,
        owner: str,
        session_id: str,
        timestamp: float,
        raw_text: str,
        status: str = "ACTIVE",
        superseded_by: Optional[str] = None,
        blocker_ref: Optional[str] = None,
    ):
        self.node_id = node_id
        self.entity_key = entity_key.lower().strip()
        self.value = value.strip()
        self.owner = owner.strip()
        self.session_id = session_id
        self.timestamp = timestamp
        self.raw_text = raw_text.strip()
        self.status = status  # ACTIVE | SUPERSEDED | RESOLVED | BLOCKED
        self.superseded_by = superseded_by
        self.blocker_ref = blocker_ref

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "entity_key": self.entity_key,
            "value": self.value,
            "owner": self.owner,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "raw_text": self.raw_text,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "blocker_ref": self.blocker_ref,
        }

    def to_summary_string(self) -> str:
        """Returns an ultra-compact, token-dense representation of the verified state."""
        if self.status == "ACTIVE":
            return f"{self.entity_key.upper()}: {self.value} ({self.owner})"
        elif self.status == "SUPERSEDED":
            return f"SUPERSEDED {self.entity_key.upper()}: {self.value}"
        elif self.status == "RESOLVED":
            return f"RESOLVED: {self.value}"
        elif self.status == "BLOCKED":
            return f"BLOCKER: {self.value}"
        return f"{self.entity_key}: {self.value}"


class TemporalStateReconciliationEngine:
    """
    Tracks, updates, and reconciles conversational states across multiple sessions,
    resolving contradictions, overrides, and task dependencies.
    """
    def __init__(self):
        self.nodes: Dict[str, EpistemicStateNode] = {}  # node_id -> node
        self.entity_timeline: Dict[str, List[str]] = {}
        self.counter = 0

    def _generate_node_id(self) -> str:
        self.counter += 1
        return f"node_{self.counter:04d}"

    def parse_state_transitions(self, speaker: str, utterance: str, session_id: str, timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Extracts candidate state changes and actions from an utterance.
        Identifies:
        1. Decisions / Overrides ("decided to", "switched to", "migrated to", "rescheduled to")
        2. Blockers & Errors ("blocked by", "error 403", "crashed", "failing")
        3. Clearances & Resolutions ("resolved", "fixed", "passed review", "increased memory", "grant")
        4. Commitments & Deadlines ("deadline is", "due by", "will deliver by")
        """
        transitions = []
        now = timestamp or time.time()

        # Pattern 1: Explicit Override / Pivot / Reschedule
        override_patterns = [
            (r"\b(?:switched|migrated|moved|changed|pivoted)\s+(?:from\s+[^,\.]+?\s+)?to\s+([A-Za-z0-9_\-\.\s]+?)(?:[,\.]|\s+to\s+use|\s+because|\s+due|\s+for|\s*$)", "technology_stack"),
            (r"\b(?:rescheduled|pushed|moved|delayed)\s+(?:to|for)\s+([A-Za-z0-9_\-\.\:\s]+?)(?:[,\.]|\s+because|\s*$)", "release_deadline"),
            (r"\b(?:officially\s+decided|agreed|confirmed|decided)\s+(?:to\s+use|on)\s+([A-Za-z0-9_\-\.\s]+?)(?:[,\.]|\s+for|\s*$)", "technology_stack"),
            (r"\b(?:new\s+target|updated\s+deadline|due\s+by|deadline\s+is|release\s+deadline\s+is)\s+([A-Za-z0-9_\-\.\:\s]+?)(?:[,\.]|\s*$)", "release_deadline"),
            (r"\b(?:pricing\s+tier|budget|cost\s+cap)\s+(?:is|set\s+to|changed\s+to)\s+([\$A-Za-z0-9_\-\.\s]+?)(?:[,\.]|\s*$)", "budget_allocation"),
        ]

        for pat, key in override_patterns:
            m = re.search(pat, utterance, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                transitions.append({
                    "entity_key": key,
                    "value": val,
                    "action": "OVERWRITE",
                    "status": "ACTIVE",
                    "owner": speaker,
                    "raw_text": utterance,
                })
                break

        # Pattern 2: Resolution / Fix of an existing issue
        resolve_patterns = [
            (r"\b(?:resolved|fixed|remedied|addressed)\s+([^,\.]+)", "issue_resolution"),
            (r"\b(?:grant|granted|added)\s+([^,\.]+?\s+(?:IAM\s+role|permissions|credentials|access)[^,\.]*)", "issue_resolution"),
            (r"\b(?:passed review|passed qa|approved|cleared)\b", "verification_pass"),
            (r"\b(?:received new credentials|increased memory|updated Info\.plist)\b", "workaround_applied"),
        ]

        for pat, key in resolve_patterns:
            m = re.search(pat, utterance, re.IGNORECASE)
            if m:
                val = m.group(0).strip()
                transitions.append({
                    "entity_key": key,
                    "value": val,
                    "action": "RESOLVE",
                    "status": "RESOLVED",
                    "owner": speaker,
                    "raw_text": utterance,
                })
                break

        # Pattern 3: Active Blocker or Failure
        blocker_patterns = [
            (r"\b(?:error\s+\d+|failing|crashed|rejected|ui freeze|out-of-memory)\b[^\.]*", "system_blocker"),
            (r"\b(?:blocked by|waiting for|taking longer)\s+([^,\.]+)", "progress_blocker"),
        ]

        for pat, key in blocker_patterns:
            m = re.search(pat, utterance, re.IGNORECASE)
            if m:
                val = m.group(0).strip()
                transitions.append({
                    "entity_key": key,
                    "value": val,
                    "action": "FLAG_BLOCKER",
                    "status": "BLOCKED",
                    "owner": speaker,
                    "raw_text": utterance,
                })
                break

        return transitions

    def process_utterance(self, speaker: str, utterance: str, session_id: str, timestamp: Optional[float] = None) -> List[EpistemicStateNode]:
        """
        Processes an utterance, updates the temporal graph, and invalidates/supersedes
        any prior contradictory or obsolete states.
        """
        now = timestamp or time.time()
        transitions = self.parse_state_transitions(speaker, utterance, session_id, now)
        created_nodes = []

        for trans in transitions:
            node_id = self._generate_node_id()
            key = trans["entity_key"]
            action = trans["action"]

            # If action is OVERWRITE, find past active states for this key and supersede them
            if action == "OVERWRITE":
                if key in self.entity_timeline:
                    for prev_id in self.entity_timeline[key]:
                        prev_node = self.nodes[prev_id]
                        if prev_node.status == "ACTIVE":
                            prev_node.status = "SUPERSEDED"
                            prev_node.superseded_by = node_id

            # If action is RESOLVE, check if any blockers can be cleared
            elif action == "RESOLVE":
                for b_key in ["system_blocker", "progress_blocker"]:
                    if b_key in self.entity_timeline:
                        for prev_id in self.entity_timeline[b_key]:
                            prev_node = self.nodes[prev_id]
                            if prev_node.status == "BLOCKED":
                                prev_node.status = "RESOLVED"
                                prev_node.blocker_ref = node_id

            new_node = EpistemicStateNode(
                node_id=node_id,
                entity_key=key,
                value=trans["value"],
                owner=trans["owner"],
                session_id=session_id,
                timestamp=now,
                raw_text=trans["raw_text"],
                status=trans["status"],
            )

            self.nodes[node_id] = new_node
            if key not in self.entity_timeline:
                self.entity_timeline[key] = []
            self.entity_timeline[key].append(node_id)
            created_nodes.append(new_node)

        return created_nodes

    def ingest_session_turns(self, turns: List[Dict[str, str]]):
        """Batch ingests a session: [{'speaker': 'Alice', 'text': '...', 'session_id': 's1'}, ...]"""
        for t in turns:
            self.process_utterance(
                speaker=t.get("speaker", "User"),
                utterance=t.get("text", ""),
                session_id=t.get("session_id", "s0"),
                timestamp=t.get("timestamp", None)
            )

    def get_reconciled_ground_truth(self) -> Dict[str, Any]:
        """
        Extracts ONLY the verified active ground-truth states,
        completely omitting superseded/stale decisions from active prompt headers,
        while retaining full node audit trail for UI visualization.
        """
        active_states = []
        resolved_blockers = []
        active_blockers = []
        superseded_history = []

        for node_id, node in self.nodes.items():
            if node.status == "ACTIVE":
                active_states.append(node)
            elif node.status == "BLOCKED":
                active_blockers.append(node)
            elif node.status == "RESOLVED":
                resolved_blockers.append(node)
            elif node.status == "SUPERSEDED":
                superseded_history.append(node)

        return {
            "active_states": active_states,
            "active_blockers": active_blockers,
            "resolved_blockers": resolved_blockers,
            "superseded_history": superseded_history,
            "superseded_count": len(superseded_history),
            "all_nodes": [n.to_dict() for n in self.nodes.values()],
        }

    def format_reconciled_prefix(self) -> str:
        """
        Generates a token-minimal, contradiction-free context header for the LLM prompt.
        """
        gt = self.get_reconciled_ground_truth()
        lines = ["[VERIFIED ACTIVE STATE]:"]

        if gt["active_states"]:
            for node in gt["active_states"]:
                lines.append(f"- {node.to_summary_string()}")
        else:
            lines.append("- None")

        if gt["active_blockers"]:
            for node in gt["active_blockers"]:
                lines.append(f"- {node.to_summary_string()}")

        if gt["resolved_blockers"]:
            for node in gt["resolved_blockers"]:
                lines.append(f"- {node.to_summary_string()}")

        return "\n".join(lines)
