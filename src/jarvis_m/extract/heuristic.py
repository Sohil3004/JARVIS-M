"""Fast, dependency-free extractor: entities by surface patterns, modality and
attribute by cue lexicons. Deliberately simple; the LLM extractor is the upgrade
and the ablation measures the gap."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ..types import Message
from .records import StateRecord

_CAP_STOP = {
    "i", "i'm", "i'll", "i've", "i'd", "the", "we", "we're", "we'll", "ok", "okay", "yes", "no", "hey", "hi",
    "thanks", "thank", "let's", "lets", "also", "so", "but", "and", "or", "if", "then", "this", "that", "it",
    "they", "you", "he", "she", "our", "my", "your", "a", "an", "good", "great", "sure", "fine", "please",
    "heads", "wait", "separately", "for", "anyone", "understood", "for my part", "rewriting", "dropping",
}
_LEAD_WORDS = {
    "scrap", "use", "using", "try", "ok", "okay", "ignore", "actually", "maybe", "moving", "move", "update", "change",
    "final", "deadline", "target", "heads", "fixed", "resolved", "seeing", "assigning", "handing", "replacing",
    "locking", "going", "pushing", "new", "separately", "sure", "yes", "no", "so", "drop", "dropping", "switch",
    "switching", "picked", "pick", "chose", "decided", "confirm", "confirmed", "approve", "approved", "also", "fine",
}
_DAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|eod|eow)"
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_TIME = re.compile(
    rf"\b(?:{_DAYS}(?:\s+\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm))?|\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)|"
    rf"{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}\s+{_MONTHS}|q[1-4]|\d{{4}}-\d{{2}}-\d{{2}})\b",
    re.I,
)
_ENT_PATTERNS = [
    re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:[ -](?:[A-Z][a-zA-Z0-9]+|\d+))+)\b"),  # Google Cloud Run, Error 403
    re.compile(r"\b[A-Z][A-Z0-9]{1,6}s?\b"),  # AWS, IAM, ACID, EC2, S3
    re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*\b"),  # PgBouncer, MongoDB, PostgreSQL
    re.compile(r"\b[A-Z][a-z]{2,}\b"),  # Grafana, Terraform (filtered below)
    re.compile(r"\b(?:User|Msg)_\d+\b"),
    re.compile(r"\b(?:v|build\s*)?\d+(?:\.\d+)+\b|\b\d+(?:\.\d+)?\s*(?:%|gb|mb|k|ms|x)\b", re.I),
    re.compile(r"\"([^\"]{3,60})\"|“([^”]{3,60})”"),
]

_NEG = re.compile(r"\b(?:not|no longer|never|won't|isn't|aren't|don't|cancel(?:led|ed)?|scrap(?:ped)?|drop(?:ped|ping)?|abandon(?:ed|ing)?)\b", re.I)
_REPLACE = re.compile(r"\b(?:instead|scrap|switch(?:ed|ing)?|mov(?:e|ed|ing) (?:to|the)|migrat\w*|replac\w*|chang(?:e|ed) to|"
                      r"push\w* (?:\w+ ){0,3}(?:out|back)|push\w* (?:it )?to|reschedul\w*|updated?|revis\w*|actually|rather than|no longer|"
                      r"pivot\w*|override|supersed\w*)\b", re.I)
_CUES: List[Tuple[str, re.Pattern]] = [
    ("done", re.compile(r"\b(?:fixed|resolved|done|finished|completed|shipped|merged|deployed|passed|signed off|"
                        r"closed|sorted|went live|is live|granted|approved)\b", re.I)),
    ("decision", re.compile(r"\b(?:decided|agreed|confirm(?:ed)?|final(?:ized|ised)?|going with|go with|it is|"
                            r"we(?:'ll| will) use|chose|chosen|picked|settled on|switch(?:ed)? (?:\w+ ){0,3}to|moving .{0,40}to|"
                            r"moved to|migrat\w+ to|official(?:ly)?|approve[ds]?|lock(?:ed)? in)\b", re.I)),
    ("blocker", re.compile(r"\b(?:blocked|blocker|failing|fails|failed|error|bug|issue|crash\w*|stuck|exhausted|"
                           r"broken|outage|regression|waiting on|delayed)\b", re.I)),
    ("commitment", re.compile(r"\b(?:i'll|i will|we'll|we will|going to|plan to|i'm owning|owning|i can take|"
                              r"assign(?:ed)? to|will (?:start|finish|send|fix|handle|deliver|update|put|grant))\b", re.I)),
    ("proposal", re.compile(r"\b(?:let's|lets|how about|maybe we|should we|propose|suggest|what if|could we)\b", re.I)),
]
_FUTURE = re.compile(r"\b(?:will|'ll|going to|plan(?:ning)? to|tomorrow|next week|later|soon|need|needs|should|must|by)\b", re.I)
_DEADLINE = re.compile(r"\b(?:deadline|due|by|release|launch|ship|target|eta|push(?:ed|ing)?|reschedul\w*|"
                       r"moved? (?:it )?to|works better)\b", re.I)


def is_time_expression(entity: str) -> bool:
    """A date / time / day entity ("april 2", "tuesday 5 pm", "q3"): a value, not a topic."""
    return bool(_TIME.fullmatch(entity.strip()))


def entities(text: str) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for pat in _ENT_PATTERNS:
        for m in pat.finditer(text):
            s = next((g for g in m.groups() if g), None) if m.groups() else m.group(0)
            s = (s or m.group(0)).strip().strip(".,;:!?")
            low = re.sub(r"^(?:the|a|an|our|my|this|that)\s+", "", s.lower())
            if len(low) < 2 or low in _CAP_STOP or low in seen:
                continue
            if pat.pattern == r"\b[A-Z][a-z]{2,}\b":
                # a lone capitalised word at sentence start is usually not an entity
                start = m.start()
                prev = text[:start].rstrip()
                if not prev or prev[-1] in ".!?\n:":
                    continue
            if " " in low and (m.start() == 0 or text[:m.start()].rstrip()[-1:] in (".", "!", "?", ":", ",")):
                head, rest = low.split(" ", 1)
                if head in _LEAD_WORDS:  # "Scrap AWS EC2" -> "aws ec2"
                    low = rest
            if low in seen:
                continue
            seen.add(low)
            out.append(low)
    for m in _TIME.finditer(text):
        low = m.group(0).lower()
        if low not in seen:
            seen.add(low)
            out.append(low)
    # drop entities fully contained in a longer one ("cloud run" inside "google cloud run")
    return tuple(e for e in out if not any(e != o and f" {e} " in f" {o} " for o in out))


_DET = re.compile(r"\b(?:the|our)\s+((?:[a-z0-9][\w\-/.]*\s+){0,3}[a-z0-9][\w\-/.]*)")
_GENERIC_NOUNS = {"rumour", "rumor", "idea", "meantime", "way", "thread", "chat", "call", "team", "moment", "record",
                  "end", "point", "same", "rest", "time", "thing", "things", "question", "link"}
_BOUNDARY = {"is", "are", "to", "from", "now", "it", "has", "have", "was", "will", "side", "instead", "after",
             "not", "be", "by", "for", "as", "with", "on", "in", "at", "of", "back", "out", "again", "this", "that",
             "and", "or", "but", "so", "done", "lands", "slips", "works", "issue", "problem", "today", "tomorrow"}


def topic_phrase(text: str, ents: Sequence[str] = ()) -> str:
    """First determiner phrase ("the backend hosting", "our release date") that is not
    itself an entity value: a cheap guess at what a statement is *about*."""
    low = text.lower()
    ent_set = set(ents)
    for m in _DET.finditer(low):
        words = []
        for w in m.group(1).split():
            w = w.strip(".,:;!?()'\"")
            if not w or w in _BOUNDARY:
                break
            words.append(w)
        phrase = " ".join(words)
        if len(phrase) >= 3 and phrase not in ent_set and not any(phrase in e for e in ent_set) \
                and phrase not in _GENERIC_NOUNS:
            return phrase
    return ""


def _clause(text: str, m: re.Match, width: int = 120) -> str:
    start = max(0, text.rfind(".", 0, m.start()) + 1)
    end = text.find(".", m.end())
    return text[start: end if end != -1 else len(text)].strip()[:width]


class HeuristicExtractor:
    name = "heuristic"

    def extract(self, messages: Sequence[Message]) -> List[StateRecord]:
        out: List[StateRecord] = []
        for m in messages:
            out.extend(self.extract_one(m))
        return out

    def extract_one(self, m: Message) -> List[StateRecord]:
        text = m.text or ""
        # the speaker's own name is who said it, not what it is about
        me = (m.speaker or "").lower()
        ents = tuple(e for e in entities(text) if e != me)
        modality, hit = "info", None
        for name, pat in _CUES:
            hit = pat.search(text)
            if hit:
                modality = name
                break
        if text.rstrip().endswith("?") and modality == "info":
            modality = "question"
        replaces = bool(_REPLACE.search(text))
        if modality in ("decision", "commitment") and hit:
            # "we have not switched to X" / "the hosting is not moving to X" / "we won't use X":
            # a negator right before the cue wins over any replacement cue in the same clause
            window = text[max(0, hit.start() - 30): hit.start() + 3]
            if re.search(r"\b(?:not|no longer|never|won't|isn't|aren't|don't|ruled out)\b", window, re.I) or \
                    re.search(r"\b(?:not happening|ruled out|no \w+ for)\b", text, re.I):
                modality, replaces = "negated", False
        elif modality in ("info", "proposal") and re.search(
                r"\b(?:not happening|ruled out|won't be using|will not be using|not going with|no longer using)\b", text, re.I):
            modality, replaces = "negated", False
        if modality == "done" and _FUTURE.search(text[: hit.start()] if hit else text):
            modality = "commitment"  # "I'll get it fixed tomorrow" is intent, not completion
        has_time = bool(_TIME.search(text))
        if has_time and _DEADLINE.search(text) and modality != "blocker":
            attribute = "deadline"
            if modality in ("info", "done"):  # "need it done by Friday", "Monday 10 AM works better"
                modality = "decision"
        elif modality == "commitment":
            attribute = "owner"
        elif modality in ("done", "blocker"):
            attribute = "status"
        elif modality in ("decision", "negated", "proposal"):
            attribute = "decision"
        else:
            attribute = "info"
        if modality == "info" and not ents:
            return []
        value = _clause(text, hit) if hit else text[:120]
        if m.meta.get("phase_name"):  # benchmarks that expose a work phase (GroupMemBench)
            subject = str(m.meta["phase_name"])
        else:
            phrase = topic_phrase(text, ents)
            subject = f"{m.channel}/{phrase}" if phrase else str(m.channel or "")
        return [StateRecord(
            msg_id=m.id, speaker=m.speaker, modality=modality, attribute=attribute, value=value,
            subject=subject, entities=ents, timestamp=m.timestamp, replaces_cue=replaces,
        )]
