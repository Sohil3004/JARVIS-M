"""Reader (answers from retrieved context) and judges (grade the answer).

Judge prompts follow each benchmark's official protocol so our numbers are
comparable with published ones:
* GroupMemBench: strict paraphrase judge, ``Final: Correct|Incorrect``
* LongMemEval: per-question-type yes/no templates (Wu et al., 2024, MIT licence)
* generic: GroupMemBench-style judge (used for LoCoMo and our own sets)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..llm import ChatModel, build_chat
from ..types import Question
from .metrics import is_abstention, normalize, token_f1

READER_SYSTEM = (
    "You are a careful QA agent. Use only the retrieved conversation passages to answer the question.\n"
    "First provide a short reasoning paragraph. Then provide the final answer on a new line using the format:\n"
    "Final: <answer>\n"
    "If the answer is not supported by the retrieved passages, say so in the Final line."
)

GMB_JUDGE_SYSTEM = (
    "You are a strict judge evaluating whether an agent's answer matches the gold answer for a question.\n"
    "Consider paraphrases correct if they have the same meaning as the gold answer.\n"
    "First provide a brief reasoning paragraph. Then provide the final judgment on a new line using the format:\n"
    "Final: Correct\nor\nFinal: Incorrect"
)

_LME_BASE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
    "the response only contains a subset of the information required by the answer, answer no. "
)
_LME = {
    "default": _LME_BASE,
    "temporal-reasoning": _LME_BASE
    + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number "
    "of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
    "is 18), the model's response is still correct. ",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
    "information along with an updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer "
    "no. The model does not need to reflect all the points in the rubric. The response is correct as long as it "
    "recalls and utilizes the user's personal information correctly.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please "
    "answer yes if the model correctly identifies the question as unanswerable. The model could say that the "
    "information is incomplete, or some other information is given but the asked information is not.",
}


def extract_final(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if low.startswith("final answer:") or low.startswith("final:"):
            return ln.split(":", 1)[1].strip()
    return lines[-1] if lines else ""


class Reader:
    def __init__(self, chat: ChatModel, max_tokens: int = 512, system: str = READER_SYSTEM) -> None:
        self.chat, self.max_tokens, self.system = chat, max_tokens, system

    def answer(self, q: Question, context: str) -> Tuple[str, str]:
        asker = f"Asking user: {q.asker}\n\n" if q.asker else ""
        user = (
            f"{asker}Question:\n{q.question}\n\nRetrieved passages:\n{context}\n\n"
            "Answer the question using the retrieved passages."
        )
        raw = self.chat.chat(self.system, user, self.max_tokens)
        return extract_final(raw), raw


@dataclass
class Verdict:
    correct: Optional[bool]
    raw: str = ""


class MatchJudge:
    """No-LLM judge for smoke tests and short-answer sets: normalised containment
    or token-F1 >= threshold; abstention questions need a refusal."""

    name = "match"

    def __init__(self, f1_threshold: float = 0.5) -> None:
        self.t = f1_threshold

    def judge(self, q: Question, pred: str) -> Verdict:
        if q.is_abstention:
            return Verdict(is_abstention(pred), "abstention-check")
        g, p = normalize(q.answer), normalize(pred)
        ok = bool(g) and (g in p or token_f1(pred, q.answer) >= self.t)
        return Verdict(ok, f"f1={token_f1(pred, q.answer):.2f}")


class LLMJudge:
    def __init__(self, chat: ChatModel, protocol: str = "generic", max_tokens: int = 256) -> None:
        self.chat, self.protocol, self.max_tokens = chat, protocol, max_tokens
        self.name = f"llm:{protocol}:{chat.name}"

    def judge(self, q: Question, pred: str) -> Verdict:
        if self.protocol == "longmemeval":
            key = "abstention" if q.is_abstention else (q.category if q.category in _LME else "default")
            tmpl = _LME[key]
            label = "Explanation" if q.is_abstention else ("Rubric" if key == "single-session-preference" else "Correct Answer")
            ask = ("Does the model correctly identify the question as unanswerable?" if q.is_abstention
                   else "Is the model response correct?")
            user = f"{tmpl}\n\nQuestion: {q.question}\n\n{label}: {q.answer}\n\nModel Response: {pred}\n\n{ask} Answer yes or no only."
            raw = self.chat.chat("", user, 10)
            low = raw.strip().lower()
            return Verdict(True if low.startswith("yes") else False if low.startswith("no") else None, raw)
        user = f"Question:\n{q.question}\n\nGold Answer:\n{q.answer}\n\nAgent Answer:\n{pred}\n"
        raw = self.chat.chat(GMB_JUDGE_SYSTEM, user, self.max_tokens)
        final = extract_final(raw).lower()
        if "incorrect" in final or "wrong" in final or "not correct" in final:
            return Verdict(False, raw)
        return Verdict(True if "correct" in final else None, raw)


DEFERRED = "deferred"


class DeferredJudge:
    """Grade later with ``jarvis-eval judge`` so the reader and judge never share
    the GPU: generate everything, free VRAM, then load the judge once."""

    name = DEFERRED

    def judge(self, q: Question, pred: str) -> Verdict:
        return Verdict(None, DEFERRED)


def build_reader(cfg: Optional[dict]) -> Reader:
    cfg = dict(cfg or {})
    max_tokens = cfg.pop("max_tokens", 512)
    return Reader(build_chat(cfg.pop("chat", None)), max_tokens=max_tokens)


def build_judge(cfg: Optional[dict]):
    cfg = dict(cfg or {"type": "match"})
    kind = cfg.pop("type")
    if kind == "match":
        return MatchJudge(**cfg)
    if kind == "llm":
        return LLMJudge(build_chat(cfg.pop("chat")), **cfg)
    if kind == DEFERRED:
        return DeferredJudge()
    raise ValueError(f"unknown judge {kind!r}")
