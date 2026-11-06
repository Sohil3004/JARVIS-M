"""Jarvis-StateBench: controlled multi-party conversations for state tracking.

Each conversation is a small team working across channels and days. Hidden
"slots" (hosting choice, release date, task owner, an open bug ...) get set,
revised, negated, proposed-but-not-adopted, resolved or only promised. Every
question carries gold evidence message ids, stale (superseded) answers and the
phenomena it probes, so per-phenomenon curves fall out of one run.

Phrasings come in *families*. ``split="test"`` uses held-out families, so a
learned extractor or retriever cannot win by memorising templates.

    jarvis-eval data make-stress --split train --n 300 --seed 0
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..types import Conversation, Dataset, Message, Question

NAMES = ["Aarav", "Bianca", "Chen", "Divya", "Elif", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Kavya", "Liam",
         "Meera", "Nikhil", "Olga", "Priya", "Quinn", "Rohan", "Sara", "Tomas", "Uma", "Victor", "Wen", "Yusuf"]
CHANNELS = ["eng", "product", "infra", "launch", "backend", "design"]

CHOICE_SLOTS = {
    "backend hosting": ["AWS EC2", "Google Cloud Run", "Azure App Service", "Fly.io", "Render", "DigitalOcean"],
    "primary database": ["PostgreSQL", "MongoDB", "MySQL", "DynamoDB", "CockroachDB", "Supabase"],
    "message queue": ["Kafka", "RabbitMQ", "Amazon SQS", "NATS", "Redis Streams", "Google Pub/Sub"],
    "frontend framework": ["Next.js", "Remix", "SvelteKit", "Nuxt", "Angular", "Astro"],
    "auth provider": ["Auth0", "Clerk", "Firebase Auth", "Keycloak", "Cognito", "Supabase Auth"],
    "CI system": ["GitHub Actions", "GitLab CI", "CircleCI", "Jenkins", "Buildkite", "Drone"],
    "analytics tool": ["Mixpanel", "Amplitude", "PostHog", "Heap", "Plausible", "Google Analytics"],
    "payment gateway": ["Stripe", "Razorpay", "Adyen", "PayPal", "Braintree", "Paddle"],
    "error tracker": ["Sentry", "Rollbar", "Bugsnag", "Datadog", "Honeybadger", "Raygun"],
}
DEADLINE_SLOTS = ["release date", "design review", "beta launch", "security audit", "investor demo", "data migration"]
DATES = ["Monday 10 AM", "Tuesday 5 PM", "Wednesday noon", "Thursday 9 AM", "Friday 5 PM", "next Monday",
         "March 14", "April 2", "May 20", "June 6", "the 15th", "end of the month"]
TASKS = ["load testing", "API documentation", "the migration script", "the onboarding flow", "the billing page",
         "the on-call runbook", "the pricing experiment", "the accessibility audit", "the SDK release notes"]
SYSTEMS = ["staging deploy", "payment webhook", "nightly ETL job", "image upload service", "search indexer",
           "push notifications", "SSO login", "CSV export"]
ERRORS = ["Error 403", "a 502 from the gateway", "an out-of-memory crash", "a timeout after 30s", "Error 500",
          "a certificate error", "a race condition", "a schema mismatch"]
FIXES = ["rotating the credentials", "bumping the memory limit", "adding a retry with backoff", "pinning the driver",
         "renewing the certificate", "adding a lock", "rerunning the migration", "fixing the IAM policy"]
CHATTER = [
    "lunch at 1?", "anyone seen the new office chairs lol", "brb, standup", "thanks!", "nice work everyone",
    "can someone share the figma link again", "my laptop just restarted itself", "coffee run, want anything?",
    "the wifi on floor 3 is dead again", "happy friday folks", "who broke the build? jk it was me, fixed",
    "reminder: all-hands moved to the big room", "ok", "sounds good", "+1", "will read after this call",
]

# ---- phrasing families: index = family id. Test split uses the last family of each list.
SET_CHOICE = [
    ["We decided to use {v} for the {s}.", "Final call: {s} is {v}.", "Agreed, {v} for the {s}."],
    ["Let's go with {v} for the {s}.", "{v} it is for the {s}.", "Locking in {v} as our {s}."],
    ["ok so {s} -> {v}, everyone fine with that?", "Going ahead with {v} on the {s} side.",
     "For the {s} we're picking {v}, no objections so far."],
]
UPDATE_CHOICE = [
    ["Change of plan: we switched the {s} to {v} instead.", "We're moving the {s} from {old} to {v}.",
     "Update: {s} is now {v}, not {old}."],
    ["Scrap {old}, {v} for the {s}.", "Actually {v} is a better fit for the {s}, let's use that.",
     "Replacing {old} with {v} for the {s}."],
    ["after the cost review {old} is out, {s} will be {v}", "heads up, {v} replaces {old} for the {s}",
     "we ended up on {v} for the {s} after all"],
]
PROPOSE = [
    ["Maybe we should try {v} for the {s}?", "What if we used {v} for the {s}?"],
    ["Has anyone looked at {v} for the {s}? Just an idea.", "Could {v} work for the {s}?"],
    ["random thought, {v} for the {s} someday?", "{v} might be nice for the {s}, not now though"],
]
NEGATE = [
    ["To be clear, we are not switching the {s} to {v}.", "No, the {s} is not moving to {v}."],
    ["Ignore the rumour, {v} is not happening for the {s}.", "We won't be using {v} for the {s}."],
    ["{v} for the {s} was ruled out, staying put", "nope, no {v} for the {s}"],
]
SET_DEADLINE = [
    ["The {s} is due {v}.", "Deadline for the {s}: {v}."],
    ["We need the {s} done by {v}.", "Target for the {s} is {v}."],
    ["{s} lands {v}, plan around that", "aiming at {v} for the {s}"],
]
UPDATE_DEADLINE = [
    ["The {s} has been rescheduled to {v}.", "New date for the {s}: {v} (was {old})."],
    ["Pushing the {s} out to {v}.", "Moving the {s} to {v}, {old} was too tight."],
    ["{s} slips to {v}", "we bumped the {s} back to {v}"],
]
OWN = [
    ["I'll take {t}.", "I will own {t} this sprint."],
    ["Assigning {t} to myself.", "I can handle {t}."],
    ["{t} is on me", "dibs on {t}"],
]
HANDOFF = [
    ["{p}, can you take over {t} from me?|Sure, I'll own {t} from now on.",
     "{p}, please pick up {t}.|On it, {t} is mine now."],
    ["Handing {t} to {p}.|Got it, I've got {t}.", "{p} will run {t} instead of me.|Yes, taking {t}."],
    ["{p} pls grab {t}|ok {t} is with me now", "passing {t} over to {p}|cool, I'll drive {t}"],
]
BLOCK = [
    ["The {x} is failing with {e}.", "We have {e} on the {x}."],
    ["{x} is broken: {e}.", "Seeing {e} in the {x} again."],
    ["{x} down, {e}", "ugh {e} on {x}"],
]
RESOLVE = [
    ["Fixed the {x} by {f}.", "Resolved: {x} works again after {f}."],
    ["The {x} issue is done, it was solved by {f}.", "{x} is back to normal, {f} did it."],
    ["{x} sorted ({f})", "{x} is green now, {f}"],
]
PROMISE = [
    ["I will fix the {x} tomorrow by {f}.", "I'll look into the {x} later, probably {f}."],
    ["Going to try {f} on the {x} next week.", "Plan is {f} for the {x}, will do it after the demo."],
    ["will poke at {x} tmrw, maybe {f}", "{x}: gonna try {f} later"],
]

Q_CURRENT = ["What is the current {s}?", "Which option did the team settle on for the {s}?", "What are we using for the {s}?"]
Q_FIRST = ["What was originally chosen for the {s}?", "What was the {s} before it was changed?"]
Q_DEADLINE = ["When is the {s}?", "What is the current date for the {s}?"]
Q_OWNER = ["Who is responsible for {t}?", "Who currently owns {t}?"]
Q_MINE = ["What am I responsible for?", "Which task did I take on?"]
Q_BLOCK = ["Is the {x} problem resolved?", "What is the status of the {x} issue?"]
Q_ABSTAIN = ["Which CDN did the team choose?", "What is the budget for the {s}?", "Who approved the hiring plan?"]


@dataclass
class _Builder:
    rng: random.Random
    cid: str
    t: float
    msgs: List[Message] = field(default_factory=list)

    def say(self, speaker: str, text: str, channel: str, reply_to: Optional[str] = None, gap: float = 0.0) -> str:
        self.t += gap + self.rng.uniform(20, 600)
        mid = f"{self.cid}-m{len(self.msgs)}"
        day = int((self.t - 1767225600) // 86400)
        self.msgs.append(Message(id=mid, speaker=speaker, text=text, timestamp=self.t, channel=channel,
                                 session_id=f"{channel}-d{day}", reply_to=reply_to))
        return mid


def _fams(split: str) -> Sequence[int]:
    return (0, 1) if split in ("train", "dev") else (2,)


def _pick(rng: random.Random, table, fams, **kw) -> str:
    return rng.choice(table[rng.choice(fams)]).format(**kw)


def generate(n: int = 100, seed: int = 0, split: str = "train", noise: float = 1.0) -> Dataset:
    base_seed = {"train": 0, "dev": 10_000, "test": 20_000}[split] + seed
    convs: Dict[str, Conversation] = {}
    questions: List[Question] = []
    fams = _fams(split)
    for ci in range(n):
        rng = random.Random(base_seed * 1_000 + ci)
        cid = f"sb-{split}-{ci}"
        team = rng.sample(NAMES, rng.randint(3, 6))
        chans = rng.sample(CHANNELS, rng.randint(1, 3))
        t0 = 1767225600.0 + rng.uniform(0, 86400 * 30)
        b = _Builder(rng, cid, t0)
        span = 86400 * rng.uniform(2, 6)

        def restart():  # each slot's thread starts at a random time -> topics interleave
            b.t = t0 + rng.uniform(0, span)
        qs: List[Question] = []
        events: List[Tuple] = []

        def chatter(k: float = 1.0):
            for _ in range(int(rng.random() * 3 * noise * k)):
                b.say(rng.choice(team), rng.choice(CHATTER), rng.choice(chans))

        def q(kind, text, ans, ev, stale=(), asker=None, phen=(), abst=False):
            qs.append(Question(
                id=f"{cid}/q{len(qs)}", conversation_id=cid, question=text, answer=ans, asker=asker or rng.choice(team),
                category=kind, evidence_ids=tuple(ev), stale_answers=tuple(stale), is_abstention=abst,
                meta={"phenomena": list(phen), "split": split},
            ))

        # --- choice slots (set / update / propose / negate), some share a channel (entity collision)
        slots = rng.sample(list(CHOICE_SLOTS), rng.randint(2, 4))
        shared_chan = rng.choice(chans)
        for s in slots:
            restart()
            vals = rng.sample(CHOICE_SLOTS[s], 4)
            ch = shared_chan if rng.random() < 0.6 else rng.choice(chans)
            history = [(vals[0], b.say(rng.choice(team), _pick(rng, SET_CHOICE, fams, s=s, v=vals[0]), ch))]
            phen = ["collision"] if ch == shared_chan and len(slots) > 1 else []
            chatter()
            r = rng.random()
            if r < 0.55:  # one or two updates
                for k in range(1 if rng.random() < 0.75 else 2):
                    new = vals[1 + k]
                    chatter(0.5)
                    gap = rng.uniform(3600, 86400 * 3)
                    history.append((new, b.say(rng.choice(team), _pick(rng, UPDATE_CHOICE, fams, s=s, v=new, old=history[-1][0]), ch, gap=gap)))
                phen += ["update"] + (["double_update"] if len(history) > 2 else [])
            elif r < 0.75:
                b.say(rng.choice(team), _pick(rng, PROPOSE, fams, s=s, v=vals[3]), ch)
                phen += ["proposal_trap"]
            elif r < 0.9:
                b.say(rng.choice(team), _pick(rng, NEGATE, fams, s=s, v=vals[3]), ch)
                phen += ["negation_trap"]
            cur, cur_mid = history[-1]
            stale = [h[0] for h in history[:-1]] + ([vals[3]] if ("proposal_trap" in phen or "negation_trap" in phen) else [])
            kind = "knowledge_update" if len(history) > 1 else "single_hop"
            q(kind, rng.choice(Q_CURRENT).format(s=s), cur, [cur_mid], stale, phen=phen or ["plain"])
            if len(history) > 1 and rng.random() < 0.5:
                q("temporal", rng.choice(Q_FIRST).format(s=s), history[0][0], [history[0][1]], [cur], phen=["history"])
            events.append(("choice", s))

        # --- deadlines
        for s in rng.sample(DEADLINE_SLOTS, rng.randint(1, 2)):
            restart()
            d = rng.sample(DATES, 3)
            ch = rng.choice(chans)
            hist = [(d[0], b.say(rng.choice(team), _pick(rng, SET_DEADLINE, fams, s=s, v=d[0]), ch))]
            chatter()
            if rng.random() < 0.6:
                hist.append((d[1], b.say(rng.choice(team), _pick(rng, UPDATE_DEADLINE, fams, s=s, v=d[1], old=d[0]), ch,
                                         gap=rng.uniform(3600, 86400 * 4))))
            q("knowledge_update" if len(hist) > 1 else "single_hop", rng.choice(Q_DEADLINE).format(s=s), hist[-1][0],
              [hist[-1][1]], [h[0] for h in hist[:-1]], phen=["deadline"] + (["update"] if len(hist) > 1 else []))

        # --- ownership (attribution, hand-off via reply, user-implicit)
        for t in rng.sample(TASKS, rng.randint(1, 2)):
            restart()
            owner = rng.choice(team)
            ch = rng.choice(chans)
            m1 = b.say(owner, _pick(rng, OWN, fams, t=t), ch)
            chatter()
            ev, stale, phen = [m1], [], ["attribution"]
            if rng.random() < 0.5:
                new_owner = rng.choice([p for p in team if p != owner])
                ask, ok = _pick(rng, HANDOFF, fams, t=t, p=new_owner).split("|")
                m2 = b.say(owner, ask, ch, gap=rng.uniform(3600, 86400))
                m3 = b.say(new_owner, ok, ch, reply_to=m2)
                ev, stale, phen = [m2, m3], [owner], ["attribution", "handoff", "reply_thread"]
                owner = new_owner
            q("multi_hop" if len(ev) > 1 else "single_hop", rng.choice(Q_OWNER).format(t=t), owner, ev, stale, phen=phen)
            if rng.random() < 0.5:
                q("user_implicit", rng.choice(Q_MINE), t, ev[-1:], asker=owner, phen=["user_implicit"])

        # --- blockers: resolved vs only promised (intent != done)
        for x in rng.sample(SYSTEMS, rng.randint(1, 2)):
            restart()
            e, f = rng.choice(ERRORS), rng.choice(FIXES)
            ch = rng.choice(chans)
            mb = b.say(rng.choice(team), _pick(rng, BLOCK, fams, x=x, e=e), ch)
            chatter()
            if rng.random() < 0.5:
                mr = b.say(rng.choice(team), _pick(rng, RESOLVE, fams, x=x, f=f), ch, reply_to=mb if rng.random() < 0.5 else None)
                q("knowledge_update", rng.choice(Q_BLOCK).format(x=x), f"Yes, it was resolved by {f}.", [mb, mr],
                  ["still open"], phen=["resolution"])
            else:
                mp = b.say(rng.choice(team), _pick(rng, PROMISE, fams, x=x, f=f), ch, reply_to=mb if rng.random() < 0.5 else None)
                q("knowledge_update", rng.choice(Q_BLOCK).format(x=x), f"No, it is still open ({e}); a fix is only planned.",
                  [mb, mp], ["resolved", "fixed"], phen=["intent_vs_done"])

        # --- abstention
        if rng.random() < 0.6:
            s = rng.choice(slots)
            q("abstention", rng.choice(Q_ABSTAIN).format(s=s), "Not mentioned in the conversation", [], abst=True,
              phen=["abstention"])
        for _ in range(3):
            restart()
            chatter(2.0)
        msgs = sorted(b.msgs, key=lambda m: m.timestamp)
        convs[cid] = Conversation(cid, msgs)
        end = msgs[-1].timestamp + 3600
        questions += [Question(**{**qq.__dict__, "question_time": end}) for qq in qs]
    return Dataset(f"statebench_{split}", convs, questions)


def save(ds: Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def mj(m: Message):
        return {k: v for k, v in m.__dict__.items() if v not in (None, {}) or k in ("id", "speaker", "text")}

    def qj(q: Question):
        d = dict(q.__dict__)
        d["evidence_ids"], d["stale_answers"], d["meta"] = list(q.evidence_ids), list(q.stale_answers), dict(q.meta)
        return d

    obj = {"name": ds.name, "conversations": [{"id": c.id, "messages": [mj(m) for m in c.messages]}
                                              for c in ds.conversations.values()],
           "questions": [qj(q) for q in ds.questions]}
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path
