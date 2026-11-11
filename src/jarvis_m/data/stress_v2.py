"""Jarvis-StateBench v2: state-tracking conversations in four domains.

    jarvis-eval data make-stress --domain personal --split train --n 300 [--background locomo]

Differences from v1 (``stress.py``, kept unchanged for v1-compatible files):

* **Domains.** ``team`` (a team across channels), ``assistant`` (a user and an AI assistant
  over sessions), ``personal`` (friends or family: a DM, sometimes a small group chat),
  ``enterprise`` (a team with channel memberships, clearance levels and compartments), and
  ``mix`` (each conversation draws one of the four).
* **Disjoint vocabularies.** Slot names, tasks, systems, personal subjects and suggestion
  topics are split into train / dev / test pools (60 / 20 / 20), on top of the phrasing
  families (train and dev share families 0-1, test uses family 2).
* **Updates without the slot name.** At least half of the update messages refer to the slot
  by ellipsis or pronoun ("too pricey, let's move it to Render"), after a message that
  brings the slot up.
* **Holders.** Per-person slots (where someone lives, their coffee order, seat preference)
  with conflicting values across people; every label carries its holder.
* **Assistant suggestions** accepted explicitly ("sounds good"), implicitly ("booked it")
  or after a rejection, and user corrections.
* **Enterprise.** A channel directory and principals (dataset annotations), a confidential
  update to a slot that also has a public value, viewer-relative questions (the gold answer
  depends on who asks), audience questions (the answer goes to a channel or a colleague)
  and out-of-order message lists.
* **Background.** ``background=`` splices the slot threads into real conversation turns
  (LoCoMo or LongMemEval-S) to reach 300-3000 messages; questions are only about the
  inserted slots.
* **Paraphrase.** :func:`paraphrase` rewrites generated messages with a chat model
  (cached; values are kept verbatim or the original is kept).

Every question carries ``evidence_ids``, ``stale_answers``, ``stale_evidence_ids`` and
``meta`` (``domain``, ``phenomena``, and for enterprise ``channel`` / ``audience``); the
dataset carries supersession labels in ``annotations["supersession"]``.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import Conversation, Dataset, Message, Question
from . import stress as v1

DOMAINS_V2 = ("team", "assistant", "personal", "enterprise")
GEN_DOMAINS = DOMAINS_V2 + ("mix",)
_POOL = {"train": (0, 1, 2), "dev": (3,), "test": (4,)}
T0 = 1767225600.0  # 2026-01-01


def pool(items: Sequence, split: str) -> List:
    """The split's share of a vocabulary: every 5th item from offset 3 is dev, from 4 test."""
    return [x for i, x in enumerate(items) if i % 5 in _POOL[split]]


def _fams(split: str) -> Tuple[int, ...]:
    return (0, 1) if split in ("train", "dev") else (2,)


# ------------------------------------------------------------------ vocabularies
CHOICE_SLOTS = {**v1.CHOICE_SLOTS, **{
    "logging stack": ["Loki", "ELK", "Graylog", "Papertrail", "Splunk", "Logtail"],
    "feature flag service": ["LaunchDarkly", "Unleash", "Flagsmith", "GrowthBook", "Split", "ConfigCat"],
    "CDN": ["Cloudflare", "Fastly", "Akamai", "CloudFront", "Bunny CDN", "KeyCDN"],
    "container registry": ["Docker Hub", "GHCR", "ECR", "Quay", "Harbor", "Artifact Registry"],
    "search engine": ["Elasticsearch", "OpenSearch", "Typesense", "Meilisearch", "Algolia", "Vespa"],
    "cache layer": ["Redis", "Memcached", "Dragonfly", "KeyDB", "Hazelcast", "Varnish"],
    "email provider": ["SendGrid", "Postmark", "Mailgun", "Amazon SES", "Resend", "Brevo"],
    "data warehouse": ["Snowflake", "BigQuery", "Redshift", "ClickHouse", "Databricks", "DuckDB"],
    "secrets manager": ["Vault", "Doppler", "AWS Secrets Manager", "1Password", "Infisical", "SOPS"],
    "design tool": ["Figma", "Sketch", "Penpot", "Framer", "Adobe XD", "Lunacy"],
    "docs platform": ["Notion", "Confluence", "GitBook", "Docusaurus", "Mintlify", "ReadMe"],
    "ORM": ["Prisma", "Drizzle", "SQLAlchemy", "TypeORM", "Sequelize", "Django ORM"],
    "load balancer": ["NGINX", "HAProxy", "Envoy", "Traefik", "Caddy", "AWS ALB"],
    "on-call tool": ["PagerDuty", "Opsgenie", "incident.io", "Grafana OnCall", "Squadcast", "FireHydrant"],
    "billing system": ["Chargebee", "Recurly", "Lago", "Orb", "Zuora", "Maxio"],
}}
DEADLINE_SLOTS = v1.DEADLINE_SLOTS + ["pentest", "pricing launch", "SOC 2 audit", "board deck", "hackathon demo",
                                      "Q3 roadmap review", "API freeze", "code freeze", "partner demo"]
TASKS = v1.TASKS + ["the SLA dashboard", "the incident postmortem", "the vendor contract review", "the hiring rubric",
                    "the data retention policy", "the release checklist", "the status page", "the GDPR export",
                    "the cost report", "the e2e test suite", "the mobile crash triage"]
SYSTEMS = v1.SYSTEMS + ["webhook retries", "the admin panel", "PDF invoices", "the mobile login", "rate limiter",
                        "audit log export", "the GraphQL gateway"]
DATES = v1.DATES + ["July 9", "August 21", "September 3", "October 12", "November 30", "December 8", "next Friday",
                    "the 28th", "Q3", "early May"]
CITIES = ["Pune", "Berlin", "Lisbon", "Austin", "Toronto", "Leeds", "Osaka", "Nairobi", "Melbourne", "Bogota",
          "Krakow", "Seattle", "Dublin", "Chennai", "Porto", "Denver", "Lyon", "Taipei", "Cape Town", "Montreal"]
# personal subjects: (subject, attribute, values, set families, update families, question, question for "me")
PERSONAL = [
    ("home", "location", CITIES,
     [["I live in {v}.", "I'm based in {v} these days."], ["Home is {v} for me now.", "I settled in {v} a while back."],
      ["been living in {v} for a bit", "{v} is home now"]],
     [["I moved to {v} last month, no longer in {old}.", "Just got the keys to the new flat in {v}!"],
      ["Big news: I relocated to {v}.", "Starting fresh in {v}, {old} is behind me."],
      ["new place in {v}, finally", "left {old}, {v} now"]],
     "Where does {p} live now?", "Where do I live now?"),
    ("diet", "preference", ["vegetarian", "vegan", "pescatarian", "keto", "gluten-free", "eating everything again"],
     [["I'm {v} these days.", "Heads up for dinner: I'm {v}."], ["Just so you know, I went {v}.", "I'm strictly {v} now."],
      ["fyi i'm {v} now", "{v} for me, remember"]],
     [["Update on food: I'm {v} now, not {old} any more.", "Dropped the {old} thing, I'm {v} now."],
      ["I switched to {v} last week.", "Plot twist, {v} from now on."],
      ["no more {old}, {v} now", "{v} now lol"]],
     "What is {p}'s diet now?", "What diet am I on now?"),
    ("coffee order", "preference", ["flat white", "oat latte", "black americano", "cortado", "chai latte", "cold brew"],
     [["My usual is a {v}.", "I always get a {v}."], ["A {v} for me, as always.", "Put me down for a {v}."],
      ["{v} pls", "the usual, {v}"]],
     [["I've switched to a {v}, the {old} was too much.", "Get me a {v} instead from now on."],
      ["Changed my order: {v}.", "No more {old}, a {v} these days."],
      ["{v} now not {old}", "{v} these days"]],
     "What coffee does {p} order now?", "What is my coffee order?"),
    ("flight seat", "preference", ["window seat", "aisle seat", "exit row", "front row", "back of the plane"],
     [["I always book a {v} when I fly.", "On flights I need the {v}."], ["Book me a {v} please.", "I'm a {v} person."],
      ["{v} for me on planes", "{v}, always"]],
     [["Changed my mind about flights: {v} from now on.", "Actually the {v} is better for me now."],
      ["I prefer the {v} these days, not the {old}.", "Swap me to a {v}."],
      ["{v} now, {old} was cramped", "{v} going forward"]],
     "Which seat does {p} prefer on flights?", "Which seat do I prefer on flights?"),
    ("job", "role", ["Stripe", "a hospital", "a startup", "the city council", "a bakery", "a law firm", "a school",
                     "a game studio"],
     [["I work at {v}.", "My job is at {v}."], ["I'm with {v} now.", "Been at {v} for two years."],
      ["working at {v}", "{v} pays the bills"]],
     [["I started a new job at {v}, left {old}.", "Big change: I joined {v}."],
      ["Last day at {old} was Friday, {v} from Monday.", "I quit {old}, {v} hired me."],
      ["new job at {v}!!", "{v} now, bye {old}"]],
     "Where does {p} work now?", "Where do I work now?"),
    ("car", "possession", ["a Honda Civic", "a Tesla Model 3", "a Vespa", "an old Golf", "a Kia EV6", "a Mini"],
     [["I drive {v}.", "My car is {v}."], ["Still driving {v}.", "{v} is my ride."],
      ["got {v}", "{v} still runs"]],
     [["Sold the old one, I got {v} now.", "I bought {v} last weekend."],
      ["Traded in {old} for {v}.", "Finally upgraded to {v}."],
      ["{v} now!", "bye {old}, hello {v}"]],
     "What does {p} drive now?", "What car do I have now?"),
    ("pet", "possession", ["a beagle", "two cats", "a parrot", "a rescue greyhound", "a hamster", "a corgi puppy"],
     [["I have {v}.", "I've got {v} at home."], ["My {v} says hi.", "Living with {v} these days."],
      ["{v} at home", "me and {v}"]],
     [["I adopted {v} last week!", "New family member: {v}."],
      ["{v} joined the family.", "Guess what, I got {v}."],
      ["got {v}!!", "{v} now, chaos"]],
     "What pet does {p} have now?", "What pet do I have now?"),
    ("gym days", "schedule", ["Mondays and Thursdays", "every morning", "weekends only", "Tuesdays", "Wednesday evenings"],
     [["I go to the gym {v}.", "Gym is {v} for me."], ["I train {v}.", "My workouts are {v}."],
      ["gym {v}", "{v} is gym time"]],
     [["New routine: gym {v} now.", "I moved my workouts to {v}."],
      ["Switching to {v} for the gym.", "{v} works better for training now."],
      ["gym {v} now", "{v} now, not {old}"]],
     "When does {p} go to the gym now?", "When do I go to the gym now?"),
    ("weekend trip", "plan", ["Lisbon", "the lake house", "Snowdonia", "Kyoto", "a beach in Goa", "the Alps"],
     [["Planning a weekend in {v}.", "I'm going to {v} next weekend."], ["Trip to {v} is booked.", "{v} this weekend!"],
      ["{v} trip soon", "off to {v} sat"]],
     [["Change of plans, I'm doing {v} instead.", "Cancelled {old}, it's {v} now."],
      ["Trip moved: {v} instead of {old}.", "I switched the trip to {v}."],
      ["{v} instead", "scrapped {old}, {v}"]],
     "Where is {p} going for the weekend trip?", "Where am I going for the weekend trip?"),
    ("phone", "possession", ["a Pixel 8", "an iPhone 15", "a Fairphone", "a Galaxy S24", "a OnePlus 12"],
     [["I use {v}.", "My phone is {v}."], ["Typing this on {v}.", "Still on {v}."], ["{v} here", "on {v}"]],
     [["Got {v}, the old one died.", "Switched to {v}."], ["Upgraded to {v} finally.", "{v} now, no more {old}."],
      ["{v} now", "new {v}!"]],
     "What phone does {p} use now?", "What phone do I use now?"),
    ("commute", "preference", ["by bike", "by bus", "on foot", "by train", "by car", "by scooter"],
     [["I get to work {v}.", "My commute is {v}."], ["I commute {v}.", "Going in {v} every day."],
      ["{v} to work", "commuting {v}"]],
     [["I've started commuting {v}.", "Now I go in {v}, not {old}."],
      ["Switched my commute to {v}.", "{v} these days."], ["{v} now", "{v} not {old}"]],
     "How does {p} commute now?", "How do I commute now?"),
    ("language class", "plan", ["Spanish", "Japanese", "German", "Portuguese", "Korean", "Italian"],
     [["I'm learning {v}.", "Signed up for {v} classes."], ["Studying {v} in the evenings.", "{v} lessons twice a week."],
      ["{v} class tonight", "learning {v}"]],
     [["I dropped {old} and started {v}.", "Switched to {v} classes."],
      ["Now it's {v}, not {old}.", "Moving on to {v}."], ["{v} now", "{v} instead"]],
     "Which language is {p} learning now?", "Which language am I learning now?"),
    ("favourite cuisine", "preference", ["Thai", "Ethiopian", "Mexican", "Sichuan", "Lebanese", "Peruvian"],
     [["{v} is my favourite food.", "Nothing beats {v} for me."], ["I'm obsessed with {v} food.", "Always up for {v}."],
      ["{v} >>> everything", "{v} pls"]],
     [["New favourite: {v}, sorry {old}.", "I'm all about {v} now."],
      ["{v} overtook {old} for me.", "These days it's {v}."], ["{v} now", "{v} > {old}"]],
     "What is {p}'s favourite cuisine now?", "What is my favourite cuisine now?"),
    ("doctor", "relationship", ["Dr. Rao", "Dr. Silva", "Dr. Okafor", "Dr. Lindqvist", "Dr. Moreau"],
     [["My doctor is {v}.", "I see {v} for check-ups."], ["{v} is my GP.", "Booked with {v}, my usual."],
      ["seeing {v}", "{v} as usual"]],
     [["I switched doctors, {v} now.", "Moved to {v} after {old} retired."],
      ["New GP: {v}.", "Seeing {v} from now on."], ["{v} now", "{v} not {old}"]],
     "Who is {p}'s doctor now?", "Who is my doctor now?"),
    ("running goal", "plan", ["a 10K", "a half marathon", "a marathon", "a 5K", "a trail ultra"],
     [["Training for {v}.", "My goal this year is {v}."], ["Signed up for {v}.", "Working towards {v}."],
      ["{v} training", "{v} this year"]],
     [["Upgraded the goal to {v}.", "Changed my mind, I'm doing {v}."],
      ["Going for {v} instead.", "{v} now, {old} felt too easy."], ["{v} now", "{v} instead"]],
     "What race is {p} training for now?", "What race am I training for now?"),
]
SUGGESTIONS = [
    ("restaurant for Friday dinner", ["Osteria Nonna", "Saffron House", "The Green Fork", "Kaiseki Mori", "Taqueria Sol"]),
    ("hotel in Lisbon", ["Casa do Rio", "Hotel Alfama", "The Lumiares", "Memmo Principe", "Lisboa Pessoa"]),
    ("laptop to buy", ["ThinkPad X1 Carbon", "MacBook Air", "Framework 13", "Dell XPS 13", "ASUS Zenbook 14"]),
    ("birthday gift for my sister", ["a pottery class", "noise-cancelling headphones", "a Kindle", "a spa voucher",
                                     "a plant subscription"]),
    ("weekend hike", ["Ridge Loop Trail", "Falls Canyon", "Eagle Peak", "Cedar Creek Trail", "Lighthouse Path"]),
    ("book for the flight", ["Project Hail Mary", "Educated", "The Overstory", "Piranesi", "Sapiens"]),
    ("podcast for the commute", ["Hard Fork", "Radiolab", "99% Invisible", "Freakonomics", "Lex Fridman"]),
    ("recipe for the dinner party", ["mushroom risotto", "lamb tagine", "paella", "butter chicken", "ramen"]),
    ("task app", ["Todoist", "TickTick", "Things", "Notion", "Microsoft To Do"]),
    ("running shoes", ["Pegasus 41", "Ghost 16", "Clifton 9", "Novablast 4", "Endorphin Speed"]),
]
PERSONAL_CHATTER = ["haha", "lol", "omg", "miss you!", "ok", "good night", "call later?", "😂", "sure", "nice",
                    "how was your day?", "running late, sorry", "did you see the match?", "ugh mondays",
                    "send pics!", "on my way", "thanks!!", "wait what", "love that", "brb"]
ASSISTANT_CHATTER = [("Can you explain how compound interest works?", "Sure: interest is added to the principal..."),
                     ("What's a good stretch for lower back pain?", "Try a gentle child's pose for 30 seconds..."),
                     ("Translate 'thank you' to Japanese.", "Arigatou gozaimasu."),
                     ("How long should I boil an egg?", "About 9 minutes for a firm yolk."),
                     ("Summarise the plot of Hamlet in one line.", "A prince avenges his father and nearly everyone dies.")]

# ellipsis updates: a context message brings the slot up, the update does not repeat its name
ELL_CTX = [["How is the {s} holding up?", "Any concerns with {old} for the {s}?"],
           ["{p}, what do you think about our {s} setup?", "Is {old} still OK for the {s}?"],
           ["thoughts on the {s}?", "{old} for the {s} still fine?"]]
ELL_UPD = [["Too expensive. Let's move it to {v}.", "Not great, switch it to {v} please."],
           ["Honestly {v} would be better, let's do that instead.", "We should drop it and go with {v}."],
           ["meh, {v} instead", "nah, moving that to {v}"]]
ELL_DL_CTX = [["Are we still on track for the {s}?", "Is {old} realistic for the {s}?"],
              ["{p}, how's the {s} looking?", "Can we make {old} for the {s}?"],
              ["{s} still on?", "{s} on {old}, really?"]]
ELL_DL_UPD = [["No, push it to {v}.", "Not really, let's do {v} instead."],
              ["We need more time, {v} is safer.", "It's moving to {v}."],
              ["nope, {v}", "{v} then"]]
SUGGEST_ASK = [["Can you recommend a {s}?", "Any ideas for a {s}?"], ["I need a {s}, what would you suggest?",
                                                                       "Help me pick a {s}."],
               ["{s} ideas?", "what {s} should i go for"]]
SUGGEST = [["I'd suggest {v}: it fits what you described.", "How about {v}? It's well reviewed."],
           ["{v} is a good option for that.", "My pick would be {v}."], ["maybe {v}?", "{v} could work"]]
ACCEPT = [["Sounds good, let's do that.", "Perfect, thanks!"], ["Yes, let's do it.", "Okay, go with that."],
          ["ok sounds good", "yep"]]
IMPLICIT = [["Booked {v}, thanks!", "Went ahead with {v}."], ["Ordered {v} just now.", "{v} it is, already done."],
            ["got {v}", "done, {v}"]]
REJECT = [["{v} is too pricey for me. Something else?", "Not {v}, I tried it before. Other options?"],
          ["Hmm, {v} doesn't work for me. Anything else?", "I'd rather not do {v}."], ["nah not {v}", "{v}? no"]]
SUGGEST2 = [["Then {v} is a great alternative.", "In that case, try {v}."], ["Then {v} could suit you.",
                                                                          "Another option is {v}."],
            ["{v} then?", "try {v}"]]
ACCEPT2 = [["Perfect, going with {v}.", "Great, {v} it is."], ["{v} sounds right, booking it.", "Done, {v}."],
           ["{v} works", "ok {v}"]]
CORRECT = [["Actually I changed my mind, I went with {v} instead.", "Correction: it's {v}, not {old}."],
           ["Update: switched to {v} after all.", "Scratch that, {v} in the end."], ["{v} in the end", "went {v} after all"]]
FACT_CORRECT = [["Sorry, correction: it's {v}, not {old}.", "Wait, I misspoke earlier: {v}, not {old}."],
                ["Let me correct that: {v}.", "Actually no, {v}."], ["oops i meant {v}", "{v}* not {old}"]]
Q_SUGGEST = ["Which {s} did I go with?", "What did I pick for the {s}?"]
Q_CUR = ["What is the current {s}?", "What are we using for the {s} now?"]
Q_DL = ["When is the {s}?", "What is the current date for the {s}?"]
# about nothing in any split's vocabulary (v1's "Which CDN ..." would name a v2 slot)
Q_ABSTAIN = ["Which office plants did the team order?", "What is the budget for the {s}?",
             "Who approved the parking policy?"]


@dataclass
class _B:
    rng: random.Random
    cid: str
    t: float
    msgs: List[Message] = field(default_factory=list)
    protect: Dict[str, List[str]] = field(default_factory=dict)  # message id -> strings a paraphrase must keep

    def say(self, speaker: str, text: str, channel: str, reply_to: Optional[str] = None, gap: float = 0.0,
            role: Optional[str] = None, keep: Sequence[str] = (), meta: Optional[Dict[str, Any]] = None) -> str:
        self.t += gap + self.rng.uniform(20, 600)
        mid = f"{self.cid}-m{len(self.msgs)}"
        day = int((self.t - T0) // 86400)
        self.msgs.append(Message(id=mid, speaker=speaker, text=text, timestamp=self.t, channel=channel,
                                 session_id=f"{channel}-d{day}", reply_to=reply_to, role=role, meta=meta or {}))
        if keep:
            self.protect[mid] = [k for k in keep if k]
        return mid


class _Conv:
    """Builds one conversation: messages, questions and labels."""

    def __init__(self, rng: random.Random, cid: str, split: str, name: str, domain: str) -> None:
        self.rng, self.cid, self.split, self.name, self.domain = rng, cid, split, name, domain
        self.fams = _fams(split)
        self.t0 = T0 + rng.uniform(0, 86400 * 30)
        self.b = _B(rng, cid, self.t0)
        self.span = 86400 * rng.uniform(2, 8)
        self.qs: List[Question] = []
        self.labels: List[Dict[str, str]] = []
        self.setters: List[Tuple[str, str, str]] = []  # (message id, slot, holder)
        self.annotations: Dict[str, Any] = {}
        self.n_updates = self.n_ellipsis = 0

    def note_update(self, slot: str, mid: str) -> None:
        """Count an update message, and whether it names its slot."""
        self.n_updates += 1
        text = next(m.text for m in reversed(self.b.msgs) if m.id == mid).lower()
        self.n_ellipsis += slot.lower() not in text

    def pick(self, table, **kw) -> str:
        return self.rng.choice(table[self.rng.choice(self.fams)]).format(**kw)

    def restart(self) -> None:
        self.b.t = self.t0 + self.rng.uniform(0, self.span)

    def q(self, kind, text, ans, ev, asker, stale=(), stale_ev=(), phen=(), abst=False, **meta) -> None:
        self.qs.append(Question(
            id=f"{self.cid}/q{len(self.qs)}", conversation_id=self.cid, question=text, answer=ans, asker=asker,
            category=kind, evidence_ids=tuple(ev), stale_answers=tuple(stale), is_abstention=abst,
            stale_evidence_ids=tuple(stale_ev),
            meta={"phenomena": list(phen), "split": self.split, "domain": self.domain,
                  **{k: v for k, v in meta.items() if v is not None}}))

    def lab(self, older, newer, label, holder) -> None:
        self.labels.append({"dataset": self.name, "conversation_id": self.cid, "older_id": older, "newer_id": newer,
                            "label": label, "holder": holder})

    def chain(self, mids, slot, holder) -> None:
        self.setters.extend((m, slot, holder) for m in mids)
        for i, older in enumerate(mids):
            for newer in mids[i + 1:]:
                self.lab(older, newer, "supersedes", holder)

    def chatter(self, speakers, lines, chans, k=1.0) -> None:
        for _ in range(int(self.rng.random() * 3 * k)):
            self.b.say(self.rng.choice(speakers), self.rng.choice(lines), self.rng.choice(chans))

    # ------------------------------------------------------------ slot threads
    def choice_slot(self, s, team, ch, values, meta_upd=None, upd_channel=None) -> Dict[str, Any]:
        """Set, then 0-2 updates (half by ellipsis), or a proposal / negation trap."""
        rng, b = self.rng, self.b
        self.restart()
        hist = [(values[0], b.say(rng.choice(team), self.pick(v1.SET_CHOICE, s=s, v=values[0]), ch, keep=[values[0]]))]
        phen, ev_extra, trap = [], [], None
        self.chatter(team, v1.CHATTER, [ch], 0.5)
        r = rng.random()
        if r < 0.6:
            for k in range(1 if rng.random() < 0.7 else 2):
                new, old = values[1 + k], hist[-1][0]
                gap = rng.uniform(3600, 86400 * 2)
                if rng.random() < 0.6:
                    ctx = b.say(rng.choice(team), self.pick(ELL_CTX, s=s, old=old, p=rng.choice(team)), ch, gap=gap)
                    mid = b.say(rng.choice(team), self.pick(ELL_UPD, v=new), ch, reply_to=ctx, keep=[new])
                    ev_extra = [ctx]
                    phen.append("ellipsis")
                else:
                    mid = b.say(rng.choice(team), self.pick(v1.UPDATE_CHOICE, s=s, v=new, old=old), ch, gap=gap,
                                keep=[new, old])
                    ev_extra = []
                hist.append((new, mid))
                self.note_update(s, mid)
            phen += ["update"] + (["double_update"] if len(hist) > 2 else [])
        elif r < 0.8:
            trap = b.say(rng.choice(team), self.pick(v1.PROPOSE, s=s, v=values[3]), ch, keep=[values[3]])
            phen.append("proposal_trap")
        elif r < 0.9:
            trap = b.say(rng.choice(team), self.pick(v1.NEGATE, s=s, v=values[3]), ch, keep=[values[3]])
            phen.append("negation_trap")
        holder = f"group:{ch}"
        self.chain([h[1] for h in hist], s, holder)
        if trap:
            self.lab(hist[-1][1], trap, "same_slot_no_change", holder)
        return {"hist": hist, "trap": trap, "phen": phen, "ev_extra": ev_extra, "trap_value": values[3]}

    def ask_choice(self, s, res, asker, **meta) -> None:
        hist, trap = res["hist"], res["trap"]
        cur, cur_mid = hist[-1]
        stale = [h[0] for h in hist[:-1]] + ([res["trap_value"]] if trap else [])
        self.q("knowledge_update" if len(hist) > 1 else "single_hop", self.rng.choice(Q_CUR).format(s=s), cur,
               [*res["ev_extra"], cur_mid] if len(hist) > 1 else [cur_mid], asker, stale,
               [h[1] for h in hist[:-1]] + ([trap] if trap else []), res["phen"] or ["plain"], **meta)
        if len(hist) > 1 and self.rng.random() < 0.4:
            self.q("temporal", self.rng.choice(v1.Q_FIRST).format(s=s), hist[0][0], [hist[0][1]], asker, [cur],
                   [cur_mid], ["history"], **meta)

    def deadline_slot(self, s, team, ch, asker=None, **meta) -> None:
        rng, b = self.rng, self.b
        self.restart()
        d = rng.sample(DATES, 3)
        hist = [(d[0], b.say(rng.choice(team), self.pick(v1.SET_DEADLINE, s=s, v=d[0]), ch, keep=[d[0]]))]
        ev_extra, phen = [], ["deadline"]
        self.chatter(team, v1.CHATTER, [ch])
        if rng.random() < 0.6:
            gap = rng.uniform(3600, 86400 * 3)
            if rng.random() < 0.6:
                ctx = b.say(rng.choice(team), self.pick(ELL_DL_CTX, s=s, old=d[0], p=rng.choice(team)), ch, gap=gap)
                hist.append((d[1], b.say(rng.choice(team), self.pick(ELL_DL_UPD, v=d[1]), ch, reply_to=ctx, keep=[d[1]])))
                ev_extra, phen = [ctx], phen + ["ellipsis"]
            else:
                hist.append((d[1], b.say(rng.choice(team), self.pick(v1.UPDATE_DEADLINE, s=s, v=d[1], old=d[0]), ch,
                                         gap=gap, keep=[d[1], d[0]])))
            self.note_update(s, hist[-1][1])
            phen.append("update")
        self.chain([h[1] for h in hist], s, f"group:{ch}")
        self.q("knowledge_update" if len(hist) > 1 else "single_hop", rng.choice(Q_DL).format(s=s), hist[-1][0],
               [*ev_extra, hist[-1][1]], asker or rng.choice(team), [h[0] for h in hist[:-1]],
               [h[1] for h in hist[:-1]], phen, **meta)

    def owner_slot(self, t, team, ch) -> None:
        rng, b = self.rng, self.b
        self.restart()
        owner = rng.choice(team)
        m1 = b.say(owner, self.pick(v1.OWN, t=t), ch)
        self.chatter(team, v1.CHATTER, [ch])
        ev, stale, phen, stale_ev = [m1], [], ["attribution"], []
        if rng.random() < 0.5:
            new = rng.choice([p for p in team if p != owner])
            ask, ok = self.pick(v1.HANDOFF, t=t, p=new).split("|")
            m2 = b.say(owner, ask, ch, gap=rng.uniform(3600, 86400))
            m3 = b.say(new, ok, ch, reply_to=m2)
            ev, stale, phen, stale_ev = [m2, m3], [owner], ["attribution", "handoff", "reply_thread"], [m1]
            owner = new
            self.chain([m1, m3], f"owner of {t}", f"group:{ch}")
        else:
            self.chain([m1], f"owner of {t}", f"group:{ch}")
        self.q("multi_hop" if len(ev) > 1 else "single_hop", rng.choice(v1.Q_OWNER).format(t=t), owner, ev,
               rng.choice(team), stale, stale_ev, phen)
        if rng.random() < 0.5:
            self.q("user_implicit", rng.choice(v1.Q_MINE), t, ev[-1:], owner, phen=["user_implicit"])

    def blocker_slot(self, x, team, ch) -> None:
        rng, b = self.rng, self.b
        self.restart()
        e, f = rng.choice(v1.ERRORS), rng.choice(v1.FIXES)
        mb = b.say(rng.choice(team), self.pick(v1.BLOCK, x=x, e=e), ch)
        self.chatter(team, v1.CHATTER, [ch])
        if rng.random() < 0.5:
            mr = b.say(rng.choice(team), self.pick(v1.RESOLVE, x=x, f=f), ch, reply_to=mb if rng.random() < 0.5 else None)
            self.chain([mb, mr], f"status of {x}", f"group:{ch}")
            self.q("knowledge_update", rng.choice(v1.Q_BLOCK).format(x=x), f"Yes, it was resolved by {f}.", [mb, mr],
                   rng.choice(team), ["still open"], phen=["resolution"])
        else:
            mp = b.say(rng.choice(team), self.pick(v1.PROMISE, x=x, f=f), ch, reply_to=mb if rng.random() < 0.5 else None)
            self.chain([mb], f"status of {x}", f"group:{ch}")
            self.lab(mb, mp, "same_slot_no_change", f"group:{ch}")
            self.q("knowledge_update", rng.choice(v1.Q_BLOCK).format(x=x),
                   f"No, it is still open ({e}); a fix is only planned.", [mb, mp], rng.choice(team),
                   ["resolved", "fixed"], phen=["intent_vs_done"])

    def personal_slot(self, spec, person, ch, others, chatter_lines, asker=None, correct=False,
                      role=None) -> Tuple[str, str]:
        """One person's fact, maybe updated (or corrected). Returns (current value, its message)."""
        subject, _attr, values, sets, upds, q_other, q_me = spec
        rng, b = self.rng, self.b
        self.restart()
        vals = rng.sample(values, 3)
        hist = [(vals[0], b.say(person, self.pick(sets, v=vals[0]), ch, keep=[vals[0]], role=role))]
        self.chatter(others + [person], chatter_lines, [ch], 0.5)
        phen = ["personal"]
        if rng.random() < 0.6:
            gap = rng.uniform(3600, 86400 * 4)
            if correct and rng.random() < 0.5:
                hist.append((vals[1], b.say(person, self.pick(FACT_CORRECT, v=vals[1], old=vals[0]), ch,
                                            gap=rng.uniform(30, 300), keep=[vals[1]], role=role)))
                phen.append("correction")
            else:
                hist.append((vals[1], b.say(person, self.pick(upds, v=vals[1], old=vals[0]), ch, gap=gap,
                                            keep=[vals[1]], role=role)))
            self.note_update(subject, hist[-1][1])
            phen.append("update")
        holder = f"speaker:{person}"
        self.chain([h[1] for h in hist], subject, holder)
        asker = asker or rng.choice(others + [person])
        text = (q_me if asker == person else q_other).format(p=person)
        self.q("knowledge_update" if len(hist) > 1 else "single_hop", text, hist[-1][0], [hist[-1][1]], asker,
               [h[0] for h in hist[:-1]], [h[1] for h in hist[:-1]], phen + (["user_implicit"] if asker == person else []),
               holder=holder)
        return hist[-1]

    def conflict(self, spec, people, ch, chatter_lines) -> None:
        """Two people state different values for the same personal subject."""
        subject, _a, values, sets, _u, q_other, _q = spec
        a, c = self.rng.sample(people, 2)
        va, vc = self.rng.sample(values, 2)
        self.restart()
        ma = self.b.say(a, self.pick(sets, v=va), ch, keep=[va])
        mc = self.b.say(c, self.pick(sets, v=vc), ch, keep=[vc], gap=self.rng.uniform(30, 900))
        self.setters += [(ma, subject, f"speaker:{a}"), (mc, subject, f"speaker:{c}")]
        self.lab(ma, mc, "unrelated", f"speaker:{c}")  # different holders never supersede
        for p, v, m, other in ((a, va, ma, vc), (c, vc, mc, va)):
            self.q("single_hop", q_other.format(p=p), v, [m], self.rng.choice(people), [other], [mc if m == ma else ma],
                   ["holder_conflict"], holder=f"speaker:{p}")

    def suggestion(self, s, values, user, bot, ch) -> None:
        rng, b = self.rng, self.b
        self.restart()
        v = rng.sample(values, 3)
        b.say(user, self.pick(SUGGEST_ASK, s=s), ch)
        a1 = b.say(bot, self.pick(SUGGEST, v=v[0]), ch, role="assistant", keep=[v[0]])
        r = rng.random()
        stale, stale_ev = [], []
        if r < 0.4:
            ok = b.say(user, self.pick(ACCEPT), ch)
            cur, cur_mid, ev, phen = v[0], a1, [a1, ok], ["suggestion_accepted"]
        elif r < 0.7:
            cur_mid = b.say(user, self.pick(IMPLICIT, v=v[0]), ch, keep=[v[0]])
            cur, ev, phen = v[0], [a1, cur_mid], ["suggestion_implicit"]
        else:
            b.say(user, self.pick(REJECT, v=v[0]), ch, keep=[v[0]])
            a2 = b.say(bot, self.pick(SUGGEST2, v=v[1]), ch, role="assistant", keep=[v[1]])
            cur_mid = b.say(user, self.pick(ACCEPT2, v=v[1]), ch, keep=[v[1]])
            cur, ev, phen = v[1], [a2, cur_mid], ["suggestion_rejected"]
            stale, stale_ev = [v[0]], [a1]
            self.lab(a1, cur_mid, "same_slot_no_change", f"speaker:{user}")  # a rejected suggestion was never a value
        chain = [cur_mid]
        if rng.random() < 0.35:
            fix = b.say(user, self.pick(CORRECT, v=v[2], old=cur), ch, gap=rng.uniform(3600, 86400 * 2), keep=[v[2]])
            self.note_update(s, fix)
            stale, stale_ev = stale + [cur], stale_ev + [cur_mid]
            cur, ev, phen = v[2], [fix], phen + ["user_correction"]
            chain.append(fix)
        self.chain(chain, s, f"speaker:{user}")
        self.q("knowledge_update" if len(chain) > 1 or stale else "single_hop", rng.choice(Q_SUGGEST).format(s=s),
               cur, ev, user, stale, stale_ev, phen, holder=f"speaker:{user}")

    def finish_labels(self, msgs: Sequence[Message], base_seed: float) -> None:
        when = {m.id: m.timestamp for m in msgs}
        cross = [(a, c, hc) for i, (a, sa, ha) in enumerate(self.setters) for c, sc, hc in self.setters[i + 1:]
                 if (sa, ha) != (sc, hc)]
        lrng = random.Random(base_seed)
        for a, c, hc in lrng.sample(cross, min(len(cross), 30)):
            older, newer = sorted((a, c), key=lambda m: when[m])
            self.lab(older, newer, "unrelated", hc)


# ------------------------------------------------------------------ domains
def _team(cv: _Conv, enterprise: bool = False) -> Tuple[List[str], List[str]]:
    rng, split = cv.rng, cv.split
    team = rng.sample(v1.NAMES, rng.randint(4, 7) if enterprise else rng.randint(3, 6))
    chans = rng.sample(v1.CHANNELS, rng.randint(1, 3))
    slots = rng.sample(pool(list(CHOICE_SLOTS), split), rng.randint(2, 3))
    shared = rng.choice(chans)
    for s in slots:
        ch = shared if rng.random() < 0.6 else rng.choice(chans)
        vals = rng.sample(CHOICE_SLOTS[s], 4)
        res = cv.choice_slot(s, team, ch, vals)
        if ch == shared and len(slots) > 1:
            res["phen"].append("collision")
        cv.ask_choice(s, res, rng.choice(team))
    for s in rng.sample(pool(DEADLINE_SLOTS, split), rng.randint(1, 2)):
        cv.deadline_slot(s, team, rng.choice(chans))
    for t in rng.sample(pool(TASKS, split), rng.randint(1, 2)):
        cv.owner_slot(t, team, rng.choice(chans))
    for x in rng.sample(pool(SYSTEMS, split), 1):
        cv.blocker_slot(x, team, rng.choice(chans))
    subjects = pool(PERSONAL, split)
    cv.conflict(rng.choice(subjects), team, rng.choice(chans), v1.CHATTER)
    if rng.random() < 0.5:
        cv.personal_slot(rng.choice(subjects), rng.choice(team), rng.choice(chans),
                         [p for p in team], v1.CHATTER)
    if rng.random() < 0.6:
        cv.q("abstention", rng.choice(Q_ABSTAIN).format(s=slots[0]), "Not mentioned in the conversation", [],
             rng.choice(team), abst=True, phen=["abstention"])
    for _ in range(3):
        cv.restart()
        cv.chatter(team, v1.CHATTER, chans, 2.0)
    return team, chans


def _assistant(cv: _Conv) -> None:
    rng, split = cv.rng, cv.split
    user, bot, ch = "user", "assistant", "chat"
    for s, values in rng.sample(pool(SUGGESTIONS, split), min(2, len(pool(SUGGESTIONS, split)))):
        cv.suggestion(s, values, user, bot, ch)
    for spec in rng.sample(pool(PERSONAL, split), 2):
        cv.personal_slot(spec, user, ch, [], ["ok", "thanks"], asker=user, correct=True)
    for _ in range(rng.randint(2, 5)):
        cv.restart()
        u, a = rng.choice(ASSISTANT_CHATTER)
        cv.b.say(user, u, ch)
        cv.b.say(bot, a, ch, role="assistant")
    if rng.random() < 0.5:
        cv.q("abstention", "Which dentist did I book?", "Not mentioned in the conversation", [], user, abst=True,
             phen=["abstention"])


def _personal(cv: _Conv) -> None:
    rng, split = cv.rng, cv.split
    group = rng.random() < 0.3
    people = rng.sample(v1.NAMES, rng.randint(3, 4) if group else 2)
    ch = "family" if group else "dm"
    subjects = pool(PERSONAL, split)
    for spec in rng.sample(subjects, min(3, len(subjects))):
        p = rng.choice(people)
        cv.personal_slot(spec, p, ch, [x for x in people if x != p], PERSONAL_CHATTER)
    cv.conflict(rng.choice(subjects), people, ch, PERSONAL_CHATTER)
    if group:  # a family plan (group-held decision)
        s = rng.choice(pool(DEADLINE_SLOTS, split))
        cv.deadline_slot(s, people, ch)
    for _ in range(3):
        cv.restart()
        cv.chatter(people, PERSONAL_CHATTER, [ch], 2.0)


def _enterprise(cv: _Conv) -> None:
    """A team plus a directory: #general open, #eng for most of the team, #leadership for the
    managers. Slots with a public value get a confidential update; viewer-relative and
    audience questions."""
    rng, split = cv.rng, cv.split
    team, _ = _team(cv, enterprise=True)
    managers, rest = team[:2], team[2:]
    intern = rest[-1]
    eng_members = sorted(team)
    principals = {p: {"level": "manager", "compartments": ["leadership"]} for p in managers}
    principals.update({p: {"level": "L1"} for p in rest})
    principals[intern] = {"level": "intern"}
    principals[managers[0]] = {"level": "exec", "compartments": ["leadership", "deal"]}
    cv.annotations = {
        "channels": {"general": {"members": "*", "level": "public"},
                     "eng": {"members": eng_members, "level": "intern"},
                     "leadership": {"members": sorted(managers), "level": "manager", "compartments": ["leadership"]}},
        "principals": principals}
    # team() used v1 channel names; the directory lists only general / eng / leadership, so
    # re-home every message so far into #eng (open to the whole team)
    cv.b.msgs = [Message(**{**m.__dict__, "channel": "eng", "session_id": f"eng-{m.session_id.split('-')[-1]}"})
                 for m in cv.b.msgs]
    cv.labels = [{**lab, "holder": "group:eng" if lab["holder"].startswith("group:") else lab["holder"]}
                 for lab in cv.labels]
    for q in cv.qs:
        q.meta.setdefault("channel", f"dm-{q.asker}")  # asked privately: the asker alone
    readers_hi = set(managers)
    for s in rng.sample([x for x in pool(list(CHOICE_SLOTS), split)], 2):
        vals = rng.sample(CHOICE_SLOTS[s], 3)
        cv.restart()
        public_ch = rng.choice(["general", "eng"])
        m_pub = cv.b.say(rng.choice(team), cv.pick(v1.SET_CHOICE, s=s, v=vals[0]), public_ch, keep=[vals[0]])
        cv.chatter(team, v1.CHATTER, [public_ch], 0.5)
        secret_ch, meta, prefix = ("leadership", None, "") if rng.random() < 0.5 else (public_ch, {"acl_level": "L2"},
                                                                                        "[CONFIDENTIAL] ")
        gap, mgr = rng.uniform(3600, 86400), rng.choice(managers)
        ctx = None
        if rng.random() < 0.6:  # by ellipsis, after a (restricted) message that brings the slot up
            ctx = cv.b.say(rng.choice(managers), prefix + cv.pick(ELL_CTX, s=s, old=vals[0], p=mgr), secret_ch,
                           gap=gap, meta=meta)
            text, gap = cv.pick(ELL_UPD, v=vals[1]), 0.0
        else:
            text = cv.pick(v1.UPDATE_CHOICE, s=s, v=vals[1], old=vals[0])
        m_sec = cv.b.say(mgr, prefix + text, secret_ch, gap=gap, keep=[vals[1]], meta=meta, reply_to=ctx)
        # #leadership: managers only; the [CONFIDENTIAL] marker: L2 and up (the managers here)
        can = readers_hi if secret_ch == "leadership" else set(managers)
        phen = ["confidential_channel"] if secret_ch == "leadership" else ["confidential_marker"]
        cv.note_update(s, m_sec)
        cv.chain([m_pub, m_sec], s, "group:" + public_ch)
        hi, lo = rng.choice(sorted(can)), rng.choice([p for p in team if p not in can])
        cv.q("viewer_relative", rng.choice(Q_CUR).format(s=s), vals[1], [*([ctx] if ctx else []), m_sec], hi,
             [vals[0]], [m_pub],
             phen + ["viewer_high"], channel=f"dm-{hi}")
        cv.q("viewer_relative", rng.choice(Q_CUR).format(s=s), vals[0], [m_pub], lo, [vals[1]], [],
             phen + ["viewer_low"], channel=f"dm-{lo}")
        # the high reader answers into #eng (default audience: everyone in it) or for a colleague
        if rng.random() < 0.5:
            cv.q("audience", rng.choice(Q_CUR).format(s=s), vals[0], [m_pub], hi, [vals[1]], [],
                 phen + ["audience_channel"], channel="eng")
        else:
            cv.q("audience", rng.choice(Q_CUR).format(s=s), vals[0], [m_pub], hi, [vals[1]], [],
                 phen + ["audience_list"], channel=f"dm-{hi}", audience=[lo])
    if rng.random() < 0.5:  # an exec-only, compartmented fact
        cv.restart()
        target = rng.choice(["Globex", "Initech", "Hooli", "Umbrella", "Vandelay"])
        m = cv.b.say(managers[0], f"We are in talks to acquire {target}, keep it quiet.", "leadership",
                     meta={"acl_compartments": ["deal"]}, keep=[target])
        cv.setters.append((m, "acquisition", "group:leadership"))
        cv.q("viewer_relative", "Which company are we in talks to acquire?", target, [m], managers[0],
             phen=["compartment"], channel=f"dm-{managers[0]}")
        cv.q("viewer_relative", "Which company are we in talks to acquire?", "Not mentioned in the conversation", [],
             managers[1], [target], abst=True, phen=["compartment", "abstention"], channel=f"dm-{managers[1]}")


# ------------------------------------------------------------------ background
def load_background(kind: str, path: Optional[str] = None) -> List[List[Message]]:
    """Real conversations to splice slot threads into (each a list of messages)."""
    from .loaders import load_locomo, load_longmemeval

    if kind == "locomo":
        ds = load_locomo(path)
    elif kind == "longmemeval":
        ds = load_longmemeval(path)
    else:
        raise ValueError(f"unknown background {kind!r}: none, locomo or longmemeval")
    return [c.messages for c in ds.conversations.values()]


def _splice(cv: _Conv, msgs: List[Message], bg: List[List[Message]], n_min: int, n_max: int,
            channel: str, rename: Dict[str, str]) -> List[Message]:
    """Interleave background turns with the slot messages, spread over the same time span,
    until there are between n_min and n_max messages."""
    rng = cv.rng
    target = rng.randint(n_min, n_max) - len(msgs)
    turns: List[Message] = []
    k = rng.randrange(len(bg))
    while len(turns) < target and bg:
        turns += bg[k % len(bg)]
        k += 1
    turns = turns[:max(0, target)]
    if not turns:
        return msgs
    lo, hi = min(m.timestamp for m in msgs) - 3600, max(m.timestamp for m in msgs) + 3600
    out = list(msgs)
    for i, m in enumerate(turns):
        t = lo + (hi - lo) * (i + 0.5) / len(turns)
        day = int((t - T0) // 86400)
        out.append(Message(id=f"{cv.cid}-bg{i}", speaker=rename.get(m.speaker, m.speaker), text=m.text, timestamp=t,
                           channel=channel, session_id=f"{channel}-d{day}",
                           role="assistant" if m.speaker == "assistant" else None, meta={"background": True}))
    return sorted(out, key=lambda m: m.timestamp)


# ------------------------------------------------------------------ entry point
def generate_v2(n: int = 100, seed: int = 0, split: str = "train", domain: str = "team",
                background: Optional[List[List[Message]]] = None, background_len: Tuple[int, int] = (300, 3000),
                out_of_order: float = 0.1) -> Dataset:
    if domain not in GEN_DOMAINS:
        raise ValueError(f"domain must be one of {GEN_DOMAINS}")
    base_seed = {"train": 0, "dev": 10_000, "test": 20_000}[split] + seed + 7_000_000
    name = f"statebench_v2_{domain}_{split}"
    convs: Dict[str, Conversation] = {}
    questions: List[Question] = []
    labels: List[Dict[str, str]] = []
    access: Dict[str, Any] = {}
    protect: Dict[str, List[str]] = {}
    n_upd = n_ell = 0
    for ci in range(n):
        rng = random.Random(base_seed * 1_000 + ci)
        dom = DOMAINS_V2[ci % 4] if domain == "mix" else domain
        cid = f"sb2-{dom}-{split}-{ci}"
        cv = _Conv(rng, cid, split, name, dom)
        {"team": _team, "assistant": _assistant, "personal": _personal, "enterprise": _enterprise}[dom](cv)
        msgs = sorted(cv.b.msgs, key=lambda m: m.timestamp)
        if background:
            main = "chat" if dom == "assistant" else ("eng" if dom == "enterprise" else msgs[0].channel)
            people = sorted({m.speaker for m in msgs if m.speaker not in ("user", "assistant")})
            bg_people = sorted({m.speaker for c in background[:1] for m in c if m.speaker not in ("user", "assistant")})
            rename = dict(zip(bg_people, people)) if dom == "personal" else {}
            msgs = _splice(cv, msgs, background, *background_len, main, rename)
        if dom == "enterprise" and out_of_order > 0:  # a late-arriving message: list order != time order
            k = max(1, int(len(msgs) * out_of_order))
            for _ in range(k):
                i = rng.randrange(len(msgs))
                j = min(len(msgs) - 1, i + rng.randint(1, 5))
                msgs[i], msgs[j] = msgs[j], msgs[i]
        if cv.annotations:  # each conversation has its own directory (names repeat across conversations)
            access[cid] = cv.annotations
        convs[cid] = Conversation(cid, msgs)
        end = max(m.timestamp for m in msgs) + 3600
        questions += [Question(**{**qq.__dict__, "question_time": end}) for qq in cv.qs]
        cv.finish_labels(msgs, base_seed * 1_000 + ci + 0.5)
        labels += cv.labels
        n_upd, n_ell = n_upd + cv.n_updates, n_ell + cv.n_ellipsis
        protect.update(cv.b.protect)
    ann: Dict[str, Any] = {"supersession": labels, "stats": {"updates": n_upd, "updates_without_slot_name": n_ell},
                           "generator": {"version": 2, "domain": domain, "split": split, "seed": seed,
                                         "background": bool(background)}}
    ann["protect"] = protect  # values each generated message states (kept verbatim by paraphrase)
    if access:
        ann["access"] = access  # per conversation: {channels, principals} (jarvis_m.access.conversation_directory)
    return Dataset(name, convs, questions, ann)


# ------------------------------------------------------------------ paraphrase (GPU, user runs)
PARAPHRASE_SYSTEM = ("Rewrite the chat message in different, natural wording, as the same person would type it. "
                     "Keep the meaning, the tone and every string listed under KEEP exactly as written. "
                     "Output only the rewritten message.")


def paraphrase(ds: Dataset, chat: Any, cache_path: str | Path, protect: Optional[Dict[str, List[str]]] = None,
               progress: bool = True) -> Dict[str, int]:
    """Rewrite generated messages (not background turns) with ``chat``. A rewrite that drops
    a value it must keep is discarded (the original stays). Rewrites are cached in a JSONL
    file keyed by (model, text, kept strings), so reruns are free."""
    cache_path = Path(cache_path)
    cache: Dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["key"]] = d["text"]
    stats = {"messages": 0, "rewritten": 0, "kept_original": 0, "cached": 0}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for cid, conv in ds.conversations.items():
            new = []
            for m in conv.messages:
                if m.meta.get("background"):
                    new.append(m)
                    continue
                keep = list((protect if protect is not None else ds.annotations.get("protect") or {}).get(m.id, []))
                stats["messages"] += 1
                key = hashlib.sha1(json.dumps([getattr(chat, "name", "?"), m.text, keep]).encode()).hexdigest()
                if key in cache:
                    stats["cached"] += 1
                    text = cache[key]
                else:
                    raw = chat.chat(PARAPHRASE_SYSTEM, f"KEEP: {json.dumps(keep)}\nMESSAGE: {m.text}", 200)
                    text = (raw or "").strip().strip('"').splitlines()[0] if (raw or "").strip() else ""
                    cache[key] = text
                    f.write(json.dumps({"key": key, "text": text}) + "\n")
                ok = text and all(k.lower() in text.lower() for k in keep)
                stats["rewritten" if ok else "kept_original"] += 1
                new.append(Message(**{**m.__dict__, "text": text}) if ok else m)
            ds.conversations[cid] = Conversation(cid, new)
    if progress:
        print(f"[paraphrase] {stats}")
    return stats
