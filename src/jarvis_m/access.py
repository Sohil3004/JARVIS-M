"""Clearance-aware access control: labels, principals, policy and the read rule.

A message carries a :class:`Label` (clearance level + compartments) and lives in a channel.
A :class:`Principal` can read it when the principal is active, is a member of the channel
(checked at query time), has at least the message's level and holds all its compartments.

Labels only ever go up. The effective label of a message is the join (higher level, union
of compartments) of: the channel default, the message's own ``acl_*`` metadata, explicit
markers in the text such as ``[CONFIDENTIAL]``, an optional extractor suggestion, and the
label of the message it replies to. See ``docs/ACCESS_CONTROL.md``.

Whether anything is enforced is ``AccessPolicy.mode``: ``auto`` (the default) enforces as
soon as any access-control metadata exists (a channel or principal directory, a channel
default or marker in the policy, or one message with a non-public label) and is a no-op
otherwise; ``on`` requires that metadata; ``off`` builds no access control at all.
Participant count never switches enforcement off.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .types import GROUP_MIN_PARTICIPANTS, Message

log = logging.getLogger("jarvis_m.access")

DEFAULT_LEVELS = ("public", "intern", "L1", "L2", "manager", "exec")
HIDDEN_UPDATE_NOTE = "this may have been updated in a restricted context"
EVERYONE = "*"
Audience = Union[None, str, Sequence[str]]


class AccessViolation(RuntimeError):
    """Retrieval was about to return something its principal may not read."""


class AccessConfigError(RuntimeError):
    """``mode: on`` without any access-control metadata to enforce."""


class Label(NamedTuple):
    level: int = 0
    compartments: FrozenSet[str] = frozenset()


PUBLIC = Label()


def join(a: Label, b: Label) -> Label:
    """Least label that is at least as restrictive as both."""
    return Label(max(a.level, b.level), a.compartments | b.compartments)


def _compartments(value) -> FrozenSet[str]:
    if not value:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, Iterable):
        return frozenset(str(c) for c in value)
    raise TypeError(f"compartments must be a list of names, got {value!r}")


def message_label(m: Message) -> Label:
    """The label a message carries in its own metadata, without a policy: ``meta["acl_level"]``
    (an integer level, or a name from :data:`DEFAULT_LEVELS`) and ``meta["acl_compartments"]``
    (a list of names). No metadata means public. With a policy, use ``AccessPolicy.base_label``."""
    level = (m.meta.get("acl_level") if m.meta else None) or 0
    if isinstance(level, str) and level in DEFAULT_LEVELS:
        level = DEFAULT_LEVELS.index(level)
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"message {m.id}: acl_level must be an integer level or one of {DEFAULT_LEVELS}, got {level!r}")
    return Label(level, _compartments(m.meta.get("acl_compartments") if m.meta else None))


@dataclass(frozen=True)
class Principal:
    """Who is asking. ``members`` is set for an audience: the principals this one stands
    for (it is their meet, and must pass every membership check for all of them)."""

    id: str
    level: int = 0
    compartments: FrozenSet[str] = frozenset()
    active: bool = True
    members: Tuple[str, ...] = ()

    @property
    def ids(self) -> Tuple[str, ...]:
        return self.members or (self.id,)


def meet(principals: Sequence[Principal], id: str = "audience") -> Principal:
    """What every one of ``principals`` may read: lowest level, shared compartments."""
    if not principals:
        return Principal(id, 0, frozenset(), active=False)
    comps = frozenset.intersection(*[p.compartments for p in principals])
    members = tuple(sorted({i for p in principals for i in p.ids}))
    return Principal(id, min(p.level for p in principals), comps, all(p.active for p in principals), members)


class ChannelDirectory:
    """Channel membership, read at query time. ``members`` is a set of principal ids, or
    ``"*"`` for an open channel (no membership restriction)."""

    def __init__(self, channels: Optional[Mapping[str, Any]] = None, unlisted: str = "closed") -> None:
        if unlisted not in ("open", "closed"):
            raise ValueError("unlisted_channels must be 'open' or 'closed'")
        self.unlisted = unlisted
        self._members: Dict[str, Optional[FrozenSet[str]]] = {}
        for name, spec in (channels or {}).items():
            members = spec.get("members", EVERYONE) if isinstance(spec, Mapping) else spec
            self.set_members(name, members)

    def set_members(self, channel: str, members: Union[str, Iterable[str], None]) -> None:
        self._members[channel] = None if members in (EVERYONE, None) else frozenset(str(m) for m in members)

    def add_member(self, channel: str, principal_id: str) -> None:
        cur = self._members.get(channel)
        if cur is None and channel in self._members:
            return  # open channel: everyone is already a member
        self._members[channel] = (cur or frozenset()) | {principal_id}

    def remove_member(self, channel: str, principal_id: str) -> None:
        cur = self._members.get(channel)
        if cur is not None:
            self._members[channel] = cur - {principal_id}

    def known(self, channel: str) -> bool:
        return channel in self._members

    def listed(self) -> bool:
        """Does the directory list any channel? An empty directory restricts nothing."""
        return bool(self._members)

    def is_open(self, channel: str) -> bool:
        if channel in self._members:
            return self._members[channel] is None
        return self.unlisted == "open" or not self._members

    def members(self, channel: str) -> Optional[FrozenSet[str]]:
        """Member ids, or None for an open channel. An unlisted channel is closed (no members)
        under ``unlisted: closed``, unless the directory lists no channel at all."""
        if channel in self._members:
            return self._members[channel]
        return None if self.is_open(channel) else frozenset()

    def is_member(self, principal: Principal, channel: str) -> bool:
        members = self.members(channel)
        return members is None or all(i in members for i in principal.ids)


@dataclass
class AccessPolicy:
    levels: Tuple[str, ...] = DEFAULT_LEVELS  # ordered, lowest first
    roles: Dict[str, str] = field(default_factory=dict)  # role -> level name
    default_level: Union[str, int] = 0  # channels with no entry
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # channel -> {level, compartments}
    markers: List[Dict[str, Any]] = field(default_factory=list)  # [{pattern, level?, compartments?}]
    unlisted_channels: str = "closed"  # with a channel directory: a channel it does not list
    signal_hidden_updates: bool = False
    inactive_audience_members: str = "block"  # block: an inactive member empties the audience | ignore
    use_extractor_suggestions: bool = True
    # off: no access control | auto: enforce when any ACL metadata exists, else a no-op | on: require it
    mode: str = "auto"
    final_check: str = "prod"  # prod = filter + log, debug = raise
    enforce: bool = True  # False = measure only (labels and audit, nothing is filtered). Never in production.

    def __post_init__(self) -> None:
        self.levels = tuple(self.levels)
        if len(set(self.levels)) != len(self.levels) or not self.levels:
            raise ValueError("levels must be a non-empty list of distinct names")
        if self.mode in ("prod", "debug"):  # before mode meant the final check
            self.final_check, self.mode = self.mode, "auto"
        if self.mode not in ("off", "auto", "on"):
            raise ValueError("access mode must be 'off', 'auto' or 'on'")
        if self.final_check not in ("prod", "debug"):
            raise ValueError("final_check must be 'prod' or 'debug'")
        if self.inactive_audience_members not in ("block", "ignore"):
            raise ValueError("inactive_audience_members must be 'block' or 'ignore'")
        self._markers = [(re.compile(m["pattern"], re.I), Label(self.level(m.get("level", 0)),
                                                                _compartments(m.get("compartments"))))
                         for m in self.markers]
        self._channel_labels = {c: Label(self.level(s.get("level", self.default_level)),
                                         _compartments(s.get("compartments")))
                                for c, s in self.channels.items()}
        self._default = Label(self.level(self.default_level))
        for role, lvl in self.roles.items():
            self.level(lvl)  # validates
            if not isinstance(role, str):
                raise ValueError(f"role names must be strings, got {role!r}")

    @classmethod
    def from_config(cls, cfg: Union[None, str, Path, Mapping[str, Any]]) -> AccessPolicy:
        """A mapping, or the path of a YAML file with a top-level ``access:`` block."""
        if isinstance(cfg, (str, Path)):
            import yaml

            raw = yaml.safe_load(Path(cfg).read_text(encoding="utf-8")) or {}
            cfg = raw.get("access", raw)
        cfg = dict(cfg or {})
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown access policy keys {sorted(unknown)}")
        return cls(**cfg)

    def level(self, value: Union[str, int]) -> int:
        """Level index from a level name, a role name or an index."""
        if isinstance(value, bool):
            raise ValueError(f"bad level {value!r}")
        if isinstance(value, int):
            if not 0 <= value < len(self.levels):
                raise ValueError(f"level {value} is outside 0..{len(self.levels) - 1}")
            return value
        if value in self.levels:
            return self.levels.index(value)
        if value in self.roles:
            return self.level(self.roles[value])
        raise ValueError(f"unknown level or role {value!r}; levels: {self.levels}, roles: {sorted(self.roles)}")

    def level_name(self, index: int) -> str:
        return self.levels[index] if 0 <= index < len(self.levels) else str(index)

    def has_label_sources(self) -> bool:
        """Does the policy itself label anything (a channel default or a text marker)?"""
        return bool(self._markers) or any(lab != PUBLIC for lab in self._channel_labels.values()) \
            or self._default != PUBLIC

    def channel_label(self, channel: str) -> Label:
        return self._channel_labels.get(channel, self._default)

    def set_channel_default(self, channel: str, level: Union[str, int, None] = None,
                            compartments: Optional[Iterable[str]] = None) -> None:
        """Raise a channel's default label (e.g. from a dataset's channel directory)."""
        extra = Label(self.level(level) if level is not None else 0, _compartments(compartments))
        self._channel_labels[channel] = join(self.channel_label(channel), extra)

    def marker_label(self, text: str) -> Label:
        out = PUBLIC
        for pat, label in self._markers:
            if pat.search(text or ""):
                out = join(out, label)
        return out

    def base_label(self, m: Message) -> Label:
        """Everything a message's label can be derived from without looking at other
        messages: channel default, its own ``acl_*`` metadata, explicit markers."""
        own = PUBLIC
        if m.meta and (m.meta.get("acl_level") is not None or m.meta.get("acl_compartments")):
            own = Label(self.level(m.meta.get("acl_level") or 0), _compartments(m.meta.get("acl_compartments")))
        return join(join(self.channel_label(m.channel), own), self.marker_label(m.text))

    def principal(self, pid: str, spec: Optional[Mapping[str, Any]] = None) -> Principal:
        """``{level: L2 | 3, role: manager, compartments: [...], active: true}``; level wins over role."""
        spec = spec or {}
        raw = spec.get("level", spec.get("role", 0))
        return Principal(str(pid), self.level(raw), _compartments(spec.get("compartments")), bool(spec.get("active", True)))


class ThreadLabeler:
    """Effective labels, in ingest order: a message's own label joined with the label of
    the message it replies to (same channel only). A parent that arrives after its replies
    raises them, so ingest order does not matter."""

    def __init__(self, base_label: Callable[[Message], Label] = message_label) -> None:
        self.base_label = base_label
        self.labels: List[Label] = []
        self.channels: List[str] = []
        self.parent: List[int] = []  # reply parent in the same channel, or -1
        self.children: Dict[int, List[int]] = {}
        self._id2node: Dict[str, int] = {}
        self._orphans: Dict[str, List[int]] = {}  # reply_to id not ingested yet -> waiting nodes

    def add(self, m: Message, suggestion: Optional[Label] = None) -> List[int]:
        """Label ``m``. Returns earlier nodes whose label was raised because ``m`` is their
        (late) thread parent."""
        v = len(self.labels)
        self._id2node[m.id] = v
        label = self.base_label(m)
        if suggestion is not None:
            label = join(label, suggestion)
        # threads live inside one channel: a cross-channel reply_to neither links nor inherits
        p = self._id2node.get(m.reply_to, -1) if m.reply_to else -1
        if m.reply_to and p == -1:
            self._orphans.setdefault(m.reply_to, []).append(v)
        if p != -1 and self.channels[p] != m.channel:
            p = -1
        self.parent.append(p)
        if p != -1:
            self.children.setdefault(p, []).append(v)
            label = join(label, self.labels[p])  # a reply into a restricted thread inherits the restriction
        self.labels.append(label)
        self.channels.append(m.channel)
        raised: List[int] = []
        for c in self._orphans.pop(m.id, []):  # replies that arrived before their parent
            if self.channels[c] == m.channel:
                self.parent[c] = v
                self.children.setdefault(v, []).append(c)
                self._raise(c, label, raised)
        return raised

    def _raise(self, v: int, label: Label, raised: List[int]) -> None:
        new = join(self.labels[v], label)
        if new == self.labels[v]:
            return
        self.labels[v] = new
        raised.append(v)
        for c in self.children.get(v, []):
            self._raise(c, new, raised)


class Redactor(Protocol):
    """Interface stub. Redaction (showing part of a message) is out of scope: the only
    behaviour today is to withhold whole messages, which needs no redactor. An
    implementation returns the text ``principal`` may see, or None to withhold."""

    def redact(self, message: Message, label: Label, principal: Principal) -> Optional[str]: ...


class AccessController:
    """Policy + channel directory + principals: resolves who is asking and what they read."""

    def __init__(self, policy: Optional[AccessPolicy] = None,
                 principals: Optional[Mapping[str, Any]] = None,
                 channels: Optional[Mapping[str, Any]] = None) -> None:
        self.policy = policy or AccessPolicy()
        env_check = os.environ.get("JARVIS_ACCESS_CHECK")  # e.g. the test suite runs with "debug"
        if env_check:
            if env_check not in ("prod", "debug"):
                raise ValueError("JARVIS_ACCESS_CHECK must be 'prod' or 'debug'")
            self.policy.final_check = env_check
        self.directory = ChannelDirectory(unlisted=self.policy.unlisted_channels)
        self.principals: Dict[str, Principal] = {}
        self.configure(principals=principals, channels=channels)

    def configure(self, annotations: Optional[Mapping[str, Any]] = None,
                  principals: Optional[Mapping[str, Any]] = None,
                  channels: Optional[Mapping[str, Any]] = None) -> None:
        """Load a directory: ``channels: {name: {members: [...] | "*", level, compartments}}``
        and ``principals: {id: {level | role, compartments, active}}`` (the same two blocks
        as ``Dataset.annotations``). Call before ingesting: channel defaults label messages."""
        ann = annotations or {}
        for name, spec in {**(ann.get("channels") or {}), **(channels or {})}.items():
            spec = spec if isinstance(spec, Mapping) else {"members": spec}
            self.directory.set_members(name, spec.get("members", EVERYONE))
            if spec.get("level") is not None or spec.get("compartments"):
                self.policy.set_channel_default(name, spec.get("level"), spec.get("compartments"))
        for pid, spec in {**(ann.get("principals") or {}), **(principals or {})}.items():
            self.principals[str(pid)] = spec if isinstance(spec, Principal) else self.policy.principal(pid, spec)

    # ------------------------------------------------------------ is anything enforced?
    def configured(self) -> bool:
        """Is there access-control metadata besides message labels: a principal or channel
        directory, or a policy that labels channels or text?"""
        return bool(self.principals) or self.directory.listed() or self.policy.has_label_sources()

    def active(self, labels: Sequence[Label] = ()) -> bool:
        """Is access control in force for a memory whose messages carry ``labels``?
        ``auto``: when any metadata exists, including one non-public label (labels only go up,
        so once true it stays true). ``on``: always, and missing metadata is an error. Never
        decided by the number of participants."""
        if self.policy.mode == "off":
            return False
        if self.configured() or any(lab != PUBLIC for lab in labels):
            return True
        if self.policy.mode == "on":
            raise AccessConfigError(
                "access mode is 'on' but there is no access-control metadata: no principals, no channel "
                "directory, no policy channel defaults or markers and no labelled message. Configure them "
                "(configure_access / dataset annotations) or use mode: auto.")
        return False

    def default_audience(self, asker: Optional[str], channel: Optional[str],
                         speakers: Iterable[str] = ()) -> Audience:
        """Who an answer posted in ``channel`` is for when the question names no audience.
        A group chat (at least GROUP_MIN_PARTICIPANTS members, counting the asker) or an open
        channel: everyone in it, so the answer must be safe for all of them. A DM, or no
        channel: the asker alone (None). Members come from the directory; for a channel it
        does not list, from ``speakers`` (who has written there)."""
        if channel is None:
            return None
        if self.directory.known(channel):
            members = self.directory.members(channel)
            if members is None:
                return channel  # open: anyone may read what is posted there
            ids = set(members) | ({str(asker)} if asker is not None else set())
            return channel if len(ids) >= GROUP_MIN_PARTICIPANTS else None
        ids = {str(s) for s in speakers} | ({str(asker)} if asker is not None else set())
        return sorted(ids) if len(ids) >= GROUP_MIN_PARTICIPANTS else None

    # ------------------------------------------------------------ who is asking
    def resolve(self, pid: Optional[str]) -> Principal:
        """A known principal, or (unknown id, no id) an anonymous one at the lowest level."""
        if pid is not None and str(pid) in self.principals:
            return self.principals[str(pid)]
        return Principal(str(pid) if pid is not None else "anonymous", 0, frozenset(), True)

    def effective(self, asker: Optional[str], audience: Audience = None) -> Principal:
        """The principal the answer must be safe for. No audience: the asker alone. A channel
        name or a list of ids: the meet of all of them and the asker."""
        me = self.resolve(asker)
        if audience is None:
            return me
        if isinstance(audience, str):
            members = self.directory.members(audience)
            if members is None:  # an open channel: anyone may be reading
                everyone = Principal(EVERYONE, 0, frozenset(), True)
                return replace(meet([me, everyone], f"channel:{audience}"), members=(EVERYONE,))
            ids, name = sorted(members), f"channel:{audience}"
            if not ids:
                return Principal(name, 0, frozenset(), active=False)  # unknown or empty channel: nobody
        else:
            ids, name = [str(i) for i in audience], "audience"
        group = [self.resolve(i) for i in ids if asker is None or str(i) != str(asker)]
        if self.policy.inactive_audience_members == "ignore":
            group = [p for p in group if p.active]
        if asker is not None:
            group.append(me)  # the asker always counts, active or not
        return meet(group, name)  # an empty group is an inactive principal: nothing is readable

    # ------------------------------------------------------------ the read rule
    def can_read(self, p: Principal, label: Label, channel: str) -> bool:
        return (p.active and self.directory.is_member(p, channel)
                and p.level >= label.level and label.compartments <= p.compartments)

    def mask(self, p: Principal, labels: Sequence[Label], channels: Sequence[str]) -> np.ndarray:
        """Per message: may ``p`` read it? (One decision per distinct (channel, label).)"""
        memo: Dict[Tuple[str, Label], bool] = {}
        out = np.zeros(len(labels), bool)
        for i, (label, channel) in enumerate(zip(labels, channels)):
            k = (channel, label)
            if k not in memo:
                memo[k] = self.can_read(p, label, channel)
            out[i] = memo[k]
        return out

    def describe(self, p: Principal) -> Dict[str, Any]:
        return {"id": p.id, "level": p.level, "level_name": self.policy.level_name(p.level),
                "compartments": sorted(p.compartments), "active": p.active, "members": list(p.members)}


_MODES = ("off", "auto", "on")


def build_access(cfg: Union[None, bool, str, Path, Mapping[str, Any], AccessController]) -> Optional[AccessController]:
    """``None`` (the default) = ``mode: auto`` with the built-in policy (labels from message
    metadata and the dataset's directory only). ``"off"`` / ``False`` = no access control (returns
    None); ``"auto"``, ``"on"`` / ``True`` = that mode with the built-in policy. Otherwise a policy
    mapping, or the path of a YAML file with an ``access:`` block (``configs/access/default.yaml``),
    or ``{policy: <path or mapping>, <key>: <override>}``. The mapping may also carry
    ``principals:`` and ``directory:`` blocks."""
    if cfg is True:
        cfg = "on"
    if cfg is False:
        cfg = "off"
    if isinstance(cfg, AccessController):
        return cfg
    if cfg is None or (isinstance(cfg, str) and cfg in _MODES):
        ac = AccessController(AccessPolicy(mode=cfg or "auto"))
        return None if ac.policy.mode == "off" else ac
    ac = _build(cfg)
    return None if ac.policy.mode == "off" else ac


def _build(cfg: Union[str, Path, Mapping[str, Any]]) -> AccessController:
    principals = channels = None
    if isinstance(cfg, Mapping):
        cfg = dict(cfg)
        principals, channels = cfg.pop("principals", None), cfg.pop("directory", None)
        if "policy" in cfg:  # {policy: path-or-mapping, mode: debug, ...}: overrides on top of a file
            base = AccessPolicy.from_config(cfg.pop("policy"))
            fields = {k: getattr(base, k) for k in AccessPolicy.__dataclass_fields__}
            unknown = set(cfg) - set(fields)
            if unknown:
                raise ValueError(f"unknown access policy keys {sorted(unknown)}")
            return AccessController(AccessPolicy(**{**fields, **cfg}), principals, channels)
    return AccessController(AccessPolicy.from_config(cfg), principals, channels)
