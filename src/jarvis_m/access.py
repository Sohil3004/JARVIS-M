"""Visibility labels on messages.

A label is a clearance level plus a set of compartments. Labels only ever go up:
:func:`join` takes the higher level and the union of the compartments.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, NamedTuple

from .types import Message


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
    raise TypeError(f"acl_compartments must be a list of names, got {value!r}")


def message_label(m: Message) -> Label:
    """The label a message carries in its own metadata: ``meta["acl_level"]`` (an integer
    level) and ``meta["acl_compartments"]`` (a list of names). No metadata means public."""
    level = m.meta.get("acl_level", 0) if m.meta else 0
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"message {m.id}: acl_level must be an integer level, got {level!r}")
    return Label(level, _compartments(m.meta.get("acl_compartments") if m.meta else None))
