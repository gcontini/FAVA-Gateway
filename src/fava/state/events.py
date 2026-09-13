"""Run identity and the append-only event log.

Hooks see **HTTP exchanges, not agent runs**, and FAVA's graphs are per-run, so
run identity has to be recovered before anything else can be trusted. Moving the
gateway off the MCP wire cost us ``Mcp-Session-Id``, but the conversation itself
is a chain: request *N+1* repeats the whole history, so the run is recoverable
from the traffic. :func:`derive_run_id` does that.

Everything observed about a run lands here as an :class:`Event`, and the log is
append-only: events are added, never rewritten or deleted. That is not
bookkeeping tidiness, it is the paper's monotone runtime repair — a decision
made against a prefix of the log stays explicable once the log has grown.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from llm_proxy.chat import ROLE_ASSISTANT, ROLE_TOOL
from llm_proxy.hooks import RequestContext

# Set by a harness that can cooperate. Cleanest source of run identity there is,
# and the only one immune to history truncation.
RUN_ID_HEADER: Final = "x-fava-run-id"

# Roles that make up the stable prefix a run is hashed from. Deliberately not
# `assistant` or `tool`: those grow every turn, and the hash must not.
PREFIX_ROLES: Final = ("system", "developer")


class EventKind(str, Enum):
    """What an :class:`Event` records.

    Deliberately few. Anything derivable from the log — an intent that never
    got a result, say — is computed on demand rather than stored as its own
    event, so the log stays a record of what was *observed*.
    """

    RUN_STARTED = "run_started"
    TOOL_INTENT = "tool_intent"
    TOOL_RESULT = "tool_result"
    RESPONSE_OPAQUE = "response_opaque"
    IR_EXTRACTED = "ir_extracted"


@dataclass(frozen=True)
class Event:
    """One observed fact about a run.

    Attributes:
        event_id: Unique within the run, and stable: ``{run prefix}-{seq}``.
        run_id: The run this belongs to.
        parent_event_id: The event this one answers or descends from — a
            result's intent, an intent's ``run_started``. ``None`` at the root.
        seq: Monotonic position in the run's log, from zero.
        timestamp: Wall-clock seconds when the event was recorded.
        kind: What happened.
        tool_name: Tool involved, when there is one.
        tool_call_id: The provider-assigned id correlating an intent to the
            result the harness reports back on the next request.
        payload: Kind-specific detail. Arguments, result content, the task
            text, the extracted IR.
    """

    event_id: str
    run_id: str
    seq: int
    kind: EventKind
    timestamp: float = field(default_factory=time.time)
    parent_event_id: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


class EventLog:
    """The append-only event sequence for one run.

    Not synchronized. Appending is a list append, and ``seq`` is assigned from
    the current length, so the worst a concurrent observer can cause is two
    events recorded in an order that does not match wall-clock time — never a
    lost or corrupted one.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._events: list[Event] = []
        self._by_call_id: dict[str, Event] = {}
        self._answered: set[str] = set()

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[Event, ...]:
        """Every event recorded so far, in observation order."""
        return tuple(self._events)

    def append(
        self,
        kind: EventKind,
        *,
        parent_event_id: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        """Record one event and return it. The only mutator on this class."""
        seq = len(self._events)
        event = Event(
            event_id=f"{self.run_id[:8]}-{seq:04d}",
            run_id=self.run_id,
            seq=seq,
            kind=kind,
            parent_event_id=parent_event_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            payload=dict(payload or {}),
        )
        self._events.append(event)
        if tool_call_id:
            # First writer wins: an intent is recorded before its result, so
            # this maps a call id to the intent rather than to the answer.
            self._by_call_id.setdefault(tool_call_id, event)
            if kind is EventKind.TOOL_RESULT:
                self._answered.add(tool_call_id)
        return event

    def intent_for(self, tool_call_id: str | None) -> Event | None:
        """The intent event a ``tool_call_id`` belongs to, if it was observed."""
        if not tool_call_id:
            return None
        event = self._by_call_id.get(tool_call_id)
        return event if event is not None and event.kind is EventKind.TOOL_INTENT else None

    def of_kind(self, *kinds: EventKind) -> tuple[Event, ...]:
        """Every event of the given kinds, in observation order."""
        return tuple(event for event in self._events if event.kind in kinds)

    def has_kind(self, kind: EventKind) -> bool:
        """Whether any event of this kind has been recorded."""
        return any(event.kind is kind for event in self._events)

    def has_result_for(self, tool_call_id: str | None) -> bool:
        """Whether a result has already been recorded for this call.

        Every request repeats the whole history, so the same ``role: "tool"``
        message arrives again on every later turn. Callers use this to record a
        result once rather than once per turn.
        """
        return bool(tool_call_id) and tool_call_id in self._answered

    def unanswered_intents(self) -> tuple[Event, ...]:
        """Intents with no matching result: refused, dropped, or still running.

        All three are worth knowing about, and none of them is an event of its
        own — the absence *is* the observation, so it is derived from the log
        rather than recorded into it.
        """
        return tuple(
            event
            for event in self._events
            if event.kind is EventKind.TOOL_INTENT and event.tool_call_id not in self._answered
        )


def derive_run_id(context: RequestContext) -> str | None:
    """Recover the run a request belongs to.

    Two sources, in order of preference:

    1. The ``X-FAVA-Run-Id`` header, when the harness can set one. Cleanest,
       and the only source immune to history rewriting.
    2. A hash of the stable prefix — the system/developer messages plus the
       first user turn — which every request in a run repeats verbatim.

    History truncation and context compaction rewrite that prefix mid-run. The
    hash then changes, and the request is simply treated as a **new run**: a
    fresh graph built from what is still visible is honest, whereas merging two
    prefixes would assert continuity the proxy cannot see.

    Args:
        context: The observed request.

    Returns:
        The run id, or ``None`` when this request is not part of a run — a
        ``GET /v1/models``, or a completion with no prefix to hash.
    """
    header = context.headers.get(RUN_ID_HEADER)
    if header and header.strip():
        return header.strip()

    if not context.is_chat_completion or context.chat is None:
        return None

    prefix = _stable_prefix(context.chat.messages)
    if not prefix:
        return None
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def task_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """The natural-language task, as the IR extractor should see it.

    The same prefix the run id is hashed from, rendered for a model to read
    rather than for a digest. This is the "initial prompt" the paper treats as
    optional context: at this boundary it arrives on every request for free.
    """
    return _stable_prefix(messages, labelled=True)


def _stable_prefix(messages: Sequence[Mapping[str, Any]], *, labelled: bool = False) -> str:
    """Render the system/developer messages plus the first user turn.

    Assistant and tool messages are excluded on purpose: they grow every turn,
    and a prefix that grows identifies a turn rather than a run.
    """
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role in (ROLE_ASSISTANT, ROLE_TOOL):
            continue
        if role in PREFIX_ROLES:
            parts.append(f"{role}: {_text(message)}" if labelled else f"{role}\x00{_text(message)}")
            continue
        if role == "user":
            parts.append(f"user: {_text(message)}" if labelled else f"user\x00{_text(message)}")
            break
    return ("\n\n" if labelled else "\x01").join(part for part in parts if part.strip())


def _text(message: Mapping[str, Any]) -> str:
    """Flatten a message's content, which may be a string or a content-part list."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal content parts: keep the text, drop images and audio.
        return "".join(part.get("text", "") for part in content if isinstance(part, Mapping))
    if content is None:
        return ""
    try:
        return json.dumps(content, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(content)


__all__ = [
    "RUN_ID_HEADER",
    "Event",
    "EventKind",
    "EventLog",
    "derive_run_id",
    "task_text",
]
