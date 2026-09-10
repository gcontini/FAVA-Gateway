"""Extension points for the pass-through relay.

The relay itself never inspects protocol semantics: it calls these hooks
around each exchange and streams bytes regardless of what the hooks do. That
keeps the proxy a drop-in transport while giving FAVA's later stages a single
place to attach.

Current hooks are **observation only** — they cannot alter or block traffic,
which is what makes the proxy safe to deploy before the authorization stage is
implemented. The intended evolution, mirroring the FAVA gateway contract, is::

    observe request -> authorize -> forward or deny -> observe result

at which point :meth:`ProxyHooks.on_request` grows a return value (allow/block/
unknown) and a block short-circuits the relay with a JSON-RPC error. Until
then, every hook is best-effort: an exception inside a hook is logged and
swallowed, because a misbehaving observer must never break the transport.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from mcp_proxy.messages import WireMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestContext:
    """Everything known about one inbound request before it is relayed.

    Attributes:
        messages: JSON-RPC envelopes parsed from the request body. Empty when
            the body carried no parseable envelope (the body is still relayed).
        method: Convenience accessor for the first message's method.
        http_method: The inbound HTTP verb (``POST``, ``GET``, ``DELETE``).
        path: The inbound request path.
        query: The inbound raw query string, empty when absent.
        headers: Lowercased inbound header mapping, filtered to what the relay
            will forward.
        body_size: Length of the buffered request body in bytes.
        session_id: ``Mcp-Session-Id`` when present. In the 2026-07-28
            stateless envelope this is ``None``, so callers must not treat a
            missing session id as an error.
        protocol_version: ``Mcp-Protocol-Version`` when present.
        client: ``(host, port)`` of the inbound connection, or ``None``.
    """

    http_method: str
    path: str
    query: str
    headers: Mapping[str, str]
    body_size: int
    messages: Sequence[WireMessage] = field(default_factory=tuple)
    session_id: str | None = None
    protocol_version: str | None = None
    client: tuple[str, int] | None = None

    @property
    def method(self) -> str | None:
        """Method of the first parsed JSON-RPC envelope, if any."""
        return self.messages[0].method if self.messages else None

    @property
    def tool_name(self) -> str | None:
        """Invoked tool name when this request is a ``tools/call``."""
        for message in self.messages:
            if message.is_tool_call:
                return message.tool_name
        return None

    @property
    def is_effectful(self) -> bool:
        """Whether the request may cause an observable effect at the backend.

        A conservative approximation: the handshake (``initialize`` or the
        2026-07-28 ``server/discover``), ``ping``, and list/get style reads are
        treated as non-effectful; anything else — including unparseable bodies
        and unknown methods — counts as effectful. This is the classification
        FAVA's authorizer will gate on, and it errs toward treating unknown
        traffic as effectful rather than assuming it is safe.
        """
        for message in self.messages:
            method = message.method
            if method is None:
                continue
            if method in _NON_EFFECTFUL_METHODS:
                continue
            return True
        # An unparsed body is opaque, so assume it may have effects.
        return any(message.validation_error for message in self.messages)


@dataclass(frozen=True)
class ResponseContext:
    """Everything known about one backend response after it is relayed.

    Attributes:
        request: The :class:`RequestContext` this response answers.
        status_code: Upstream HTTP status.
        headers: Lowercased response header mapping as relayed to the client.
        content_type: Response content type, or ``None``.
        messages: JSON-RPC envelopes parsed from the response body. For an SSE
            stream this holds every frame's message, in arrival order.
        body_size: Length of the observed response prefix in bytes. This may be
            smaller than what the client received: observation is capped so a
            long-lived stream cannot grow hooks' memory without bound, while
            the relay itself forwards every byte.
        truncated: True when :attr:`body_size` is only a prefix of the real
            response body, so :attr:`messages` may be incomplete.
        stream_closed_early: True when the client disconnected or the relay
            errored mid-stream, so the response is known-incomplete.
    """

    request: RequestContext
    status_code: int
    headers: Mapping[str, str]
    content_type: str | None
    messages: Sequence[WireMessage] = field(default_factory=tuple)
    body_size: int = 0
    truncated: bool = False
    stream_closed_early: bool = False

    @property
    def is_error(self) -> bool:
        """Whether the response reports a failure at HTTP or JSON-RPC level."""
        if self.status_code >= 400:
            return True
        return any(message.jsonrpc_is_error() for message in self.messages)


# Methods that cannot produce an effect at the backend: the connection
# handshake, liveness probes, and read-only catalog/document listings. Both
# envelope eras are covered — the 2025 `initialize` handshake and the
# 2026-07-28 `server/discover` that supersedes it (SDK v2 clients in default
# `auto` negotiate mode send `server/discover`, not `initialize`).
_NON_EFFECTFUL_METHODS = frozenset(
    {
        "initialize",
        "server/discover",
        "ping",
        "tools/list",
        "prompts/list",
        "resources/list",
        "resources/templates/list",
        "subscriptions/listen",
    }
)


@runtime_checkable
class ProxyHooks(Protocol):
    """Observer called around each relayed exchange.

    Implementations must be safe to call concurrently: a single proxy serves
    many client connections at once, and nothing here owns cross-call state.
    All methods are async and must not block for long — they run inline on the
    request path.
    """

    async def on_request(self, context: RequestContext) -> None:
        """Called after the request body is buffered and before it is relayed."""

    async def on_response(self, context: ResponseContext) -> None:
        """Called after the backend response has been fully relayed."""

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Called when relaying failed, so no response reached the client."""


class NullHooks:
    """The default :class:`ProxyHooks`: no observation, no overhead."""

    async def on_request(self, context: RequestContext) -> None:
        """No-op."""

    async def on_response(self, context: ResponseContext) -> None:
        """No-op."""

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """No-op."""


HooksLike: TypeAlias = ProxyHooks | Sequence[ProxyHooks] | None


def normalize_hooks(hooks: HooksLike) -> ProxyHooks:
    """Coerce a hook, a sequence of hooks, or nothing into one hook object.

    Args:
        hooks: A single :class:`ProxyHooks`, a sequence of them, or ``None``.

    Returns:
        :class:`NullHooks` for ``None`` or an empty sequence, the hook itself
        for a single hook, and a failure-isolating :class:`CompositeHooks` for
        several. Passing a single hook avoids the per-call fan-out cost when
        there is nothing to fan out to.
    """
    if hooks is None:
        return NullHooks()
    if isinstance(hooks, Sequence):
        entries = tuple(h for h in hooks if h is not None)
        if not entries:
            return NullHooks()
        if len(entries) == 1:
            return entries[0]
        return CompositeHooks(entries)
    return hooks


class LoggingHooks:
    """Hooks that log a one-line summary of each exchange, for smoke tests."""

    def __init__(self, logger_name: str = "mcp_proxy.access", level: int = logging.INFO) -> None:
        self._log = logging.getLogger(logger_name)
        self._level = level

    async def on_request(self, context: RequestContext) -> None:
        """Log the inbound method, tool name and session id."""
        self._log.log(
            self._level,
            "-> %s %s method=%s tool=%s session=%s bytes=%d",
            context.http_method,
            context.path,
            context.method,
            context.tool_name,
            context.session_id,
            context.body_size,
        )

    async def on_response(self, context: ResponseContext) -> None:
        """Log the upstream status and how many JSON-RPC messages came back."""
        self._log.log(
            self._level,
            "<- %d method=%s messages=%d bytes=%d truncated=%s",
            context.status_code,
            context.request.method,
            len(context.messages),
            context.body_size,
            context.stream_closed_early,
        )

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Log a relay failure with its exception type."""
        self._log.warning("!! %s %s failed: %s: %s", context.http_method, context.path, type(error).__name__, error)


class CompositeHooks:
    """Fan-out to several :class:`ProxyHooks`, isolating each one's failures.

    Hooks may implement a subset of the protocol: a missing method is skipped
    rather than raising, so a duck-typed observer needs only the callbacks it
    cares about.
    """

    def __init__(self, hooks: Sequence[ProxyHooks]) -> None:
        self._hooks = tuple(hooks)

    async def on_request(self, context: RequestContext) -> None:
        """Dispatch to every hook, logging and suppressing individual failures."""
        await self._dispatch("on_request", context)

    async def on_response(self, context: ResponseContext) -> None:
        """Dispatch to every hook, logging and suppressing individual failures."""
        await self._dispatch("on_response", context)

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Dispatch to every hook, logging and suppressing individual failures."""
        await self._dispatch("on_relay_error", context, error)

    async def _dispatch(self, name: str, *args: Any) -> None:
        """Call ``name`` on each hook, skipping those that don't define it."""
        for hook in self._hooks:
            method = getattr(hook, name, None)
            if method is None:
                continue
            await call_swallowing(method, *args)


async def call_swallowing(func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
    """Await ``func(*args)``, turning hook exceptions into log entries.

    A hook is an observer: if it raises, the transport must still work. Only
    :exc:`Exception` is caught, so cancellation propagates and the relay's
    task group stays in control of shutdown.
    """
    if func is None:
        return None
    try:
        return await func(*args)
    except Exception:  # noqa: BLE001 - deliberate: hooks are best-effort observers
        logger.exception("Proxy hook %s raised; continuing relay", getattr(func, "__qualname__", func))
        return None
