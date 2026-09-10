"""Extension points for the pass-through relay.

The relay itself never inspects semantics: it calls these hooks around each
exchange and streams bytes regardless of what the hooks do. That keeps the proxy
a drop-in transport while giving FAVA's later stages a single place to attach.

Current hooks are **observation only** — nothing a hook does can alter or block
traffic, which is what makes the proxy safe to deploy before the authorization
stage exists. The intended evolution, mirroring the FAVA gateway contract, is::

    observe request -> observe response -> authorize -> release or deny

Note where authorization lands. On an MCP wire the decision point is the
*request*, because that is where a tool invocation appears. Here it is the
**response**: the model's tool-call intents arrive from the provider, and the
harness dispatches them only after the response reaches it. So
:meth:`ProxyHooks.on_response` — not ``on_request`` — is the seam that will
grow an allow/block return value.

Until then every hook is best-effort: an exception inside a hook is logged and
swallowed, because a misbehaving observer must never break the transport.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from llm_proxy.chat import ChatRequest, ChatResponse, ToolIntent, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestContext:
    """Everything known about one inbound request before it is relayed.

    Attributes:
        http_method: The inbound HTTP verb.
        path: The inbound request path as the client sent it.
        upstream_suffix: The portion of the path forwarded upstream, i.e. the
            path with the proxy's mount prefix removed.
        query: The inbound raw query string, empty when absent.
        headers: Lowercased inbound header mapping. Carries the client's
            credential — never log it unredacted, see
            :func:`llm_proxy.headers.redact`.
        body_size: Length of the buffered request body in bytes.
        chat: The parsed Chat Completions view, or ``None`` when the request
            carried no body (a ``GET /v1/models``, say). A body that failed to
            parse yields a view with ``validation_error`` set, not ``None``.
        client: ``(host, port)`` of the inbound connection, or ``None``.
    """

    http_method: str
    path: str
    upstream_suffix: str
    query: str
    headers: Mapping[str, str]
    body_size: int
    chat: ChatRequest | None = None
    client: tuple[str, int] | None = None

    @property
    def is_chat_completion(self) -> bool:
        """Whether this request is a completion call rather than a side endpoint."""
        return self.chat is not None and self.upstream_suffix.rstrip("/").endswith("/chat/completions")

    @property
    def model(self) -> str | None:
        """Requested model name, when the body named one."""
        return self.chat.model if self.chat else None

    @property
    def streaming(self) -> bool:
        """Whether the client asked for a streamed response."""
        return bool(self.chat and self.chat.stream)

    @property
    def declared_tools(self) -> Sequence[ToolSpec]:
        """The tool catalog the harness declared on this request."""
        return self.chat.tools if self.chat else ()

    @property
    def tool_results(self) -> Sequence[ToolResult]:
        """Results of previously dispatched tool calls carried in the history."""
        return self.chat.tool_results if self.chat else ()


@dataclass(frozen=True)
class ResponseContext:
    """Everything known about one upstream response after it is relayed.

    This is the interception point that matters: :attr:`tool_intents` holds what
    the model asked the harness to run, observed before the harness has run it.

    Attributes:
        request: The :class:`RequestContext` this response answers.
        status_code: Upstream HTTP status.
        headers: Lowercased response header mapping as relayed to the client.
        content_type: Response content type, or ``None``.
        chat: The parsed completion view, or ``None`` when nothing was parsed.
        body_size: Total response bytes observed. The relay forwarded every one
            of them; only the retained *sample* is bounded.
        stream_closed_early: True when the client disconnected or the relay
            errored mid-stream, so the response is known-incomplete.
    """

    request: RequestContext
    status_code: int
    headers: Mapping[str, str]
    content_type: str | None
    chat: ChatResponse | None = None
    body_size: int = 0
    stream_closed_early: bool = False

    @property
    def is_error(self) -> bool:
        """Whether the provider reported a failure."""
        return self.status_code >= 400

    @property
    def tool_intents(self) -> Sequence[ToolIntent]:
        """Tool calls the model requested, with streamed fragments reassembled."""
        return self.chat.tool_intents if self.chat else ()

    @property
    def has_tool_intents(self) -> bool:
        """Whether at least one tool call was positively observed."""
        return bool(self.tool_intents)

    @property
    def intents_unknown(self) -> bool:
        """Whether the response was too opaque to rule tool calls out.

        True when a successful completion could not be parsed, or when a stream
        ended without its terminating ``[DONE]`` — in both cases the model may
        have asked for a call the proxy failed to see.
        """
        if self.is_error or not self.request.is_chat_completion:
            return False
        if self.chat is None or self.chat.validation_error is not None:
            return True
        if self.stream_closed_early:
            return True
        return self.chat.is_stream and not self.chat.complete

    @property
    def may_dispatch_tools(self) -> bool:
        """The conservative gate FAVA's authorizer will act on.

        True when tool calls were observed **or** when the proxy could not tell.
        Unknown traffic is never assumed safe: an authorizer that only checked
        :attr:`has_tool_intents` would wave through exactly the responses it
        failed to understand.
        """
        return self.has_tool_intents or self.intents_unknown

    @property
    def finish_reasons(self) -> Sequence[str | None]:
        """Finish reason per choice, in choice-index order."""
        return self.chat.finish_reasons if self.chat else ()


@runtime_checkable
class ProxyHooks(Protocol):
    """Observer called around each relayed exchange.

    Implementations must be safe to call concurrently: a single proxy serves
    many client connections at once, and nothing here owns cross-call state.
    All methods are async and must not block for long — they run inline on the
    request path, and for a streamed response ``on_response`` runs after the
    last byte has already reached the client.
    """

    async def on_request(self, context: RequestContext) -> None:
        """Called after the request body is buffered and before it is relayed."""

    async def on_response(self, context: ResponseContext) -> None:
        """Called after the upstream response has been fully relayed."""

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Called when relaying failed, so no complete response reached the client."""


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
        several. Passing a single hook through avoids the per-call fan-out cost
        when there is nothing to fan out to.
    """
    if hooks is None:
        return NullHooks()
    if isinstance(hooks, Sequence):
        entries = tuple(hook for hook in hooks if hook is not None)
        if not entries:
            return NullHooks()
        if len(entries) == 1:
            return entries[0]
        return CompositeHooks(entries)
    return hooks


class LoggingHooks:
    """Hooks that log a one-line summary of each exchange, for smoke tests.

    Credentials never appear: only header *names* the relay chose to forward
    are ever summarized, and values are not logged at all.
    """

    def __init__(self, logger_name: str = "llm_proxy.access", level: int = logging.INFO) -> None:
        self._log = logging.getLogger(logger_name)
        self._level = level

    async def on_request(self, context: RequestContext) -> None:
        """Log the model, history size, declared tools and returned results."""
        self._log.log(
            self._level,
            "-> %s %s model=%s messages=%d tools=%d tool_results=%d stream=%s bytes=%d",
            context.http_method,
            context.path,
            context.model,
            context.chat.message_count if context.chat else 0,
            len(context.declared_tools),
            len(context.tool_results),
            context.streaming,
            context.body_size,
        )

    async def on_response(self, context: ResponseContext) -> None:
        """Log the status and, crucially, the tool calls the model asked for."""
        intents = context.tool_intents
        summary = ",".join(f"{intent.name}({len(intent.arguments_json)}B)" for intent in intents) or "-"
        self._log.log(
            self._level,
            "<- %d model=%s intents=%d [%s] finish=%s unknown=%s bytes=%d",
            context.status_code,
            context.chat.model if context.chat else None,
            len(intents),
            summary,
            ",".join(str(reason) for reason in context.finish_reasons) or "-",
            context.intents_unknown,
            context.body_size,
        )

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Log a relay failure with its exception type."""
        self._log.warning(
            "!! %s %s failed: %s: %s",
            context.http_method,
            context.path,
            type(error).__name__,
            error,
        )


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
            await dispatch(hook, name, *args)


async def dispatch(hooks: ProxyHooks, name: str, *args: Any) -> None:
    """Invoke one callback on a hook object, skipping it when unimplemented.

    Hooks may implement a subset of the protocol. That tolerance has to live
    here rather than in :class:`CompositeHooks`, because
    :func:`normalize_hooks` passes a lone hook straight through: routing every
    call through this helper is what stops "partial hooks are allowed" from
    silently meaning "allowed only when you register two of them".
    """
    method = getattr(hooks, name, None)
    if method is None:
        return
    await call_swallowing(method, *args)


async def call_swallowing(func: Callable[..., Awaitable[Any]] | None, *args: Any) -> Any:
    """Await ``func(*args)``, turning hook exceptions into log entries.

    A hook is an observer: if it raises, the transport must still work. Only
    :exc:`Exception` is caught, so cancellation propagates and the relay's task
    group stays in control of shutdown.
    """
    if func is None:
        return None
    try:
        return await func(*args)
    except Exception:  # noqa: BLE001 - deliberate: hooks are best-effort observers
        logger.exception("Proxy hook %s raised; continuing relay", getattr(func, "__qualname__", func))
        return None


__all__ = [
    "CompositeHooks",
    "HooksLike",
    "LoggingHooks",
    "NullHooks",
    "ProxyHooks",
    "RequestContext",
    "ResponseContext",
    "call_swallowing",
    "dispatch",
    "normalize_hooks",
]
