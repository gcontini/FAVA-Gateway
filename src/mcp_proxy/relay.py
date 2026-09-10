"""The forwarding relay: a byte-exact MCP reverse proxy over Streamable HTTP.

This module is the only component that touches the wire. It holds no protocol
state and never rewrites a message: it buffers an inbound request, relays it to
the backend, streams the backend's response straight back to the client, and
notifies :class:`~mcp_proxy.hooks.ProxyHooks` around the exchange.

Design notes:

* **Raw ASGI, not Starlette.** A framework would re-encode bodies and manage
  its own headers; relaying at the ASGI level keeps every byte, status code and
  header the backend produced.
* **Sessions stay where they belong.** ``Mcp-Session-Id`` is relayed in both
  directions untouched, so the client's session is with the *backend*, not with
  the proxy. The proxy can therefore be restarted or scaled without dropping
  agent sessions.
* **Both envelope eras work.** The 2025-era stateful flow (session id, GET
  stream, resumable ``Last-Event-ID``) and the stateless ``2026-07-28`` header
  envelope are both just bytes here; the proxy relays whichever the client uses.
* **Observation never mutates.** Request/response bodies are parsed for hooks
  only; a parse failure still results in a faithful forward, and a raising hook
  is logged rather than propagated.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import anyio
import httpx2
from starlette.types import Message, Receive, Scope, Send

from mcp.types.jsonrpc import INTERNAL_ERROR

from mcp_proxy.config import ProxySettings
from mcp_proxy.headers import forward_request_headers, forward_response_headers
from mcp_proxy.hooks import (
    ProxyHooks,
    RequestContext,
    ResponseContext,
    call_swallowing,
    normalize_hooks,
)
from mcp_proxy.messages import parse_response_body, parse_wire_messages

logger = logging.getLogger(__name__)

# Default observation cap for streamed responses. An SSE stream may run for
# minutes and grow without bound, so hooks see only a prefix; the relay itself
# is unaffected and still forwards every byte.
DEFAULT_OBSERVE_LIMIT: Final = 256 * 1024

# Hop-by-hop and server-owned headers are handled in `mcp_proxy.headers`.
_JSON_CONTENT_TYPE: Final = b"application/json"

# HTTP methods that may carry a request body. The standalone SSE `GET` stream
# and session-terminating `DELETE` must not gain one.
_BODY_METHODS: Final = frozenset({"POST", "PUT", "PATCH"})


class ForwardingProxy:
    """An ASGI application that relays MCP traffic to a single backend endpoint.

    Call instances directly as an ASGI app, or wrap them with
    :func:`mcp_proxy.app.create_proxy_app` for lifespan and routing conveniences.

    Attributes:
        settings: The immutable :class:`~mcp_proxy.config.ProxySettings`.
        hooks: The observer invoked around each exchange.
    """

    def __init__(
        self,
        settings: ProxySettings,
        *,
        hooks: ProxyHooks | Sequence[ProxyHooks] | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create a relay.

        Args:
            settings: Backend URL, mount path and relay limits.
            hooks: One hook, a sequence of hooks (composed and failure-isolated),
                or ``None`` for no observation.
            http_client: Optional pre-built upstream client. When ``None`` the
                relay builds one on ASGI lifespan startup and closes it on
                shutdown; passing a client moves that lifecycle to the caller.
        """
        self.settings = settings
        self.hooks: ProxyHooks = normalize_hooks(hooks)
        self._client: httpx2.AsyncClient | None = http_client
        self._own_client = http_client is None
        self._observe_limit = max(0, min(settings.max_body_size, DEFAULT_OBSERVE_LIMIT))

    # -- ASGI plumbing ---------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Dispatch one ASGI event stream: lifespan, HTTP, or neither."""
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._handle_lifespan(receive, send)
            return
        if scope_type != "http":
            logger.debug("Ignoring ASGI scope type %r", scope_type)
            return
        await self.handle_http(scope, receive, send)

    async def _handle_lifespan(self, receive: Receive, send: Send) -> None:
        """Own the upstream HTTP client for the process lifetime."""
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await self.startup()
                except Exception as exc:  # pragma: no cover - defensive
                    await send({"type": "lifespan.startup.failed", "message": str(exc)})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                try:
                    await self.shutdown()
                except Exception as exc:  # pragma: no cover - defensive
                    await send({"type": "lifespan.shutdown.failed", "message": str(exc)})
                    return
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def startup(self) -> None:
        """Create the upstream HTTP client when the relay owns it."""
        if self._client is None and self._own_client:
            self._client = self._build_client()
            logger.info("Upstream client ready for %s", self.settings.backend_url)

    async def shutdown(self) -> None:
        """Close the upstream HTTP client when the relay owns it."""
        client, self._client = self._client, None
        if client is not None and self._own_client:
            await client.aclose()
            logger.info("Upstream client closed")

    def _build_client(self) -> httpx2.AsyncClient:
        """Build an upstream client with MCP-appropriate timeouts."""
        return httpx2.AsyncClient(
            # The 300s read timeout mirrors the MCP SDK: a streamable HTTP server
            # may hold a response stream open far longer than a normal request.
            timeout=httpx2.Timeout(self.settings.timeout, read=self.settings.sse_read_timeout),
            # Redirects belong to the client and the backend, not to the relay:
            # forward them instead of silently following them upstream.
            follow_redirects=False,
        )

    # -- HTTP relay ------------------------------------------------------

    async def handle_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Relay one HTTP exchange, or answer a path the proxy owns locally."""
        http_method: str = scope["method"]
        path: str = scope.get("path", "/")
        query: str = scope.get("query_string", b"").decode("latin-1")

        health_path = self.settings.health_path
        if health_path is not None and _norm(path) == _norm(health_path):
            await _send_local_json(
                send,
                200,
                {
                    "status": "ok",
                    "backend_url": self.settings.backend_url,
                    "mount_path": self.settings.mount_path,
                    "health_path": path,
                    "upstream_client_ready": self._client is not None,
                },
            )
            return

        if not self._path_accepted(path):
            await _send_local_json(
                send,
                404,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": INTERNAL_ERROR,
                        "message": f"No MCP endpoint at {path}",
                    },
                },
                extra_message=f"this proxy serves {self.settings.mount_path!r}",
            )
            return

        headers = _scope_headers(scope)
        try:
            body = await _read_body(receive, limit=self.settings.max_body_size)
        except _BodyTooLarge:
            await _send_local_json(
                send,
                413,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": INTERNAL_ERROR,
                        "message": f"Request body exceeds {self.settings.max_body_size} bytes",
                    },
                },
            )
            return

        context = self._build_request_context(scope, http_method, path, query, headers, body)
        await call_swallowing(self.hooks.on_request, context)

        client = self._client
        if client is not None:
            await self._relay(context, client, body, receive, send)
            return

        # No lifespan ran (a direct `__call__`, as in tests), so the relay builds
        # a client for this exchange only and closes it afterwards.
        ephemeral = self._build_client()
        try:
            await self._relay(context, ephemeral, body, receive, send)
        finally:
            await ephemeral.aclose()

    def _path_accepted(self, path: str) -> bool:
        """Whether an inbound path may be relayed to the backend."""
        mount = self.settings.mount_path
        if mount is None:
            return True
        return _norm(path) == _norm(mount)

    def _build_request_context(
        self,
        scope: Scope,
        http_method: str,
        path: str,
        query: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> RequestContext:
        """Assemble the :class:`RequestContext` handed to hooks."""
        client = scope.get("client")
        return RequestContext(
            http_method=http_method,
            path=path,
            query=query,
            headers=headers,
            body_size=len(body) if body else 0,
            messages=parse_wire_messages(body, headers=headers, kind="request"),
            session_id=headers.get("mcp-session-id"),
            protocol_version=headers.get("mcp-protocol-version"),
            client=(client[0], client[1]) if client else None,
        )

    async def _relay(
        self,
        context: RequestContext,
        client: httpx2.AsyncClient,
        body: bytes | None,
        receive: Receive,
        send: Send,
    ) -> None:
        """Relay one exchange, cancelling it when the client goes away.

        Two tasks race inside a task group: the disconnect watcher cancels the
        scope when the client's connection dies, and the relay streams the
        response. Errors are captured on :class:`_ResponseState` rather than
        raised, because an exception escaping a task becomes an ExceptionGroup
        and a client disconnect must not be reported as a failure. The exchange
        is always terminated exactly once, then hooks are notified.

        The watcher is cancelled once the relay finishes: it awaits
        ``receive()``, which only returns on a disconnect, so without the
        explicit cancel the task group would wait forever on a healthy
        exchange.
        """
        state = _ResponseState()
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._watch_disconnect, context, receive, tg.cancel_scope)
                try:
                    await self._relay_stream(client, context, body, send, state)
                finally:
                    tg.cancel_scope.cancel()
        finally:
            if state.notify is not None:
                await call_swallowing(self.hooks.on_response, state.notify)

    async def _watch_disconnect(
        self,
        context: RequestContext,
        receive: Receive,
        cancel_scope: anyio.CancelScope,
    ) -> None:
        """Cancel the relay as soon as the client drops the connection."""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                logger.debug("Client disconnected during %s %s", context.http_method, context.path)
                cancel_scope.cancel()
                return

    async def _relay_stream(
        self,
        client: httpx2.AsyncClient,
        context: RequestContext,
        body: bytes | None,
        send: Send,
        state: _ResponseState,
    ) -> None:
        """Send the request upstream and stream the response back byte-for-byte."""
        request = self._build_upstream_request(client, context, body)

        try:
            state.upstream = await client.send(request, stream=True)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exc:  # reachability, DNS, TLS, timeout
            logger.warning("Cannot reach backend at %s: %s", self.settings.backend_url, exc)
            await self._fail(context, send, state, exc, status_code=502)
            return

        upstream = state.upstream
        assert upstream is not None  # noqa: S101 - set or returned above
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": upstream.status_code,
                    "headers": forward_response_headers(upstream.headers.items()),
                }
            )
            state.started = True

            async for chunk in upstream.aiter_raw():
                if not chunk:
                    continue
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
                state.observe(chunk, self._observe_limit)
            # `state.completed` is deliberately left unset here: `terminate` in
            # the `finally` below owns sending the final `more_body: False`
            # message, and marking completion early would make it skip that and
            # leave the ASGI response unfinished.
        except anyio.get_cancelled_exc_class():
            state.client_gone = True
            raise
        except Exception as exc:  # mid-stream transport error
            logger.warning("Relay failed mid-response for %s %s: %s", context.http_method, context.path, exc)
            state.error = exc
        finally:
            # Shielded: the response must be closed and the exchange ended even
            # while the enclosing task group is being cancelled.
            with anyio.CancelScope(shield=True):
                await upstream.aclose()
                await state.terminate(send)
                state.notify = self._response_context(context, state)
                if not state.started and state.error is not None:
                    await self._fail(context, send, state, state.error, status_code=502)

    def _response_context(self, context: RequestContext, state: _ResponseState) -> ResponseContext | None:
        """Build the :class:`ResponseContext` hooks receive, or None if no response."""
        upstream = state.upstream
        if upstream is None:
            return None
        lowered = {name.lower(): value for name, value in upstream.headers.items()}
        content_type = lowered.get("content-type")
        payload = b"".join(state.observed)
        # A truncated capture is not parsable as a whole JSON body, but the
        # completed frames of an SSE stream are still worth surfacing.
        is_stream = content_type is not None and "text/event-stream" in content_type.lower()
        messages = parse_response_body(payload, content_type=content_type) if is_stream or not state.truncated else ()
        return ResponseContext(
            request=context,
            status_code=upstream.status_code,
            headers=lowered,
            content_type=content_type,
            messages=messages,
            body_size=len(payload),
            truncated=state.truncated,
            stream_closed_early=state.client_gone or state.error is not None,
        )

    async def _fail(
        self,
        context: RequestContext,
        send: Send,
        state: _ResponseState,
        error: BaseException,
        *,
        status_code: int,
    ) -> None:
        """Answer a request that could not be relayed with a JSON-RPC error.

        The request id is echoed from the parsed envelope so an MCP client can
        correlate the failure with the call it made. Nothing is written when the
        response already started, since the status line is on the wire.
        """
        await call_swallowing(self.hooks.on_relay_error, context, error)
        if state.started:
            return
        request_id = next((message.id for message in context.messages if message.is_request), None)
        # `_send_local_json` writes both the start and the terminating body
        # message, so mark the exchange ended for `terminate`'s idempotence.
        state.started = True
        state.completed = True
        try:
            await _send_local_json(
                send,
                status_code,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": INTERNAL_ERROR,
                        "message": f"Bad gateway relaying to backend: {type(error).__name__}",
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 - the exchange may already be dead
            logger.debug("Could not write gateway error response: %s", exc)

    def _build_upstream_request(
        self,
        client: httpx2.AsyncClient,
        context: RequestContext,
        body: bytes | None,
    ) -> httpx2.Request:
        """Build the upstream request from the relayed bytes and headers.

        The backend URL's own path always wins: the proxy serves one MCP
        endpoint, so the inbound path is not appended. The inbound query string
        is preserved when :attr:`ProxySettings.query_passthrough` is set.
        """
        url = self.settings.backend_url
        if self.settings.query_passthrough and context.query:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{context.query}"

        headers = forward_request_headers(
            context.headers.items(),
            host=httpx2.URL(url).netloc.decode("ascii"),
            extra=dict(self.settings.extra_headers),
            trust_client_headers=self.settings.trust_client_headers,
        )
        # A `GET` for the standalone SSE stream and a session-terminating
        # `DELETE` arrive with no body and must not gain one upstream.
        content = body if context.http_method.upper() in _BODY_METHODS and body else None
        return client.build_request(context.http_method, url, content=content, headers=headers)


@dataclass
class _ResponseState:
    """Mutable bookkeeping for one in-flight relayed response.

    Tracks what has been sent to the client so the relay terminates the ASGI
    exchange exactly once — on success, on a mid-stream error, and on client
    disconnect alike — and accumulates the bounded observation prefix that hooks
    are shown.

    Attributes:
        upstream: The opened backend response.
        started: ``http.response.start`` reached the client, so the status line
            is committed and a locally generated error can no longer be sent.
        completed: The response body was fully relayed.
        client_gone: The client disconnected mid-exchange.
        error: The relay failure, when one occurred.
        truncated: The observation prefix dropped bytes (the relay did not).
        observed: Chunks of the observed prefix.
        notify: The :class:`ResponseContext` awaiting hook dispatch.
    """

    upstream: httpx2.Response | None = None
    started: bool = False
    completed: bool = False
    client_gone: bool = False
    error: BaseException | None = None
    truncated: bool = False
    observed: list[bytes] = field(default_factory=list)
    notify: ResponseContext | None = None
    _observed_size: int = 0

    def observe(self, chunk: bytes, limit: int) -> None:
        """Record up to ``limit`` bytes of a streamed response for hooks.

        The relay's own forwarding is untouched by this cap, which exists only
        so a long-lived SSE stream cannot grow hooks' memory without bound.
        """
        if self._observed_size >= limit:
            self.truncated = True
            return
        keep = limit - self._observed_size
        self.observed.append(chunk[:keep])
        self._observed_size += min(len(chunk), keep)
        self.truncated = self.truncated or len(chunk) > keep

    async def terminate(self, send: Send) -> None:
        """Send the final ASGI body message when a response was started."""
        if not self.started or self.completed:
            return
        try:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            self.completed = True
        except Exception as exc:  # noqa: BLE001 - the exchange may already be dead
            logger.debug("Could not terminate relayed response: %s", exc)


# -- module-level helpers --------------------------------------------------


class _BodyTooLarge(Exception):
    """Raised while buffering an inbound body that exceeds the configured limit."""


def _norm(path: str) -> str:
    """Normalize a path for comparison by stripping a trailing slash.

    ``/mcp`` and ``/mcp/`` must behave identically: the SDK's own session
    manager treats them as the same endpoint, and a proxy that disagreed would
    send one of them to its 404 branch.
    """
    return path.rstrip("/") or "/"


def _scope_headers(scope: Scope) -> dict[str, str]:
    """Flatten ASGI header pairs into a lowercase-valued mapping."""
    headers: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers") or ():
        headers[raw_name.decode("latin-1").lower()] = raw_value.decode("latin-1")
    return headers


async def _read_body(receive: Receive, *, limit: int) -> bytes | None:
    """Buffer the inbound body, rejecting anything over ``limit`` bytes.

    Returns ``None`` when the request carried no body at all (as distinguished
    from an empty body), because a relayed ``GET``/``DELETE`` must not gain one.
    """
    chunks: list[bytes] = []
    total = 0
    saw_body = False
    while True:
        message: Message = await receive()
        if message["type"] != "http.request":
            # A disconnect before the body arrived: relay nothing.
            return None
        saw_body = True
        chunk: bytes = message.get("body") or b""
        if chunk:
            total += len(chunk)
            if total > limit:
                raise _BodyTooLarge
            chunks.append(chunk)
        if not message.get("more_body", False):
            break
    return b"".join(chunks) if saw_body else None


async def _send_local_json(
    send: Send,
    status_code: int,
    payload: Mapping[str, Any],
    *,
    extra_message: str | None = None,
) -> None:
    """Write a locally generated JSON response.

    Args:
        send: The ASGI send callable.
        status_code: HTTP status to report.
        payload: JSON-serializable body.
        extra_message: Appended to a JSON-RPC error message when present.
    """
    body_payload = dict(payload)
    if extra_message is not None and isinstance(body_payload.get("error"), dict):
        error = dict(body_payload["error"])
        error["message"] = f"{error.get('message', '')} ({extra_message})"
        body_payload["error"] = error

    body = json.dumps(body_payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", _JSON_CONTENT_TYPE),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
