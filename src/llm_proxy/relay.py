"""The forwarding relay: a byte-exact reverse proxy for an OpenAI-compatible API.

This module is the only component that touches the wire. It holds no protocol
state and never rewrites a message: it buffers an inbound request, relays it
upstream, streams the provider's response straight back to the client, and
notifies :class:`~llm_proxy.hooks.ProxyHooks` around the exchange.

Design notes:

* **Raw ASGI, not Starlette.** A framework would re-encode bodies and manage its
  own headers; relaying at the ASGI level keeps every byte, status code and
  header the provider produced. That fidelity is the point — an agent harness
  must not be able to tell the proxy is there.
* **The path is a prefix, not an endpoint.** The integration is a ``base_url``
  swap, so everything under the mount prefix routes: ``/chat/completions``,
  ``/models``, ``/embeddings``. The inbound suffix is appended to the configured
  upstream base.
* **Observation is incremental.** Response bytes are fed to a
  :class:`~llm_proxy.chat.ResponseObserver` as they stream past, so a long
  answer can never push the model's tool calls out of view. The relay's own
  forwarding is unaffected by what the observer does or fails to do.
* **Observation never mutates.** A parse failure still results in a faithful
  forward, and a raising hook is logged rather than propagated.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import anyio
import httpx
from starlette.types import Message, Receive, Scope, Send

from llm_proxy.chat import ChatResponse, ResponseObserver, parse_chat_request
from llm_proxy.config import ProxySettings
from llm_proxy.headers import forward_request_headers, forward_response_headers
from llm_proxy.hooks import (
    ProxyHooks,
    RequestContext,
    ResponseContext,
    dispatch,
    normalize_hooks,
)

logger = logging.getLogger(__name__)

_JSON_CONTENT_TYPE: Final = b"application/json"

# HTTP methods that may carry a request body.
_BODY_METHODS: Final = frozenset({"POST", "PUT", "PATCH"})


class ForwardingProxy:
    """An ASGI application that relays LLM API traffic to a single upstream.

    Call instances directly as an ASGI app, or wrap them with
    :func:`llm_proxy.app.create_proxy_app` for lifespan and CORS conveniences.

    Attributes:
        settings: The immutable :class:`~llm_proxy.config.ProxySettings`.
        hooks: The observer invoked around each exchange.
    """

    def __init__(
        self,
        settings: ProxySettings,
        *,
        hooks: ProxyHooks | Sequence[ProxyHooks] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a relay.

        Args:
            settings: Upstream base URL, mount prefix and relay limits.
            hooks: One hook, a sequence of hooks (composed and failure-isolated),
                or ``None`` for no observation.
            http_client: Optional pre-built upstream client. When ``None`` the
                relay builds one on ASGI lifespan startup and closes it on
                shutdown; passing a client moves that lifecycle to the caller.
        """
        self.settings = settings
        self.hooks: ProxyHooks = normalize_hooks(hooks)
        self._client: httpx.AsyncClient | None = http_client
        self._own_client = http_client is None

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
            logger.info("Upstream client ready for %s", self.settings.upstream_url)

    async def shutdown(self) -> None:
        """Close the upstream HTTP client when the relay owns it."""
        client, self._client = self._client, None
        if client is not None and self._own_client:
            await client.aclose()
            logger.info("Upstream client closed")

    def _build_client(self) -> httpx.AsyncClient:
        """Build an upstream client with completion-appropriate timeouts."""
        return httpx.AsyncClient(
            # The read timeout is generous: a model may pause a long time
            # between tokens, and a tool-calling turn holds the stream open
            # throughout.
            timeout=httpx.Timeout(self.settings.timeout, read=self.settings.stream_read_timeout),
            # Redirects belong to the client and the provider, not to the relay:
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
                    "upstream_url": self.settings.upstream_url,
                    "mount_prefix": self.settings.mount_prefix,
                    "health_path": path,
                    "upstream_client_ready": self._client is not None,
                },
            )
            return

        suffix = self._upstream_suffix(path)
        if suffix is None:
            await _send_api_error(
                send,
                404,
                message=f"No API endpoint at {path}; this proxy serves "
                f"{self.settings.mount_prefix!r}",
                error_type="invalid_request_error",
                code="unknown_path",
            )
            return

        headers = _scope_headers(scope)
        try:
            body = await _read_body(receive, limit=self.settings.max_body_size)
        except _BodyTooLarge:
            await _send_api_error(
                send,
                413,
                message=f"Request body exceeds {self.settings.max_body_size} bytes",
                error_type="invalid_request_error",
                code="request_too_large",
            )
            return

        context = RequestContext(
            http_method=http_method,
            path=path,
            upstream_suffix=suffix,
            query=query,
            headers=headers,
            body_size=len(body) if body else 0,
            chat=parse_chat_request(body),
            client=_peer(scope),
        )
        await dispatch(self.hooks, "on_request", context)

        client = self._client
        if client is not None:
            await self._relay(context, client, body, receive, send)
            return

        # No lifespan ran (a direct `__call__`, as in tests), so the relay
        # builds a client for this exchange only and closes it afterwards.
        ephemeral = self._build_client()
        try:
            await self._relay(context, ephemeral, body, receive, send)
        finally:
            await ephemeral.aclose()

    def _upstream_suffix(self, path: str) -> str | None:
        """The path portion to append upstream, or ``None`` if unmounted.

        With a mount prefix of ``/v1``, an inbound ``/v1/chat/completions``
        yields ``/chat/completions``. With no prefix the whole path is
        forwarded, in which case the configured upstream base should not
        already repeat it.
        """
        prefix = self.settings.mount_prefix
        if prefix is None:
            return path
        if path == prefix:
            return ""
        if path.startswith(f"{prefix}/"):
            return path[len(prefix) :]
        return None

    async def _relay(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
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
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    self._watch_disconnect, context, receive, task_group.cancel_scope
                )
                try:
                    await self._relay_stream(client, context, body, send, state)
                finally:
                    task_group.cancel_scope.cancel()
        finally:
            if state.notify is not None:
                await dispatch(self.hooks, "on_response", state.notify)

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
        client: httpx.AsyncClient,
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
            logger.warning("Cannot reach upstream at %s: %s", self.settings.upstream_url, exc)
            await self._fail(context, send, state, exc, status_code=502)
            return

        upstream = state.upstream
        assert upstream is not None  # noqa: S101 - set or returned above
        state.observer = ResponseObserver(content_type=upstream.headers.get("content-type"))
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
                # Forward first, observe second: observation must never sit
                # between the provider's bytes and the client.
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
                state.observe(chunk)
            # `state.completed` is deliberately left unset here: `terminate` in
            # the `finally` below owns sending the final `more_body: False`
            # message, and marking completion early would make it skip that and
            # leave the ASGI response unfinished.
        except anyio.get_cancelled_exc_class():
            state.client_gone = True
            raise
        except Exception as exc:  # mid-stream transport error
            logger.warning(
                "Relay failed mid-response for %s %s: %s", context.http_method, context.path, exc
            )
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

    def _response_context(
        self, context: RequestContext, state: _ResponseState
    ) -> ResponseContext | None:
        """Build the :class:`ResponseContext` hooks receive, or None if no response."""
        upstream = state.upstream
        if upstream is None:
            return None
        lowered = {name.lower(): value for name, value in upstream.headers.items()}
        chat: ChatResponse | None = state.observer.result() if state.observer is not None else None
        return ResponseContext(
            request=context,
            status_code=upstream.status_code,
            headers=lowered,
            content_type=lowered.get("content-type"),
            chat=chat,
            body_size=state.observer.observed_size if state.observer else 0,
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
        """Answer a request that could not be relayed with an API-shaped error.

        The body mirrors the provider's own error envelope so a client SDK
        surfaces it as an API error rather than failing to parse the response.
        Nothing is written when the response already started, since the status
        line is on the wire.
        """
        await dispatch(self.hooks, "on_relay_error", context, error)
        if state.started:
            return
        # `_send_api_error` writes both the start and the terminating body
        # message, so mark the exchange ended for `terminate`'s idempotence.
        state.started = True
        state.completed = True
        try:
            await _send_api_error(
                send,
                status_code,
                message=f"Bad gateway relaying to upstream: {type(error).__name__}",
                error_type="upstream_error",
                code="bad_gateway",
            )
        except Exception as exc:  # noqa: BLE001 - the exchange may already be dead
            logger.debug("Could not write gateway error response: %s", exc)

    def _build_upstream_request(
        self,
        client: httpx.AsyncClient,
        context: RequestContext,
        body: bytes | None,
    ) -> httpx.Request:
        """Build the upstream request from the relayed bytes and headers."""
        url = f"{self.settings.upstream_url}{context.upstream_suffix}"
        if self.settings.query_passthrough and context.query:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{context.query}"

        headers = forward_request_headers(
            context.headers.items(),
            host=httpx.URL(url).netloc.decode("ascii"),
            extra=dict(self.settings.extra_headers),
            trust_client_headers=self.settings.trust_client_headers,
            forward_client_auth=self.settings.forward_client_auth,
        )
        # A bodyless `GET` (e.g. `/v1/models`) must not gain one upstream.
        content = body if context.http_method.upper() in _BODY_METHODS and body else None
        return client.build_request(context.http_method, url, content=content, headers=headers)


@dataclass
class _ResponseState:
    """Mutable bookkeeping for one in-flight relayed response.

    Tracks what has been sent to the client so the relay terminates the ASGI
    exchange exactly once — on success, on a mid-stream error, and on client
    disconnect alike — and holds the incremental observer.

    Attributes:
        upstream: The opened upstream response.
        started: ``http.response.start`` reached the client, so the status line
            is committed and a locally generated error can no longer be sent.
        completed: The response body was fully relayed.
        client_gone: The client disconnected mid-exchange.
        error: The relay failure, when one occurred.
        observer: Parses the response as it streams past.
        notify: The :class:`ResponseContext` awaiting hook dispatch.
    """

    upstream: httpx.Response | None = None
    started: bool = False
    completed: bool = False
    client_gone: bool = False
    error: BaseException | None = None
    observer: ResponseObserver | None = None
    notify: ResponseContext | None = None

    def observe(self, chunk: bytes) -> None:
        """Feed a relayed chunk to the observer, never disturbing the relay."""
        if self.observer is None:
            return
        try:
            self.observer.feed(chunk)
        except Exception:  # noqa: BLE001 - observation is best-effort by design
            logger.exception("Response observation failed; continuing relay")

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
    """Normalize a path for comparison by stripping a trailing slash."""
    return path.rstrip("/") or "/"


def _peer(scope: Scope) -> tuple[str, int] | None:
    """Extract the inbound ``(host, port)`` from an ASGI scope, if present."""
    client = scope.get("client")
    return (client[0], client[1]) if client else None


def _scope_headers(scope: Scope) -> dict[str, str]:
    """Flatten ASGI header pairs into a lowercase-keyed mapping."""
    headers: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers") or ():
        headers[raw_name.decode("latin-1").lower()] = raw_value.decode("latin-1")
    return headers


async def _read_body(receive: Receive, *, limit: int) -> bytes | None:
    """Buffer the inbound body, rejecting anything over ``limit`` bytes.

    Returns ``None`` when the request carried no body at all (as distinguished
    from an empty body), because a relayed ``GET`` must not gain one.
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


async def _send_local_json(send: Send, status_code: int, payload: Mapping[str, Any]) -> None:
    """Write a locally generated JSON response."""
    body = json.dumps(dict(payload)).encode("utf-8")
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


async def _send_api_error(
    send: Send,
    status_code: int,
    *,
    message: str,
    error_type: str,
    code: str,
) -> None:
    """Write an error in the provider's own envelope.

    Client SDKs parse ``{"error": {...}}`` and raise a typed API error from it.
    Anything else surfaces to the agent as a decoding failure, which hides what
    actually went wrong.
    """
    await _send_local_json(
        send,
        status_code,
        {"error": {"message": message, "type": error_type, "param": None, "code": code}},
    )
