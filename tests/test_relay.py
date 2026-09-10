"""Relay-level tests driven directly against the ASGI interface.

These assert the transport contract rather than the API semantics: that the
exchange is terminated exactly once, that the disconnect watcher never deadlocks
a healthy request, that failures are answered in the provider's error envelope,
and that bytes cross unchanged.

The two termination behaviours are **regressions guarded on purpose**. Both
were real bugs in the MCP-era relay this package replaces:

1. marking the response complete after the streaming loop skipped the final
   ``more_body: False`` message, leaving the ASGI response unfinished;
2. not cancelling the disconnect watcher deadlocked every healthy exchange,
   because ``receive()`` only returns when the client goes away.

A clean rewrite is exactly where such bugs come back, so every test here runs
under a hard timeout: a deadlock must fail the suite, not hang it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx
import pytest

from llm_proxy.config import ProxySettings
from llm_proxy.relay import ForwardingProxy

from tests.conftest import RecordingHooks

# A generous ceiling: every exchange here is local and finishes in milliseconds.
# It exists so a regression surfaces as a failure instead of a hung suite.
DEADLINE = 10.0


# -- upstream doubles -------------------------------------------------------


class ChunkStream(httpx.AsyncByteStream):
    """A streaming upstream body under full test control.

    Unlike ``httpx.ASGITransport``, which buffers a response before returning
    it, this yields chunk by chunk — which is the only way to exercise the
    relay's streaming path and its mid-stream failure handling.
    """

    def __init__(
        self,
        chunks: Sequence[bytes],
        *,
        fail_after: int | None = None,
        delay: float = 0.0,
    ) -> None:
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for position, chunk in enumerate(self._chunks):
            if self._fail_after is not None and position >= self._fail_after:
                raise RuntimeError("upstream exploded mid-stream")
            if self._delay:
                await anyio.sleep(self._delay)
            yield chunk

    async def aclose(self) -> None:
        """Nothing to release."""


def streaming_transport(
    chunks: Sequence[bytes],
    *,
    content_type: str = "text/event-stream",
    status_code: int = 200,
    fail_after: int | None = None,
    delay: float = 0.0,
    headers: dict[str, str] | None = None,
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """A transport whose response streams ``chunks`` back to the relay."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(
            status_code,
            headers={"content-type": content_type, **(headers or {})},
            stream=ChunkStream(chunks, fail_after=fail_after, delay=delay),
        )

    return httpx.MockTransport(handler)


def build_proxy(
    transport: httpx.MockTransport | None = None,
    **settings_kwargs: Any,
) -> tuple[ForwardingProxy, RecordingHooks, httpx.AsyncClient | None]:
    """Build a relay wired to a mock upstream, plus its recording hooks."""
    settings_kwargs.setdefault("upstream_url", "http://upstream.test/v1")
    settings = ProxySettings(**settings_kwargs)
    hooks = RecordingHooks()
    client = httpx.AsyncClient(transport=transport) if transport is not None else None
    return ForwardingProxy(settings, hooks=[hooks], http_client=client), hooks, client


# -- ASGI exchange driver ---------------------------------------------------


@dataclass
class Exchange:
    """The ASGI messages one relayed exchange produced."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def starts(self) -> list[dict[str, Any]]:
        """Every ``http.response.start`` message sent."""
        return [m for m in self.messages if m["type"] == "http.response.start"]

    @property
    def terminators(self) -> list[dict[str, Any]]:
        """Every body message that ends the response."""
        return [m for m in self.messages if m["type"] == "http.response.body" and not m.get("more_body", False)]

    @property
    def status(self) -> int:
        """Status code of the single response start."""
        return self.starts[0]["status"]

    @property
    def headers(self) -> dict[str, str]:
        """Response headers as a lowercase-keyed mapping."""
        return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in self.starts[0]["headers"]}

    @property
    def body(self) -> bytes:
        """Every body byte the client received, in order."""
        return b"".join(m.get("body", b"") for m in self.messages if m["type"] == "http.response.body")

    def json(self) -> Any:
        """Decode the response body as JSON."""
        return json.loads(self.body)

    def assert_terminated_exactly_once(self) -> None:
        """The ASGI contract: one start, one terminator, and it comes last."""
        assert len(self.starts) == 1, f"expected 1 response.start, got {len(self.starts)}"
        assert len(self.terminators) == 1, f"expected 1 terminating body, got {len(self.terminators)}"
        assert self.messages[-1] is self.terminators[0], "the terminating body message must be last"


async def run_exchange(
    proxy: ForwardingProxy,
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
    body: bytes | None = b'{"model":"m","messages":[]}',
    headers: dict[str, str] | None = None,
    query: bytes = b"",
    disconnect_after: float | None = None,
    body_chunks: Sequence[bytes] | None = None,
) -> Exchange:
    """Drive one HTTP exchange straight through the ASGI interface.

    After the request body is delivered, ``receive()`` behaves like a live
    connection: it never returns again unless ``disconnect_after`` is set. That
    is what makes the watcher-cancellation regression observable — an
    uncancelled watcher would await this forever.
    """
    exchange = Exchange()

    if body_chunks is not None:
        inbound: list[dict[str, Any]] = [
            {"type": "http.request", "body": chunk, "more_body": position < len(body_chunks) - 1}
            for position, chunk in enumerate(body_chunks)
        ]
    else:
        inbound = [{"type": "http.request", "body": body or b"", "more_body": False}]

    async def receive() -> dict[str, Any]:
        if inbound:
            return inbound.pop(0)
        if disconnect_after is not None:
            await anyio.sleep(disconnect_after)
            return {"type": "http.disconnect"}
        # A healthy, still-open connection: nothing more ever arrives.
        await anyio.sleep_forever()
        raise AssertionError("unreachable")  # pragma: no cover

    async def send(message: dict[str, Any]) -> None:
        exchange.messages.append(message)

    raw_headers = [(b"content-type", b"application/json")]
    for name, value in (headers or {}).items():
        raw_headers.append((name.encode("latin-1"), value.encode("latin-1")))

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query,
        "headers": raw_headers,
        "client": ("127.0.0.1", 54321),
    }

    await asyncio.wait_for(proxy(scope, receive, send), timeout=DEADLINE)
    return exchange


# -- termination contract ---------------------------------------------------


async def test_terminates_exactly_once_on_a_buffered_response() -> None:
    """A plain JSON response ends the ASGI exchange once and only once."""
    payload = b'{"id":"c","model":"m","choices":[{"index":0,"message":{"content":"hi"},"finish_reason":"stop"}]}'
    proxy, _, client = build_proxy(streaming_transport([payload], content_type="application/json"))

    exchange = await run_exchange(proxy)

    exchange.assert_terminated_exactly_once()
    assert exchange.status == 200
    assert exchange.body == payload
    await client.aclose()  # type: ignore[union-attr]


async def test_terminates_exactly_once_while_streaming() -> None:
    """A multi-chunk SSE response still terminates exactly once."""
    chunks = [b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n' for _ in range(5)]
    chunks.append(b"data: [DONE]\n\n")
    proxy, _, client = build_proxy(streaming_transport(chunks))

    exchange = await run_exchange(proxy)

    exchange.assert_terminated_exactly_once()
    assert exchange.body == b"".join(chunks)
    await client.aclose()  # type: ignore[union-attr]


async def test_terminates_exactly_once_when_upstream_fails_midstream() -> None:
    """The response already started, so the relay must close it, not restart it."""
    chunks = [b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'] * 4
    proxy, hooks, client = build_proxy(streaming_transport(chunks, fail_after=2))

    exchange = await run_exchange(proxy)

    # A locally generated error cannot be sent once the status line is on the
    # wire; the client gets a truncated-but-well-formed response instead.
    exchange.assert_terminated_exactly_once()
    assert exchange.status == 200
    assert hooks.responses[0].stream_closed_early is True
    await client.aclose()  # type: ignore[union-attr]


async def test_healthy_exchange_does_not_deadlock_on_the_watcher() -> None:
    """The disconnect watcher must be cancelled when the relay finishes.

    `receive()` here never returns after the body, exactly like a real open
    connection. If the watcher is left running, the relay's task group waits on
    it forever and this test times out.
    """
    proxy, _, client = build_proxy(streaming_transport([b"{}"], content_type="application/json"))

    exchange = await asyncio.wait_for(run_exchange(proxy), timeout=DEADLINE)

    exchange.assert_terminated_exactly_once()
    await client.aclose()  # type: ignore[union-attr]


async def test_client_disconnect_is_not_reported_as_a_relay_failure() -> None:
    """A client hanging up is normal, not an error to log or surface."""
    chunks = [b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'] * 20
    proxy, hooks, client = build_proxy(streaming_transport(chunks, delay=0.05))

    exchange = await run_exchange(proxy, disconnect_after=0.02)

    # The exchange unwinds cleanly rather than raising out of the task group.
    assert hooks.errors == []
    if hooks.responses:
        assert hooks.responses[0].stream_closed_early is True
    assert len(exchange.starts) <= 1
    await client.aclose()  # type: ignore[union-attr]


# -- locally answered requests ---------------------------------------------


async def test_unknown_path_returns_an_api_shaped_error() -> None:
    """A path outside the mount prefix never reaches the provider."""
    proxy, _, client = build_proxy(streaming_transport([b"{}"]))

    exchange = await run_exchange(proxy, path="/not-the-api")

    exchange.assert_terminated_exactly_once()
    assert exchange.status == 404
    error = exchange.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "unknown_path"
    await client.aclose()  # type: ignore[union-attr]


async def test_oversized_body_is_rejected_before_relaying() -> None:
    """The buffered body is bounded, and the refusal is API-shaped."""
    proxy, hooks, client = build_proxy(streaming_transport([b"{}"]), max_body_size=64)

    exchange = await run_exchange(proxy, body=b"x" * 256)

    assert exchange.status == 413
    assert exchange.json()["error"]["code"] == "request_too_large"
    # Nothing was observed or forwarded: the request never became an exchange.
    assert hooks.requests == []
    await client.aclose()  # type: ignore[union-attr]


async def test_health_endpoint_is_answered_locally() -> None:
    """The health path is owned by the proxy and never relayed."""
    seen: list[httpx.Request] = []
    proxy, _, client = build_proxy(streaming_transport([b"{}"], seen=seen))

    exchange = await run_exchange(proxy, method="GET", path="/healthz", body=None)

    assert exchange.status == 200
    payload = exchange.json()
    assert payload["status"] == "ok"
    assert payload["upstream_url"] == "http://upstream.test/v1"
    assert seen == []
    await client.aclose()  # type: ignore[union-attr]


async def test_unreachable_upstream_yields_a_gateway_error() -> None:
    """With no upstream at all the client gets a typed API error, not a crash."""
    # No transport: the relay builds a real client and genuinely fails to connect.
    proxy, hooks, _ = build_proxy(upstream_url="http://127.0.0.1:1/v1", timeout=2.0)

    exchange = await run_exchange(proxy)

    exchange.assert_terminated_exactly_once()
    assert exchange.status == 502
    error = exchange.json()["error"]
    assert error["type"] == "upstream_error"
    assert error["code"] == "bad_gateway"
    assert len(hooks.errors) == 1


# -- routing and fidelity ---------------------------------------------------


async def test_path_suffix_is_appended_to_the_upstream_base() -> None:
    """The base-URL swap is the integration, so every suffix must route."""
    seen: list[httpx.Request] = []
    proxy, _, client = build_proxy(streaming_transport([b"{}"], content_type="application/json", seen=seen))

    await run_exchange(proxy, method="GET", path="/v1/models", body=None)

    assert str(seen[0].url) == "http://upstream.test/v1/models"
    # A bodyless GET must not gain a body upstream.
    assert seen[0].content == b""
    await client.aclose()  # type: ignore[union-attr]


async def test_query_string_is_preserved() -> None:
    """Query parameters belong to the provider's endpoint, not the proxy."""
    seen: list[httpx.Request] = []
    proxy, _, client = build_proxy(streaming_transport([b"{}"], content_type="application/json", seen=seen))

    await run_exchange(proxy, method="GET", path="/v1/models", body=None, query=b"limit=5")

    assert str(seen[0].url) == "http://upstream.test/v1/models?limit=5"
    await client.aclose()  # type: ignore[union-attr]


async def test_unmounted_proxy_forwards_the_whole_path() -> None:
    """With no mount prefix every path is relayed verbatim."""
    seen: list[httpx.Request] = []
    proxy, _, client = build_proxy(
        streaming_transport([b"{}"], content_type="application/json", seen=seen),
        upstream_url="http://upstream.test",
        mount_prefix=None,
    )

    await run_exchange(proxy, path="/v1/chat/completions")

    assert str(seen[0].url) == "http://upstream.test/v1/chat/completions"
    await client.aclose()  # type: ignore[union-attr]


async def test_request_bytes_cross_unchanged() -> None:
    """The body the provider receives is the body the client sent, byte for byte."""
    seen: list[httpx.Request] = []
    proxy, _, client = build_proxy(streaming_transport([b"{}"], content_type="application/json", seen=seen))
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "héllo — ünicode"}]}).encode("utf-8")

    # Delivered in several ASGI chunks, as a large request really arrives.
    await run_exchange(proxy, body_chunks=[body[:10], body[10:25], body[25:]])

    assert seen[0].content == body
    await client.aclose()  # type: ignore[union-attr]


async def test_upstream_response_headers_reach_the_client() -> None:
    """Rate-limit headers a client acts on must survive the relay."""
    proxy, _, client = build_proxy(
        streaming_transport(
            [b"{}"],
            content_type="application/json",
            headers={"x-ratelimit-remaining-requests": "42"},
        )
    )

    exchange = await run_exchange(proxy)

    assert exchange.headers["x-ratelimit-remaining-requests"] == "42"
    await client.aclose()  # type: ignore[union-attr]


async def test_hooks_observe_tool_intents_through_the_relay() -> None:
    """The whole point: a streamed tool call is reassembled by the time hooks run."""
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        b'"function":{"name":"run_shell","arguments":""}}]}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":"{\\"cmd\\":"}}]}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":"\\"whoami\\"}"}}]}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    proxy, hooks, client = build_proxy(streaming_transport(chunks))

    await run_exchange(proxy)

    (response,) = hooks.responses
    (intent,) = response.tool_intents
    assert intent.name == "run_shell"
    assert intent.arguments == {"cmd": "whoami"}
    assert response.may_dispatch_tools is True
    assert response.intents_unknown is False
    await client.aclose()  # type: ignore[union-attr]


async def test_observation_failure_never_breaks_the_relay() -> None:
    """A hook that explodes must not cost the client its response."""

    class Exploding:
        async def on_response(self, context: Any) -> None:
            raise RuntimeError("observer is broken")

    settings = ProxySettings(upstream_url="http://upstream.test/v1")
    client = httpx.AsyncClient(transport=streaming_transport([b"{}"], content_type="application/json"))
    proxy = ForwardingProxy(settings, hooks=[Exploding()], http_client=client)

    exchange = await run_exchange(proxy)

    exchange.assert_terminated_exactly_once()
    assert exchange.body == b"{}"
    await client.aclose()
