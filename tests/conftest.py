"""Shared fixtures: a fake OpenAI-compatible upstream and a proxy in front of it.

The upstream is deliberately hand-rolled rather than mocked at the library
level: the behaviours that matter here are wire-level (SSE framing, arguments
split across frames, frames split across TCP chunks) and a mock would paper over
exactly those.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_proxy.app import create_proxy_app
from llm_proxy.config import ProxySettings
from llm_proxy.hooks import RequestContext, ResponseContext

# Fixed so two identical requests produce byte-identical responses, which is what
# the byte-fidelity test compares.
CREATED = 1_700_000_000

# -- fake upstream ----------------------------------------------------------


@dataclass
class ToolCallScript:
    """One tool call the fake model should ask for."""

    name: str
    arguments: str
    id: str = "call_0"


@dataclass
class FakeUpstream:
    """A scriptable OpenAI-compatible provider.

    Tests mutate the attributes to steer the next response. Every received
    request is recorded, so header and body fidelity can be asserted from the
    provider's side of the relay.

    Attributes:
        content: Assistant text to return.
        tool_calls: Tool calls the model should request.
        fragment_size: Characters of ``arguments`` per SSE frame. Small values
            force the accumulator to reassemble many fragments.
        interleave: Emit fragments round-robin across parallel tool calls, as a
            real provider does, so arrival order cannot be relied on.
        chunk_size: When set, the SSE body is sliced into byte chunks of this
            size, splitting frames across chunk boundaries.
        status_code: Return this status instead of a completion.
        error_body: Body to return with a non-200 ``status_code``.
        delay: Seconds to sleep before responding, for disconnect tests.
        fail_after_chunks: Raise mid-stream once this many chunks have been
            emitted, so the relay's "response already started" error path can
            be exercised.
    """

    content: str = "Hello from the fake provider."
    tool_calls: list[ToolCallScript] = field(default_factory=list)
    fragment_size: int = 6
    interleave: bool = True
    chunk_size: int | None = None
    status_code: int = 200
    error_body: dict[str, Any] = field(default_factory=lambda: {"error": {"message": "boom", "type": "server_error"}})
    delay: float = 0.0
    fail_after_chunks: int | None = None
    model: str = "fake-model-1"
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last_request(self) -> dict[str, Any]:
        """The most recent request body and headers the upstream received."""
        return self.requests[-1]

    def _finish_reason(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"

    def completion(self) -> dict[str, Any]:
        """Build a non-streaming completion payload."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": CREATED,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": self._finish_reason()}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

    def sse_body(self) -> bytes:
        """Build the complete SSE body for a streamed completion."""
        frames = [self._frame({"role": "assistant", "content": ""})]

        if self.content:
            frames.extend(self._frame({"content": piece}) for piece in _slice(self.content, self.fragment_size))

        if self.tool_calls:
            # Opening fragment per call: id and full name, empty arguments.
            for index, call in enumerate(self.tool_calls):
                frames.append(
                    self._frame(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call.id,
                                    "type": "function",
                                    "function": {"name": call.name, "arguments": ""},
                                }
                            ]
                        }
                    )
                )
            for index, fragment in self._argument_fragments():
                frames.append(
                    self._frame({"tool_calls": [{"index": index, "function": {"arguments": fragment}}]})
                )

        frames.append(self._frame({}, finish_reason=self._finish_reason()))
        frames.append(b"data: [DONE]\n\n")
        return b"".join(frames)

    def _argument_fragments(self) -> Iterator[tuple[int, str]]:
        """Yield ``(tool call index, argument fragment)`` in emission order."""
        per_call = [_slice(call.arguments, self.fragment_size) for call in self.tool_calls]
        if not self.interleave:
            for index, fragments in enumerate(per_call):
                for fragment in fragments:
                    yield index, fragment
            return
        # Round-robin: a real provider interleaves parallel calls, so fragments
        # cannot be accumulated in arrival order.
        for position in range(max((len(f) for f in per_call), default=0)):
            for index, fragments in enumerate(per_call):
                if position < len(fragments):
                    yield index, fragments[position]

    def _frame(self, delta: dict[str, Any], *, finish_reason: str | None = None) -> bytes:
        """Render one SSE chunk frame."""
        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": CREATED,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _slice(text: str, size: int) -> list[str]:
    """Split text into fixed-size pieces, never returning an empty list."""
    if not text:
        return []
    size = max(1, size)
    return [text[start : start + size] for start in range(0, len(text), size)]


def create_upstream_app(fake: FakeUpstream) -> Starlette:
    """Build the ASGI app for a :class:`FakeUpstream`."""

    async def chat_completions(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            payload = {}
        fake.requests.append({"headers": dict(request.headers), "body": payload, "raw": raw, "path": request.url.path})

        if fake.delay:
            await asyncio.sleep(fake.delay)

        if fake.status_code >= 400:
            return JSONResponse(fake.error_body, status_code=fake.status_code)

        if payload.get("stream"):
            body = fake.sse_body()

            async def stream() -> AsyncIterator[bytes]:
                size = fake.chunk_size or len(body)
                emitted = 0
                for start in range(0, len(body), size):
                    if fake.fail_after_chunks is not None and emitted >= fake.fail_after_chunks:
                        raise RuntimeError("upstream exploded mid-stream")
                    yield body[start : start + size]
                    emitted += 1
                    # Yield control so chunks really cross the wire apart.
                    await asyncio.sleep(0)

            return StreamingResponse(stream(), media_type="text/event-stream")

        return JSONResponse(fake.completion())

    async def models(request: Request) -> Response:
        fake.requests.append({"headers": dict(request.headers), "body": None, "raw": b"", "path": request.url.path})
        return JSONResponse(
            {"object": "list", "data": [{"id": fake.model, "object": "model", "owned_by": "fava-tests"}]}
        )

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
        ]
    )


# -- HTTP servers -----------------------------------------------------------


@dataclass
class ServiceHandle:
    """A uvicorn server running in a background task on a free port."""

    name: str
    host: str
    port: int
    _server: Any = field(repr=False)
    _task: Any = field(repr=False)

    @property
    def base_url(self) -> str:
        """The OpenAI-style base URL of this service."""
        return f"http://{self.host}:{self.port}/v1"

    @property
    def root_url(self) -> str:
        """The service root, for non-API paths such as the health endpoint."""
        return f"http://{self.host}:{self.port}"

    async def stop(self) -> None:
        """Shut the server down and wait for its task to finish."""
        self._server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(self._task, timeout=10)


def free_port() -> int:
    """Return an unused TCP port on localhost.

    Binding to port 0 and reading back the assigned port is the only race-free
    way to do this without an external dependency. Never hardcode a port: the
    suite must stay parallel-safe.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def run_uvicorn(app: Any, port: int, *, name: str) -> ServiceHandle:
    """Start a uvicorn server for an ASGI app in the current event loop."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(), name=f"uvicorn-{name}")

    # Wait until the server reports itself started; otherwise the first request
    # races with the socket bind.
    for _ in range(300):
        if server.started:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover - defensive
        task.cancel()
        raise RuntimeError(f"uvicorn {name} did not start")

    return ServiceHandle(name=name, host="127.0.0.1", port=port, _server=server, _task=task)


# -- recording hooks --------------------------------------------------------


@dataclass
class RecordingHooks:
    """Hook implementation that records every observation for assertions."""

    requests: list[RequestContext] = field(default_factory=list)
    responses: list[ResponseContext] = field(default_factory=list)
    errors: list[tuple[RequestContext, BaseException]] = field(default_factory=list)

    async def on_request(self, context: RequestContext) -> None:
        """Record an observed inbound request."""
        self.requests.append(context)

    async def on_response(self, context: ResponseContext) -> None:
        """Record an observed upstream response."""
        self.responses.append(context)

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Record a relay failure."""
        self.errors.append((context, error))

    def intent_names(self) -> list[str]:
        """Names of every tool call observed across all responses, in order."""
        return [intent.name or "" for response in self.responses for intent in response.tool_intents]

    def intents(self) -> list[Any]:
        """Every observed tool intent, flattened across responses."""
        return [intent for response in self.responses for intent in response.tool_intents]


# -- composite fixtures -----------------------------------------------------


@dataclass
class ProxyFixture:
    """A live fake provider + a proxy in front of it + recording hooks."""

    upstream: ServiceHandle
    proxy: ServiceHandle
    fake: FakeUpstream
    hooks: RecordingHooks
    settings: ProxySettings

    @property
    def proxy_base_url(self) -> str:
        """The base URL an OpenAI client should be pointed at."""
        return self.proxy.base_url

    @property
    def upstream_base_url(self) -> str:
        """The provider's own base URL, for comparing proxied vs direct."""
        return self.upstream.base_url

    async def wait_for_responses(self, count: int = 1, timeout: float = 5.0) -> None:
        """Block until the proxy has observed ``count`` responses.

        The relay dispatches `on_response` from the proxy's own task, *after*
        the last byte reaches the client. A test that asserts the moment its
        client call returns is therefore racing that task — especially for a
        stream, where the client stops reading at `[DONE]` while the relay is
        still unwinding. Waiting here removes the race without weakening the
        assertion.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while len(self.hooks.responses) < count:
            if loop.time() > deadline:
                raise AssertionError(
                    f"expected {count} observed response(s), got {len(self.hooks.responses)} within {timeout}s"
                )
            await asyncio.sleep(0.01)


@pytest.fixture
async def fake_upstream() -> FakeUpstream:
    """A scriptable fake provider, shared with the proxy fixture."""
    return FakeUpstream()


@pytest.fixture
async def proxy_pair(fake_upstream: FakeUpstream) -> AsyncIterator[ProxyFixture]:
    """Start a fake provider and a forwarding proxy in front of it.

    Both run on real TCP ports in the test's event loop, so requests cross the
    actual HTTP wire rather than an in-memory simulation.
    """
    upstream_port = free_port()
    proxy_port = free_port()
    while proxy_port == upstream_port:  # pragma: no cover - extremely unlikely
        proxy_port = free_port()

    upstream = await run_uvicorn(create_upstream_app(fake_upstream), upstream_port, name="upstream")

    hooks = RecordingHooks()
    settings = ProxySettings(
        upstream_url=f"http://127.0.0.1:{upstream_port}/v1",
        host="127.0.0.1",
        port=proxy_port,
    )
    proxy = await run_uvicorn(create_proxy_app(settings, hooks=[hooks]), proxy_port, name="proxy")

    try:
        yield ProxyFixture(
            upstream=upstream,
            proxy=proxy,
            fake=fake_upstream,
            hooks=hooks,
            settings=settings,
        )
    finally:
        await proxy.stop()
        await upstream.stop()
