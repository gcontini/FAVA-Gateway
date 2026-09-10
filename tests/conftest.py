"""Shared fixtures: an in-process MCP backend and a proxy relay pointed at it."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
import uvicorn

from mcp.server.mcpserver import MCPServer

from mcp_proxy.app import create_proxy_app
from mcp_proxy.config import ProxySettings
from mcp_proxy.hooks import RequestContext, ResponseContext


# -- backend ----------------------------------------------------------------

def make_backend_server() -> MCPServer:
    """Build a small MCP server exposing tools a proxy test can drive.

    Tools:
        ``echo``: returns its arguments verbatim — proves argument fidelity
            through the relay.
        ``slow``: sleeps before answering, so tests can exercise mid-stream
            client disconnect.
        ``boom``: raises, so tests can see a tool error relayed unchanged.
    """
    server = MCPServer("fava-test-backend", instructions="Fixture backend for proxy tests.")

    @server.tool()
    def echo(text: str) -> str:
        """Return the input text unchanged."""
        return text

    @server.tool()
    async def slow(seconds: float) -> str:
        """Sleep, then report how long it slept."""
        await asyncio.sleep(seconds)
        return f"slept {seconds}s"

    @server.tool()
    def boom() -> str:
        """Always raise, to test error relaying."""
        raise RuntimeError("backend tool failure")

    return server


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
    def mcp_url(self) -> str:
        """The streamable HTTP MCP endpoint of this service."""
        return f"http://{self.host}:{self.port}/mcp"

    async def stop(self) -> None:
        """Shut the server down and wait for its task to finish."""
        self._server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(self._task, timeout=10)


def _free_port() -> int:
    """Return an unused TCP port on localhost.

    Binding to port 0 and reading back the assigned port is the only race-free
    way to do this without an external dependency.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _run_uvicorn(app: Any, port: int, *, name: str) -> ServiceHandle:
    """Start a uvicorn server for an ASGI app in the current event loop."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(), name=f"uvicorn-{name}")

    # Wait until the server reports itself started; otherwise the first request
    # races with the socket bind.
    for _ in range(200):
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
        """Record an observed backend response."""
        self.responses.append(context)

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Record a relay failure."""
        self.errors.append((context, error))

    def methods(self) -> list[str | None]:
        """Methods of all observed requests, in order (skipping non-JSON-RPC ones)."""
        return [request.method for request in self.requests]

    def tool_calls(self) -> list[str | None]:
        """Tool names of all observed ``tools/call`` requests."""
        return [request.tool_name for request in self.requests if request.tool_name is not None]


# -- composite fixtures -----------------------------------------------------

@dataclass
class ProxyFixture:
    """A live backend + a proxy in front of it + recording hooks."""

    backend: ServiceHandle
    proxy: ServiceHandle
    hooks: RecordingHooks
    settings: ProxySettings

    @property
    def backend_url(self) -> str:
        """Direct backend endpoint, for comparing proxied vs direct behavior."""
        return self.backend.mcp_url

    @property
    def proxy_url(self) -> str:
        """Proxy endpoint the MCP client should be pointed at."""
        return self.proxy.mcp_url


@pytest.fixture
async def proxy_pair() -> AsyncIterator[ProxyFixture]:
    """Start a real MCP backend and a forwarding proxy in front of it.

    Both run on real TCP ports in the test's event loop, so requests cross the
    actual streamable HTTP wire rather than an in-memory simulation.
    """
    backend_port = _free_port()
    proxy_port = _free_port()
    while proxy_port == backend_port:  # pragma: no cover - extremely unlikely
        proxy_port = _free_port()

    backend_app = make_backend_server().streamable_http_app()
    backend = await _run_uvicorn(backend_app, backend_port, name="backend")

    hooks = RecordingHooks()
    settings = ProxySettings(
        backend_url=backend.mcp_url,
        host="127.0.0.1",
        port=proxy_port,
    )
    proxy = await _run_uvicorn(create_proxy_app(settings, hooks=[hooks]), proxy_port, name="proxy")

    try:
        yield ProxyFixture(backend=backend, proxy=proxy, hooks=hooks, settings=settings)
    finally:
        await proxy.stop()
        await backend.stop()
