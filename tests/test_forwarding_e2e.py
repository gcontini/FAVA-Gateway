"""End-to-end tests: a real MCP client talking to a backend through the proxy.

These exercises are the ones that matter for a drop-in reverse proxy. An
unmodified MCP client (the official SDK's own transport) points at the proxy
URL; the proxy relays to a real backend over TCP. Nothing here knows the proxy
exists except by its URL — which is precisely the deployment contract.
"""

from __future__ import annotations

import asyncio

import httpx2
import pytest

from mcp import Client

from tests.conftest import ProxyFixture


@pytest.mark.asyncio
async def test_list_tools_through_proxy(proxy_pair: ProxyFixture) -> None:
    """`tools/list` returns the backend's catalog, relayed unchanged."""
    async with Client(proxy_pair.proxy_url) as client:
        result = await client.list_tools()

    names = {tool.name for tool in result.tools}
    assert {"echo", "slow", "boom"} <= names
    # The relay observed the exchange.
    assert "tools/list" in proxy_pair.hooks.methods()


@pytest.mark.asyncio
async def test_call_tool_argument_fidelity(proxy_pair: ProxyFixture) -> None:
    """Tool arguments and results cross the proxy byte-for-byte."""
    async with Client(proxy_pair.proxy_url) as client:
        result = await client.call_tool("echo", {"text": "hello \u2014 fava"})

    assert result.is_error is False
    assert "hello \u2014 fava" in str(result.content[0].text)
    assert "echo" in proxy_pair.hooks.tool_calls()


@pytest.mark.asyncio
async def test_tool_error_relays_unchanged(proxy_pair: ProxyFixture) -> None:
    """A backend tool exception arrives as a tool error, not a proxy failure."""
    async with Client(proxy_pair.proxy_url) as client:
        result = await client.call_tool("boom", {})

    assert result.is_error is True
    # The relay itself succeeded (it returned a response), so no relay error fired.
    assert proxy_pair.hooks.errors == []


@pytest.mark.asyncio
async def test_multiple_sequential_calls(proxy_pair: ProxyFixture) -> None:
    """A session makes several calls in a row; each is observed once."""
    async with Client(proxy_pair.proxy_url) as client:
        for index in range(5):
            result = await client.call_tool("echo", {"text": f"call-{index}"})
            assert f"call-{index}" in str(result.content[0].text)

    tool_calls = proxy_pair.hooks.tool_calls()
    assert tool_calls.count("echo") == 5


@pytest.mark.asyncio
async def test_concurrent_clients(proxy_pair: ProxyFixture) -> None:
    """Parallel sessions each get their own correct result, no cross-talk."""

    async def one_client(value: str) -> str:
        async with Client(proxy_pair.proxy_url) as client:
            result = await client.call_tool("echo", {"text": value})
            return str(result.content[0].text)

    values = [f"session-{n}" for n in range(8)]
    results = await asyncio.gather(*[one_client(value) for value in values])

    for value, result in zip(values, results, strict=True):
        assert value in result


@pytest.mark.asyncio
async def test_hooks_see_handshake_and_call(proxy_pair: ProxyFixture) -> None:
    """The observe-request/observe-response hooks fire for a full session."""
    async with Client(proxy_pair.proxy_url) as client:
        await client.call_tool("echo", {"text": "observed"})

    methods = proxy_pair.hooks.methods()
    # Which handshake is used depends on the negotiated protocol revision: the
    # SDK v2 client sends the 2026-07-28 `server/discover`, while a legacy-mode
    # client still sends `initialize`. Either proves the handshake was observed.
    assert {"initialize", "server/discover"} & set(methods)
    assert "tools/call" in methods
    assert len(proxy_pair.hooks.responses) >= 1
    assert all(response.status_code < 400 for response in proxy_pair.hooks.responses)


@pytest.mark.asyncio
async def test_health_endpoint_is_local(proxy_pair: ProxyFixture) -> None:
    """The proxy's health path is answered locally, never relayed."""
    async with httpx2.AsyncClient() as http_client:
        response = await http_client.get(f"http://127.0.0.1:{proxy_pair.proxy.port}/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_unknown_path_returns_jsonrpc_error(proxy_pair: ProxyFixture) -> None:
    """A request to a path the proxy doesn't mount gets a JSON-RPC error."""
    async with httpx2.AsyncClient() as http_client:
        response = await http_client.post(
            f"http://127.0.0.1:{proxy_pair.proxy.port}/not-mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 404
    assert "No MCP endpoint" in response.text


@pytest.mark.asyncio
async def test_backend_down_yields_gateway_error() -> None:
    """When no backend answers, the proxy returns a JSON-RPC error, not a crash."""
    from mcp_proxy.app import create_proxy_app
    from mcp_proxy.config import ProxySettings

    settings = ProxySettings(backend_url="http://127.0.0.1:1/mcp", timeout=2.0)
    app = create_proxy_app(settings)

    # Reach the app directly: no server needed, just the ASGI call.
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
            headers={"content-type": "application/json", "accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 502
    payload = response.json()
    assert payload["id"] == 7
    assert "Bad gateway" in payload["error"]["message"]

