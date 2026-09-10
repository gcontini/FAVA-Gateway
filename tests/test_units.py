"""Unit tests for the wire parsing, header relay and hook contracts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from mcp_proxy.config import ENV_BACKEND_URL, ProxySettings
from mcp_proxy.headers import forward_request_headers, forward_response_headers, is_mcp_wire_header
from mcp_proxy.hooks import (
    CompositeHooks,
    LoggingHooks,
    NullHooks,
    RequestContext,
    ResponseContext,
    normalize_hooks,
)
from mcp_proxy.messages import (
    METHOD_TOOLS_CALL,
    WireMessage,
    parse_response_body,
    parse_sse_events,
    parse_wire_messages,
)


# -- body parsing -----------------------------------------------------------

def test_parse_tools_call_extracts_name_and_arguments() -> None:
    """A `tools/call` envelope exposes the tool name and arguments to hooks."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": METHOD_TOOLS_CALL,
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        }
    ).encode()

    (message,) = parse_wire_messages(body)

    assert message.is_tool_call is True
    assert message.tool_name == "echo"
    assert message.tool_arguments == {"text": "hi"}
    assert message.id == 3
    assert message.is_request is True
    assert message.validation_error is None


def test_parse_notification_has_no_id() -> None:
    """A notification (no id) parses as a request-shaped message without an id."""
    body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()

    (message,) = parse_wire_messages(body)

    assert message.method == "notifications/initialized"
    assert message.id is None
    assert message.is_request is True


def test_parse_batch_yields_every_envelope() -> None:
    """A JSON-RPC batch produces one observed message per element."""
    body = json.dumps(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": METHOD_TOOLS_CALL, "params": {"name": "echo"}},
        ]
    ).encode()

    messages = parse_wire_messages(body)

    assert [message.method for message in messages] == ["tools/list", METHOD_TOOLS_CALL]
    assert messages[1].tool_name == "echo"


def test_parse_modern_envelope_records_header_method() -> None:
    """The 2026-07-28 `MCP-Method` header mirror is surfaced alongside the body."""
    body = json.dumps({"jsonrpc": "2.0", "id": 9, "method": METHOD_TOOLS_CALL, "params": {"name": "echo"}}).encode()

    (message,) = parse_wire_messages(body, headers={"mcp-method": "tools/call"})

    assert message.header_method == "tools/call"
    assert message.tool_name == "echo"


def test_parse_garbage_body_records_error_without_raising() -> None:
    """An unparseable body still yields a view so hooks know it happened."""
    (message,) = parse_wire_messages(b"{not json")

    assert message.validation_error is not None
    assert "not JSON" in message.validation_error


def test_parse_invalid_jsonrpc_envelope_keeps_method() -> None:
    """A body that decodes but is not a valid envelope keeps its method for logs."""
    body = json.dumps({"jsonrpc": "1.0", "id": 1, "method": "tools/list"}).encode()

    (message,) = parse_wire_messages(body)

    assert message.method == "tools/list"
    assert message.validation_error is not None


def test_parse_empty_body_observes_nothing() -> None:
    """A bodyless request (GET stream, DELETE) observes no messages."""
    assert parse_wire_messages(None) == ()
    assert parse_wire_messages(b"") == ()


def test_sse_event_parsing() -> None:
    """SSE frames are split into their `data:` payloads, skipping comments."""
    body = b": keep-alive\n\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n"

    events = list(parse_sse_events(body))

    assert events == [b'{"jsonrpc":"2.0","id":1,"result":{}}']


def test_parse_sse_response_body_finds_messages() -> None:
    """An `text/event-stream` response body yields its JSON-RPC messages."""
    frame = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"tools": []}})
    body = f"event: message\ndata: {frame}\n\n".encode()

    messages = parse_response_body(body, content_type="text/event-stream")

    assert len(messages) == 1
    assert messages[0].id == 5
    assert messages[0].jsonrpc_is_error() is False


def test_jsonrpc_error_detection() -> None:
    """`jsonrpc_is_error` and `error_code` read the raw envelope."""
    body = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "unknown"}}).encode()

    (message,) = parse_response_body(body, content_type="application/json")

    assert message.jsonrpc_is_error() is True
    assert message.error_code == -32601


# -- request context classification -----------------------------------------

def _request_context(body: dict[str, Any] | None) -> RequestContext:
    """Build a RequestContext from a JSON-RPC payload."""
    encoded = json.dumps(body).encode() if body is not None else b""
    return RequestContext(
        http_method="POST",
        path="/mcp",
        query="",
        headers={"content-type": "application/json"},
        body_size=len(encoded),
        messages=parse_wire_messages(encoded, kind="request"),
    )


def test_tools_call_is_effectful() -> None:
    """A `tools/call` is classified as effectful, so an authorizer gates it."""
    context = _request_context(
        {"jsonrpc": "2.0", "id": 1, "method": METHOD_TOOLS_CALL, "params": {"name": "write_file"}}
    )

    assert context.is_effectful is True
    assert context.tool_name == "write_file"
    assert context.method == METHOD_TOOLS_CALL


def test_read_methods_are_not_effectful() -> None:
    """Protocol reads and handshakes are not treated as effectful."""
    for method in (
        "initialize",
        "server/discover",  # the 2026-07-28 handshake replacing `initialize`
        "ping",
        "tools/list",
        "prompts/list",
        "resources/list",
    ):
        context = _request_context({"jsonrpc": "2.0", "id": 1, "method": method})
        assert context.is_effectful is False, method


def test_unknown_method_is_conservatively_effectful() -> None:
    """An unrecognized method counts as effectful rather than assumed safe."""
    context = _request_context({"jsonrpc": "2.0", "id": 1, "method": "vendor/launch-rockets"})

    assert context.is_effectful is True


def test_unparseable_body_is_conservatively_effectful() -> None:
    """An opaque body is treated as effectful, since its intent is unknown."""
    context = RequestContext(
        http_method="POST",
        path="/mcp",
        query="",
        headers={},
        body_size=9,
        messages=parse_wire_messages(b"{not json"),
    )

    assert context.is_effectful is True


# -- headers ----------------------------------------------------------------

def test_hop_by_hop_headers_are_not_relayed() -> None:
    """Connection-scoped headers stop at the proxy in both directions."""
    inbound = [
        ("connection", "keep-alive"),
        ("keep-alive", "timeout=5"),
        ("transfer-encoding", "chunked"),
        ("accept", "application/json, text/event-stream"),
        ("content-type", "application/json"),
    ]

    relayed = forward_request_headers(inbound, host="backend:8901")

    assert "connection" not in relayed
    assert "keep-alive" not in relayed
    assert "transfer-encoding" not in relayed
    assert relayed["accept"] == "application/json, text/event-stream"
    assert relayed["host"] == "backend:8901"


def test_connection_nominated_headers_are_dropped() -> None:
    """Headers named by `Connection:` are treated as hop-by-hop."""
    inbound = [("connection", "x-custom-token"), ("x-custom-token", "secret"), ("accept", "*/*")]

    relayed = forward_request_headers(inbound, trust_client_headers=True)

    assert "x-custom-token" not in relayed
    assert relayed["accept"] == "*/*"


def test_untrusted_mode_relays_only_mcp_wire_headers() -> None:
    """Without `trust_client_headers`, cookies and credentials are dropped."""
    inbound = [
        ("cookie", "session=abc"),
        ("authorization", "Bearer client-token"),
        ("user-agent", "agent/1.0"),
        ("mcp-session-id", "backend-session-9"),
        ("mcp-protocol-version", "2026-07-28"),
        ("mcp-method", "tools/call"),
        ("mcp-param-verbosity", "high"),
    ]

    relayed = forward_request_headers(inbound)

    assert relayed == {
        "mcp-session-id": "backend-session-9",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": "tools/call",
        "mcp-param-verbosity": "high",
    }


def test_extra_headers_override_inbound() -> None:
    """Proxy-supplied headers win, so a backend credential can be injected."""
    inbound = [("authorization", "Bearer client-token"), ("accept", "*/*")]

    relayed = forward_request_headers(inbound, extra={"Authorization": "Bearer backend-token"})

    assert relayed["authorization"] == "Bearer backend-token"


def test_mcp_param_prefix_recognition() -> None:
    """`Mcp-Param-*` is recognized as an MCP wire header by prefix."""
    assert is_mcp_wire_header("mcp-param-verbosity") is True
    assert is_mcp_wire_header("mcp-session-id") is True
    assert is_mcp_wire_header("cookie") is False


def test_response_headers_relayed_as_asgi_pairs() -> None:
    """Response headers come back as raw pairs, keeping repeated names."""
    backend = [
        ("content-type", "text/event-stream"),
        ("mcp-session-id", "s1"),
        ("transfer-encoding", "chunked"),
        ("set-cookie", "a=1"),
        ("set-cookie", "b=2"),
    ]

    relayed = forward_response_headers(backend)
    names = [name.decode() for name, _ in relayed]

    assert b"transfer-encoding" not in [name for name, _ in relayed]
    assert names.count("set-cookie") == 2
    assert ("content-type", "text/event-stream") == tuple(x.decode() for x in relayed[0])


# -- hooks ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_composite_hooks_dispatch_to_all() -> None:
    """Every composed hook sees an event; order is preserved."""
    seen: list[str] = []

    class Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        async def on_request(self, context: RequestContext) -> None:
            seen.append(self.name)

    composite = CompositeHooks([Recorder("first"), Recorder("second")])
    await composite.on_request(_request_context({"jsonrpc": "2.0", "id": 1, "method": "ping"}))

    assert seen == ["first", "second"]


@pytest.mark.asyncio
async def test_raising_hook_does_not_break_the_others() -> None:
    """A hook exception is swallowed so the relay keeps forwarding."""
    seen: list[str] = []

    class Broken:
        async def on_request(self, context: RequestContext) -> None:
            raise RuntimeError("hook bug")

    class Fine:
        async def on_request(self, context: RequestContext) -> None:
            seen.append("fine")

    composite = CompositeHooks([Broken(), Fine()])
    await composite.on_request(_request_context({"jsonrpc": "2.0", "id": 1, "method": "ping"}))

    assert seen == ["fine"]


@pytest.mark.asyncio
async def test_partial_hook_object_is_tolerated() -> None:
    """A hook implementing one callback works; missing ones are skipped."""

    class RequestOnly:
        async def on_request(self, context: RequestContext) -> None:
            """Only observes requests."""

    composite = CompositeHooks([RequestOnly()])
    await composite.on_response(
        ResponseContext(
            request=_request_context({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
            status_code=200,
            headers={},
            content_type="application/json",
        )
    )  # no AttributeError


def test_normalize_hooks() -> None:
    """Hook normalization collapses None, single and multiple observers."""
    single = NullHooks()

    assert isinstance(normalize_hooks(None), NullHooks)
    assert normalize_hooks(single) is single
    assert normalize_hooks([single]) is single
    assert isinstance(normalize_hooks([]), NullHooks)
    assert isinstance(normalize_hooks([single, single]), CompositeHooks)


@pytest.mark.asyncio
async def test_null_and_logging_hooks_are_noops() -> None:
    """NullHooks does nothing; LoggingHooks emits without raising."""
    context = _request_context({"jsonrpc": "2.0", "id": 1, "method": "ping"})

    await NullHooks().on_request(context)
    await LoggingHooks(level=50).on_request(context)
    await LoggingHooks(level=50).on_response(
        ResponseContext(request=context, status_code=200, headers={}, content_type="application/json")
    )


# -- settings ---------------------------------------------------------------

def test_settings_from_env() -> None:
    """Settings read the documented environment variables."""
    settings = ProxySettings.from_env(
        {ENV_BACKEND_URL: "http://backend:9001/mcp", "MCP_PROXY_PORT": "9100", "MCP_PROXY_MOUNT_PATH": "*"}
    )

    assert settings.backend_url == "http://backend:9001/mcp"
    assert settings.port == 9100
    assert settings.mount_path is None  # "*" means no path restriction


def test_settings_reject_relative_backend_url() -> None:
    """A non-absolute backend URL is rejected at construction."""
    with pytest.raises(ValueError, match="absolute HTTP"):
        ProxySettings(backend_url="backend:9001/mcp")


def test_settings_from_env_requires_backend() -> None:
    """Missing `MCP_PROXY_BACKEND_URL` is a configuration error."""
    with pytest.raises(ValueError, match=ENV_BACKEND_URL):
        ProxySettings.from_env({})


def test_wire_message_defaults_are_inert() -> None:
    """A bare WireMessage is safe to inspect without attributes set."""
    message = WireMessage()

    assert message.is_tool_call is False
    assert message.tool_name is None
    assert message.jsonrpc_is_error() is False
    assert message.error_code is None
