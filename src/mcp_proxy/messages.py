"""Read-only views over JSON-RPC payloads crossing the proxy.

The relay forwards bytes without rewriting them. Parsing happens *alongside*
the relay, purely so observers (hooks, logs, and later FAVA's authorizer) can
see which method and arguments are in flight. Nothing here can mutate the wire
format: a payload that fails to parse is still forwarded byte-for-byte.

Two envelope eras are supported, both from SDK `mcp` v2:

* **2025-era ("monolithic")**: the POST body holds the full JSON-RPC envelope
  (`jsonrpc`, `id`, `method`, `params`), and server responses arrive as
  ``text/event-stream`` SSE frames or a single ``application/json`` body.
* **2026-07-28 ("modern")**: routing lives in ``MCP-Method``/``MCP-Name``/
  ``Mcp-Param-*`` headers, the body still carries the JSON-RPC envelope, and
  each POST is stateless (no ``Mcp-Session-Id``).

Both put ``method``/``params`` in the body, so one parser covers them; the
modern header mirror is attached when present.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import ValidationError

from mcp.types.jsonrpc import (
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)

logger = logging.getLogger(__name__)

# Method whose params carry the tool name and arguments — the interception
# point FAVA's authorizer will act on.
METHOD_TOOLS_CALL: Final = "tools/call"
METHOD_TOOLS_LIST: Final = "tools/list"
METHOD_INITIALIZE: Final = "initialize"


@dataclass(frozen=True)
class WireMessage:
    """One JSON-RPC message observed on the wire.

    Attributes:
        method: The request/notification method, or ``None`` for responses and
            messages that carry no method.
        params: Request params as a mapping; ``None`` when absent or not an object.
        id: JSON-RPC id for requests/responses; ``None`` for notifications.
        is_request: True when the message expects a response.
        tool_name: For ``tools/call``, the invoked tool name.
        tool_arguments: For ``tools/call``, the argument mapping (``{}`` if absent).
        kind: Origin of the message: ``"request"`` (client -> server),
            ``"response"`` (server -> client) or ``"unknown"``.
        jsonrpc: The raw decoded payload, for diagnostics only. Never re-sent.
        header_method: The ``MCP-Method`` header value, when the modern
            envelope mirrored the method into headers.
        validation_error: Set when the body decoded as JSON but did not match a
            JSON-RPC envelope. The proxy still relays such a body untouched.
    """

    method: str | None = None
    params: Mapping[str, Any] | None = None
    id: Any = None
    is_request: bool = False
    tool_name: str | None = None
    tool_arguments: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "unknown"
    jsonrpc: Any = None
    header_method: str | None = None
    validation_error: str | None = None

    @property
    def is_tool_call(self) -> bool:
        """Whether this message invokes a tool."""
        return self.method == METHOD_TOOLS_CALL

    @property
    def response_id(self) -> Any:
        """The id a response answers, aliasing :attr:`id` for readability."""
        return self.id

    def jsonrpc_is_error(self) -> bool:
        """Whether this message is a JSON-RPC error response.

        Checks the raw payload's ``error`` member rather than the validated
        model, since a message that failed validation still needs an answer
        here.
        """
        return isinstance(self.jsonrpc, dict) and self.jsonrpc.get("error") is not None

    @property
    def error_code(self) -> int | None:
        """The JSON-RPC ``error.code`` when this is an error response."""
        if isinstance(self.jsonrpc, dict):
            error = self.jsonrpc.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), int):
                return error["code"]
        return None


def parse_wire_messages(
    body: bytes | None,
    *,
    headers: Mapping[str, str] | None = None,
    kind: str = "request",
) -> Sequence[WireMessage]:
    """Parse an HTTP body into zero or more :class:`WireMessage` views.

    Args:
        body: The raw request or response body. ``None`` or empty yields nothing.
        headers: Lowercased inbound header mapping, consulted for the modern
            ``MCP-Method`` envelope mirror.
        kind: ``"request"`` or ``"response"``, stamped onto each message.

    Returns:
        One message per JSON-RPC envelope. A JSON batch yields one per element.
        Unparseable, non-JSON and non-envelope bodies yield a single message
        with :attr:`WireMessage.validation_error` set. Never raises: the relay
        must keep forwarding whatever the client sent.
    """
    if not body:
        return ()

    try:
        decoded = json.loads(body)
    except (ValueError, RecursionError) as exc:
        # Not just JSONDecodeError: oversized int literals raise bare ValueError,
        # deep nesting raises RecursionError.
        logger.debug("Wire body is not JSON (%s); relaying unchanged", exc)
        return (WireMessage(kind=kind, validation_error=f"not JSON: {exc}"),)

    header_method = None
    if headers:
        header_method = headers.get("mcp-method")

    if isinstance(decoded, list):
        if not decoded:
            return (WireMessage(kind=kind, validation_error="empty JSON batch"),)
        return tuple(_from_payload(item, headers=header_method, kind=kind) for item in decoded)

    return (_from_payload(decoded, headers=header_method, kind=kind),)


def _from_payload(payload: Any, *, headers: str | None, kind: str) -> WireMessage:
    """Build a :class:`WireMessage` from one decoded JSON value."""
    if not isinstance(payload, dict):
        return WireMessage(kind=kind, jsonrpc=payload, validation_error="JSON-RPC envelope must be an object")

    method = payload.get("method")
    params = payload.get("params")

    try:
        if kind == "response":
            _validated = JSONRPCResponse.model_validate(payload)
        elif method is None:
            _validated = JSONRPCError.model_validate(payload)
        elif "id" in payload and payload["id"] is not None:
            _validated = JSONRPCRequest.model_validate(payload)
        else:
            _validated = JSONRPCNotification.model_validate(payload)
    except ValidationError as exc:
        # Still relayed verbatim — an opaque body is the backend's problem, and
        # a proxy that rejects it would break forward compatibility.
        errors = exc.errors()
        detail = f"{errors[0]['msg']} at {'.'.join(str(p) for p in errors[0]['loc'])}" if errors else "invalid envelope"
        return WireMessage(
            method=method if isinstance(method, str) else None,
            id=payload.get("id"),
            header_method=headers,
            jsonrpc=payload,
            kind=kind,
            validation_error=f"invalid JSON-RPC envelope: {detail}",
        )

    is_request = isinstance(_validated, (JSONRPCRequest, JSONRPCNotification))

    if isinstance(_validated, JSONRPCNotification):
        resolved_method, resolved_id = method, None
    elif isinstance(_validated, JSONRPCRequest):
        resolved_method, resolved_id = method, _validated.id
    else:
        resolved_method, resolved_id = None, getattr(_validated, "id", None)

    params_map: Mapping[str, Any] | None
    if isinstance(params, Mapping):
        params_map = params
    else:
        params_map = None

    tool_name: str | None = None
    tool_args: Mapping[str, Any] = {}
    if resolved_method == METHOD_TOOLS_CALL and params_map is not None:
        raw_name = params_map.get("name")
        tool_name = raw_name if isinstance(raw_name, str) else None
        raw_args = params_map.get("arguments")
        if isinstance(raw_args, Mapping):
            tool_args = raw_args

    return WireMessage(
        method=resolved_method if isinstance(resolved_method, str) else None,
        params=params_map,
        id=resolved_id,
        is_request=bool(is_request),
        tool_name=tool_name,
        tool_arguments=tool_args,
        kind=kind,
        jsonrpc=payload,
        header_method=headers,
    )


def parse_sse_events(body: bytes) -> Iterable[bytes]:
    """Yield the ``data:`` payload of each SSE frame in a response body.

    Only used for observation of already-forwarded bytes; the relay streams the
    original frames to the client unmodified. Multi-line ``data:`` fields within
    one event are joined with newlines, per the SSE spec, and comment lines
    (``:`` prefix, e.g. keep-alive pings) are ignored.
    """
    for event_block in body.split(b"\n\n"):
        if not event_block.strip():
            continue
        data_lines: list[bytes] = []
        for line in event_block.split(b"\n"):
            if line.startswith(b":"):
                continue
            field, _, value = line.partition(b":")
            if field.strip() != b"data":
                continue
            if value.startswith(b" "):
                value = value[1:]
            data_lines.append(value)
        if data_lines:
            yield b"\n".join(data_lines)


def parse_response_body(body: bytes, *, content_type: str | None) -> Sequence[WireMessage]:
    """Parse a backend response body, dispatching on content type.

    ``text/event-stream`` bodies are split into SSE frames and each ``data``
    payload parsed as JSON-RPC; everything else is parsed as a single JSON
    envelope. Returns an empty sequence when no frame contains JSON-RPC.
    """
    if content_type and "text/event-stream" in content_type.lower():
        messages: list[WireMessage] = []
        for payload in parse_sse_events(body):
            messages.extend(parse_wire_messages(payload, kind="response"))
        return tuple(messages)
    return parse_wire_messages(body, kind="response")
