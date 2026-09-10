"""HTTP header handling for the pass-through relay.

The proxy must not confuse the agent's HTTP connection with the backend's.
Hop-by-hop headers are therefore never relayed in either direction, while the
MCP wire headers that carry protocol routing (session id, protocol version,
method/name/param mirrors) are always relayed verbatim.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping

logger = logging.getLogger(__name__)

# RFC 9110 §7.6.1 hop-by-hop headers: meaningful to a single transport-level
# connection and MUST NOT be forwarded by an intermediary.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Headers the ASGI server owns: never sent upstream, never read from it.
# `host` is excluded deliberately — it is rewritten per-request instead.
# `content-length` is excluded too: httpx2 recomputes it from the relayed body.
HOPPED_REQUEST_EXTRA = frozenset({"host", "content-length"})

# Response headers the ASGI server sets itself.
HOPPED_RESPONSE_EXTRA = frozenset({"content-length"})

# MCP protocol headers that MUST reach the backend untouched. Relayed even when
# `trust_client_headers` is disabled, because the protocol cannot work without
# them (`mcp-session-id` identifies the backend session, the others carry the
# 2026-07-28 per-request envelope).
MCP_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        "last-event-id",
        "mcp-session-id",
        "mcp-protocol-version",
        "mcp-method",
        "mcp-name",
    }
)

# `Mcp-Param-*`, the 2026-07-28 custom-parameter header prefix. Matched by
# prefix rather than enumerated, since parameter names are tool-defined.
MCP_PARAM_HEADER_PREFIX = "mcp-param-"

MCP_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "mcp-session-id",
        "mcp-protocol-version",
        "last-event-id",
        "retry",
        "cache-control",
        "content-location",
        "location",
    }
)


def is_mcp_wire_header(name: str) -> bool:
    """Whether a lowercase header name carries MCP protocol routing."""
    return name in MCP_REQUEST_HEADERS or name.startswith(MCP_PARAM_HEADER_PREFIX)


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    """Yield headers, dropping any named by a `Connection:` header."""
    pairs = list(headers)
    connection = ",".join(value for name, value in pairs if name == "connection")
    nominated = {token.strip().lower() for token in connection.split(",") if token.strip()}
    for name, value in pairs:
        if name in HOP_BY_HOP_HEADERS or name in nominated:
            continue
        yield name, value


def forward_request_headers(
    headers: Iterable[tuple[str, str]],
    *,
    host: str | None = None,
    extra: Mapping[str, str] | None = None,
    trust_client_headers: bool = False,
) -> dict[str, str]:
    """Build the header mapping sent to the backend for an inbound request.

    Args:
        headers: Raw ``(lowercase_name, value)`` pairs from the inbound ASGI scope.
        host: Value for the upstream ``Host`` header (the backend's own host).
        extra: Headers that override inbound ones of the same name, e.g. a
            static ``Authorization`` token for the backend.
        trust_client_headers: When true, relay all inbound headers except
            hop-by-hop ones. When false, relay only :data:`MCP_REQUEST_HEADERS`
            and ``Mcp-Param-*``, which drops client cookies, user agents and
            credentials unless they were explicitly listed in ``extra``.

    Returns:
        A single-valued header mapping. Duplicate inbound values for the same
        name collapse, last one winning — acceptable for the MCP wire headers
        because the protocol rejects duplicated routing headers itself (see
        `mcp.shared.inbound.find_duplicated_routing_header`).
    """
    result: dict[str, str] = {}
    for name, value in _strip_hop_by_hop(headers):
        if name in HOPPED_REQUEST_EXTRA:
            continue
        if not trust_client_headers and not is_mcp_wire_header(name):
            continue
        result[name] = value

    if host is not None:
        result["host"] = host
    if extra:
        # Lowercase keys so caller-supplied casing cannot introduce duplicates.
        for name, value in extra.items():
            result[name.lower()] = value
    return result


def forward_response_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    """Convert backend response headers into raw ASGI header pairs.

    Hop-by-hop headers are dropped. `Set-Cookie` is preserved as-is because it
    may legitimately carry backend session state; multi-value headers survive
    because the return type is a list of pairs, not a mapping.
    """
    out: list[tuple[bytes, bytes]] = []
    for name, value in _strip_hop_by_hop(headers):
        if name in HOPPED_RESPONSE_EXTRA:
            continue
        out.append((name.encode("latin-1"), value.encode("latin-1")))
    return out
