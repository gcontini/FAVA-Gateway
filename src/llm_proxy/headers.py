"""HTTP header handling for the pass-through relay.

The proxy must not confuse the agent's HTTP connection with the provider's.
Hop-by-hop headers are therefore never relayed in either direction, while the
headers an LLM API actually routes on — content negotiation, the credential,
and the vendor's own request-scoping headers — are relayed verbatim.

Unlike an MCP proxy, this one sits on the **credential path**: without an
``Authorization`` header the upstream request cannot succeed at all. See
:data:`API_REQUEST_HEADERS` and ``ProxySettings.forward_client_auth``.
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
# `content-length` is excluded too: httpx recomputes it from the relayed body.
HOPPED_REQUEST_EXTRA = frozenset({"host", "content-length"})

# Response headers the ASGI server sets itself.
HOPPED_RESPONSE_EXTRA = frozenset({"content-length"})

# Never forwarded upstream regardless of `trust_client_headers`.
#
# `accept-encoding` is the load-bearing one. The relay streams the upstream's
# *raw* bytes, so a gzipped response would reach the client correctly but reach
# the observer as compressed noise — tool-call intents would be invisible,
# which is the one thing this proxy exists to see. Forcing identity encoding on
# the upstream leg keeps "what we relay" and "what we observe" the same bytes.
STRIPPED_REQUEST_HEADERS = frozenset({"accept-encoding"})

# Headers that MUST reach the provider for a request to work, relayed even when
# `trust_client_headers` is off. `authorization` is gated separately by
# `ProxySettings.forward_client_auth`.
API_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "authorization",
        "content-type",
        "user-agent",
        # Azure OpenAI deployments authenticate with `api-key` instead.
        "api-key",
    }
)

# Vendor request-scoping and SDK telemetry headers, matched by prefix because
# the names are open-ended: `openai-organization`, `openai-project`,
# `openai-beta`, and the OpenAI SDK's own `x-stainless-*` build metadata.
API_REQUEST_HEADER_PREFIXES = ("openai-", "x-stainless-")


def is_api_wire_header(name: str, *, forward_client_auth: bool = True) -> bool:
    """Whether a lowercase header name must reach the provider.

    Args:
        name: Lowercased inbound header name.
        forward_client_auth: When false, ``authorization`` is not treated as a
            wire header, so the client's credential is dropped and only a
            deployment-supplied one (via ``extra_headers``) reaches upstream.
    """
    if name == "authorization":
        return forward_client_auth
    return name in API_REQUEST_HEADERS or name.startswith(API_REQUEST_HEADER_PREFIXES)


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    """Yield headers, dropping hop-by-hop ones and any named by `Connection:`."""
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
    forward_client_auth: bool = True,
) -> dict[str, str]:
    """Build the header mapping sent upstream for an inbound request.

    Args:
        headers: Raw ``(lowercase_name, value)`` pairs from the inbound ASGI scope.
        host: Value for the upstream ``Host`` header (the provider's own host).
        extra: Headers that override inbound ones of the same name, e.g. the
            deployment's own ``Authorization`` token.
        trust_client_headers: When true, relay all inbound headers except
            hop-by-hop and stripped ones. When false, relay only the API wire
            headers, dropping client cookies and unrelated credentials.
        forward_client_auth: Relay the client's own ``Authorization`` header.

    Returns:
        A single-valued header mapping with identity content-coding forced.
        Duplicate inbound values for the same name collapse, last one winning.
    """
    result: dict[str, str] = {}
    for name, value in _strip_hop_by_hop(headers):
        if name in HOPPED_REQUEST_EXTRA or name in STRIPPED_REQUEST_HEADERS:
            continue
        if name == "authorization" and not forward_client_auth:
            continue
        if not trust_client_headers and not is_api_wire_header(
            name, forward_client_auth=forward_client_auth
        ):
            continue
        result[name] = value

    # Set explicitly rather than merely dropping the inbound value: httpx adds
    # its own `accept-encoding` default otherwise, and the upstream would
    # compress a response the observer then cannot read.
    result["accept-encoding"] = "identity"

    if host is not None:
        result["host"] = host
    if extra:
        # Lowercase keys so caller-supplied casing cannot introduce duplicates.
        for name, value in extra.items():
            result[name.lower()] = value
    return result


def forward_response_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    """Convert upstream response headers into raw ASGI header pairs.

    Hop-by-hop headers are dropped and ``content-length`` is left to the ASGI
    server. Everything else survives — including ``set-cookie`` and the
    ``x-ratelimit-*`` family, which a client may legitimately act on — because
    the return type is a list of pairs rather than a mapping, so repeated
    headers are not collapsed.
    """
    out: list[tuple[bytes, bytes]] = []
    for name, value in _strip_hop_by_hop(headers):
        if name in HOPPED_RESPONSE_EXTRA:
            continue
        out.append((name.encode("latin-1"), value.encode("latin-1")))
    return out


def redact(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy a header mapping with credential values masked, for logging.

    The proxy is on the credential path, so no log line may ever carry a raw
    key. Applied at the point of logging rather than at parse time, because
    hooks legitimately need the real value to forward it.
    """
    masked = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in ("authorization", "api-key", "cookie", "set-cookie"):
            masked[name] = "<redacted>"
        else:
            masked[name] = value
    return masked
