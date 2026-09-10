"""FAVA pass-through MCP reverse proxy.

A transport-level relay that forwards JSON-RPC traffic between an MCP client
(an agent runtime) and a remote MCP server over Streamable HTTP, without
interpreting or rewriting the protocol.

The relay is deliberately dumb: bytes in, bytes out. Every method, id, header
and SSE event is passed through untouched, so it can be dropped in front of an
existing agent/MCP deployment with no changes on either side.

The single extension point is :class:`mcp_proxy.hooks.ProxyHooks`, which
observes requests and responses without being able to alter them yet. That is
where FAVA's authorization stage will attach:
``observe request -> authorize -> forward or deny -> observe result``.
"""

from __future__ import annotations

from mcp_proxy.config import ProxySettings
from mcp_proxy.hooks import NullHooks, ProxyHooks, RequestContext, ResponseContext
from mcp_proxy.messages import WireMessage, parse_wire_messages
from mcp_proxy.relay import ForwardingProxy
from mcp_proxy.app import create_proxy_app

__all__ = [
    "ForwardingProxy",
    "NullHooks",
    "ProxyHooks",
    "ProxySettings",
    "RequestContext",
    "ResponseContext",
    "WireMessage",
    "create_proxy_app",
    "parse_wire_messages",
]
