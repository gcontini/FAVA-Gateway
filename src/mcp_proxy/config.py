"""Configuration for the pass-through MCP proxy."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

# Environment variable names read by `ProxySettings.from_env`.
ENV_BACKEND_URL = "MCP_PROXY_BACKEND_URL"
ENV_HOST = "MCP_PROXY_HOST"
ENV_PORT = "MCP_PROXY_PORT"
ENV_MOUNT_PATH = "MCP_PROXY_MOUNT_PATH"
ENV_TIMEOUT = "MCP_PROXY_TIMEOUT"
ENV_SSE_TIMEOUT = "MCP_PROXY_SSE_READ_TIMEOUT"
ENV_MAX_BODY = "MCP_PROXY_MAX_BODY_SIZE"


@dataclass(frozen=True)
class ProxySettings:
    """Static proxy configuration.

    Attributes:
        backend_url: Absolute URL of the remote (upstream) MCP server endpoint,
            e.g. ``http://tools.example.internal:8901/mcp``. Every relayed
            request is sent here.
        mount_path: Path the proxy itself listens on. ``None`` accepts any path
            and forwards to ``backend_url`` regardless.
        host: Bind address for the proxy's own HTTP server.
        port: Bind port for the proxy's own HTTP server.
        extra_headers: Headers added to every upstream request (e.g. a static
            ``Authorization`` bearer token for the backend). They override any
            client-supplied header of the same name.
        query_passthrough: Forward the inbound query string when relaying.
        timeout: Seconds for connect/write/pool on the upstream request.
        sse_read_timeout: Seconds between upstream reads; must be generous
            because Streamable HTTP holds response streams open.
        max_body_size: Upper bound on a client request body, in bytes. Bodies
            are buffered so that hooks can parse them; this bounds memory use.
        health_path: Path served locally by the proxy (not relayed), or ``None``
            to disable it.
        trust_client_headers: When true, forward client request headers
            verbatim except hop-by-hop fields. When false, only the MCP wire
            headers in :data:`mcp_proxy.headers.FORWARDED_REQUEST_HEADERS` are
            relayed, which is the safer default for an exposed proxy.
    """

    backend_url: str
    mount_path: str | None = "/mcp"
    host: str = "127.0.0.1"
    port: int = 8900
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    query_passthrough: bool = True
    timeout: float = 30.0
    sse_read_timeout: float = 300.0
    max_body_size: int = 4 * 1024 * 1024
    health_path: str | None = "/healthz"
    trust_client_headers: bool = False

    def __post_init__(self) -> None:
        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.backend_url.startswith(("http://", "https://")):
            raise ValueError(f"backend_url must be an absolute HTTP(S) URL, got {self.backend_url!r}")
        if self.max_body_size <= 0:
            raise ValueError("max_body_size must be positive")
        if self.timeout <= 0 or self.sse_read_timeout <= 0:
            raise ValueError("timeouts must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProxySettings:
        """Build settings from environment variables.

        Raises:
            ValueError: If ``MCP_PROXY_BACKEND_URL`` is unset or a value is not
                parseable.
        """
        env = os.environ if environ is None else environ
        backend_url = env.get(ENV_BACKEND_URL)
        if not backend_url:
            raise ValueError(f"{ENV_BACKEND_URL} is required")

        mount_path = env.get(ENV_MOUNT_PATH, "/mcp")
        return cls(
            backend_url=backend_url,
            mount_path=None if mount_path in ("", "*", "/") else mount_path,
            host=env.get(ENV_HOST, "127.0.0.1"),
            port=_env_int(env, ENV_PORT, 8900),
            timeout=_env_float(env, ENV_TIMEOUT, 30.0),
            sse_read_timeout=_env_float(env, ENV_SSE_TIMEOUT, 300.0),
            max_body_size=_env_int(env, ENV_MAX_BODY, 4 * 1024 * 1024),
        )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
