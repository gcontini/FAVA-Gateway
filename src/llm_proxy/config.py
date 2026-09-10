"""Configuration for the pass-through LLM API proxy."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

# Environment variable names read by `ProxySettings.from_env`.
ENV_UPSTREAM_URL = "LLM_PROXY_UPSTREAM_URL"
ENV_HOST = "LLM_PROXY_HOST"
ENV_PORT = "LLM_PROXY_PORT"
ENV_MOUNT_PREFIX = "LLM_PROXY_MOUNT_PREFIX"
ENV_TIMEOUT = "LLM_PROXY_TIMEOUT"
ENV_STREAM_READ_TIMEOUT = "LLM_PROXY_STREAM_READ_TIMEOUT"
ENV_MAX_BODY = "LLM_PROXY_MAX_BODY_SIZE"

DEFAULT_MOUNT_PREFIX = "/v1"
DEFAULT_PORT = 8900
DEFAULT_TIMEOUT = 30.0
DEFAULT_STREAM_READ_TIMEOUT = 600.0
DEFAULT_MAX_BODY_SIZE = 32 * 1024 * 1024


@dataclass(frozen=True)
class ProxySettings:
    """Static proxy configuration.

    Attributes:
        upstream_url: Absolute **base** URL of the LLM API, e.g.
            ``https://api.openai.com/v1``. The inbound path suffix is appended
            to it, so one proxy serves every endpoint under the base rather
            than a single pinned route.
        mount_prefix: Path prefix the proxy serves, e.g. ``/v1``. The remainder
            of the inbound path is the suffix forwarded upstream. ``None``
            accepts any path and forwards it whole.
        host: Bind address for the proxy's own HTTP server.
        port: Bind port for the proxy's own HTTP server.
        extra_headers: Headers added to every upstream request. They override
            any client-supplied header of the same name, which is how a
            deployment injects its own ``Authorization`` and stops the agent's
            key from ever reaching the provider.
        query_passthrough: Forward the inbound query string when relaying.
        timeout: Seconds for connect/write/pool on the upstream request.
        stream_read_timeout: Seconds between upstream reads. Generous, because
            a streamed completion may pause between tokens for a long time,
            and a long tool-calling turn holds the response open throughout.
        max_body_size: Upper bound on a client request body, in bytes. Bodies
            are buffered so hooks can parse them; this bounds memory use. The
            default is large because a chat request grows with the whole
            conversation history, which for a long agent run is megabytes.
        health_path: Path served locally by the proxy (never relayed), or
            ``None`` to disable it.
        trust_client_headers: When true, forward client request headers
            verbatim except hop-by-hop fields. When false, only the headers in
            :data:`llm_proxy.headers.API_REQUEST_HEADERS` and the recognized
            vendor prefixes are relayed.
        forward_client_auth: Relay the client's own ``Authorization`` header
            upstream. True by default: unlike an MCP proxy, this proxy sits on
            the credential path and the request cannot succeed without a key.
            Set false when the deployment supplies its own via
            ``extra_headers``.
    """

    upstream_url: str
    mount_prefix: str | None = DEFAULT_MOUNT_PREFIX
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    query_passthrough: bool = True
    timeout: float = DEFAULT_TIMEOUT
    stream_read_timeout: float = DEFAULT_STREAM_READ_TIMEOUT
    max_body_size: int = DEFAULT_MAX_BODY_SIZE
    health_path: str | None = "/healthz"
    trust_client_headers: bool = False
    forward_client_auth: bool = True

    def __post_init__(self) -> None:
        if not self.upstream_url:
            raise ValueError("upstream_url is required")
        if not self.upstream_url.startswith(("http://", "https://")):
            raise ValueError(f"upstream_url must be an absolute HTTP(S) URL, got {self.upstream_url!r}")
        if self.max_body_size <= 0:
            raise ValueError("max_body_size must be positive")
        if self.timeout <= 0 or self.stream_read_timeout <= 0:
            raise ValueError("timeouts must be positive")
        # A trailing slash would produce `//chat/completions` once the suffix is
        # appended. Normalize here so the relay can join without thinking.
        object.__setattr__(self, "upstream_url", self.upstream_url.rstrip("/"))

    @property
    def base_url_for_clients(self) -> str:
        """The ``base_url`` an OpenAI client should be pointed at."""
        prefix = self.mount_prefix or ""
        return f"http://{self.host}:{self.port}{prefix}"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProxySettings:
        """Build settings from environment variables.

        Raises:
            ValueError: If ``LLM_PROXY_UPSTREAM_URL`` is unset or a value is not
                parseable.
        """
        env = os.environ if environ is None else environ
        upstream_url = env.get(ENV_UPSTREAM_URL)
        if not upstream_url:
            raise ValueError(f"{ENV_UPSTREAM_URL} is required")

        return cls(
            upstream_url=upstream_url,
            mount_prefix=normalize_mount_prefix(env.get(ENV_MOUNT_PREFIX, DEFAULT_MOUNT_PREFIX)),
            host=env.get(ENV_HOST, "127.0.0.1"),
            port=_env_int(env, ENV_PORT, DEFAULT_PORT),
            timeout=_env_float(env, ENV_TIMEOUT, DEFAULT_TIMEOUT),
            stream_read_timeout=_env_float(env, ENV_STREAM_READ_TIMEOUT, DEFAULT_STREAM_READ_TIMEOUT),
            max_body_size=_env_int(env, ENV_MAX_BODY, DEFAULT_MAX_BODY_SIZE),
        )


def normalize_mount_prefix(raw: str | None) -> str | None:
    """Normalize a mount prefix, mapping ``*``, ``/`` and ``""`` to no restriction.

    A prefix is stored without its trailing slash so that ``/v1`` and ``/v1/``
    behave identically when the relay strips it off an inbound path.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text in ("", "*", "/"):
        return None
    if not text.startswith("/"):
        text = f"/{text}"
    return text.rstrip("/") or None


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
