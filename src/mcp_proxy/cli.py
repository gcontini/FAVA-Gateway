"""Command-line entrypoint for the forwarding MCP proxy.

Usage::

    uv run fava-mcp-proxy --backend-url http://127.0.0.1:8901/mcp --port 8900

Or entirely from the environment (``ProxySettings.from_env``)::

    MCP_PROXY_BACKEND_URL=http://127.0.0.1:8901/mcp uv run fava-mcp-proxy

The backend is an ordinary remote MCP server speaking Streamable HTTP; this
process is a reverse proxy that relays its JSON-RPC traffic unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence

import uvicorn

from mcp_proxy.app import create_proxy_app
from mcp_proxy.config import (
    ENV_BACKEND_URL,
    ENV_HOST,
    ENV_MAX_BODY,
    ENV_MOUNT_PATH,
    ENV_PORT,
    ENV_SSE_TIMEOUT,
    ENV_TIMEOUT,
    ProxySettings,
)
from mcp_proxy.hooks import LoggingHooks

_LOG_LEVELS = ("critical", "error", "warning", "info", "debug")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcp-proxy",
        description="Pass-through MCP reverse proxy: relays JSON-RPC to a remote MCP server.",
    )
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Remote MCP server endpoint, e.g. http://127.0.0.1:8901/mcp. "
        f"Defaults to the MCP_PROXY_BACKEND_URL environment variable.",
    )
    parser.add_argument("--host", default=None, help="Bind address for the proxy. Default 127.0.0.1.")
    parser.add_argument("--port", type=int, default=None, help="Bind port for the proxy. Default 8900.")
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Path the proxy serves. Default /mcp. Pass '*' to accept any path.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="Extra header sent to the backend, repeatable. Use for a static "
        "backend credential, e.g. --header 'Authorization: Bearer ...'.",
    )
    parser.add_argument(
        "--trust-client-headers",
        action="store_true",
        help="Relay all client request headers upstream except hop-by-hop ones. "
        "Off by default: only MCP wire headers are relayed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Upstream connect/write timeout in seconds. Default 30.",
    )
    parser.add_argument(
        "--sse-read-timeout",
        type=float,
        default=None,
        help="Upstream read timeout in seconds. Default 300, because a "
        "streamable HTTP server may hold a response stream open.",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="CORS origin, repeatable. Off by default (a server-side agent "
        "runtime does not need it).",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Log one line per relayed exchange (method, tool name, status).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=_LOG_LEVELS,
        help="Logging verbosity. Default info.",
    )
    return parser


def parse_extra_headers(raw: Sequence[str]) -> dict[str, str]:
    """Parse ``--header NAME:VALUE`` arguments into a mapping.

    Raises:
        SystemExit: If a value is not in ``NAME:VALUE`` form.
    """
    headers: dict[str, str] = {}
    for entry in raw:
        name, separator, value = entry.partition(":")
        if not separator or not name.strip():
            raise SystemExit(f"--header expects NAME:VALUE, got {entry!r}")
        headers[name.strip()] = value.strip()
    return headers


# Documented CLI defaults, used for every flag not passed. The defaults are
# spelled out here rather than read off `ProxySettings` so `--help` stays honest
# when the dataclass evolves.
_CLI_DEFAULTS = ProxySettings(backend_url="http://127.0.0.1:0/mcp")

_ARG_DEFAULTS: dict[str, object] = {
    "host": "127.0.0.1",
    "port": 8900,
    "mount_path": "/mcp",
    "timeout": 30.0,
    "sse_read_timeout": 300.0,
}


def settings_from_args(args: argparse.Namespace, environ: Mapping[str, str] | None = None) -> ProxySettings:
    """Combine CLI arguments, then environment, then defaults into settings.

    Precedence per field is: explicit CLI flag, environment variable, default.
    The backend URL is the only required value and is read from
    ``--backend-url`` or ``MCP_PROXY_BACKEND_URL``.

    Args:
        args: Parsed CLI namespace.
        environ: Environment mapping, defaulting to ``os.environ``. Injected for
            tests.

    Raises:
        SystemExit: If no backend URL was supplied by either source.
    """
    env = os.environ if environ is None else environ
    backend_url = args.backend_url or env.get(ENV_BACKEND_URL)
    if not backend_url:
        raise SystemExit(f"a backend URL is required: pass --backend-url or set {ENV_BACKEND_URL}")

    return ProxySettings(
        backend_url=backend_url,
        mount_path=_mount_path(_resolve(args.mount_path, env.get(ENV_MOUNT_PATH), _ARG_DEFAULTS["mount_path"])),
        host=_resolve(args.host, env.get(ENV_HOST), _ARG_DEFAULTS["host"]),
        port=int(_resolve(args.port, env.get(ENV_PORT), _ARG_DEFAULTS["port"])),
        extra_headers=parse_extra_headers(args.header),
        query_passthrough=_CLI_DEFAULTS.query_passthrough,
        timeout=float(_resolve(args.timeout, env.get(ENV_TIMEOUT), _ARG_DEFAULTS["timeout"])),
        sse_read_timeout=float(
            _resolve(args.sse_read_timeout, env.get(ENV_SSE_TIMEOUT), _ARG_DEFAULTS["sse_read_timeout"])
        ),
        max_body_size=int(env.get(ENV_MAX_BODY) or _CLI_DEFAULTS.max_body_size),
        health_path=_CLI_DEFAULTS.health_path,
        trust_client_headers=args.trust_client_headers,
    )


def _resolve(arg_value: object, env_value: str | None, default: object) -> object:
    """Pick the first supplied value among CLI arg, environment, default."""
    if arg_value is not None:
        return arg_value
    if env_value not in (None, ""):
        return env_value
    return default


def _mount_path(raw: object) -> str | None:
    """Normalize the mount path, mapping ``*`` and ``/`` to no restriction."""
    if raw is None:
        return None
    text = str(raw)
    return None if text in ("", "*", "/") else text


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, build the app and run it under uvicorn.

    Returns:
        The process exit code (0 on a clean shutdown).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    settings = settings_from_args(args)
    app = create_proxy_app(
        settings,
        hooks=[LoggingHooks()] if args.access_log else None,
        access_log=False,
        allow_origins=args.allow_origin,
    )

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level, lifespan="on")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
