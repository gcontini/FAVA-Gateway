"""Command-line entrypoint for the forwarding LLM API proxy.

Usage::

    uv run fava-llm-proxy --upstream-url https://api.openai.com/v1 --port 8900

Or entirely from the environment (``ProxySettings.from_env``)::

    LLM_PROXY_UPSTREAM_URL=https://api.openai.com/v1 uv run fava-llm-proxy

The agent harness is then pointed at ``http://127.0.0.1:8900/v1`` as its
OpenAI base URL — that swap is the whole integration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence

import uvicorn

from llm_proxy.app import create_proxy_app
from llm_proxy.config import (
    DEFAULT_MAX_BODY_SIZE,
    DEFAULT_MOUNT_PREFIX,
    DEFAULT_PORT,
    DEFAULT_STREAM_READ_TIMEOUT,
    DEFAULT_TIMEOUT,
    ENV_HOST,
    ENV_MAX_BODY,
    ENV_MOUNT_PREFIX,
    ENV_PORT,
    ENV_STREAM_READ_TIMEOUT,
    ENV_TIMEOUT,
    ENV_UPSTREAM_URL,
    ProxySettings,
    normalize_mount_prefix,
)
from llm_proxy.hooks import LoggingHooks

_LOG_LEVELS = ("critical", "error", "warning", "info", "debug")

# Read by `--api-key` when the flag is omitted. Deliberately not `OPENAI_API_KEY`:
# a proxy that silently adopted the ambient provider key would inject a
# credential the operator never asked it to.
ENV_API_KEY = "LLM_PROXY_API_KEY"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="fava-llm-proxy",
        description="Pass-through LLM API reverse proxy: relays OpenAI-compatible traffic to a provider.",
    )
    parser.add_argument(
        "--upstream-url",
        default=None,
        help="Upstream API base URL, e.g. https://api.openai.com/v1. "
        f"Defaults to the {ENV_UPSTREAM_URL} environment variable.",
    )
    parser.add_argument("--host", default=None, help="Bind address for the proxy. Default 127.0.0.1.")
    parser.add_argument("--port", type=int, default=None, help=f"Bind port for the proxy. Default {DEFAULT_PORT}.")
    parser.add_argument(
        "--mount-prefix",
        default=None,
        help=f"Path prefix the proxy serves. Default {DEFAULT_MOUNT_PREFIX}. Pass '*' to accept any path.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Inject 'Authorization: Bearer <key>' on every upstream request, overriding "
        f"whatever the client sent. Defaults to the {ENV_API_KEY} environment variable. "
        "Pair with --no-forward-client-auth to keep agent-supplied keys off the wire.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="Extra header sent upstream, repeatable. Overrides an inbound header of the same name.",
    )
    parser.add_argument(
        "--no-forward-client-auth",
        action="store_true",
        help="Drop the client's own Authorization header instead of relaying it. "
        "Use with --api-key so the proxy, not the agent, holds the provider credential.",
    )
    parser.add_argument(
        "--trust-client-headers",
        action="store_true",
        help="Relay all client request headers upstream except hop-by-hop ones. "
        "Off by default: only API wire headers are relayed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Upstream connect/write timeout in seconds. Default {DEFAULT_TIMEOUT:g}.",
    )
    parser.add_argument(
        "--stream-read-timeout",
        type=float,
        default=None,
        help=f"Upstream read timeout in seconds. Default {DEFAULT_STREAM_READ_TIMEOUT:g}, because a "
        "model may pause a long time between tokens.",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="CORS origin, repeatable. Off by default (a server-side agent harness does not need it).",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Log one line per relayed exchange, including the tool calls the model requested.",
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


# Documented CLI defaults, used for every flag not passed. Spelled out here
# rather than read off `ProxySettings` so `--help` stays honest as it evolves.
_ARG_DEFAULTS: dict[str, object] = {
    "host": "127.0.0.1",
    "port": DEFAULT_PORT,
    "mount_prefix": DEFAULT_MOUNT_PREFIX,
    "timeout": DEFAULT_TIMEOUT,
    "stream_read_timeout": DEFAULT_STREAM_READ_TIMEOUT,
}


def settings_from_args(args: argparse.Namespace, environ: Mapping[str, str] | None = None) -> ProxySettings:
    """Combine CLI arguments, then environment, then defaults into settings.

    Precedence per field is: explicit CLI flag, environment variable, default.
    The upstream URL is the only required value.

    Args:
        args: Parsed CLI namespace.
        environ: Environment mapping, defaulting to ``os.environ``. Injected for
            tests.

    Raises:
        SystemExit: If no upstream URL was supplied by either source.
    """
    env = os.environ if environ is None else environ
    upstream_url = args.upstream_url or env.get(ENV_UPSTREAM_URL)
    if not upstream_url:
        raise SystemExit(f"an upstream URL is required: pass --upstream-url or set {ENV_UPSTREAM_URL}")

    extra_headers = parse_extra_headers(args.header)
    api_key = args.api_key or env.get(ENV_API_KEY)
    if api_key:
        # Set after --header parsing so an explicit --header 'Authorization: ...'
        # can still be overridden predictably by the more specific flag.
        extra_headers["Authorization"] = f"Bearer {api_key}"

    return ProxySettings(
        upstream_url=upstream_url,
        mount_prefix=normalize_mount_prefix(
            str(_resolve(args.mount_prefix, env.get(ENV_MOUNT_PREFIX), _ARG_DEFAULTS["mount_prefix"]))
        ),
        host=str(_resolve(args.host, env.get(ENV_HOST), _ARG_DEFAULTS["host"])),
        port=int(_resolve(args.port, env.get(ENV_PORT), _ARG_DEFAULTS["port"])),
        extra_headers=extra_headers,
        timeout=float(_resolve(args.timeout, env.get(ENV_TIMEOUT), _ARG_DEFAULTS["timeout"])),
        stream_read_timeout=float(
            _resolve(args.stream_read_timeout, env.get(ENV_STREAM_READ_TIMEOUT), _ARG_DEFAULTS["stream_read_timeout"])
        ),
        max_body_size=int(env.get(ENV_MAX_BODY) or DEFAULT_MAX_BODY_SIZE),
        trust_client_headers=args.trust_client_headers,
        forward_client_auth=not args.no_forward_client_auth,
    )


def _resolve(arg_value: object, env_value: str | None, default: object) -> object:
    """Pick the first supplied value among CLI arg, environment, default."""
    if arg_value is not None:
        return arg_value
    if env_value not in (None, ""):
        return env_value
    return default


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

    logger = logging.getLogger(__name__)
    logger.info("Point your agent harness at base_url=%s", settings.base_url_for_clients)

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level, lifespan="on")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
