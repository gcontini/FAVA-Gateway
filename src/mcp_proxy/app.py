"""ASGI application factory for the forwarding MCP proxy.

:func:`create_proxy_app` is the integration point for a deployment: it owns
settings, hooks and CORS, and returns an ASGI app you hand to ``uvicorn`` or
mount inside an existing Starlette/FastAPI router.

The returned app is deliberately thin. :class:`~mcp_proxy.relay.ForwardingProxy`
already implements the full ASGI lifecycle (including owning the upstream HTTP
client across startup/shutdown); this module only attaches process-wide
concerns around it, so swapping in the real FAVA gateway later means replacing
the relay — not the app.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from mcp_proxy.config import ProxySettings
from mcp_proxy.hooks import LoggingHooks, ProxyHooks, normalize_hooks
from mcp_proxy.relay import ForwardingProxy

logger = logging.getLogger(__name__)


def create_proxy_app(
    settings: ProxySettings,
    *,
    hooks: ProxyHooks | Sequence[ProxyHooks] | None = None,
    access_log: bool = False,
    allow_origins: Sequence[str] = (),
    expose_session_header: bool = True,
) -> ASGIApp:
    """Build the proxy's ASGI application.

    Args:
        settings: Backend URL, mount path and relay limits.
        hooks: Observers invoked around each exchange. See
            :func:`mcp_proxy.hooks.normalize_hooks`.
        access_log: Add a :class:`~mcp_proxy.hooks.LoggingHooks` observer that
            logs one line per exchange. Convenient when running by hand.
        allow_origins: Origins permitted by CORS preflight. Empty disables the
            CORS middleware entirely, which is the right default for a proxy
            reached by a server-side agent runtime rather than a browser.
        expose_session_header: When CORS is enabled, expose ``Mcp-Session-Id``
            to browser clients so a JavaScript MCP transport can read the
            backend session id from the relayed response.

    Returns:
        An ASGI app. Pass it to ``uvicorn.Server``, ``uvicorn.run``, or mount it
        under a path in an existing router.
    """
    observers: list[ProxyHooks] = list(hooks) if isinstance(hooks, Sequence) else [hooks] if hooks is not None else []
    if access_log:
        observers.append(LoggingHooks())

    proxy = ForwardingProxy(settings, hooks=normalize_hooks(observers or None))

    app: ASGIApp = proxy
    if allow_origins:
        app = CORSMiddleware(
            app,
            allow_origins=list(allow_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id", "MCP-Protocol-Version"] if expose_session_header else [],
        )
        logger.info("CORS enabled for origins %s", list(allow_origins))

    logger.info(
        "Forwarding MCP proxy: %s -> %s (mount=%r)",
        f"{settings.host}:{settings.port}",
        settings.backend_url,
        settings.mount_path,
    )
    return app
