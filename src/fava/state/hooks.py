"""The seam: a :class:`~llm_proxy.hooks.ProxyHooks` that feeds the store.

Attaching this to the relay is the whole integration::

    from llm_proxy import ProxySettings, create_proxy_app
    from fava.state import RecordStore, StateHooks

    store = RecordStore()
    app = create_proxy_app(settings, hooks=StateHooks(store))

The hook stays **read-only**, which is what makes it safe to deploy now. The
allow/block return value belongs on ``on_response`` — tool-call intent arrives
from the provider, and the harness dispatches only after the response reaches it
— but that is the authorizer's phase, not this one. Until then a failure here
costs an observation, never a relayed byte.
"""

from __future__ import annotations

import logging

from fava.state.store import RecordStore
from llm_proxy.hooks import RequestContext, ResponseContext

logger = logging.getLogger(__name__)


class StateHooks:
    """Records every observed exchange into a :class:`RecordStore`.

    Args:
        store: Where runs are kept. Share one across the process: run identity
            is recovered per request, so two stores would split one run's
            evidence in half.
    """

    def __init__(self, store: RecordStore) -> None:
        self.store = store

    async def on_request(self, context: RequestContext) -> None:
        """Record the run's task, catalog and returned tool results."""
        await self.store.observe_request(context)

    async def on_response(self, context: ResponseContext) -> None:
        """Record the tool calls the model asked for, before they are run."""
        run = await self.store.observe_response(context)
        if run is None or not logger.isEnabledFor(logging.DEBUG):
            return
        graph = run.graph()
        logger.debug(
            "run %s graph v%d: %d nodes, %d edges, posture=%s",
            run.run_id[:8],
            graph.version,
            len(graph.nodes),
            len(graph.edges),
            run.ir.risk_posture.value,
        )

    async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
        """Nothing to record: no response reached the client, so no intent exists."""


__all__ = ["StateHooks"]
