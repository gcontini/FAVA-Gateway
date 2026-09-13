#!/usr/bin/env python3
"""Start the proxy in front of a real LLM provider, for manual testing.

Holds only the upstream provider's URL. The credential stays with the
caller: the proxy relays whatever `Authorization` the client sends
(`forward_client_auth` defaults to True) rather than holding it itself — see
``call_model.py``.

A :class:`~fava.state.RecordStore` is attached, so every exchange is recorded
into a run and lowered into a permission graph, printed after each response.
The IR extractor reuses the same provider and the same credential the agent
sent, unless ``FAVA_EXTRACTOR_*`` says otherwise; with no credential at all it
is skipped and every run stays ``ambiguous``.

Setup::

    cp .env.example .env
    # edit .env: set MODEL_BASE_URL for your provider

Run (blocks until Ctrl+C)::

    uv run python scripts/start_proxy.py
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from fava.state import ExtractorSettings, LlmIRExtractor, RecordStore, StateHooks
from fava.state.graph import PermissionGraph
from llm_proxy.app import create_proxy_app
from llm_proxy.config import ProxySettings
from llm_proxy.hooks import ResponseContext

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8901


def load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from a plain ``KEY=VALUE`` .env file.

    Deliberately hand-rolled rather than a `python-dotenv` dependency: the
    format needed here is a couple of flat variables, nothing more.
    """
    if not path.exists():
        raise SystemExit(f"missing {path} — copy .env.example to .env and fill in your provider's details")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class PrintGraph(StateHooks):
    """Records into the store, then prints the run's graph after each response."""

    async def on_response(self, context: ResponseContext) -> None:
        """Record the exchange, then show what the graph looks like now."""
        await super().on_response(context)
        run = self.store.get_for(context.request)
        if run is None:
            return
        print(f"\nrun {run.run_id[:8]}  posture={run.ir.risk_posture.value}  intent={run.ir.intent or '-'}")
        _print_graph(run.graph())
        for violation in run.validate():
            print(f"  !! {violation.code}: {violation.detail}")


def _print_graph(graph: PermissionGraph) -> None:
    """Render a permission graph as two indented lists."""
    print(f"  graph v{graph.version}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    for node in graph.nodes:
        requests = " ".join(node.requests)
        labels = ",".join(label.name for label in node.labels)
        print(f"    [{node.kind.value:<7}] {node.id}  op={node.op or '-'}  {requests}  {labels}".rstrip())
    for edge in graph.edges:
        print(f"    {edge.type.value:<7} {edge.src} -> {edge.dst}  ({edge.trust.value})")


def main() -> int:
    load_dotenv(ENV_PATH)
    model_base_url = os.environ["MODEL_BASE_URL"]

    settings = ProxySettings(upstream_url=model_base_url, host=PROXY_HOST, port=PROXY_PORT)
    extractor = LlmIRExtractor(ExtractorSettings.from_env(default_base_url=model_base_url))
    store = RecordStore(extractor=extractor)
    app = create_proxy_app(settings, hooks=PrintGraph(store), access_log=True)

    print(f"proxy listening: {settings.base_url_for_clients} -> {model_base_url}")
    print(f"IR extraction:   {extractor.settings.base_url} (model={extractor.settings.model or 'as requested'})")
    print("point call_model.py's PROXY_URL at the line above, then run it in another terminal")

    uvicorn.run(app, host=settings.host, port=settings.port, lifespan="on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
