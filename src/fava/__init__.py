"""FAVA: formal authorization for verified agents, above the proxy transport.

:mod:`llm_proxy` is the wire — a byte-exact relay at the harness↔LLM boundary
that observes traffic and forwards it unchanged. This package is everything
that reasons about what it saw.

The dependency arrow points one way and must stay that way: ``fava`` imports
``llm_proxy`` for its parsed views of the wire, and ``llm_proxy`` never imports
``fava``. That is what keeps the relay deployable on its own and keeps a bug in
the authorization stage from being able to break the transport.

Today only :mod:`fava.state` exists — run identity, an append-only event log,
the Permission IR, and the permission graph lowered from both. The authorizer
and the enforcing gateway follow; see ``docs/implementation_plan.md``.
"""
