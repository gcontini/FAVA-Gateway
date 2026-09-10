# Project Guidelines

## What this repo is

Phase 1 of FAVA: a **pass-through MCP reverse proxy** over Streamable HTTP.
It relays JSON-RPC bytes unchanged and observes them via hooks. There is no
authorization yet — that is what the hooks are reserved for.

Background: [`README.md`](README.md) (usage, API, design rationale).
Roadmap: [`docs/implementation_plan.md`](docs/implementation_plan.md).
Paper: `docs/FAVA_ Formal Authorization for Verified Agents….html`.

## Build and Test

Always use `uv`. Never install Python packages globally.

```bash
uv sync --group dev     # install/create .venv
uv run pytest           # 38 tests, ~3.5s
```

If a test run hangs, suspect the relay's task group, not pytest. Prefer a hard
cut so a hang costs 200s instead of forever:

```bash
timeout -s KILL 200 uv run pytest -q --no-header
```

`tests/test_forwarding_e2e.py` binds **real free TCP ports** and runs two
uvicorn servers (`MCPServer` backend + proxy). Ports are allocated with a
`bind(0)` probe, so tests are parallel-safe. Never hardcode ports.

## Architecture

The layering is load-bearing — respect the direction of dependencies:

```
cli ──► app ──► relay ──► headers, messages, config
                         │
                         └──► hooks (observation only, read-only)
```

- **`relay.py` is the only module touching the wire.** Raw ASGI on purpose: a
  Starlette app would re-encode bodies and own headers, breaking byte fidelity.
  Do not introduce framework-level request/response objects here.
- **`messages.py` parsing is strictly side-band.** It may never gate, mutate or
  reject traffic. A body that fails to parse is still forwarded verbatim, with
  the failure recorded on `WireMessage.validation_error`.
- **`hooks.py` cannot alter traffic today.** Keep it that way until the
  authorizer exists; read-only hooks are why this is safe to deploy early.

## Conventions that differ from the obvious

- **SDK v2, not v1.** `mcp` 2.2.0, protocol `2026-07-28`. v2 uses **`httpx2`**,
  not `httpx`. `MCPServer` replaces `mcp.server.Server`; `request_ctx` is gone.
  `CallToolResult.is_error` (v2), not `isError` (v1). Any snippet found online
  for `mcp-proxy` 0.12.0 is v1-era and will not import.
- **Two envelope eras, one parser.** 2025-era is stateful (`Mcp-Session-Id`,
  `GET` SSE stream, `Last-Event-ID`). `2026-07-28` is stateless and mirrors
  routing into `Mcp-Method` / `Mcp-Name` / `Mcp-Param-*` headers. Both carry
  `method`/`params` in the JSON body, so handle both by parsing the body and
  attaching the header mirror — never by branching on version.
- **Handshake method varies.** A v2 client in default `auto` mode sends
  **`server/discover`**, not `initialize`. `_NON_EFFECTFUL_METHODS` in
  `hooks.py` must list both. Tests assert the intersection, not one name.
- **Fail closed on the unknown.** `RequestContext.is_effectful` returns `True`
  for unrecognized methods, unparseable bodies, and anything not in the
  allowlist of reads. Never invert this default.
- **The proxy owns no session.** `Mcp-Session-Id` is relayed untouched, so the
  client's session is with the backend. Do not add session tracking keyed on
  the HTTP connection — it must be keyed on the run, see the implementation plan.
- **ASGI responses terminate exactly once.** `_ResponseState` tracks
  `started`/`completed`; `terminate()` is idempotent and shielded from
  cancellation. Two bugs already came from this: setting `completed` after the
  stream loop (skipped the final `more_body: False`), and not cancelling the
  disconnect watcher (deadlock, since `receive()` only returns on disconnect).

## Documentation style

Docstrings use Google style with `Args:`/`Returns:`/`Raises:`. Every module has
a prose docstring explaining *why*, not just *what* — the relay and messages
modules are the model to follow.

## Editing this repo

- The workspace root is owned by `nobody:nogroup` while `gab` owns the content;
  `git status` reports *dubious ownership*. Do not "fix" this by changing global
  git config without asking.
- `docs/` holds the source paper. Treat it as read-only reference material.
- Never print file-change diffs to chat; edit files directly.
