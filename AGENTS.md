# Project Guidelines

## What this repo is

Phase 1 of FAVA: a **pass-through reverse proxy for the OpenAI-compatible Chat
Completions API**, sitting between an agent harness and the LLM provider. It
relays bytes unchanged and observes the model's tool-call intents via hooks.
There is no authorization yet — that is what the hooks are reserved for.

Background: [`README.md`](README.md) (usage, API, design rationale).
Roadmap: [`docs/implementation_plan.md`](docs/implementation_plan.md).
Paper: `docs/FAVA_ Formal Authorization for Verified Agents….html`.

The MCP-wire proxy this replaces is preserved on the `master` branch. It is
**superseded, not a reference**: it mediated only MCP-routed tools and was blind
to a harness's local ones, which is the whole reason for this boundary.

## Build and Test

Always use `uv`. Never install Python packages globally.

```bash
uv sync --group dev     # install/create .venv
uv run pytest           # 92 tests, ~7s
```

If a test run hangs, suspect the relay's task group, not pytest. Prefer a hard
cut so a hang costs 200s instead of forever:

```bash
timeout -s KILL 200 uv run pytest -q --no-header
```

`tests/test_e2e.py` binds **real free TCP ports** and runs two uvicorn servers
(fake provider + proxy), driven by a real `openai` client. Ports are allocated
with a `bind(0)` probe, so tests are parallel-safe. Never hardcode ports.

`tests/test_relay.py` drives the ASGI interface directly through an
`httpx.MockTransport`. Do **not** reach for `httpx.ASGITransport` there: it
buffers the whole response body before returning, so it cannot exercise the
streaming path or a mid-stream upstream failure at all.

## Architecture

The layering is load-bearing — respect the direction of dependencies:

```
cli ──► app ──► relay ──► headers, chat, config
                       │
                       └──► hooks (observation only, read-only)
```

- **`relay.py` is the only module touching the wire.** Raw ASGI on purpose: a
  Starlette app would re-encode bodies and own headers, breaking byte fidelity.
  Do not introduce framework-level request/response objects here.
- **`chat.py` parsing is strictly side-band.** It may never gate, mutate or
  reject traffic. A body that fails to parse is still forwarded verbatim, with
  the failure recorded on `validation_error`.
- **`hooks.py` cannot alter traffic today.** Keep it that way until the
  authorizer exists; read-only hooks are why this is safe to deploy early.

## Conventions that differ from the obvious

- **Tool-call intent lives in the *response*, not the request.** This inverts the
  interception model of an MCP proxy, where the client's request carries the
  invocation. Here the request carries *context* — conversation, declared tool
  catalog, and the previous turn's tool **results** — while the model's intent
  arrives from the provider. `on_response` is the authorization seam that will
  grow an allow/block return value; `on_request` never will.
- **Streaming fragments tool calls.** `function.arguments` arrives as a string
  split across many SSE frames, identified only by `(choice index, tool call
  index)`, and frames split across TCP chunk boundaries too. Parallel calls
  interleave, so fragments must be keyed rather than appended in arrival order.
  Accumulate incrementally while bytes stream past (`StreamAccumulator`) — never
  parse a buffered prefix, or a long answer pushes the tool calls out of view.
- **`Accept-Encoding` is forced to `identity` upstream.** The relay forwards raw
  bytes, so a gzipped response would reach the client fine and reach the observer
  as compressed noise. Identity encoding keeps "what we relay" and "what we
  observe" the same bytes. Do not "restore" compression on the upstream leg.
- **The path is a prefix, not a pinned endpoint.** The base-URL swap is the whole
  integration, so `/chat/completions`, `/models` and `/embeddings` must all
  route: the inbound suffix is appended to the configured upstream base. An
  endpoint-pinning relay would break every non-completion call.
- **The proxy is on the credential path.** `Authorization` is relayed by default
  — the opposite of the MCP proxy, which dropped client credentials — because
  the request simply fails without it. Never log a credential; use
  `headers.redact` at the point of logging, not at parse time.
- **Fail closed on the unknown.** `ResponseContext.may_dispatch_tools` is true
  when intents were observed *or* when the response could not be read: an
  unparseable success body, or a stream that ended without `[DONE]`. Never invert
  this default — it would wave through exactly the responses the proxy failed to
  understand.
- **Hook dispatch goes through `hooks.dispatch`, never `hooks.on_x(...)`.**
  `normalize_hooks` passes a lone hook straight through instead of wrapping it,
  so calling the attribute directly makes partial hooks work with two observers
  registered and crash with one. Routing every call through the helper is what
  keeps "hooks may implement a subset" true.
- **ASGI responses terminate exactly once.** `_ResponseState` tracks
  `started`/`completed`; `terminate()` is idempotent and shielded from
  cancellation. Two bugs already came from this and are guarded by tests in
  `tests/test_relay.py`: setting `completed` after the stream loop (skipped the
  final `more_body: False`), and not cancelling the disconnect watcher (deadlock,
  since `receive()` only returns on disconnect).

## Documentation style

Docstrings use Google style with `Args:`/`Returns:`/`Raises:`. Every module has a
prose docstring explaining *why*, not just *what* — the relay and chat modules
are the model to follow.

## Editing this repo

- The workspace root is owned by `nobody:nogroup` while `gab` owns the content;
  `git status` reports *dubious ownership*. Do not "fix" this by changing global
  git config without asking.
- `docs/` holds the source paper. Treat it as read-only reference material.
- Never print file-change diffs to chat; edit files directly.
