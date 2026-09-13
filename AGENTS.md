# Project Guidelines

## What this repo is

FAVA's runtime gateway. Two packages under `src/`:

- **`llm_proxy`** — a **pass-through reverse proxy for the OpenAI-compatible
  Chat Completions API**, sitting between an agent harness and the LLM
  provider. It relays bytes unchanged and observes the model's tool-call
  intents via hooks.
- **`fava`** — everything that reasons about what was relayed. Today that is
  `fava.state`: run identity, an append-only event log, the Permission IR, and
  the permission graph lowered from both.

There is still no authorization: the graph is built, nothing decides on it.

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
uv run pytest           # 151 tests, ~9s
```

If a test run hangs, suspect the relay's task group, not pytest. Prefer a hard
cut so a hang costs 200s instead of forever:

```bash
timeout -s KILL 200 uv run pytest -q --no-header
```

`tests/test_e2e.py` binds **real free TCP ports** and runs two uvicorn servers
(fake provider + proxy), driven by a real `openai` client. Ports are allocated
with a `bind(0)` probe, so tests are parallel-safe. Never hardcode ports.

`tests/test_state.py` is mostly in-process, but its end-to-end section uses the
same two-server setup: correlating an intent observed in response *N* to the
result reported in request *N+1* only happens across two genuine exchanges.

`tests/test_relay.py` drives the ASGI interface directly through an
`httpx.MockTransport`. Do **not** reach for `httpx.ASGITransport` there: it
buffers the whole response body before returning, so it cannot exercise the
streaming path or a mid-stream upstream failure at all.

## Architecture

The layering is load-bearing — respect the direction of dependencies:

```
fava.state.hooks ──► fava.state.store ──► events, ir, graph
        │                                        │
        ▼                                        │
cli ──► app ──► relay ──► headers, chat, config ◄┘
                       │
                       └──► hooks (observation only, read-only)
```

**`fava` imports `llm_proxy`; `llm_proxy` never imports `fava`.** That is what
keeps the relay deployable on its own, and keeps a bug in the authorization
stage from being able to break the transport. Wiring happens at the call site
(`create_proxy_app(settings, hooks=StateHooks(store))`), not inside the relay.

- **`relay.py` is the only module touching the wire.** Raw ASGI on purpose: a
  Starlette app would re-encode bodies and own headers, breaking byte fidelity.
  Do not introduce framework-level request/response objects here.
- **`chat.py` parsing is strictly side-band.** It may never gate, mutate or
  reject traffic. A body that fails to parse is still forwarded verbatim, with
  the failure recorded on `validation_error`.
- **`hooks.py` cannot alter traffic today.** Keep it that way until the
  authorizer exists; read-only hooks are why this is safe to deploy early.
- **`fava.state` observes and records; it never decides.** `lower()` and
  `validate()` are total functions that do not raise, because their caller sits
  on the relay's observation path. The allow/block return value belongs on
  `on_response`, and that is a later phase.

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
- **Run identity is derived, never assumed from the connection.** There is no
  session header at this boundary. `derive_run_id` takes `X-FAVA-Run-Id` if a
  harness sets one, else hashes the stable conversation prefix. A rewritten
  prefix (truncation, compaction) is a **new run**, not a merge — do not "fix"
  that by trying to stitch runs back together, because the proxy cannot see
  that they are the same one.
- **Every request replays the whole history.** A `role: "tool"` result arrives
  again on every later turn, so anything that records per-message must
  de-duplicate (`EventLog.has_result_for`) or the log grows quadratically.
- **The IR extractor calls the provider directly, never through the proxy.**
  Routed through the gateway its own request would be observed, minted as a
  run, and extracted from — recursively. `X-FAVA-Internal` marks its calls.
- **Extraction lands out of order.** It runs as a background task, so the
  `ir_extracted` event's position in the log is genuinely nondeterministic
  relative to traffic. Tests must not assert a fixed event order; filter it out
  (`_traffic_kinds` in `tests/test_state.py`) instead.
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
