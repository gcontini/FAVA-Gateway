# FAVA — pass-through LLM API reverse proxy

A **reverse proxy for the OpenAI-compatible Chat Completions API**. It sits
between an agent harness and the LLM provider and relays traffic
**byte-for-byte**, observing the tool calls the model asks for on the way past.

This is the first component of FAVA (*Formal Authorization for Verified Agents
with Evidence-Backed Permission Graphs*, arXiv:2607.27267v1); the source paper is
vendored under `docs/`. Today the proxy is a dumb relay with observation hooks;
the hooks are where FAVA's authorization stage will attach. See
[`docs/implementation_plan.md`](docs/implementation_plan.md).

```
┌─────────────┐         ┌──────────────────────┐         ┌────────────────┐
│ agent       │  HTTP   │     fava proxy       │  HTTP   │  LLM provider  │
│ harness     │◄───────►│  POST /v1/…          │◄───────►│  (OpenAI, …)   │
└─────────────┘         │  raw ASGI relay      │         └────────────────┘
      │                 │                      │
      │ dispatches      │  ProxyHooks ─────────┼──► observe request
      │ tool calls      │  (read-only today)   │──► observe tool intents
      ▼                 └──────────────────────┘
  bash, edit,
  MCP servers…
```

## Why this boundary and not the MCP wire

An earlier iteration of this proxy sat on the MCP wire, between the harness and
an MCP server. That placement cannot hold the paper's first assumption — that
**all** security-relevant effects are mediated before execution — because it only
sees tools routed through MCP servers. A harness's local tools (shell, file
edits, web fetch) never touch MCP at all, so an agent doing damage through `Bash`
was simply invisible.

Intercepting at the harness↔LLM boundary fixes that, for two reasons:

1. **Every tool call passes through here first.** Local and MCP-backed tools
   alike begin as a tool-call *intent* in the model's response. From the model's
   side both are just declared tools with a name and a schema — how the harness
   dispatches them is invisible to the API wire format.
2. **It is harness-agnostic.** The integration is the standard base-URL override
   nearly every agent runtime already supports. No plugin, no patch.

The trade-off is that the proxy observes *intent*, not execution. It sees what
the model asked for, and — on the following request — what the harness reported
back. It never sees the tool actually run.

## Install

Requires Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev     # creates .venv/, installs fava editable
```

Nothing is installed globally; everything lives in the project `.venv/`.

## Run the proxy

```bash
uv run fava-llm-proxy --upstream-url https://api.openai.com/v1 --port 8900
```

The harness is then pointed at `http://127.0.0.1:8900/v1` as its OpenAI base URL
— that swap is the whole integration:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8900/v1", api_key="sk-…")
```

With access logging (one line per exchange, including the tool calls the model
requested) and debug verbosity:

```bash
uv run fava-llm-proxy \
    --upstream-url https://api.openai.com/v1 \
    --port 8900 --access-log --log-level debug
```

To hold the provider credential in the proxy rather than the agent:

```bash
uv run fava-llm-proxy \
    --upstream-url https://api.openai.com/v1 \
    --api-key "$OPENAI_API_KEY" --no-forward-client-auth
```

### Environment configuration

Every setting can come from the environment instead of flags. Precedence per
field is CLI flag → environment variable → default.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROXY_UPSTREAM_URL` | — (required) | Upstream API **base** URL, e.g. `https://api.openai.com/v1` |
| `LLM_PROXY_HOST` | `127.0.0.1` | Proxy bind address |
| `LLM_PROXY_PORT` | `8900` | Proxy bind port |
| `LLM_PROXY_MOUNT_PREFIX` | `/v1` | Path prefix the proxy serves; `*` accepts any path |
| `LLM_PROXY_TIMEOUT` | `30` | Upstream connect/write seconds |
| `LLM_PROXY_STREAM_READ_TIMEOUT` | `600` | Upstream read seconds; generous because a model may pause between tokens |
| `LLM_PROXY_MAX_BODY_SIZE` | `33554432` | Max client request body in bytes (32 MiB) |
| `LLM_PROXY_API_KEY` | — | Injected as `Authorization: Bearer …`, same as `--api-key` |

```bash
LLM_PROXY_UPSTREAM_URL=https://api.openai.com/v1 uv run fava-llm-proxy
```

### CLI flags

| Flag | Purpose |
|---|---|
| `--upstream-url URL` | Upstream API base URL |
| `--host` / `--port` | Where the proxy listens |
| `--mount-prefix PATH` | Served path prefix; `*` for any |
| `--api-key KEY` | Inject `Authorization: Bearer KEY` upstream |
| `--header NAME:VALUE` | Extra header sent upstream, repeatable |
| `--no-forward-client-auth` | Drop the client's own `Authorization` instead of relaying it |
| `--trust-client-headers` | Relay *all* client headers upstream (minus hop-by-hop). Off by default |
| `--allow-origin ORIGIN` | CORS origin, repeatable. Off by default; a server-side harness does not need it |
| `--access-log` | Log one line per exchange |
| `--timeout` / `--stream-read-timeout` | Upstream timeouts |
| `--log-level` | `critical`…`debug` |

A health endpoint at `/healthz` is answered **locally** and never relayed:

```bash
curl http://127.0.0.1:8900/healthz
# {"status":"ok","upstream_url":"https://api.openai.com/v1","mount_prefix":"/v1",
#  "health_path":"/healthz","upstream_client_ready":true}
```

## Test

```bash
uv run pytest                       # 92 tests
uv run pytest tests/test_units.py   # no servers needed, ~0.2s
uv run pytest -q -x                 # stop at first failure
```

The end-to-end tests point a **real** `openai` client at the proxy, which sits in
front of a fake provider on another real TCP port. Requests cross the genuine
HTTP wire in both hops — no in-memory simulation:

```
tests:  openai.AsyncOpenAI ──TCP──► proxy (uvicorn) ──TCP──► fake provider
```

They cover streaming and non-streaming completions, tool-call observation, tool
arguments fragmented across SSE frames *and* TCP chunks, interleaved parallel
tool calls, the tool-result round trip, non-ASCII fidelity, byte-identical
proxied-vs-direct responses, `/v1/models` passthrough, provider errors, the local
health endpoint and 8 concurrent clients.

## The API

### Package layout

| Module | Responsibility |
|---|---|
| `llm_proxy.config` | `ProxySettings` — immutable, validated configuration |
| `llm_proxy.relay` | `ForwardingProxy` — the ASGI app; the only code touching the wire |
| `llm_proxy.app` | `create_proxy_app()` — deployment entry point (settings + hooks + CORS) |
| `llm_proxy.cli` | `fava-llm-proxy` console script |
| `llm_proxy.chat` | Read-only parsed views of Chat Completions traffic |
| `llm_proxy.headers` | Hop-by-hop and API wire header relay rules |
| `llm_proxy.hooks` | `ProxyHooks`, `RequestContext`, `ResponseContext` — the interception seam |

### As an ASGI app

`ForwardingProxy` *is* an ASGI application and implements the lifespan protocol,
so it owns its upstream `httpx.AsyncClient` across startup/shutdown. Hand it to
uvicorn directly:

```python
import uvicorn
from llm_proxy import ProxySettings, create_proxy_app

settings = ProxySettings(upstream_url="https://api.openai.com/v1", port=8900)
app = create_proxy_app(settings, access_log=True)   # adds a LoggingHooks observer

uvicorn.run(app, host=settings.host, port=settings.port, lifespan="on")
```

With your own observers, pass them as `hooks` and leave `access_log` off —
`access_log=True` appends a `LoggingHooks`, so setting both logs each exchange
twice:

```python
from llm_proxy.hooks import LoggingHooks

app = create_proxy_app(settings, hooks=[MyAuditHooks(), LoggingHooks()])
```

It can also be mounted inside an existing Starlette/FastAPI router. Do not
disable lifespan: without it the relay falls back to building and closing an HTTP
client per exchange, which costs a connection pool every request.

### Observing traffic with hooks

Hooks are the extension point. They are **read-only today** — nothing a hook does
can alter or block traffic, which is precisely why the proxy is safe to deploy
before the authorization stage exists.

The important asymmetry: **tool-call intent arrives in the response**, so
`on_response` is the seam that matters. The request carries context — the
conversation, the declared tool catalog, and the results of the *previous* turn's
tool calls.

```python
from llm_proxy.hooks import ProxyHooks, RequestContext, ResponseContext

class AuditHooks:
    async def on_request(self, context: RequestContext) -> None:
        """Called after the body is buffered, before it is relayed."""
        for result in context.tool_results:          # what the harness just ran
            print(f"result for {result.tool_call_id}: {result.content_size} chars")

    async def on_response(self, context: ResponseContext) -> None:
        """Called after the provider's response is fully relayed."""
        for intent in context.tool_intents:          # what the model wants to run
            print(f"{intent.name}({intent.arguments})")
        if context.may_dispatch_tools:               # the conservative gate
            ...

    async def on_relay_error(self, context, error) -> None:
        """Called when relaying failed and no response reached the client."""
        print(f"relay failed: {error!r}")
```

Pass one hook or a list; several are composed and failure-isolated, so a raising
hook is logged and never breaks the transport. Hooks may implement a subset of
the protocol — missing callbacks are skipped, whether you register one hook or
several.

`RequestContext` fields: `http_method`, `path`, `upstream_suffix`, `query`,
`headers`, `body_size`, `chat`, `client`. Properties: `is_chat_completion`,
`model`, `streaming`, `declared_tools`, `tool_results`.

`ResponseContext` fields: `request`, `status_code`, `headers`, `content_type`,
`chat`, `body_size`, `stream_closed_early`. Properties: `is_error`,
`tool_intents`, `has_tool_intents`, `intents_unknown`, `may_dispatch_tools`,
`finish_reasons`.

### Parsing what crossed the wire

`llm_proxy.chat` turns bytes into views. Parsing is strictly side-band: an
unparseable body is still relayed unchanged, with the failure recorded on the
view rather than acted upon.

```python
from llm_proxy.chat import parse_chat_request, parse_chat_response

request = parse_chat_request(b'{"model":"gpt-4o-mini","messages":[…],"tools":[…]}')
request.model              # "gpt-4o-mini"
request.tool_names         # ("run_shell", "read_file") — the catalog, in-band
request.tool_results       # role:"tool" messages from the previous turn
request.prior_tool_calls   # the intents those results answer

response = parse_chat_response(body)
response.tool_intents      # (ToolIntent(name="run_shell", arguments={"cmd": "ls"}),)
response.finish_reasons    # ("tool_calls",)
```

A `ToolIntent` keeps `arguments_json` exactly as the model emitted it, because a
model can and does emit invalid JSON; `arguments` is the decoded mapping, or
`None` when it would not decode.

### Streaming

A streamed response fragments tool calls: `function.arguments` arrives as a
string split across many SSE frames, keyed only by `index`, and the frames
themselves split across TCP chunks. `StreamAccumulator` reassembles both, fed
incrementally as bytes pass:

```python
from llm_proxy.chat import StreamAccumulator

accumulator = StreamAccumulator()
for chunk in response_chunks:      # any chunking at all
    accumulator.feed(chunk)
accumulator.close()

result = accumulator.result()
result.tool_intents        # fragments reassembled, parallel calls kept separate
result.complete            # False when the stream was cut before `[DONE]`
```

`ResponseObserver` picks the right strategy off the content type, so the relay
never has to know which shape it is relaying.

### Effect classification

`ResponseContext.may_dispatch_tools` is the predicate FAVA's authorizer will gate
on. It is conservative: true when tool calls were observed **or** when the proxy
could not tell — an unparseable success body, or a stream that ended without its
terminating `[DONE]`. Unknown traffic is never assumed safe, because an
authorizer that only trusted `has_tool_intents` would wave through exactly the
responses it failed to understand. Provider errors and non-completion endpoints
dispatch nothing and are not gated.

### Headers and credentials

`llm_proxy.headers` applies RFC 9110 §7.6.1 rules. Hop-by-hop headers
(`connection`, `keep-alive`, `te`, `trailer`, `transfer-encoding`, `upgrade`,
`proxy-authenticate`, `proxy-authorization`) are dropped in both directions, as
are any headers nominated by a `Connection:` value. `Host` is rewritten to the
provider's.

Unlike an MCP proxy, this one is **on the credential path**: `Authorization` is
relayed by default, because the request cannot succeed without it. Use
`--no-forward-client-auth` with `--api-key` to keep agent-supplied keys off the
wire entirely. Credentials are never logged.

By default only API wire headers are relayed upstream — `accept`,
`authorization`, `content-type`, `user-agent`, `api-key`, and anything prefixed
`openai-` or `x-stainless-`. Cookies and unrelated headers are dropped. Use
`--trust-client-headers` to relay everything else, or `--header` to inject
specific values.

`Accept-Encoding` is always forced to `identity` upstream. The relay streams raw
bytes, so a gzipped response would reach the client correctly but reach the
observer as compressed noise — and tool-call intents are the one thing this proxy
exists to see.

## Known limits

Deliberate for a first stub; each is tracked in
[`docs/implementation_plan.md`](docs/implementation_plan.md).

- **No authorization.** Every response is forwarded. `may_dispatch_tools`
  classifies but nothing acts on it.
- **No per-run correlation.** Hooks see HTTP exchanges, not agent runs. FAVA
  needs a `run_id`; the nearest available keys are a client-supplied header, the
  conversation prefix, and the `(host, port)` peer.
- **Intent, not execution.** The proxy sees what the model asked for and what the
  harness reported back, never the tool actually running. A harness that ignores
  a denial, or runs something it never asked the model about, is outside the
  guarantee.
- **Chat Completions only.** The `/v1/responses` API and Anthropic's Messages API
  are not parsed. They still relay — they are just not understood.
- **Request bodies are buffered** so hooks can parse them, bounded by
  `--max-body-size`. Response bodies are streamed and observed incrementally;
  only the retained text sample is capped.
- **One upstream per process.** Fan-out to several providers, or per-model
  routing, would need a routing table.
