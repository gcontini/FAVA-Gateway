# Configuration and extension API

Full reference for running and extending the proxy. For what it is and how to
launch it, see the [README](../README.md); for the architecture rationale and
roadmap, see [`implementation_plan.md`](implementation_plan.md).

## Configuration

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

The IR extractor (`fava.state`) has its own settings, read by
`ExtractorSettings.from_env()`. It calls the provider **directly**, never back
through the proxy — and it is a **separate, standalone configuration**: none
of the three variables below default from `LLM_PROXY_*` above. The model that
extracts the IR must be a different model from the one under observation,
reached with its own credential, never the monitored agent's — otherwise the
very traffic FAVA reasons about could shape its own risk assessment. Leaving
any of the three unset leaves extraction unconfigured, and every run then stays
`ambiguous`.

| Variable | Default | Meaning |
|---|---|---|
| `FAVA_EXTRACTOR_BASE_URL` | — (required) | OpenAI-compatible base URL for the extraction model's *own* provider |
| `FAVA_EXTRACTOR_MODEL` | — (required) | Model to extract the IR with — not the model the harness asked for |
| `FAVA_EXTRACTOR_API_KEY` | — (required) | Credential for the extraction call — not the agent's own |
| `FAVA_EXTRACTOR_TIMEOUT` | `20` | Seconds for the whole extraction call |

`LlmIRExtractor` calls the model through the `openai` package's
`AsyncOpenAI` client, so **any OpenAI-compatible endpoint works**, not just
OpenAI's own — point `FAVA_EXTRACTOR_BASE_URL` at it.

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

To hold the provider credential in the proxy rather than the agent:

```bash
uv run fava-llm-proxy \
    --upstream-url https://api.openai.com/v1 \
    --api-key "$OPENAI_API_KEY" --no-forward-client-auth
```

A health endpoint at `/healthz` is answered **locally** and never relayed:

```bash
curl http://127.0.0.1:8900/healthz
# {"status":"ok","upstream_url":"https://api.openai.com/v1","mount_prefix":"/v1",
#  "health_path":"/healthz","upstream_client_ready":true}
```

## Package layout

| Module | Responsibility |
|---|---|
| `llm_proxy.config` | `ProxySettings` — immutable, validated configuration |
| `llm_proxy.relay` | `ForwardingProxy` — the ASGI app; the only code touching the wire |
| `llm_proxy.app` | `create_proxy_app()` — deployment entry point (settings + hooks + CORS) |
| `llm_proxy.cli` | `fava-llm-proxy` console script |
| `llm_proxy.chat` | Read-only parsed views of Chat Completions traffic |
| `llm_proxy.headers` | Hop-by-hop and API wire header relay rules |
| `llm_proxy.hooks` | `ProxyHooks`, `RequestContext`, `ResponseContext` — the interception seam |

Above the transport, `fava` holds everything that reasons about what was
relayed. The dependency arrow only ever points `fava` → `llm_proxy`.

| Module | Responsibility |
|---|---|
| `fava.state.events` | `derive_run_id()`, `Event`, `EventLog` — append-only per run |
| `fava.state.ir` | `PermissionIR` and the LLM extractor that builds it from task text |
| `fava.state.graph` | `PermissionGraph`, `lower()`, `validate()`, capability normalization |
| `fava.state.store` | `RecordStore`, `RunState` — per-run state and the lowered graph |
| `fava.state.hooks` | `StateHooks` — the `ProxyHooks` that feeds the store |

## As an ASGI app

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

## Observing traffic with hooks

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

## Parsing what crossed the wire

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

## Streaming

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

## Effect classification

`ResponseContext.may_dispatch_tools` is the predicate FAVA's authorizer will gate
on. It is conservative: true when tool calls were observed **or** when the proxy
could not tell — an unparseable success body, or a stream that ended without its
terminating `[DONE]`. Unknown traffic is never assumed safe, because an
authorizer that only trusted `has_tool_intents` would wave through exactly the
responses it failed to understand. Provider errors and non-completion endpoints
dispatch nothing and are not gated.

## Per-run state and the permission graph

`fava.state` turns the proxy's stream of exchanges into the substrate an
authorizer can reason over. Attaching it is the whole integration:

```python
from llm_proxy import ProxySettings, create_proxy_app
from fava.state import ExtractorSettings, LlmIRExtractor, RecordStore, StateHooks

store = RecordStore(extractor=LlmIRExtractor(ExtractorSettings.from_env()))
app = create_proxy_app(settings, hooks=StateHooks(store))
```

**Run identity.** There is no session header at this boundary, so
`derive_run_id` takes the `X-FAVA-Run-Id` header when a harness sets one, and
otherwise hashes the stable conversation prefix — the system/developer messages
plus the first user turn, which every request in a run repeats. Truncation and
context compaction rewrite that prefix; the hash then changes and the traffic is
treated as a **new run**, because merging two prefixes would assert a continuity
the proxy cannot see.

**The event log** is append-only: `run_started`, `tool_intent`, `tool_result`,
`response_opaque`, `ir_extracted`. Intents are observed in response *N*; the
results answering them arrive in request *N+1* as `role: "tool"` messages keyed
by `tool_call_id`, which is what `EventLog.intent_for` correlates. An intent
with no result — refused, dropped, or still running — is derived by
`unanswered_intents()` rather than recorded, because the absence is the
observation.

**The Permission IR** is the paper's five fields (`intent`, `assets`, `actions`,
`obligations`, `sinks`) plus a `risk_posture`, extracted from the task text by an
LLM used strictly as a semantic extractor. Extraction runs once per run, in the
background, and can never raise: any failure yields `RiskPosture.AMBIGUOUS`,
which is the value the gateway fails closed on.

**The graph** (`lower(log, ir)`) follows the paper's Table 1 exactly — nodes
carry `id`, `kind`, `op`, `args`, `outputs`, `labels`, `requests`, `time`; edges
carry `src`, `dst`, `type`, `evidence`, `trust`. The mapping:

| Edge | Trust | Where it comes from |
|---|---|---|
| `parent` | `observed` | a call descends from the run context; a result from the call that produced it |
| `data` | `observed` | a tool result's content reappearing in a later call's arguments |
| `data` | `policy` | a requested capability matching a sink the IR named |
| `data` | `inferred` | an IR asset named in a call's arguments — a string match, not a flow |
| `control` | `policy` | an IR obligation guarding the call it must precede |

The observed `data` edge is the one this boundary gets for free: the whole
conversation crosses the wire every turn, so exfiltration through a later tool
call is directly visible without instrumenting the harness. Only `observed` and
`policy` edges are meant to be trusted — `inferred` ones exist to audit
extraction errors.

`graph_version` is the number of events the graph was lowered from, so a
decision can be bound to the exact log prefix it was made against.
`validate(graph)` returns structural and evidential violations (dangling edges,
duplicate ids, ungrounded labels, unknown evidence) rather than raising.

Capabilities come from policy, not the catalog: an OpenAI tool definition has no
`destructiveHint`, so `capabilities_for` maps a tool name and its arguments onto
`tool:{name}` plus, when recognized, a coarser `proc:exec:…`, `file:write:…`,
`file:read:…`, `net:send:…` or `net:fetch:…`. It is one small table, meant to be
replaced by a real policy module.

## Headers and credentials

`llm_proxy.headers` applies RFC 9110 §7.6.1 rules. Hop-by-hop headers
(`connection`, `keep-alive`, `te`, `trailer`, `transfer-encoding`, `upgrade`,
`proxy-authenticate`, `proxy-authorization`) are dropped in both directions, as
are any headers nominated by a `Connection:` value. `Host` is rewritten to the
provider's.

This proxy is **on the credential path**: `Authorization` is relayed by
default, because the request cannot succeed without it. Use
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
[`implementation_plan.md`](implementation_plan.md).

- **No authorization.** Every response is forwarded. `may_dispatch_tools`
  classifies and `fava.state` builds the graph, but nothing acts on either yet.
- **Run state is in memory.** `RecordStore` keeps the most recent `max_runs`
  runs and loses everything on restart. The interface is the place to put Redis
  behind, not the relay.
- **Extraction is best-effort and not on the critical path.** It runs in the
  background, so the first turn of a run is usually lowered against an
  `ambiguous` IR. Once an authorizer exists it will need to wait for, or fail
  closed on, an IR that has not landed.
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
