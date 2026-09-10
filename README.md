# FAVA — pass-through MCP reverse proxy

A **transport-level reverse proxy for the Model Context Protocol**. It sits
between an MCP client (an agent runtime) and a remote MCP server speaking
Streamable HTTP, and relays JSON-RPC traffic **byte-for-byte**.

This is the first component of FAVA (*Formal Authorization for Verified Agents
with Evidence-Backed Permission Graphs*, arXiv:2607.27267v1); the source paper
is vendored under `docs/`. Today the proxy is a dumb relay with observation
hooks; the hooks are where FAVA's authorization stage will attach. See
[`docs/implementation_plan.md`](docs/implementation_plan.md).

```
┌─────────────┐         ┌──────────────────────┐         ┌────────────────┐
│ MCP client  │  HTTP   │     fava proxy       │  HTTP   │  remote MCP    │
│ (agent)     │◄───────►│  POST /mcp           │◄───────►│  server        │
└─────────────┘         │  raw ASGI relay      │         └────────────────┘
                        │                      │
                        │  ProxyHooks ─────────┼──► observe request
                        │  (read-only today)   │──► observe response
                        └──────────────────────┘
```

## Why raw ASGI relay rather than a session-level proxy

Two designs are possible for an MCP proxy:

1. **Session-level**: open an `ClientSession` to the backend, run an
   `MCPServer` facing the agent, and re-implement a handler per method
   (`tools/list`, `tools/call`, `prompts/get`, …). This is what
   [`mcp-proxy`](https://pypi.org/project/mcp-proxy/) does.
2. **Transport-level**: forward the bytes, parsing only enough to observe.

This project uses **(2)**, because a session-level proxy must be updated every
time the protocol grows a method, and it re-encodes payloads. A byte relay is
forward-compatible: it carries the 2025-era session flow and the stateless
`2026-07-28` envelope identically, with no per-method code.

Consequences worth knowing:

- The proxy owns **no** MCP session. `Mcp-Session-Id` is relayed untouched, so
  the client's session is with the *backend*. The proxy can restart without
  dropping agent sessions.
- The proxy does not need to know the tool catalog. There is no `tools/list`
  cache, because nothing is rewritten.
- The trade-off is that interception requires parsing rather than dispatch.
  `mcp_proxy.messages` provides those read-only parsed views.

## Install

Requires Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev     # creates .venv/, installs fava + mcp 2.2.0 editable
```

Nothing is installed globally; everything lives in the project `.venv/`.

## Run the proxy

```bash
uv run fava-mcp-proxy --backend-url http://127.0.0.1:8901/mcp --port 8900
```

The agent is then pointed at `http://127.0.0.1:8900/mcp` instead of the backend
— that swap is the whole integration.

With access logging (one line per exchange) and debug verbosity:

```bash
uv run fava-mcp-proxy \
    --backend-url http://127.0.0.1:8901/mcp \
    --port 8900 --access-log --log-level debug
```

### Environment configuration

Every setting can come from the environment instead of flags. Precedence per
field is CLI flag → environment variable → default.

| Variable | Default | Meaning |
|---|---|---|
| `MCP_PROXY_BACKEND_URL` | — (required) | Remote MCP endpoint, e.g. `http://backend:8901/mcp` |
| `MCP_PROXY_HOST` | `127.0.0.1` | Proxy bind address |
| `MCP_PROXY_PORT` | `8900` | Proxy bind port |
| `MCP_PROXY_MOUNT_PATH` | `/mcp` | Path the proxy serves; `*` accepts any path |
| `MCP_PROXY_TIMEOUT` | `30` | Upstream connect/write seconds |
| `MCP_PROXY_SSE_READ_TIMEOUT` | `300` | Upstream read seconds; generous because a server may hold a response stream open |
| `MCP_PROXY_MAX_BODY_SIZE` | `4194304` | Max client request body in bytes (4 MiB) |

```bash
MCP_PROXY_BACKEND_URL=http://127.0.0.1:8901/mcp uv run fava-mcp-proxy
```

### CLI flags

| Flag | Purpose |
|---|---|
| `--backend-url URL` | Upstream MCP endpoint |
| `--host` / `--port` | Where the proxy listens |
| `--mount-path PATH` | Served path; `*` for any |
| `--header NAME:VALUE` | Extra header sent upstream, repeatable — the way to inject a static backend credential |
| `--trust-client-headers` | Relay *all* client headers upstream (minus hop-by-hop). Off by default |
| `--allow-origin ORIGIN` | CORS origin, repeatable. Off by default; a server-side agent runtime does not need it |
| `--access-log` | Log one line per exchange |
| `--timeout` / `--sse-read-timeout` | Upstream timeouts |
| `--log-level` | `critical`…`debug` |

A health endpoint at `/healthz` is answered **locally** and never relayed:

```bash
curl http://127.0.0.1:8900/healthz
# {"status":"ok","backend_url":"http://127.0.0.1:8901/mcp","mount_path":"/mcp",
#  "health_path":"/healthz","upstream_client_ready":true}
```

## Test

```bash
uv run pytest                       # 38 tests: 29 unit + 9 end-to-end
uv run pytest tests/test_units.py   # no servers needed, ~0.2s
uv run pytest -q -x                 # stop at first failure
```

The end-to-end tests start a **real** SDK v2 `MCPServer` on a real TCP port, put
the proxy in front of it on another port, and drive it with the official SDK
`Client`. Requests cross the genuine Streamable HTTP wire in both hops — no
in-memory simulation:

```
tests:  mcp.Client ──TCP──► proxy (uvicorn) ──TCP──► MCPServer (uvicorn)
```

They cover `tools/list`, argument/result fidelity (including non-ASCII), tool
exceptions surfacing as tool errors rather than proxy failures, sequential
calls, 8 concurrent sessions, hook observation of handshake + call, the local
health endpoint, unknown-path rejection, and unreachable-backend → JSON-RPC
`502`.

## The API

### Package layout

| Module | Responsibility |
|---|---|
| `mcp_proxy.config` | `ProxySettings` — immutable, validated configuration |
| `mcp_proxy.relay` | `ForwardingProxy` — the ASGI app; the only code touching the wire |
| `mcp_proxy.app` | `create_proxy_app()` — deployment entry point (settings + hooks + CORS) |
| `mcp_proxy.cli` | `fava-mcp-proxy` console script |
| `mcp_proxy.messages` | `WireMessage` — read-only parsed JSON-RPC views |
| `mcp_proxy.headers` | Hop-by-hop and MCP wire header relay rules |
| `mcp_proxy.hooks` | `ProxyHooks`, `RequestContext`, `ResponseContext` — the interception seam |

### As an ASGI app

`ForwardingProxy` *is* an ASGI application and implements the lifespan
protocol, so it owns its upstream `httpx2.AsyncClient` across
startup/shutdown. Hand it to uvicorn directly:

```python
import uvicorn
from mcp_proxy import ProxySettings, create_proxy_app

settings = ProxySettings(backend_url="http://127.0.0.1:8901/mcp", port=8900)
app = create_proxy_app(settings, access_log=True)   # adds a LoggingHooks observer

uvicorn.run(app, host=settings.host, port=settings.port, lifespan="on")
```

With your own observers, pass them as `hooks` and leave `access_log` off —
`access_log=True` appends a `LoggingHooks`, so setting both logs each exchange
twice:

```python
from mcp_proxy.hooks import LoggingHooks

app = create_proxy_app(settings, hooks=[MyAuditHooks(), LoggingHooks()])
```

It can also be mounted inside an existing Starlette/FastAPI router. Do not
disable lifespan: without it the relay falls back to building and closing an
HTTP client per exchange, which costs a connection pool every request.

### Observing traffic with hooks

Hooks are the extension point. They are **read-only today** — nothing a hook
does can alter or block traffic, which is precisely why the proxy is safe to
deploy before the authorization stage exists.

```python
from mcp_proxy.hooks import ProxyHooks, RequestContext, ResponseContext

class AuditHooks:
    async def on_request(self, context: RequestContext) -> None:
        """Called after the body is buffered, before it is relayed."""
        if context.is_effectful:                      # gates writes, not reads
            print(f"tool={context.tool_name} method={context.method}")

    async def on_response(self, context: ResponseContext) -> None:
        """Called after the backend response is fully relayed."""
        print(f"status={context.status_code} error={context.is_error}")

    async def on_relay_error(self, context, error) -> None:
        """Called when relaying failed and no response reached the client."""
        print(f"relay failed: {error!r}")
```

Pass one hook or a list; several are composed and failure-isolated, so a raising
hook is logged and never breaks the transport. Hooks may implement a subset of
the protocol — missing callbacks are skipped.

`RequestContext` fields: `http_method`, `path`, `query`, `headers`,
`body_size`, `messages`, `session_id`, `protocol_version`, `client`.
Properties: `method`, `tool_name`, `is_effectful`.

`ResponseContext` fields: `request`, `status_code`, `headers`, `content_type`,
`messages`, `body_size`, `truncated`, `stream_closed_early`. Property:
`is_error`.

### Parsing what crossed the wire

`mcp_proxy.messages` turns bytes into views. Parsing is strictly
side-band: an unparseable body is still relayed unchanged, with the failure
recorded on the message rather than acted upon.

```python
from mcp_proxy.messages import parse_wire_messages

(message,) = parse_wire_messages(
    b'{"jsonrpc":"2.0","id":3,"method":"tools/call",'
    b'"params":{"name":"echo","arguments":{"text":"hi"}}}'
)
message.is_tool_call       # True
message.tool_name          # "echo"
message.tool_arguments     # {"text": "hi"}
message.is_request         # True
message.validation_error   # None
```

Both envelope eras parse through the same entry point, since both carry
`method`/`params` in the body:

* **2025-era**: stateful — `Mcp-Session-Id`, a standalone `GET` SSE stream,
  resumable via `Last-Event-ID`.
* **2026-07-28**: stateless — routing mirrored into `Mcp-Method`, `Mcp-Name`
  and `Mcp-Param-*` headers; the mirror surfaces as `WireMessage.header_method`.

Use `parse_response_body(body, content_type=...)` for responses: it splits
`text/event-stream` frames and parses each `data:` payload.

### Effectful classification

`is_effectful` is the predicate FAVA's authorizer will gate on. It is
conservative — treated as **not** effectful: the handshake (`initialize` and
its 2026-07-28 successor `server/discover`), `ping`, and the list/catalog reads
(`tools/list`, `prompts/list`, `resources/list`,
`resources/templates/list`, `subscriptions/listen`). **Everything else is
effectful**, including unrecognized methods and unparseable bodies: unknown
traffic is never assumed safe.

### Headers

`mcp_proxy.headers` applies RFC 9110 §7.6.1 rules. Hop-by-hop headers
(`connection`, `keep-alive`, `te`, `trailer`, `transfer-encoding`, `upgrade`,
`proxy-authenticate`, `proxy-authorization`) are dropped in both directions, as
are any headers nominated by a `Connection:` value. `Host` is rewritten to the
backend's, so upstream `Host`-based validation and DNS-rebinding protection
still see their own address.

By default only MCP wire headers are relayed upstream — `accept`,
`content-type`, `last-event-id`, `mcp-session-id`, `mcp-protocol-version`,
`mcp-method`, `mcp-name`, and `Mcp-Param-*`. Cookies, user agents and client
credentials are dropped. Use `--trust-client-headers` to relay everything else,
or `--header` to inject specific values upstream.

## Compatibility notes

- Built against **`mcp` 2.2.0** (SDK v2, protocol `2026-07-28`). SDK v2
  relays HTTP over `httpx2`, not `httpx`.
- v1 code does not import: `mcp.server.Server` is `MCPServer` in v2, and the
  v1 `request_ctx` pattern is gone. Anything written against `mcp-proxy`
  0.12.0's `create_proxy_server` needs adaptation.
- A v2 SDK client in default `auto` negotiate mode sends **`server/discover`**,
  not `initialize`. Tests and the effectful list account for both.
- `CallToolResult` uses `is_error` in v2 (`isError` was v1).

## Known limits

Deliberate for a first stub; each is tracked in
[`docs/implementation_plan.md`](docs/implementation_plan.md).

- **No authorization.** Every request is forwarded. `is_effectful` classifies
  but nothing acts on it.
- **No per-run correlation.** Hooks see HTTP connections, not agent runs.
  FAVA needs a `run_id` — today the nearest available keys are
  `Mcp-Session-Id` (absent in the stateless 2026 envelope) and the
  `(host, port)` peer.
- **Request bodies are buffered** so hooks can parse them, bounded by
  `--max-body-size`. Response bodies are *streamed*, so hooks see only a
  256 KiB observation prefix; `ResponseContext.truncated` reports this. The
  relay forwards every byte regardless.
- **One backend per process.** Fan-out to several upstreams, or per-tool
  routing, would need a routing table.
- **No resumability store.** `Last-Event-ID` is relayed, and the backend's own
  event store answers it; the proxy keeps no store of its own.
