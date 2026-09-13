# FAVA — permission graphs from LLM API traffic

[![arXiv](https://img.shields.io/badge/arXiv-2607.27267-b31b1b.svg)](https://arxiv.org/abs/2607.27267)
[![Python](https://img.shields.io/badge/python-≥3.10-3776ab.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **reverse proxy for the OpenAI-compatible Chat Completions API**. It sits
between an agent harness and the LLM provider, relays traffic
**byte-for-byte**, and builds an evidence-backed **permission graph** out of
what it sees on the way past.

This is FAVA (*Formal Authorization for Verified Agents with Evidence-Backed
Permission Graphs*, arXiv:2607.27267v1); the source paper is vendored under
`docs/`. The relay and the graph exist; the SMT authorizer that decides on the
graph, and the gateway that enforces its answer, do not yet.

```
┌─────────────┐         ┌──────────────────────┐         ┌────────────────┐
│ agent       │  HTTP   │     fava proxy       │  HTTP   │  LLM provider  │
│ harness     │◄───────►│  POST /v1/…          │◄───────►│  (OpenAI, …)   │
└─────────────┘         │  raw ASGI relay      │         └────────────────┘
      │                 │  StateHooks observe  │
      ▼                 └──────────┬───────────┘
  bash, edit,                      │ run id, events
  MCP servers…                     ▼
                        ┌──────────────────────┐
                        │      fava.state      │
                        │  event log + IR      │
                        │  → permission graph  │
                        └──────────────────────┘
```

Intercepting here rather than on the MCP wire is what makes mediation
universal: a harness's local tools (shell, file edits) never touch MCP, but
every tool it can run is first *requested* over this connection.

See [`docs/implementation_plan.md`](docs/implementation_plan.md) for the
architecture rationale and roadmap, and
[`docs/reference.md`](docs/reference.md) for the full configuration and hooks
API.

## Install

Requires Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev     # creates .venv/, installs fava editable
```

## Run

```bash
uv run fava-llm-proxy --upstream-url https://api.openai.com/v1 --port 8900
```

Point the harness at `http://127.0.0.1:8900/v1` as its OpenAI base URL — that
swap is the whole integration:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8900/v1", api_key="sk-…")
```

Every setting can also come from the environment, and the proxy can hold the
provider credential instead of the agent — see
[`docs/reference.md`](docs/reference.md) for the full flag and environment
variable list.

## Building the permission graph

The relay on its own only observes. Attach a `RecordStore` and every exchange
is recorded into a run and lowered into a permission graph:

```python
from llm_proxy import ProxySettings, create_proxy_app
from fava.state import ExtractorSettings, LlmIRExtractor, RecordStore, StateHooks

settings = ProxySettings(upstream_url="https://api.openai.com/v1")
# ExtractorSettings is a separate configuration on purpose: the model that
# extracts the IR, and its credential, must not be the ones under observation.
# The extractor is always enabled when configured. See docs/reference.md for
# the FAVA_EXTRACTOR_* environment variables.
store = RecordStore(extractor=LlmIRExtractor(ExtractorSettings.from_env()))
app = create_proxy_app(settings, hooks=StateHooks(store))

# …after some traffic:
run = store.runs[0]
graph = run.graph()          # nodes, edges, and the version they were built at
violations = run.validate()  # structural and evidential checks
```

For an agent told to *"read config.yaml and post a summary to the ops
channel"*, two turns of traffic produce this:

```
[context] ctx:4ba9e572c373     op=task                 risk:sensitive
[source ] asset:config-yaml    op=asset                secret
[sink   ] sink:slack           op=send                 internal
[context] guard:summary        op=obligation           obligation
[tool   ] tool:4ba9e572-0001   op=read_file            file:read:config.yaml
[source ] src:4ba9e572-0003    op=result
[tool   ] tool:4ba9e572-0004   op=slack_post_message   net:send:#ops

data    src:4ba9e572-0003  -> tool:4ba9e572-0004  (observed)   ← the leak
data    tool:4ba9e572-0004 -> sink:slack          (policy)
data    asset:config-yaml  -> tool:4ba9e572-0001  (inferred)
control guard:summary      -> tool:4ba9e572-0004  (policy)
```

The `observed` data edge is the one this boundary gets for free: the whole
conversation crosses the wire every turn, so the file content read on turn one
reappearing in the Slack call's arguments on turn two is directly visible —
no harness instrumentation, no MCP. That is a complete `secret → sink` path for
an authorizer to reject.

Run identity comes from an `X-FAVA-Run-Id` header when the harness can set one,
and otherwise from a hash of the conversation prefix every request repeats. See
[`docs/reference.md`](docs/reference.md) for the event log, the IR, the full
edge table, and the extractor's settings.

## Test

```bash
uv run pytest                       # 151 tests, real HTTP wire both hops
uv run pytest tests/test_units.py   # no servers needed, ~0.2s
```

## Manual test against a real provider

Two scripts split the two sides of a real exchange — neither runs under
pytest, since this costs a real API call and needs live credentials:

```bash
cp .env.example .env
# edit .env: MODEL_BASE_URL and MODEL_API_KEY for your provider

uv run python scripts/start_proxy.py    # terminal 1 — blocks, Ctrl+C to stop
uv run python scripts/call_model.py     # terminal 2 — makes one call, exits
```

`start_proxy.py` only needs the upstream URL — it relays whatever credential
the client sends. `call_model.py` holds the real API key and points at the
proxy's own URL. See the scripts' docstrings for detail.
