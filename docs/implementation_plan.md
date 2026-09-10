# FAVA implementation plan

Status of Phase 1 and the path to the full FAVA gateway.

Phase 1 is complete and tested: a pass-through reverse proxy for the
OpenAI-compatible Chat Completions API (`src/llm_proxy/`) that relays bytes
exactly and exposes read-only observation hooks. Everything below builds on that
seam.

The paper is `docs/FAVA_ Formal Authorization for Verified Agents with
Evidence-Backed Permission Graphs.html` (arXiv:2607.27267v1 [cs.CR]).

## Where Phase 1 already lands

The proxy is the **runtime enforcement gateway** of the paper's macro
architecture (§ Runtime Enforcement). It is now in the right *place* — the
harness↔LLM boundary, where every tool call first appears as an intent
regardless of how the harness will dispatch it — but today it mediates without
deciding: it observes and forwards. Nothing yet builds a Permission IR, an
evidence graph, or consults an authorizer.

What Phase 1 settled, and that later phases should not re-litigate:

- **The interception point.** A raw ASGI relay at the LLM API boundary, reached
  by a base-URL override. This covers local and MCP-backed tools uniformly,
  which an MCP-wire proxy cannot (see `README.md`, "Why this boundary").
- **The decision side is the response.** Tool-call intent arrives from the
  provider; the harness dispatches only afterwards. `on_response` is the seam.
- **Hooks are read-only and best-effort.** The gateway degrades to a pure relay
  if an observer breaks, which is the correct failure mode for Phase 1 but must
  **not** survive into the enforcement phase — see "Fail closed" below.
- **`ResponseContext.may_dispatch_tools`** already implements the conservative
  classification the authorizer will gate on: observed intents *or* a response
  too opaque to rule them out.
- **The tool catalog is in-band.** Every request carries `tools[]`, so there is
  no catalog to fetch, cache or invalidate.

## Target component decomposition

Five components, matching the paper's macro blocks. Only the forwarder exists.

```mermaid
graph LR
    A[agent harness] -->|POST /v1/chat/completions| B[fava-adapter]
    B -->|observe request + tool results| C[fava-state]
    C -->|append-only events| D[fava-graph]
    D -->|PermissionGraph + version| E[fava-authorizer]
    E -->|Allow / Block / Unknown| B
    B -->|release iff Allow| F[fava-forwarder]
    F -->|HTTPS| G[LLM provider]
    G -->|response + tool intents| F
    F -->|observe intents| C
    B --> H[in-band tool catalog]
    H -.->|tools[] metadata| D
```

| Component | Status | Responsibility |
|---|---|---|
| `fava-forwarder` | **done** (`relay.py`) | Byte-exact upstream forwarding |
| `fava-adapter` | skeleton (`relay.py`, `hooks.py`, `chat.py`) | Interception, `run_id` propagation, catalog extraction, event recording |
| `fava-state` | not started | Per-run state: `run_id → PermissionGraph + events + policy context`, append-only, graph versioning |
| `fava-graph` | not started | Node/edge construction, deterministic lowering, structural + evidential validation |
| `fava-authorizer` | not started | `Allow / Block / Unknown` decision; SMT backend later; fail closed |

Keep them as modules of one package initially (`fava.state`, `fava.graph`, …).
Splitting into separate distributions is premature: the interfaces matter, the
process boundaries do not.

## Phase 2 — run identity and event recording

**Goal:** attach every observed exchange to a run, and record it append-only.

The critical gap today is that hooks see **HTTP exchanges, not agent runs.**
FAVA's graphs are per-run, so this must be solved first; nothing else can be
trusted without it (§ Concurrency Model).

Moving off the MCP wire cost us `Mcp-Session-Id` — but bought something better.
The conversation history *is* a chain: request *N+1* contains the assistant turn
from response *N* plus the `role: "tool"` results of the calls it requested. Run
identity can therefore be recovered from the traffic itself.

- Derive or propagate a `run_id`. Options, in order of preference:
  1. Trust a client-supplied header (e.g. `X-FAVA-Run-Id`) when the harness can
     set one — cleanest, needs cooperation.
  2. Hash a stable prefix of the message array (system prompt + first user turn).
     Free and harness-agnostic, since every request in a run repeats it. Beware
     history truncation and context compaction, which rewrite the prefix
     mid-run; treat a prefix change as a *new* run rather than silently merging.
  3. Fall back to peer `(host, port)` — coarse; many runs share a connection
     pool.
  Whichever wins, `run_id` must be explicit state, never inferred from the
  connection alone.
- Event record (per paper § Evidence): `event_id`, `run_id`, `parent_event_id`,
  monotonic sequence + timestamp, tool name, argument metadata (not necessarily
  full argument values — decide and document the retention policy), the intent's
  provider-assigned `tool_call_id`, and the evidence source.
- **Correlate intent to outcome across two exchanges.** The intent is observed on
  response *N*; its result arrives as a `role: "tool"` message on request *N+1*,
  keyed by `tool_call_id`. `chat.ChatRequest.tool_results` and `prior_tool_calls`
  already expose both halves. An intent with no matching result was refused,
  dropped, or is still running — all three are worth recording.
- Append-only. Repair is monotone: events and edges are added, never rewritten or
  deleted (§ Monotonic Runtime Repair).

**Deliverable:** `fava.state.RecordStore` with `observe_request` /
`observe_response`, wired into `hooks.py`. In-memory first; Redis behind the same
interface if state must survive a proxy restart.

## Phase 3 — tool catalog normalization

**Goal:** know what a tool *is* before its call is authorized.

This phase is much smaller than it was on the MCP wire, because the catalog is
**in-band**: every request carries `tools[]`, so there is no discovery call, no
cache key, no invalidation and no TTL. `chat.ToolSpec` already extracts it.

What remains is genuinely harder, though:

- **OpenAI tool definitions carry no effect annotations.** MCP gave
  `readOnlyHint` / `destructiveHint` / `idempotentHint` for free; here there is
  only a name, a description and a JSON Schema. Whether `run_shell` is dangerous
  cannot be read off the catalog and must come from **policy**, keyed on tool
  name plus (optionally) argument patterns.
- Normalize into a `ToolSpec` the graph can consume, and detect catalog drift
  within a run: a harness that adds or redefines a tool mid-run is a
  policy-relevant event, not a no-op.
- Catalog-less fallback: a request declaring no tools cannot produce an intent,
  and an intent naming an undeclared tool is a red flag worth recording.

**Deliverable:** `fava.adapter.ToolCatalog` with `describe(run_id, tool_name) →
ToolSpec | None` plus drift detection.

## Phase 4 — permission graph

**Goal:** build the evidence-backed graph the authorizer reasons over.

- Nodes: `kind ∈ {context, source, tool, sink}` with `id`, `op`,
  `args`/`outputs`, `labels`, `requests`, `time`.
- Edges: `type ∈ {data, control, parent}` with `src`, `dst`, `evidence`,
  `trust ∈ {observed, policy, inferred}`.
- Deterministic lowering from the recorded events to the graph. Same events ⇒
  same graph, byte-for-byte — required for auditing and for replaying a decision.
- Structural validation (well-formed graph) and evidential validation (every edge
  cites evidence that exists).
- Version the graph: `(run_id, graph_version)`. A decision is bound to the
  version it was made against, so a later append cannot silently widen an earlier
  authorization.
- **Data flow is unusually visible here.** The whole conversation crosses the
  wire on every request, so a tool result flowing into a later tool call's
  arguments is directly observable — the prompt-injection path the paper cares
  about, available without instrumenting the harness.

**Deliverable:** `fava.graph.PermissionGraph` plus `fava.graph.lower(events) →
(graph, version)` and `validate(graph) → list[Violation]`.

## Phase 5 — authorizer

**Goal:** a decision on every requested tool call, before the harness runs it.

Interface, fixed now so the backend can change underneath it:

```python
authorize(
    run_id: str,
    graph_version: int,
    candidate_action: CandidateAction,   # one ToolIntent, in context
    policy_context: PolicyContext,
) -> Allow | Block | Unknown
```

- `Allow` carries `capability_scope` and `decision_id`.
- `Block` carries `reason`, the violating **evidence references**, and
  `decision_id`.
- **Fail closed on both `Block` and `Unknown`.** An authorizer that errors, times
  out, or cannot decide must deny. This is the single most important behavioral
  change from Phase 1, where hooks are advisory and a failure forwards anyway.
- A response may carry **several parallel intents**. Decide per intent, and
  define the composition rule explicitly: denying one call while allowing its
  siblings leaves the model with a partial result it did not expect. Blocking the
  whole turn is simpler to reason about and is the recommended starting point.
- Start with a placeholder policy (deny-by-default on a configurable dangerous
  list, allow the rest) so the plumbing and the denial path are exercised end to
  end before any solver work. Z3/SMT comes later, behind this same interface.

**Deliverable:** `fava.authorizer.Authorizer` protocol + `PlaceholderAuthorizer`
+ a `DenyResult` the relay can render.

## Phase 6 — enforcement in the relay

**Goal:** make the gateway actually gate.

This is the phase that changes `relay.py` control flow, and it is **harder here
than on an MCP wire**. There, denying meant not forwarding a request. Here the
intent is in the *response*, already on its way to the harness, so enforcement
means **altering the response before the harness sees it**:

- Replace the `tool_calls` with an assistant message explaining the denial, so
  the model can react and the harness has nothing to dispatch; or
- return an API-shaped error for the whole turn.

Either way, byte fidelity is deliberately abandoned **for denied responses only**.
An allowed response must still reach the client byte-identically — that
invariant is what keeps the proxy a drop-in.

### The streaming decision, deferred from Phase 1

For a non-streamed response the body is whole before it is relayed, so the
decision fits naturally. Streaming is the real design question, left open on
purpose because it only matters once a decision can block:

- **Buffer everything** until `[DONE]`, decide, then release. Simple and always
  correct, but destroys streaming latency for every turn, including the vast
  majority that request nothing.
- **Buffer from the first tool-call delta** (recommended). Content tokens stream
  through untouched; the moment a `delta.tool_calls` fragment appears, hold
  frames until `finish_reason`, decide, then release or replace. Costs latency
  only on turns that actually request a call — and those are exactly the turns
  where a pause is defensible.

Note the ordering constraint this imposes on `relay.py`: today the relay forwards
each chunk and *then* observes it. Enforcement inverts that for tool-call frames
— observe, decide, then forward. Keep the two paths clearly separated so the
allow path stays a pure passthrough.

Other constraints to preserve:

- Gate on `may_dispatch_tools`. A plain answer, a provider error and a
  `/v1/models` call must not pay for authorization.
- Deny with a well-formed response the client SDK can parse, so the harness
  surfaces a denial rather than a decoding failure. `relay._send_api_error`
  already has the right shape to reuse.
- Authorization latency sits inline on the tool-call path. Budget it, measure it,
  and keep the placeholder fast enough that Phase 6 does not regress current
  round trips.

### The assumption enforcement rests on

Blocking the intent only helps if the harness cannot run the tool anyway. A
harness that dispatches a call it never received back is outside the guarantee —
as is one reaching the provider directly. Deployment must ensure the provider is
unreachable except through the proxy; see "Assumptions" below.

## Phase 7 — repair and metrics

- Monotone append-only runtime repair: on new evidence, append and re-decide;
  never mutate history. JIT re-authorization for long-running runs.
- Instrument the gateway's decisions and measure **DCR** (=
  (TP+TN)/(TP+TN+FP+FN)) against labelled traces. Without this the authorization
  stage cannot be tuned — a gateway that denies everything is perfectly safe and
  perfectly useless.

## Assumptions that bound the guarantee

The paper is explicit (§ Assumptions, `#L818-L824`) and these transfer here, but
the move to the LLM boundary changes what each one costs:

1. **The gateway mediates all security-relevant effects.** This is what the
   pivot bought: local tools and MCP-backed tools alike are requested over this
   wire. It now requires that the harness reach the provider *only* through the
   proxy (no fallback base URL, no second provider key) and that it dispatch only
   the calls it receives back.
2. **The harness is call-conformant:** it runs what the response asked for, and
   reports results faithfully in the next request. Tool *execution* is never
   observed directly, so a lying harness is outside the TCB boundary.
3. **Policy translation and sanitizer specifications are inside the TCB.**

Also worth stating plainly: **the initial prompt is not optional here** — it
arrives on every request as part of the message array, whether or not FAVA asks
for it. Task context therefore sharpens decisions for free, and fewer requests
should resolve to `Unknown` than on the MCP wire.

## Suggested order and first task

Phases 2 → 6 are strictly sequential: run identity before events, events before a
graph, a graph before a decision, a decision before enforcement. Phase 3 (catalog
normalization) is independent and can run in parallel with 4 and 5.

The first concrete task is **Phase 2's `run_id`**, because it is a prerequisite
for everything else and because the obvious answers are gone: there is no session
header at this boundary. Decide the propagation mechanism first — including how a
truncated or compacted history is handled — and the store design follows from it.
