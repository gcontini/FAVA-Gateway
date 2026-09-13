# FAVA implementation plan

Status, and the path to the full FAVA gateway.

**Phase 1 (relay) and Phases 2 and 4 (state and permission graph) are complete
and tested.** `src/llm_proxy/` relays bytes exactly and exposes read-only
observation hooks; `src/fava/state/` turns those observations into per-run
append-only event logs, an extracted Permission IR, and the evidence-backed
permission graph lowered from both. Nothing decides on the graph yet — that is
Phase 5.

The paper is `docs/FAVA_ Formal Authorization for Verified Agents with
Evidence-Backed Permission Graphs.html` (arXiv:2607.27267v1 [cs.CR]).

## Why this boundary

The gateway sits between the **harness and the LLM API**, not on the MCP wire.
That placement is what makes the paper's first assumption — that **all**
security-relevant effects are mediated before execution — holdable at all: a
proxy on the MCP wire only sees tools routed through MCP servers, so a
harness's local tools (shell, file edits, web fetch) would stay invisible.
Intercepting at the harness↔LLM boundary fixes that:

1. **Every tool call passes through here first.** Local and MCP-backed tools
   alike begin as a tool-call *intent* in the model's response. From the
   model's side both are just declared tools with a name and a schema — how
   the harness dispatches them is invisible to the API wire format.
2. **It is harness-agnostic.** The integration is the standard base-URL
   override nearly every agent runtime already supports. No plugin, no patch.

The trade-off is that the proxy observes *intent*, not execution. It sees what
the model asked for, and — on the following request — what the harness
reported back. It never sees the tool actually run.

## Where Phase 1 already lands

The proxy is the **runtime enforcement gateway** of the paper's macro
architecture (§ Runtime Enforcement). It is now in the right *place* — the
harness↔LLM boundary, where every tool call first appears as an intent
regardless of how the harness will dispatch it — but it still mediates without
deciding: it observes, builds the graph, and forwards regardless. The Permission
IR and the evidence graph now exist (Phases 2 and 4); no authorizer consults
them.

What Phase 1 settled, and that later phases should not re-litigate:

- **The interception point.** A raw ASGI relay at the LLM API boundary, reached
  by a base-URL override. This covers local and MCP-backed tools uniformly,
  which an MCP-wire proxy cannot (see "Why this boundary" above).
- **The decision side is the response.** Tool-call intent arrives from the
  provider; the harness dispatches only afterwards. `on_response` is the seam.
- **Hooks are read-only and best-effort.** The gateway degrades to a pure relay
  if an observer breaks, which is the correct failure mode for Phase 1 but must
  **not** survive into the enforcement phase — see "Fail closed" below.
- **`ResponseContext.may_dispatch_tools`** already implements the conservative
  classification the authorizer will gate on: observed intents *or* a response
  too opaque to rule them out.
- **The tool catalog is in-band.** Every request carries `tools[]`, so there is
  no catalog to fetch, cache or invalidate. `RunState.tool_catalog` holds the
  most recently declared names; Phase 3's drift detection is still open.

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
| `fava-adapter` | **done** (`relay.py`, `hooks.py`, `chat.py`, `state/hooks.py`) | Interception, `run_id` propagation, catalog extraction, event recording |
| `fava-state` | **done** (`state/events.py`, `state/store.py`) | Per-run state: `run_id → PermissionGraph + events + IR`, append-only, graph versioning |
| `fava-graph` | **done** (`state/graph.py`, `state/ir.py`) | Node/edge construction, lowering, structural + evidential validation |
| `fava-authorizer` | not started | `Allow / Block / Unknown` decision; SMT backend later; fail closed |

They are modules of one package (`fava.state`), not separate distributions: the
interfaces matter, the process boundaries do not. `fava-graph` landed inside
`fava.state` rather than beside it, because a graph is a pure function of a run's
log and the two have no independent life; promoting `graph.py` to `fava.graph`
later is a move, not a redesign.

## Phase 2 — run identity and event recording — **done**

**Goal:** attach every observed exchange to a run, and record it append-only.

Delivered as `fava.state.events` + `fava.state.store`, wired in through
`fava.state.hooks.StateHooks`.

What was settled:

- **`run_id` comes from the traffic.** `X-FAVA-Run-Id` when a harness sets one,
  otherwise a SHA-256 of the stable prefix — the system/developer messages plus
  the first user turn, which every request in a run repeats. The `(host, port)`
  peer fallback was dropped: a completion always has a prefix to hash, and a
  request without one has no run to belong to.
- **A rewritten prefix is a new run.** Truncation and compaction change the
  hash, and that is the answer, not a problem to solve. A fresh graph built from
  what is still visible is honest; merging two prefixes would assert a
  continuity the proxy cannot see.
- **Five event kinds**, no more: `run_started`, `tool_intent`, `tool_result`,
  `response_opaque`, `ir_extracted`. Anything derivable is derived — an intent
  with no result is `EventLog.unanswered_intents()`, not an event, because the
  absence is the observation.
- **Intent↔outcome correlation works across the two exchanges**, keyed on
  `tool_call_id` (`EventLog.intent_for`). Since every request replays the whole
  history, results are de-duplicated on the way in or the log grows
  quadratically.
- **Retention:** arguments whole, result content capped at 64 KiB, in memory,
  bounded to the most recent `max_runs` runs. No hashing and no redaction mode —
  add them when there is a deployment that needs one.

State is in-memory. Redis goes behind `RecordStore`, not inside the relay.

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

## Phase 4 — permission graph — **done**

**Goal:** build the evidence-backed graph the authorizer reasons over.

Delivered as `fava.state.ir` (the Permission IR and its LLM extractor) and
`fava.state.graph` (`lower`, `validate`, capability normalization).

What was settled:

- **Table 1, field for field.** Nodes carry `id`, `kind`, `op`, `args`,
  `outputs`, `labels`, `requests`, `time`; edges carry `src`, `dst`, `type`,
  `evidence`, `trust`.
- **The IR is extracted by an LLM, once per run, in the background.** It never
  raises: any failure yields `RiskPosture.AMBIGUOUS`, the value the gateway
  fails closed on. It calls the provider **directly** — routed through the proxy
  its own request would be observed, minted as a run, and extracted from,
  recursively.
- **Data flow is observed, as predicted.** A tool result's content reappearing
  in a later call's arguments is a `data`/`observed` edge, found by substring
  match over the decoded argument values. That is the prompt-injection path,
  available with no harness instrumentation.
- **Trust is three-valued and load-bearing.** `observed` (seen on the wire),
  `policy` (a capability matching a named sink, an obligation guarding a call)
  and `inferred` (an IR asset name appearing in arguments — a string match, not
  a flow). Only the first two are meant to be trusted; `inferred` edges exist to
  audit extraction errors.
- **Capabilities come from policy, not the catalog.** An OpenAI tool definition
  has no `destructiveHint`, so `capabilities_for` is one small ordered table
  mapping a tool name and its arguments onto `tool:{name}` plus a coarser
  `proc:exec:…` / `file:write:…` / `net:send:…`. Explicitly a placeholder.
- **`graph_version` is `len(log)`** — monotone, and it binds a decision to the
  exact log prefix it was made against.
- **Validation returns, never raises**: dangling edges, duplicate ids,
  ungrounded labels, unknown evidence, a child predating its parent.

**Not built, deliberately:** byte-for-byte graph replay. "Deterministic
lowering" in the paper means the pass is a plain function rather than an LLM
judgment, which it is. Reproducing a past graph exactly is an audit feature, and
it belongs with the first decision worth auditing — not before.

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

## Suggested order and next task

Phases 2 → 6 are strictly sequential: run identity before events, events before a
graph, a graph before a decision, a decision before enforcement. Phase 3 (catalog
normalization) is independent and can run in parallel with 4 and 5.

Phases 2 and 4 are done, so **the next task is Phase 5, the authorizer**. The
graph it needs exists and validates; what remains is to decide on it. Start with
the placeholder policy — deny-by-default on a configurable dangerous list — so
the denial path is exercised end to end before any solver work, and so Phase 6's
control-flow change has something real to gate on.

Two things Phase 5 inherits and must resolve rather than ignore:

- **An IR that has not landed yet.** Extraction is a background task, so the
  first turn of a run is usually lowered against an `ambiguous` IR. The
  authorizer must either wait for it with a budget or treat "not yet extracted"
  as a `Block`, and either way the choice belongs in the decision, not in the
  store.
- **`trust`.** Only `observed` and `policy` edges may carry a decision;
  `inferred` edges are for auditing extraction errors. Nothing enforces that
  today because nothing reads the graph yet.
