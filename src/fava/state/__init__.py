"""Per-run state: events observed, the Permission IR, the permission graph.

`fava-state` in the component decomposition. It turns the proxy's stream of
HTTP exchanges into the substrate the authorizer will reason over:

1. **Run identity** recovered from the traffic — there is no session header at
   this boundary, but every request repeats the conversation prefix.
2. **An append-only event log** per run. Intents are observed in response *N*;
   the results answering them arrive in request *N+1*, keyed by
   ``tool_call_id``. Events are added, never rewritten.
3. **The Permission IR**, extracted from the task text by an LLM used strictly
   as a semantic extractor, never as a decision-maker.
4. **The permission graph**, lowered from both — nodes carrying evidence-backed
   labels, typed edges carrying data flow, temporal guards and ancestry.

Nothing here decides anything. The authorizer that consumes the graph, and the
gateway that enforces its answer, are later phases; see
``docs/implementation_plan.md``.
"""

from fava.state.events import RUN_ID_HEADER, Event, EventKind, EventLog, derive_run_id, task_text
from fava.state.graph import (
    Edge,
    EdgeType,
    Label,
    Node,
    NodeKind,
    PermissionGraph,
    Trust,
    Violation,
    capabilities_for,
    lower,
    validate,
)
from fava.state.hooks import StateHooks
from fava.state.ir import (
    Action,
    Asset,
    Evidence,
    ExtractorSettings,
    IRExtractor,
    LlmIRExtractor,
    NullIRExtractor,
    Obligation,
    PermissionIR,
    RiskPosture,
    Sink,
)
from fava.state.store import RecordStore, RunState

__all__ = [
    "RUN_ID_HEADER",
    "Action",
    "Asset",
    "Edge",
    "EdgeType",
    "Event",
    "EventKind",
    "EventLog",
    "Evidence",
    "ExtractorSettings",
    "IRExtractor",
    "Label",
    "LlmIRExtractor",
    "Node",
    "NodeKind",
    "NullIRExtractor",
    "Obligation",
    "PermissionGraph",
    "PermissionIR",
    "RecordStore",
    "RiskPosture",
    "RunState",
    "Sink",
    "StateHooks",
    "Trust",
    "Violation",
    "capabilities_for",
    "derive_run_id",
    "lower",
    "task_text",
    "validate",
]
