"""The evidence-backed permission graph, and the pass that lowers events into it.

A flat permission list cannot express "review before commit", so the paper
lowers the Permission IR into a graph *G = (V, E)* whose nodes carry
evidence-grounded security labels and whose typed edges carry dependency
relations. :data:`PermissionGraph` is that schema, field for field (paper,
Table 1), and :func:`lower` is the pass that builds it.

Two things distinguish this boundary from the paper's setting, and both make the
graph better rather than worse:

* **Data flow is directly observable.** The whole conversation crosses the wire
  on every request, so a tool result flowing into a later tool call's arguments
  is visible without instrumenting the harness. That is the prompt-injection
  path the paper cares about, and here it is an ``observed`` edge rather than an
  inferred one.
* **Effects are not.** An OpenAI tool definition has no ``destructiveHint``, so
  what a tool *does* cannot be read off the catalog. It comes from policy —
  :func:`capabilities_for` below, which is deliberately one small table.

Lowering is total and never raises. A malformed graph is caught by
:func:`validate`, which returns violations rather than throwing, because the
caller sits on the relay's observation path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Final

from fava.state.events import Event, EventKind, EventLog
from fava.state.ir import Evidence, PermissionIR

# Shortest run of result text that counts as having flowed into a later call.
# Long enough that shared boilerplate ("true", a path fragment) does not trip
# it; short enough to catch a pasted sentence.
MIN_FLOW_MATCH: Final = 24

# How much of a tool result is retained on a node, and searched for flows.
MAX_RETAINED_CONTENT: Final = 64 * 1024

# Policy rule identifiers, usable as edge evidence. An edge asserted by policy
# rather than observed still has to name *which* policy said so.
POLICY_CAPABILITY_SINK: Final = "policy:capability-sink"
POLICY_OBLIGATION: Final = "policy:obligation"
POLICY_RULES: Final = frozenset({POLICY_CAPABILITY_SINK, POLICY_OBLIGATION})


class NodeKind(str, Enum):
    """What a node represents."""

    CONTEXT = "context"
    SOURCE = "source"
    TOOL = "tool"
    SINK = "sink"


class EdgeType(str, Enum):
    """What an edge means.

    ``data`` carries value flow, ``control`` carries temporal guards, and
    ``parent`` carries runtime ancestry. Labels propagate along ``data`` and
    policy ``control`` edges only — ancestry is not a flow.
    """

    DATA = "data"
    CONTROL = "control"
    PARENT = "parent"


class Trust(str, Enum):
    """Where an edge came from.

    The authorizer trusts ``observed`` and ``policy`` edges. ``inferred`` edges
    — the ones that exist because the extractor said so and a string match
    agreed — are retained purely for auditing extraction errors.
    """

    OBSERVED = "observed"
    POLICY = "policy"
    INFERRED = "inferred"


@dataclass(frozen=True)
class Label:
    """A security label on a node, with what grounds it."""

    name: str
    evidence: Evidence | None = None


@dataclass(frozen=True)
class Node:
    """One vertex of the permission graph (paper, Table 1).

    Attributes:
        id: Unique within the graph.
        kind: ``context``, ``source``, ``tool`` or ``sink``.
        op: The action — a tool name, ``read``, ``task``.
        args: Input arguments.
        outputs: Produced values.
        labels: Security labels, each carrying its evidence.
        requests: Normalized capabilities this node needs, e.g.
            ``file:write:/tmp/x``.
        time: Execution order — the event's sequence number in the run.
    """

    id: str
    kind: NodeKind
    op: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    labels: tuple[Label, ...] = ()
    requests: tuple[str, ...] = ()
    time: int = 0


@dataclass(frozen=True)
class Edge:
    """One dependency of the permission graph (paper, Table 1).

    Attributes:
        src: Id of the node the dependency comes from.
        dst: Id of the node it reaches.
        type: ``data``, ``control`` or ``parent``.
        evidence: The ``event_id`` that was observed, or the policy rule that
            asserted this. Never empty — :func:`validate` rejects an edge whose
            evidence does not resolve.
        trust: ``observed``, ``policy`` or ``inferred``.
    """

    src: str
    dst: str
    type: EdgeType
    evidence: str
    trust: Trust


@dataclass(frozen=True)
class PermissionGraph:
    """A run's permission graph at one point in its event log.

    Attributes:
        run_id: The run this describes.
        version: The number of events it was lowered from. A decision can be
            bound to this, so a later append cannot silently widen an earlier
            authorization.
        nodes: Vertices, in lowering order.
        edges: Dependencies, in lowering order.
        evidence_ids: Everything an edge is allowed to cite — the event ids in
            the log plus the policy rule names.
    """

    run_id: str
    version: int
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    evidence_ids: frozenset[str] = frozenset()

    def node(self, node_id: str) -> Node | None:
        """The node with this id, or ``None``."""
        return next((node for node in self.nodes if node.id == node_id), None)

    def of_kind(self, kind: NodeKind) -> tuple[Node, ...]:
        """Every node of one kind, in lowering order."""
        return tuple(node for node in self.nodes if node.kind is kind)

    def edges_of(self, edge_type: EdgeType) -> tuple[Edge, ...]:
        """Every edge of one type, in lowering order."""
        return tuple(edge for edge in self.edges if edge.type is edge_type)

    def predecessors(self, node_id: str) -> tuple[str, ...]:
        """Ids reaching ``node_id`` over data and control edges.

        The paper's ``Pred(v)``: the set labels propagate from. Parent edges are
        excluded — ancestry is not a flow.
        """
        return tuple(
            edge.src for edge in self.edges if edge.dst == node_id and edge.type is not EdgeType.PARENT
        )


# -- capability normalization ----------------------------------------------


@dataclass(frozen=True)
class CapabilityRule:
    """Maps a tool name onto a normalized capability.

    Attributes:
        keywords: Word fragments in the tool name that select this rule.
        capability: Capability prefix, e.g. ``file:write``.
        arg_fields: Argument keys whose value completes the capability, tried
            in order. Empty means the capability needs no target.
    """

    keywords: tuple[str, ...]
    capability: str
    arg_fields: tuple[str, ...] = ()


# Ordered most-dangerous first: `read_write_file` should normalize to a write.
# This stands in for the effect annotations an MCP catalog would have given us
# for free, and is meant to be replaced by a real policy module — keep it one
# table and one function so that swap stays cheap.
CAPABILITY_RULES: Final = (
    CapabilityRule(("shell", "bash", "exec", "command", "terminal", "subprocess"), "proc:exec", ("command", "cmd", "script")),
    CapabilityRule(("send", "post", "publish", "email", "slack", "upload", "webhook", "notify"), "net:send", ("url", "endpoint", "channel", "to", "recipient", "host")),
    CapabilityRule(("fetch", "http", "curl", "request", "browse", "download"), "net:fetch", ("url", "endpoint", "host")),
    CapabilityRule(("delete", "remove", "rm", "drop", "truncate"), "file:delete", ("path", "file", "file_path", "filename", "target")),
    CapabilityRule(("write", "edit", "create", "append", "save", "patch"), "file:write", ("path", "file", "file_path", "filename", "target")),
    CapabilityRule(("read", "cat", "open", "view", "load", "get", "list"), "file:read", ("path", "file", "file_path", "filename", "target")),
)

_WORD_SPLIT: Final = re.compile(r"[^a-z0-9]+")


def capabilities_for(tool_name: str | None, args: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Normalize a tool call into the capabilities it requests.

    Always yields ``tool:{name}``, so a policy can forbid a specific tool the
    way the paper does (``pii -> tool:slack.post_message``) without this table
    having to recognize it. A recognized effect adds a second, coarser
    capability — ``file:write:/tmp/x``, ``net:send:hooks.slack.com`` — which is
    what a label-to-sink rule matches against.

    Args:
        tool_name: Function name the model asked for.
        args: The decoded arguments, when they parsed.

    Returns:
        Capabilities, most specific first, without duplicates.
    """
    if not tool_name:
        return ()

    capabilities = [f"tool:{tool_name}"]
    words = {word for word in _WORD_SPLIT.split(tool_name.lower()) if word}
    for rule in CAPABILITY_RULES:
        if not words & set(rule.keywords):
            continue
        target = _argument_target(args, rule.arg_fields)
        capabilities.append(f"{rule.capability}:{target}" if target else rule.capability)
        break

    return tuple(dict.fromkeys(capabilities))


def _argument_target(args: Mapping[str, Any] | None, fields: Sequence[str]) -> str:
    """The first named argument that carries a usable target string."""
    if not args:
        return ""
    for name in fields:
        value = args.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def matches_sink(capability: str, sink_target: str) -> bool:
    """Whether a requested capability reaches a sink the IR named.

    Glob patterns work (``net:*``, ``tool:slack.*``). Otherwise the sink is
    matched as a substring, because an extractor names sinks the way a person
    does — "slack", "the customer's email" — not as capability patterns.
    """
    target = sink_target.strip().lower()
    if not target:
        return False
    lowered = capability.lower()
    if any(char in target for char in "*?["):
        return fnmatch(lowered, target)
    return target in lowered


# -- lowering ---------------------------------------------------------------


def lower(log: EventLog, ir: PermissionIR) -> PermissionGraph:
    """Build the permission graph for a run's log and extracted IR.

    Mapping rules, per the paper: IR assets become source nodes, observed tool
    calls become tool nodes, IR sinks become sink nodes, and the run's task
    becomes the single context node everything descends from.

    Args:
        log: The run's append-only event log.
        ir: The extracted Permission IR. An unextracted one still produces a
            graph — it just carries an ambiguous risk posture and no task-side
            assets or sinks.

    Returns:
        The graph at ``version == len(log)``. Total: no input raises.
    """
    events = log.events
    nodes: list[Node] = []
    edges: list[Edge] = []

    ir_event = next((event for event in events if event.kind is EventKind.IR_EXTRACTED), None)
    root = events[0].event_id if events else f"{log.run_id[:8]}-0000"
    ir_ref = ir_event.event_id if ir_event is not None else root

    context_id = f"ctx:{log.run_id[:12]}"
    nodes.append(_context_node(context_id, ir))

    # IR-derived nodes. They exist because the extractor said so, so everything
    # anchoring them to the run is `inferred` — auditable, and ignored by the
    # authorizer when it comes to trust decisions.
    emitted: set[str] = {context_id}

    asset_ids: dict[str, str] = {}
    for asset in ir.assets:
        node_id = f"asset:{_slug(asset.name)}"
        if node_id in emitted:
            continue
        emitted.add(node_id)
        asset_ids[asset.name] = node_id
        nodes.append(
            Node(
                id=node_id,
                kind=NodeKind.SOURCE,
                op="asset",
                args={"name": asset.name},
                labels=tuple(Label(name, asset.evidence) for name in asset.labels),
            )
        )
        edges.append(Edge(node_id, context_id, EdgeType.PARENT, ir_ref, Trust.INFERRED))

    sink_ids: dict[str, str] = {}
    for sink in ir.sinks:
        node_id = f"sink:{_slug(sink.target)}"
        if node_id in emitted:
            continue
        emitted.add(node_id)
        sink_ids[sink.target] = node_id
        nodes.append(
            Node(
                id=node_id,
                kind=NodeKind.SINK,
                op="send",
                args={"target": sink.target},
                labels=tuple(Label(name, sink.evidence) for name in sink.labels),
            )
        )
        edges.append(Edge(node_id, context_id, EdgeType.PARENT, ir_ref, Trust.INFERRED))

    guard_ids: dict[str, str] = {}
    for obligation in ir.obligations:
        node_id = f"guard:{_slug(obligation.requires)}"
        if node_id in emitted:
            continue
        emitted.add(node_id)
        guard_ids[obligation.requires] = node_id
        nodes.append(
            Node(
                id=node_id,
                kind=NodeKind.CONTEXT,
                op="obligation",
                args={"requires": obligation.requires, "before": obligation.before},
                labels=(Label("obligation", obligation.evidence),),
            )
        )
        edges.append(Edge(node_id, context_id, EdgeType.PARENT, ir_ref, Trust.INFERRED))

    # Observed nodes. Both lists keep the event beside the node it produced, so
    # the flow passes below can read `seq` and arguments without re-scanning.
    intents: list[tuple[Event, Node]] = []
    results: list[tuple[Event, Node]] = []
    by_event: dict[str, Node] = {}

    for event in events:
        if event.kind is EventKind.TOOL_INTENT:
            node = _tool_node(event)
            intents.append((event, node))
            by_event[event.event_id] = node
            nodes.append(node)
            edges.append(Edge(node.id, context_id, EdgeType.PARENT, event.event_id, Trust.OBSERVED))

        elif event.kind is EventKind.TOOL_RESULT:
            node = _result_node(event)
            results.append((event, node))
            nodes.append(node)
            intent = log.intent_for(event.tool_call_id)
            parent = by_event.get(intent.event_id) if intent is not None else None
            if parent is not None:
                # The result descends from the call that produced it — the
                # correlation that spans two HTTP exchanges.
                edges.append(Edge(node.id, parent.id, EdgeType.PARENT, event.event_id, Trust.OBSERVED))

    # Argument text per intent, computed once: every flow pass searches it.
    haystacks = {event.event_id: _haystack(event.payload) for event, _ in intents}

    # Observed data flow: a result's content reappearing in a later call's
    # arguments. Directly visible here, and the edge the injection policy needs.
    for result_event, result_node in results:
        content = str(result_node.outputs.get("content") or "")
        if len(content) < MIN_FLOW_MATCH:
            continue
        for intent_event, tool_node in intents:
            if intent_event.seq <= result_event.seq:
                continue
            if _flows_into(content, haystacks[intent_event.event_id]):
                edges.append(
                    Edge(result_node.id, tool_node.id, EdgeType.DATA, intent_event.event_id, Trust.OBSERVED)
                )

    # Policy flow: a requested capability that reaches a sink the task named.
    for _, tool_node in intents:
        for target, sink_id in sink_ids.items():
            if any(matches_sink(capability, target) for capability in tool_node.requests):
                edges.append(Edge(tool_node.id, sink_id, EdgeType.DATA, POLICY_CAPABILITY_SINK, Trust.POLICY))

    # Inferred flow: an asset the task named appearing in a call's arguments.
    for name, asset_id in asset_ids.items():
        if len(name) < 3:
            continue
        needle = name.lower()
        for intent_event, tool_node in intents:
            if needle in haystacks[intent_event.event_id].lower():
                edges.append(Edge(asset_id, tool_node.id, EdgeType.DATA, ir_ref, Trust.INFERRED))

    # Temporal guards: the obligation must hold before the call it guards.
    for obligation in ir.obligations:
        guard_id = guard_ids.get(obligation.requires)
        if guard_id is None:
            continue
        for _, tool_node in intents:
            if _guards(obligation.before, tool_node.op):
                edges.append(Edge(guard_id, tool_node.id, EdgeType.CONTROL, POLICY_OBLIGATION, Trust.POLICY))

    evidence_ids = frozenset({event.event_id for event in events} | POLICY_RULES | {root})
    return PermissionGraph(
        run_id=log.run_id,
        version=len(log),
        nodes=tuple(nodes),
        edges=tuple(edges),
        evidence_ids=evidence_ids,
    )


def _context_node(node_id: str, ir: PermissionIR) -> Node:
    """The single node every call in the run descends from.

    Carries the risk posture as a label, because that is what the gateway fails
    closed on, and the declared actions as arguments so an action the task asked
    for but never performed is still visible.
    """
    return Node(
        id=node_id,
        kind=NodeKind.CONTEXT,
        op="task",
        args={
            "intent": ir.intent,
            "actions": [{"op": action.op, "target": action.target} for action in ir.actions],
            "extracted": ir.extracted,
        },
        labels=(Label(f"risk:{ir.risk_posture.value}", Evidence(ref="task", quote=ir.intent)),),
    )


def _tool_node(event: Event) -> Node:
    """A tool node from an observed intent."""
    arguments = event.payload.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    return Node(
        id=f"tool:{event.event_id}",
        kind=NodeKind.TOOL,
        op=event.tool_name or "",
        args=dict(arguments),
        requests=capabilities_for(event.tool_name, arguments),
        time=event.seq,
    )


def _result_node(event: Event) -> Node:
    """A source node from an observed tool result."""
    content = event.payload.get("content")
    content = content if isinstance(content, str) else ""
    return Node(
        id=f"src:{event.event_id}",
        kind=NodeKind.SOURCE,
        op="result",
        args={"tool_call_id": event.tool_call_id},
        outputs={"content": content[:MAX_RETAINED_CONTENT]},
        time=event.seq,
    )


def _flows_into(content: str, haystack: str) -> bool:
    """Whether a result's content reappears in a later call's arguments.

    ``haystack`` comes from :func:`_haystack`, which reads the *decoded*
    argument values rather than the raw JSON string: the arguments carry the
    content JSON-escaped, so a quote or newline in the result would defeat a
    comparison against the wire text.

    A deliberately plain substring test. It finds the case that matters — text
    from a tool result pasted into the next call — and is cheap to replace with
    something better once there is a decision riding on it.
    """
    if not haystack:
        return False
    normalized = _squeeze(content)
    if len(normalized) >= MIN_FLOW_MATCH and normalized in haystack:
        return True
    return any(
        len(segment) >= MIN_FLOW_MATCH and segment in haystack
        for segment in (_squeeze(line) for line in content.splitlines())
    )


def _haystack(intent_payload: Mapping[str, Any]) -> str:
    """The searchable text of a tool call's arguments."""
    arguments = intent_payload.get("arguments")
    if isinstance(arguments, Mapping):
        return _squeeze(" ".join(_render(value) for value in arguments.values()))
    raw = intent_payload.get("arguments_json")
    return _squeeze(raw) if isinstance(raw, str) else ""


def _render(value: Any) -> str:
    """Flatten one argument value to text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(value)


def _squeeze(text: str) -> str:
    """Collapse runs of whitespace, so reindented text still matches."""
    return " ".join(text.split())


def _guards(before: str, tool_name: str) -> bool:
    """Whether an obligation's ``before`` clause names this tool."""
    if not before or not tool_name:
        return False
    tool_words = {word for word in _WORD_SPLIT.split(tool_name.lower()) if word}
    before_words = {word for word in _WORD_SPLIT.split(before.lower()) if len(word) >= 3}
    return bool(tool_words & before_words)


def _slug(text: str) -> str:
    """A stable, readable node-id fragment."""
    slug = _WORD_SPLIT.sub("-", text.strip().lower()).strip("-")
    return slug[:64] or "unnamed"


# -- validation -------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """A malformed construct found by :func:`validate`.

    Attributes:
        code: Machine-readable kind, e.g. ``dangling-edge``.
        detail: Human-readable explanation.
        subject: The node or edge id it concerns.
    """

    code: str
    detail: str
    subject: str = ""


def validate(graph: PermissionGraph) -> list[Violation]:
    """Check a graph structurally and evidentially.

    The paper validates before the authorizer runs, to keep an extraction error
    from reaching the solver as if it were a fact. Two families of check:

    * **Structural** — ids are unique, kinds are legal, every edge lands on
      nodes that exist, nothing points at itself, and a child never predates
      its parent.
    * **Evidential** — every edge cites evidence that resolves to an event in
      the log or to a named policy rule.

    Returns:
        The violations found, empty when the graph is well formed. Never
        raises: the caller is an observer on the relay's path.
    """
    violations: list[Violation] = []

    seen: set[str] = set()
    times: dict[str, int] = {}
    for node in graph.nodes:
        if node.id in seen:
            violations.append(Violation("duplicate-node", f"node id {node.id!r} appears more than once", node.id))
        seen.add(node.id)
        times[node.id] = node.time
        if not isinstance(node.kind, NodeKind):
            violations.append(Violation("bad-node-kind", f"node {node.id!r} has kind {node.kind!r}", node.id))
        for label in node.labels:
            if label.evidence is None:
                violations.append(
                    Violation("ungrounded-label", f"label {label.name!r} on {node.id!r} cites no evidence", node.id)
                )

    for edge in graph.edges:
        subject = f"{edge.src}->{edge.dst}"
        if edge.src not in seen:
            violations.append(Violation("dangling-edge", f"edge source {edge.src!r} is not a node", subject))
        if edge.dst not in seen:
            violations.append(Violation("dangling-edge", f"edge target {edge.dst!r} is not a node", subject))
        if edge.src == edge.dst:
            violations.append(Violation("self-edge", f"node {edge.src!r} depends on itself", subject))
        if not isinstance(edge.type, EdgeType):
            violations.append(Violation("bad-edge-type", f"edge {subject} has type {edge.type!r}", subject))
        if not isinstance(edge.trust, Trust):
            violations.append(Violation("bad-edge-trust", f"edge {subject} has trust {edge.trust!r}", subject))
        if not edge.evidence:
            violations.append(Violation("unevidenced-edge", f"edge {subject} cites no evidence", subject))
        elif edge.evidence not in graph.evidence_ids:
            violations.append(
                Violation("unknown-evidence", f"edge {subject} cites unknown evidence {edge.evidence!r}", subject)
            )
        if (
            edge.type is EdgeType.PARENT
            and edge.src in times
            and edge.dst in times
            and times[edge.src] < times[edge.dst]
        ):
            violations.append(
                Violation("non-monotonic-parent", f"{edge.src!r} precedes its parent {edge.dst!r}", subject)
            )

    return violations


__all__ = [
    "CAPABILITY_RULES",
    "CapabilityRule",
    "Edge",
    "EdgeType",
    "Label",
    "Node",
    "NodeKind",
    "PermissionGraph",
    "Trust",
    "Violation",
    "capabilities_for",
    "lower",
    "matches_sink",
    "validate",
]
