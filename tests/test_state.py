"""Tests for `fava.state`: run identity, the event log, and graph lowering.

Most of this runs in-process against hand-built contexts, in the style of
`test_units.py`. The end-to-end section at the bottom puts a real client, a real
proxy and a real fake provider on TCP sockets, because the thing most worth
testing — correlating an intent observed in response *N* to the result reported
in request *N+1* — only happens across two genuine exchanges.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from llm_proxy.app import create_proxy_app
from llm_proxy.chat import parse_chat_request, parse_chat_response
from llm_proxy.config import ProxySettings
from llm_proxy.hooks import RequestContext, ResponseContext

from fava.state import (
    EdgeType,
    EventKind,
    ExtractorSettings,
    LlmIRExtractor,
    NodeKind,
    NullIRExtractor,
    PermissionIR,
    RecordStore,
    RiskPosture,
    RunState,
    StateHooks,
    Trust,
    capabilities_for,
    derive_run_id,
    lower,
    task_text,
    validate,
)
from fava.state.graph import Edge, Label, Node, PermissionGraph, matches_sink
from fava.state.ir import Evidence, Obligation, parse_ir
from tests.conftest import (
    FakeUpstream,
    RecordingHooks,
    ServiceHandle,
    ToolCallScript,
    create_upstream_app,
    free_port,
    run_uvicorn,
)

SYSTEM = "You are a deploy bot. Never leak credentials."
TASK = "Read config.yaml and post a summary to the ops channel."

# Long enough to clear MIN_FLOW_MATCH, so a flow into a later call is detectable.
SECRET = "AKIA0123456789EXAMPLE deploy key, do not share with anyone"


# -- helpers ----------------------------------------------------------------


def _request(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[str] = (),
    headers: Mapping[str, str] | None = None,
    suffix: str = "/chat/completions",
) -> RequestContext:
    """A RequestContext carrying a parsed chat body, as the relay builds it."""
    body = {
        "model": "gpt-4o-mini",
        "messages": list(messages),
        "tools": [{"type": "function", "function": {"name": name}} for name in tools],
    }
    return RequestContext(
        http_method="POST",
        path=f"/v1{suffix}",
        upstream_suffix=suffix,
        query="",
        headers=dict(headers or {}),
        body_size=0,
        chat=parse_chat_request(json.dumps(body).encode()),
    )


def _response(request: RequestContext, *calls: tuple[str, str, Mapping[str, Any]]) -> ResponseContext:
    """A ResponseContext whose model asked for ``(id, name, arguments)`` calls."""
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for call_id, name, args in calls
        ]
    body = {
        "id": "chatcmpl-x",
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}
        ],
    }
    return ResponseContext(
        request=request,
        status_code=200,
        headers={},
        content_type="application/json",
        chat=parse_chat_response(json.dumps(body).encode()),
    )


def _turn_one() -> list[dict[str, Any]]:
    """The opening conversation of a run."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]


def _turn_two(content: str = SECRET) -> list[dict[str, Any]]:
    """The same run one turn later: the assistant's call, and its result."""
    return [
        *_turn_one(),
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": content},
    ]


def _traffic_kinds(run: RunState) -> list[EventKind]:
    """The run's event kinds, minus the IR.

    Extraction runs in the background, so where its event lands relative to the
    traffic is genuinely unordered — asserting a fixed position would be
    asserting something that is not true of the running system.
    """
    return [event.kind for event in run.log.events if event.kind is not EventKind.IR_EXTRACTED]


async def _two_turn_run(store: RecordStore, *, leak: bool = True) -> RunState:
    """Drive one run through the store: read a file, then post its content."""
    first = _request(_turn_one(), tools=["read_file", "slack_post_message"])
    await store.observe_request(first)
    await store.observe_response(first_ := _response(first, ("call_1", "read_file", {"path": "config.yaml"})))
    assert first_.has_tool_intents

    second = _request(_turn_two(), tools=["read_file", "slack_post_message"])
    await store.observe_request(second)
    text = SECRET if leak else "an innocuous summary of the configuration file"
    await store.observe_response(
        _response(second, ("call_2", "slack_post_message", {"channel": "#ops", "text": text}))
    )

    run = store.get(derive_run_id(first) or "")
    assert run is not None
    return run


# -- run identity -----------------------------------------------------------


def test_run_id_prefers_a_harness_supplied_header() -> None:
    """A cooperating harness gives the cleanest identity there is."""
    context = _request(_turn_one(), headers={"x-fava-run-id": "run-42"})

    assert derive_run_id(context) == "run-42"


def test_run_id_is_stable_across_turns_of_one_run() -> None:
    """Every request repeats the prefix, so the hash must not move with it."""
    first = derive_run_id(_request(_turn_one()))
    later = derive_run_id(_request(_turn_two()))

    assert first is not None
    assert first == later


def test_run_id_differs_across_runs() -> None:
    """A different task is a different run, hence a different graph."""
    other = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Delete the build cache."}]

    assert derive_run_id(_request(_turn_one())) != derive_run_id(_request(other))


def test_rewritten_prefix_is_a_new_run_not_a_merge() -> None:
    """Compaction rewrites the prefix; asserting continuity would be a guess."""
    compacted = [{"role": "system", "content": "[summary of earlier turns]"}, {"role": "user", "content": TASK}]

    assert derive_run_id(_request(_turn_one())) != derive_run_id(_request(compacted))


def test_request_without_a_prefix_belongs_to_no_run() -> None:
    """`GET /v1/models` and a bodyless completion cannot produce a tool call."""
    assert derive_run_id(_request([], suffix="/models")) is None
    assert derive_run_id(_request([])) is None


def test_task_text_reads_the_prefix_not_the_whole_history() -> None:
    """The extractor sees the task, not the transcript that grew from it."""
    text = task_text(_turn_two())

    assert SYSTEM in text
    assert TASK in text
    assert SECRET not in text


def test_task_text_flattens_multimodal_content_parts() -> None:
    """A content-part list keeps its text and drops its images."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "look at this"}, {"type": "image_url", "url": "x"}]}
    ]

    assert task_text(messages) == "user: look at this"


# -- event log --------------------------------------------------------------


async def test_run_starts_once_and_records_the_task() -> None:
    """Two turns of one run share a log, and `run_started` fires once."""
    store = RecordStore()
    run = await _two_turn_run(store)

    starts = run.log.of_kind(EventKind.RUN_STARTED)
    assert len(store.runs) == 1
    assert len(starts) == 1
    assert starts[0].payload["task_text"].endswith(TASK)
    assert run.tool_catalog == ("read_file", "slack_post_message")


async def test_intents_and_results_correlate_across_two_exchanges() -> None:
    """The evidence loop: intent in response N, result in request N+1."""
    store = RecordStore()
    run = await _two_turn_run(store)

    assert _traffic_kinds(run) == [
        EventKind.RUN_STARTED,
        EventKind.TOOL_INTENT,
        EventKind.TOOL_RESULT,
        EventKind.TOOL_INTENT,
    ]

    result = run.log.of_kind(EventKind.TOOL_RESULT)[0]
    intent = run.log.intent_for("call_1")
    assert intent is not None
    assert result.parent_event_id == intent.event_id
    assert result.payload["content"] == SECRET


async def test_sequence_numbers_are_monotonic() -> None:
    """Ordering is the graph's `time`, so it has to come from the log."""
    store = RecordStore()
    run = await _two_turn_run(store)

    assert [event.seq for event in run.log.events] == list(range(len(run.log)))


async def test_a_repeated_result_is_recorded_once() -> None:
    """Every request replays the whole history; the log must not grow with it."""
    store = RecordStore()
    run = await _two_turn_run(store)

    # A third turn carrying the same `role: "tool"` message again.
    await store.observe_request(_request(_turn_two()))

    assert len(run.log.of_kind(EventKind.TOOL_RESULT)) == 1


async def test_an_intent_with_no_result_is_derived_not_recorded() -> None:
    """Refused, dropped or still running — the absence is the observation."""
    store = RecordStore()
    run = await _two_turn_run(store)

    assert [event.tool_call_id for event in run.log.unanswered_intents()] == ["call_2"]


async def test_an_opaque_response_is_recorded_rather_than_read_as_silence() -> None:
    """`may_dispatch_tools` is true when the proxy could not tell; so is the log."""
    store = RecordStore()
    request = _request(_turn_one())
    await store.observe_request(request)

    opaque = ResponseContext(
        request=request,
        status_code=200,
        headers={},
        content_type="application/json",
        chat=parse_chat_response(b"<html/>"),
    )
    assert opaque.may_dispatch_tools is True
    run = await store.observe_response(opaque)

    assert run is not None
    assert run.log.has_kind(EventKind.RESPONSE_OPAQUE)


async def test_exchanges_outside_a_run_are_ignored() -> None:
    """A side endpoint has no run, and must not mint one."""
    store = RecordStore()

    assert await store.observe_request(_request([], suffix="/models")) is None
    assert store.runs == ()


async def test_looking_a_run_up_does_not_create_one() -> None:
    """Asking what is known must not itself become evidence."""
    store = RecordStore()
    request = _request(_turn_one())

    assert store.get_for(request) is None
    await store.observe_request(request)
    assert store.get_for(request) is store.runs[0]


async def test_the_store_evicts_the_oldest_run() -> None:
    """A long-lived proxy must not accumulate one log per conversation ever seen."""
    store = RecordStore(max_runs=2)
    for index in range(3):
        await store.observe_request(_request([{"role": "user", "content": f"task {index}"}]))

    assert len(store.runs) == 2


# -- capability normalization ----------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("run_shell", {"command": "rm -rf /"}, "proc:exec:rm -rf /"),
        ("write_file", {"path": "/tmp/x"}, "file:write:/tmp/x"),
        ("read_file", {"file_path": "/etc/passwd"}, "file:read:/etc/passwd"),
        ("slack_post_message", {"channel": "#ops"}, "net:send:#ops"),
        ("http_fetch", {"url": "https://example.com"}, "net:fetch:https://example.com"),
    ],
)
def test_capabilities_name_the_effect_and_its_target(name: str, args: dict[str, Any], expected: str) -> None:
    """OpenAI tool definitions carry no effect hints, so policy supplies them."""
    capabilities = capabilities_for(name, args)

    assert capabilities[0] == f"tool:{name}"
    assert expected in capabilities


def test_an_unrecognized_tool_still_gets_a_capability() -> None:
    """A policy can forbid a specific tool without this table knowing it."""
    assert capabilities_for("frobnicate", {}) == ("tool:frobnicate",)


def test_a_dangerous_effect_wins_over_a_benign_one() -> None:
    """`read_write_file` is a write; the table is ordered for that reason."""
    assert "file:write:/tmp/x" in capabilities_for("read_write_file", {"path": "/tmp/x"})


def test_sink_matching_handles_both_globs_and_plain_names() -> None:
    """An extractor names sinks the way a person does, not as glob patterns."""
    assert matches_sink("net:send:#ops", "net:*") is True
    assert matches_sink("tool:slack_post_message", "slack") is True
    assert matches_sink("file:read:/etc/passwd", "slack") is False


# -- lowering ---------------------------------------------------------------


async def test_lowering_builds_the_table_1_schema() -> None:
    """Nodes and edges carry the fields the authorizer will reason over."""
    store = RecordStore()
    run = await _two_turn_run(store)
    graph = run.graph()

    assert graph.version == len(run.log)
    assert len(graph.of_kind(NodeKind.CONTEXT)) == 1

    tools = graph.of_kind(NodeKind.TOOL)
    assert [node.op for node in tools] == ["read_file", "slack_post_message"]
    assert "file:read:config.yaml" in tools[0].requests
    assert tools[0].time < tools[1].time

    results = graph.of_kind(NodeKind.SOURCE)
    assert results[0].outputs["content"] == SECRET


async def test_every_call_descends_from_the_run_context() -> None:
    """Ancestry is what makes a prefix graph a graph rather than a list."""
    store = RecordStore()
    run = await _two_turn_run(store)
    graph = run.graph()
    context = graph.of_kind(NodeKind.CONTEXT)[0]

    parents = graph.edges_of(EdgeType.PARENT)
    assert all(edge.trust is Trust.OBSERVED for edge in parents)
    assert {edge.dst for edge in parents} >= {context.id}


async def test_a_result_flowing_into_a_later_call_is_an_observed_data_edge() -> None:
    """The exfiltration path, visible because the history crosses the wire."""
    store = RecordStore()
    run = await _two_turn_run(store, leak=True)
    graph = run.graph()

    flows = [edge for edge in graph.edges_of(EdgeType.DATA) if edge.trust is Trust.OBSERVED]
    assert len(flows) == 1

    source = graph.node(flows[0].src)
    sink_side = graph.node(flows[0].dst)
    assert source is not None and source.op == "result"
    assert sink_side is not None and sink_side.op == "slack_post_message"
    assert flows[0].src in graph.predecessors(flows[0].dst)


async def test_a_summary_that_does_not_carry_the_content_is_not_a_flow() -> None:
    """Detection has to be able to say no, or the edge means nothing."""
    store = RecordStore()
    run = await _two_turn_run(store, leak=False)

    observed = [edge for edge in run.graph().edges_of(EdgeType.DATA) if edge.trust is Trust.OBSERVED]
    assert observed == []


async def test_a_reindented_result_still_matches() -> None:
    """Whitespace is squeezed, so a pretty-printed paste is still a flow."""
    store = RecordStore()
    first = _request(_turn_one())
    await store.observe_request(first)
    await store.observe_response(_response(first, ("call_1", "read_file", {"path": "c.yaml"})))

    second = _request(_turn_two(f"key:\n  {SECRET}"))
    await store.observe_request(second)
    await store.observe_response(
        _response(second, ("call_2", "http_post", {"url": "https://x", "body": f"key: {SECRET}"}))
    )

    run = store.get(derive_run_id(first) or "")
    assert run is not None
    assert any(edge.trust is Trust.OBSERVED for edge in run.graph().edges_of(EdgeType.DATA))


async def test_ir_sinks_and_assets_become_nodes_and_edges() -> None:
    """IR assets lower to sources, IR sinks to sinks, per the mapping rules."""
    store = RecordStore()
    run = await _two_turn_run(store)
    run.ir = parse_ir(
        json.dumps(
            {
                "intent": "summarize config and post it",
                "risk_posture": "sensitive",
                "assets": [{"name": "config.yaml", "labels": ["secret"], "evidence": "Read config.yaml"}],
                "sinks": [{"target": "slack", "labels": ["public"], "evidence": "post a summary"}],
            }
        )
    )
    graph = lower(run.log, run.ir)

    assets = [node for node in graph.of_kind(NodeKind.SOURCE) if node.op == "asset"]
    assert [label.name for label in assets[0].labels] == ["secret"]

    sinks = graph.of_kind(NodeKind.SINK)
    assert sinks[0].args["target"] == "slack"

    # The Slack call requests `tool:slack_post_message`, which reaches the sink.
    policy = [edge for edge in graph.edges_of(EdgeType.DATA) if edge.trust is Trust.POLICY]
    assert [edge.dst for edge in policy] == [sinks[0].id]

    # The asset was named in a call's arguments — a string match, so `inferred`.
    inferred = [edge for edge in graph.edges_of(EdgeType.DATA) if edge.trust is Trust.INFERRED]
    assert [edge.src for edge in inferred] == [assets[0].id]


async def test_an_obligation_becomes_a_control_edge() -> None:
    """Temporal guards are the reason a flat permission list is insufficient."""
    store = RecordStore()
    run = await _two_turn_run(store)
    run.ir = PermissionIR(
        intent="post a summary",
        obligations=(
            Obligation(requires="review", before="post the message", evidence=Evidence("task", "after review")),
        ),
        risk_posture=RiskPosture.SENSITIVE,
        extracted=True,
    )
    graph = lower(run.log, run.ir)

    controls = graph.edges_of(EdgeType.CONTROL)
    assert len(controls) == 1
    assert controls[0].trust is Trust.POLICY

    guard = graph.node(controls[0].src)
    guarded = graph.node(controls[0].dst)
    assert guard is not None and guard.op == "obligation"
    assert guarded is not None and guarded.op == "slack_post_message"


async def test_the_context_node_carries_the_risk_posture() -> None:
    """The gateway fails closed on this label, so it has to be on the graph."""
    store = RecordStore()
    run = await _two_turn_run(store)

    context = run.graph().of_kind(NodeKind.CONTEXT)[0]
    assert [label.name for label in context.labels] == ["risk:ambiguous"]


async def test_the_graph_version_tracks_the_log() -> None:
    """A decision binds to a version, so an append must move it."""
    store = RecordStore(extractor=NullIRExtractor())
    run = await _two_turn_run(store)
    first = run.graph()
    assert first.version == len(run.log)

    run.log.append(EventKind.TOOL_INTENT, tool_name="run_shell", tool_call_id="call_3", payload={})
    second = run.graph()

    assert second.version == len(run.log)
    assert second.version > first.version
    assert len(second.of_kind(NodeKind.TOOL)) == len(first.of_kind(NodeKind.TOOL)) + 1


def test_lowering_an_empty_log_still_produces_a_graph() -> None:
    """The pass is total: no input raises, however little was observed."""
    from fava.state.events import EventLog

    graph = lower(EventLog("run-0"), PermissionIR.unknown())

    assert graph.version == 0
    assert len(graph.of_kind(NodeKind.CONTEXT)) == 1
    assert validate(graph) == []


# -- validation -------------------------------------------------------------


async def test_a_well_formed_graph_has_no_violations() -> None:
    """Everything lowering emits must survive its own validator."""
    store = RecordStore()
    run = await _two_turn_run(store)

    assert run.validate() == []


def _graph(nodes: Sequence[Node], edges: Sequence[Edge], evidence: Sequence[str] = ("e0",)) -> PermissionGraph:
    return PermissionGraph(
        run_id="r", version=1, nodes=tuple(nodes), edges=tuple(edges), evidence_ids=frozenset(evidence)
    )


def test_a_dangling_edge_is_a_violation() -> None:
    """An edge to a node that does not exist is a malformed construct."""
    graph = _graph(
        [Node(id="a", kind=NodeKind.TOOL)],
        [Edge("a", "missing", EdgeType.DATA, "e0", Trust.OBSERVED)],
    )

    assert [v.code for v in validate(graph)] == ["dangling-edge"]


def test_a_duplicate_node_id_is_a_violation() -> None:
    """Ids address nodes; two nodes sharing one makes the graph unreadable."""
    graph = _graph([Node(id="a", kind=NodeKind.TOOL), Node(id="a", kind=NodeKind.SINK)], [])

    assert [v.code for v in validate(graph)] == ["duplicate-node"]


def test_an_edge_citing_unknown_evidence_is_a_violation() -> None:
    """Evidential validation: an edge may not invent its own justification."""
    graph = _graph(
        [Node(id="a", kind=NodeKind.TOOL), Node(id="b", kind=NodeKind.SINK)],
        [Edge("a", "b", EdgeType.DATA, "e-nope", Trust.POLICY)],
    )

    assert [v.code for v in validate(graph)] == ["unknown-evidence"]


def test_an_ungrounded_label_is_a_violation() -> None:
    """A label with no evidence is an LLM assertion wearing a fact's clothes."""
    graph = _graph([Node(id="a", kind=NodeKind.SOURCE, labels=(Label("secret"),))], [])

    assert [v.code for v in validate(graph)] == ["ungrounded-label"]


def test_a_child_predating_its_parent_is_a_violation() -> None:
    """Ancestry runs forward in time or the ordering means nothing."""
    graph = _graph(
        [Node(id="a", kind=NodeKind.SOURCE, time=1), Node(id="b", kind=NodeKind.TOOL, time=5)],
        [Edge("a", "b", EdgeType.PARENT, "e0", Trust.OBSERVED)],
    )

    assert [v.code for v in validate(graph)] == ["non-monotonic-parent"]


def test_validation_reports_rather_than_raises() -> None:
    """The caller is an observer on the relay's path and must not be broken."""
    graph = _graph(
        [Node(id="a", kind=NodeKind.TOOL)],
        [Edge("a", "a", EdgeType.DATA, "", Trust.OBSERVED)],
    )

    assert {v.code for v in validate(graph)} == {"self-edge", "unevidenced-edge"}


# -- the IR extractor -------------------------------------------------------


def _extractor_reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]},
    )


GOOD_IR = json.dumps(
    {
        "intent": "summarize the config and post it to ops",
        "risk_posture": "sensitive",
        "assets": [{"name": "config.yaml", "labels": ["secret"], "evidence": "Read config.yaml"}],
        "actions": [{"op": "read", "target": "config.yaml", "evidence": "Read config.yaml"}],
        "obligations": [{"requires": "summary", "before": "post", "evidence": "post a summary"}],
        "sinks": [{"target": "ops channel", "labels": ["internal"], "evidence": "the ops channel"}],
    }
)


def _stub(handler: Any) -> LlmIRExtractor:
    return LlmIRExtractor(
        ExtractorSettings(base_url="https://provider.invalid/v1", model="extractor-model", api_key="sk-x"),
        transport=httpx.MockTransport(handler),
    )


async def test_the_extractor_parses_a_well_formed_reply() -> None:
    """The five fields plus a posture, each element carrying its evidence."""
    extractor = _stub(lambda request: _extractor_reply(GOOD_IR))

    ir = await extractor.extract(TASK, model="gpt-4o-mini")

    assert ir.extracted is True
    assert ir.risk_posture is RiskPosture.SENSITIVE
    assert [asset.name for asset in ir.assets] == ["config.yaml"]
    assert ir.assets[0].labels == ("secret",)
    assert ir.assets[0].evidence == Evidence(ref="task", quote="Read config.yaml")
    assert [sink.target for sink in ir.sinks] == ["ops channel"]
    assert [obligation.requires for obligation in ir.obligations] == ["summary"]


async def test_the_extractor_never_calls_through_the_proxy_unmarked() -> None:
    """Its own call must be identifiable, or a misconfiguration loops forever."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _extractor_reply(GOOD_IR)

    await _stub(handler).extract(TASK, model="gpt-4o-mini")

    assert seen[0].headers["x-fava-internal"] == "extract"
    assert seen[0].url.path.endswith("/chat/completions")
    assert json.loads(seen[0].content)["stream"] is False


async def test_the_extractor_never_uses_the_monitored_model_or_credential() -> None:
    """The extraction model and key are configured separately from the traffic under watch.

    A monitored agent's own model name and bearer token must not leak into the
    call that reasons about that agent's task — otherwise the traffic FAVA is
    supposed to be watching could shape its own risk assessment.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _extractor_reply(GOOD_IR)

    await _stub(handler).extract(TASK, model="gpt-4o-mini", authorization="Bearer sk-agent-key")

    assert json.loads(seen[0].content)["model"] == "extractor-model"
    assert seen[0].headers["authorization"] == "Bearer sk-x"


async def test_prose_wrapped_json_is_still_parsed() -> None:
    """Models fence their JSON even when told not to."""
    extractor = _stub(lambda request: _extractor_reply(f"Sure!\n```json\n{GOOD_IR}\n```\n"))

    assert (await extractor.extract(TASK, model="m")).risk_posture is RiskPosture.SENSITIVE


@pytest.mark.parametrize("content", ["not json at all", "{", '{"risk_posture": }'])
async def test_an_unparseable_reply_falls_closed_to_ambiguous(content: str) -> None:
    """Nothing was established, so the task must not read as benign."""
    ir = await _stub(lambda request: _extractor_reply(content)).extract(TASK, model="m")

    assert ir.extracted is False
    assert ir.risk_posture is RiskPosture.AMBIGUOUS
    assert ir.error


async def test_a_provider_failure_falls_closed_without_raising() -> None:
    """An observer on the relay's path may not propagate an exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    ir = await _stub(handler).extract(TASK, model="m")

    assert ir.risk_posture is RiskPosture.AMBIGUOUS
    assert "500" in (ir.error or "")


async def test_a_transport_error_falls_closed_without_raising() -> None:
    """DNS, TLS and timeouts included."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to provider")

    ir = await _stub(handler).extract(TASK, model="m")

    assert ir.risk_posture is RiskPosture.AMBIGUOUS
    # The OpenAI SDK wraps transport failures in its own exception type
    # rather than passing the underlying httpx one through.
    assert "APIConnectionError" in (ir.error or "")


async def test_an_unknown_posture_is_read_as_ambiguous() -> None:
    """A model inventing a fifth posture must not widen the boundary."""
    ir = parse_ir(json.dumps({"intent": "x", "risk_posture": "probably_fine"}))

    assert ir.risk_posture is RiskPosture.AMBIGUOUS


def test_every_posture_but_benign_fails_closed() -> None:
    """The paper's conservative default, stated once so it cannot drift."""
    assert RiskPosture.BENIGN.fails_closed is False
    assert all(posture.fails_closed for posture in RiskPosture if posture is not RiskPosture.BENIGN)


async def test_extraction_is_skipped_without_its_own_credential() -> None:
    """No fallback to the agent's own key: with none configured, nothing is claimed."""
    extractor = LlmIRExtractor(ExtractorSettings(base_url="https://provider.invalid/v1", model="extractor-model"))

    ir = await extractor.extract(TASK, model="m", authorization="Bearer sk-agent-key")

    assert ir.risk_posture is RiskPosture.AMBIGUOUS
    assert "credential" in (ir.error or "")


async def test_extraction_is_skipped_without_its_own_model() -> None:
    """No fallback to the harness-requested model: with none configured, nothing is claimed."""
    extractor = LlmIRExtractor(ExtractorSettings(base_url="https://provider.invalid/v1", api_key="sk-x"))

    ir = await extractor.extract(TASK, model="m")

    assert ir.risk_posture is RiskPosture.AMBIGUOUS
    assert "model" in (ir.error or "")


async def test_the_null_extractor_leaves_a_run_ambiguous() -> None:
    """The default store extracts nothing and must say so."""
    store = RecordStore(extractor=NullIRExtractor())
    run = await _two_turn_run(store)

    assert run.ir.extracted is False
    assert run.ir.risk_posture is RiskPosture.AMBIGUOUS


async def test_a_landed_extraction_is_appended_to_the_log() -> None:
    """The IR is evidence like anything else, and moves the graph version."""
    store = RecordStore()
    run = await _two_turn_run(store)
    before = run.graph_version

    run.set_ir(parse_ir(GOOD_IR))

    assert run.graph_version > before
    assert run.log.events[-1].kind is EventKind.IR_EXTRACTED
    assert run.log.events[-1].payload["risk_posture"] == "sensitive"
    assert run.graph().of_kind(NodeKind.SINK)[0].args["target"] == "ops channel"


def test_extractor_settings_read_the_environment() -> None:
    """Deployment knobs — nothing here is inherited from the proxy's own settings."""
    settings = ExtractorSettings.from_env(
        {
            "FAVA_EXTRACTOR_BASE_URL": "https://api.openai.com/v1/",
            "FAVA_EXTRACTOR_MODEL": "gpt-4o",
            "FAVA_EXTRACTOR_API_KEY": "sk-extract",
            "FAVA_EXTRACTOR_TIMEOUT": "5",
        }
    )

    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o"
    assert settings.api_key == "sk-extract"
    assert settings.timeout == 5.0
    assert settings.enabled is True
    assert settings.configured is True

    assert ExtractorSettings.from_env({"FAVA_EXTRACTOR_ENABLED": "0"}).enabled is False


def test_extractor_settings_default_to_unconfigured() -> None:
    """With no `FAVA_EXTRACTOR_*` set, extraction has nothing of its own to use."""
    settings = ExtractorSettings.from_env({})

    assert settings.base_url == ""
    assert settings.model == ""
    assert settings.api_key == ""
    assert settings.configured is False


# -- end to end -------------------------------------------------------------


@dataclass
class StateFixture:
    """A live provider, a proxy recording into a store, and a real client."""

    upstream: ServiceHandle
    proxy: ServiceHandle
    fake: FakeUpstream
    store: RecordStore
    seen: RecordingHooks

    def client(self) -> AsyncOpenAI:
        """An unmodified OpenAI client pointed at the proxy."""
        return AsyncOpenAI(base_url=self.proxy.base_url, api_key="sk-test-key", max_retries=0)

    async def wait_for_responses(self, count: int, timeout: float = 5.0) -> None:
        """Block until the proxy has finished observing ``count`` responses."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while len(self.seen.responses) < count:
            if loop.time() > deadline:
                raise AssertionError(f"only {len(self.seen.responses)} of {count} responses observed")
            await asyncio.sleep(0.01)

    async def wait_for_extraction(self, timeout: float = 5.0) -> RunState:
        """Block until the single tracked run's extraction has landed."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            runs = self.store.runs
            if runs and runs[0].log.has_kind(EventKind.IR_EXTRACTED):
                return runs[0]
            if loop.time() > deadline:
                raise AssertionError("extraction did not land")
            await asyncio.sleep(0.01)

    @property
    def run(self) -> RunState:
        """The single run the fixture's traffic belongs to."""
        assert len(self.store.runs) == 1, f"expected one run, got {len(self.store.runs)}"
        return self.store.runs[0]


async def _state_proxy(fake: FakeUpstream, store: RecordStore) -> AsyncIterator[StateFixture]:
    """Start a fake provider and a proxy recording into ``store``."""
    upstream_port = free_port()
    proxy_port = free_port()
    while proxy_port == upstream_port:  # pragma: no cover - extremely unlikely
        proxy_port = free_port()

    upstream = await run_uvicorn(create_upstream_app(fake), upstream_port, name="upstream")
    seen = RecordingHooks()
    settings = ProxySettings(
        upstream_url=f"http://127.0.0.1:{upstream_port}/v1", host="127.0.0.1", port=proxy_port
    )
    # StateHooks first: `seen` is what the tests wait on, so the store must
    # already have recorded the exchange by the time it appears there.
    proxy = await run_uvicorn(
        create_proxy_app(settings, hooks=[StateHooks(store), seen]), proxy_port, name="proxy"
    )
    try:
        yield StateFixture(upstream=upstream, proxy=proxy, fake=fake, store=store, seen=seen)
    finally:
        await store.aclose()
        await proxy.stop()
        await upstream.stop()


@pytest.fixture
async def state_pair(fake_upstream: FakeUpstream) -> AsyncIterator[StateFixture]:
    """A proxy recording into a store, with extraction switched off."""
    async for fixture in _state_proxy(fake_upstream, RecordStore(extractor=NullIRExtractor())):
        yield fixture


async def _drive_two_turns(fixture: StateFixture, *, stream: bool) -> None:
    """A real client asking for a tool call, then reporting its result back."""
    fixture.fake.tool_calls = [ToolCallScript(name="read_file", arguments=json.dumps({"path": "config.yaml"}), id="call_1")]
    fixture.fake.content = ""

    async with fixture.client() as client:
        first = {"model": "gpt-4o-mini", "messages": _turn_one(), "stream": stream}
        if stream:
            async for _ in await client.chat.completions.create(**first):  # type: ignore[arg-type]
                pass
        else:
            await client.chat.completions.create(**first)  # type: ignore[arg-type]
        await fixture.wait_for_responses(1)

        fixture.fake.tool_calls = [
            ToolCallScript(name="slack_post_message", arguments=json.dumps({"channel": "#ops", "text": SECRET}), id="call_2")
        ]
        second = {"model": "gpt-4o-mini", "messages": _turn_two(), "stream": stream}
        if stream:
            async for _ in await client.chat.completions.create(**second):  # type: ignore[arg-type]
                pass
        else:
            await client.chat.completions.create(**second)  # type: ignore[arg-type]
        await fixture.wait_for_responses(2)


@pytest.mark.parametrize("stream", [False, True])
async def test_a_real_two_turn_exchange_builds_the_graph(state_pair: StateFixture, stream: bool) -> None:
    """The whole path, over TCP, streamed and not: traffic in, graph out."""
    await _drive_two_turns(state_pair, stream=stream)
    run = state_pair.run
    graph = run.graph()

    assert _traffic_kinds(run) == [
        EventKind.RUN_STARTED,
        EventKind.TOOL_INTENT,
        EventKind.TOOL_RESULT,
        EventKind.TOOL_INTENT,
    ]
    assert [node.op for node in graph.of_kind(NodeKind.TOOL)] == ["read_file", "slack_post_message"]

    flows = [edge for edge in graph.edges_of(EdgeType.DATA) if edge.trust is Trust.OBSERVED]
    assert len(flows) == 1, "the secret read on turn one reached the post on turn two"
    assert validate(graph) == []


async def test_observation_does_not_disturb_the_relay(state_pair: StateFixture) -> None:
    """Recording is read-only; the client must see an untouched response."""
    state_pair.fake.tool_calls = []
    state_pair.fake.content = "plain answer"

    async with state_pair.client() as client:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )

    await state_pair.wait_for_responses(1)
    assert completion.choices[0].message.content == "plain answer"
    assert state_pair.run.log.of_kind(EventKind.TOOL_INTENT) == ()


async def test_extraction_reaches_the_provider_and_lands_on_the_run(
    fake_upstream: FakeUpstream,
) -> None:
    """The extractor's own call goes upstream, not back through the proxy."""
    fake_upstream.content = GOOD_IR
    fake_upstream.tool_calls = []

    upstream_port = free_port()
    upstream = await run_uvicorn(create_upstream_app(fake_upstream), upstream_port, name="upstream-extract")
    try:
        store = RecordStore(
            extractor=LlmIRExtractor(
                ExtractorSettings(
                    base_url=f"http://127.0.0.1:{upstream_port}/v1",
                    model="extractor-model",
                    api_key="sk-extract",
                )
            )
        )
        async for fixture in _state_proxy(fake_upstream, store):
            async with fixture.client() as client:
                await client.chat.completions.create(
                    model="gpt-4o-mini", messages=_turn_one()
                )
            run = await fixture.wait_for_extraction()

            assert run.ir.extracted is True
            assert run.ir.risk_posture is RiskPosture.SENSITIVE
            assert [asset.name for asset in run.ir.assets] == ["config.yaml"]

            context = run.graph().of_kind(NodeKind.CONTEXT)[0]
            assert [label.name for label in context.labels] == ["risk:sensitive"]

            extraction_calls = [
                request for request in fake_upstream.requests if request["headers"].get("x-fava-internal")
            ]
            assert len(extraction_calls) == 1
            break
    finally:
        await upstream.stop()
