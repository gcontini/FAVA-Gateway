"""Per-run state: the event log, the IR, and the graph lowered from both.

This is where an HTTP exchange becomes part of an agent run. The relay hands
over a request and, later, the response it answered with; the store resolves
which run they belong to, records what was observed, and produces the run's
permission graph on demand.

The two halves of the evidence loop land in different exchanges, and that is the
whole reason this class exists:

* Response *N* carries the model's tool-call **intents** — what it wants run,
  before the harness has run it.
* Request *N+1* carries the **results**, as ``role: "tool"`` messages keyed by
  ``tool_call_id``, plus the same conversation prefix that identifies the run.

Nothing here can alter traffic. It is an observer, and every path through it
swallows its own failures.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Final

from llm_proxy.hooks import RequestContext, ResponseContext

from fava.state.events import EventKind, EventLog, derive_run_id, task_text
from fava.state.graph import PermissionGraph, Violation, lower, validate
from fava.state.ir import IRExtractor, NullIRExtractor, PermissionIR

logger = logging.getLogger(__name__)

# Retained per tool result. A result can be an entire file; the relay forwarded
# every byte regardless, and this bounds only what the store holds in memory.
MAX_RESULT_CONTENT: Final = 64 * 1024

# Runs tracked at once. A long-lived proxy would otherwise accumulate one log
# per conversation it has ever seen. Eviction is oldest-first: a run nobody has
# sent a request for in a while is over.
DEFAULT_MAX_RUNS: Final = 256


class RunState:
    """Everything FAVA knows about one agent run.

    Attributes:
        run_id: Identity recovered from the traffic.
        log: The run's append-only event log.
        ir: The extracted Permission IR, ambiguous until extraction lands.
        tool_catalog: Tool names the harness declared, most recently.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.log = EventLog(run_id)
        self.ir: PermissionIR = PermissionIR.unknown("extraction has not run")
        self.tool_catalog: tuple[str, ...] = ()
        self._graph: PermissionGraph | None = None

    @property
    def graph_version(self) -> int:
        """The version a graph lowered right now would carry."""
        return len(self.log)

    def graph(self) -> PermissionGraph:
        """The run's permission graph, lowered on demand and cached.

        Cached on the log length, so a run that is observed but never asked
        about never pays for lowering, and repeated reads between two events
        pay once. A race between two callers costs a redundant lowering and
        nothing else — the pass is a pure function of the log and the IR.
        """
        cached = self._graph
        if cached is not None and cached.version == len(self.log):
            return cached
        graph = lower(self.log, self.ir)
        self._graph = graph
        return graph

    def validate(self) -> list[Violation]:
        """Structural and evidential violations in the current graph."""
        return validate(self.graph())

    def set_ir(self, ir: PermissionIR) -> None:
        """Record an extraction result, appending it to the log.

        The IR is evidence like anything else: it goes in the log so an edge
        that exists because the extractor said so can cite the moment it said
        it. Appending also invalidates the cached graph by advancing the
        version.
        """
        self.ir = ir
        self.log.append(EventKind.IR_EXTRACTED, payload=ir.as_payload())


class RecordStore:
    """Observes exchanges and keeps the per-run state they belong to.

    Not synchronized. Appends are list appends and run creation goes through
    ``dict.setdefault``, so concurrent exchanges cannot lose an event; the worst
    a race costs is a graph lowered twice.

    Args:
        extractor: Turns a run's task text into a Permission IR. Defaults to
            :class:`~fava.state.ir.NullIRExtractor`, which leaves every run
            ambiguous — the honest posture when nothing has been extracted.
        max_runs: How many runs to retain before evicting the oldest.
    """

    def __init__(
        self,
        *,
        extractor: IRExtractor | None = None,
        max_runs: int = DEFAULT_MAX_RUNS,
    ) -> None:
        self.extractor = extractor or NullIRExtractor()
        self.max_runs = max_runs
        self._runs: dict[str, RunState] = {}
        # Strong references to in-flight extractions. Without these the tasks
        # are only weakly held by the loop and may be collected mid-flight.
        self._extractions: set[asyncio.Task[None]] = set()

    @property
    def runs(self) -> tuple[RunState, ...]:
        """Every tracked run, oldest first."""
        return tuple(self._runs.values())

    def get(self, run_id: str) -> RunState | None:
        """The state for a run id, or ``None`` if it is not tracked."""
        return self._runs.get(run_id)

    def get_for(self, context: RequestContext) -> RunState | None:
        """The run a request belongs to, resolving its identity first.

        Read-only: unlike :meth:`observe_request` this never creates a run, so
        an authorizer can ask "what do we know about this exchange?" without
        the question itself becoming evidence.
        """
        run_id = derive_run_id(context)
        return self._runs.get(run_id) if run_id else None

    async def observe_request(self, context: RequestContext) -> RunState | None:
        """Record what an inbound request says about its run.

        Appends ``run_started`` the first time a run is seen — carrying the task
        text and the declared tool catalog — and one ``tool_result`` per result
        the harness reported back for the intents observed last turn.

        Args:
            context: The observed request.

        Returns:
            The run's state, or ``None`` when the request is not part of a run
            (a ``GET /v1/models``, or a completion with no prefix to hash).
        """
        run_id = derive_run_id(context)
        if run_id is None or context.chat is None:
            return None

        run = self._runs.get(run_id)
        if run is None:
            run = self._create(run_id, context)

        run.tool_catalog = tuple(context.chat.tool_names)

        root = run.log.events[0].event_id if len(run.log) else None
        for result in context.chat.tool_results:
            # Every request repeats the whole history, so a result already
            # recorded arrives again on every later turn. Record it once.
            if run.log.has_result_for(result.tool_call_id):
                continue
            intent = run.log.intent_for(result.tool_call_id)
            content = result.content or ""
            run.log.append(
                EventKind.TOOL_RESULT,
                parent_event_id=intent.event_id if intent is not None else root,
                tool_name=result.name or (intent.tool_name if intent is not None else None),
                tool_call_id=result.tool_call_id,
                payload={
                    "content": content[:MAX_RESULT_CONTENT],
                    "content_size": len(content),
                    "truncated": len(content) > MAX_RESULT_CONTENT,
                },
            )
        return run

    async def observe_response(self, context: ResponseContext) -> RunState | None:
        """Record the tool calls the model asked for, before the harness runs them.

        This is the interception point that matters. A response the proxy could
        not read is recorded as ``response_opaque`` rather than as "no calls":
        ``may_dispatch_tools`` is true exactly when intents were observed *or*
        the response was too opaque to rule them out, and the second case must
        stay visible in the log.

        Args:
            context: The observed response.

        Returns:
            The run's state, or ``None`` when the exchange is not part of a run.
        """
        run_id = derive_run_id(context.request)
        if run_id is None:
            return None
        run = self._runs.get(run_id)
        if run is None:
            return None

        root = run.log.events[0].event_id if len(run.log) else None
        for intent in context.tool_intents:
            run.log.append(
                EventKind.TOOL_INTENT,
                parent_event_id=root,
                tool_name=intent.name,
                tool_call_id=intent.id,
                payload={
                    "arguments": dict(intent.arguments) if intent.arguments is not None else None,
                    "arguments_json": intent.arguments_json,
                    "arguments_valid": intent.arguments_valid,
                    "choice_index": intent.choice_index,
                    "index": intent.index,
                },
            )

        if context.intents_unknown:
            run.log.append(
                EventKind.RESPONSE_OPAQUE,
                parent_event_id=root,
                payload={
                    "status_code": context.status_code,
                    "reason": _opacity_reason(context),
                },
            )
        return run

    async def aclose(self) -> None:
        """Cancel any extraction still in flight and wait for it to unwind."""
        pending = tuple(self._extractions)
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(BaseException):
                await task
        self._extractions.clear()

    def _create(self, run_id: str, context: RequestContext) -> RunState:
        """Track a new run and start extracting its IR."""
        run = RunState(run_id)
        self._runs[run_id] = run
        self._evict()

        text = task_text(context.chat.messages) if context.chat else ""
        run.log.append(
            EventKind.RUN_STARTED,
            payload={
                "task_text": text,
                "model": context.model,
                "tools": list(context.chat.tool_names) if context.chat else [],
                "run_id_source": "header" if context.headers.get("x-fava-run-id") else "prefix",
            },
        )
        self._start_extraction(run, text, context)
        return run

    def _start_extraction(self, run: RunState, text: str, context: RequestContext) -> None:
        """Extract the run's IR in the background, once.

        Deliberately not awaited on the request path: extraction is a whole
        model call, and every turn of every run would pay for it. It is also
        deliberately *not* spawned into the relay's task group, which is scoped
        to a single exchange and would cancel it when the response finishes.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no loop outside the server
            return

        authorization = context.headers.get("authorization")
        task = loop.create_task(self._extract(run, text, context.model, authorization))
        self._extractions.add(task)
        task.add_done_callback(self._extractions.discard)

    async def _extract(self, run: RunState, text: str, model: str | None, authorization: str | None) -> None:
        """Run the extractor and record its result. Never raises."""
        try:
            ir = await self.extractor.extract(text, model=model, authorization=authorization)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - an observer must not be able to fail
            logger.warning("IR extraction raised for run %s: %s: %s", run.run_id[:8], type(exc).__name__, exc)
            ir = PermissionIR.unknown(f"{type(exc).__name__}: {exc}")
        run.set_ir(ir)

    def _evict(self) -> None:
        """Drop the oldest runs once the store is over its bound."""
        while len(self._runs) > self.max_runs:
            self._runs.pop(next(iter(self._runs)))


def _opacity_reason(context: ResponseContext) -> str:
    """Why a response could not be read, for the ``response_opaque`` payload."""
    if context.chat is None:
        return "no response body was parsed"
    if context.chat.validation_error:
        return context.chat.validation_error
    if context.stream_closed_early:
        return "stream closed before the response finished"
    return "stream ended without [DONE]"


__all__ = ["MAX_RESULT_CONTENT", "RecordStore", "RunState"]
