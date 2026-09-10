"""Read-only views over OpenAI Chat Completions traffic crossing the proxy.

The relay forwards bytes without rewriting them. Parsing happens *alongside* the
relay, purely so observers (hooks, logs, and later FAVA's authorizer) can see
what is in flight. Nothing here can mutate the wire format: a payload that fails
to parse is still forwarded byte-for-byte, with the failure recorded.

The shape of this boundary differs from an MCP proxy in one decisive way:

* An MCP proxy sees a tool **invocation** in the client's request.
* Here the client's request carries *context* — the message history, the
  declared tool catalog, and the **results** of the tool calls the model asked
  for last turn — while the tool-call **intent** appears in the model's
  *response*, before the harness has dispatched anything.

That inversion is why :class:`ResponseObserver`, not the request parser, is the
interception point FAVA's authorizer will attach to.

Streaming makes intents harder still: ``function.arguments`` arrives as a string
split across many SSE frames, keyed by ``index``, and frames themselves split
across TCP chunk boundaries. :class:`StreamAccumulator` reassembles both, fed
incrementally as bytes pass so that a long answer can never push the tool calls
out of the observation window.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_SUFFIX: Final = "/chat/completions"
ROLE_TOOL: Final = "tool"
ROLE_ASSISTANT: Final = "assistant"
FINISH_TOOL_CALLS: Final = "tool_calls"
SSE_DONE: Final = b"[DONE]"

# A stream with no frame boundary in this many bytes is malformed; the carry
# buffer is dropped rather than grown without bound. Frames are a few hundred
# bytes in practice, so this is orders of magnitude of headroom.
MAX_SSE_CARRY: Final = 1024 * 1024

# How much assistant text to retain for diagnostics. The relay forwards every
# byte regardless; this only bounds what an observer is handed.
DEFAULT_CONTENT_SAMPLE: Final = 4096

# A non-streaming completion body is small. Beyond this it is not buffered for
# parsing, and the response is reported as unparsed rather than held in memory.
DEFAULT_JSON_BUFFER_LIMIT: Final = 8 * 1024 * 1024


# -- tool-facing views ------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool the harness declared to the model, from the request's ``tools``.

    Attributes:
        name: Function name the model will use to call it.
        description: Natural-language description shown to the model.
        parameters: JSON Schema for the arguments, or ``None`` when absent.
        strict: The provider's structured-output flag, when set.

    Note:
        Unlike an MCP tool definition, this carries **no effect annotations** —
        there is no ``readOnlyHint`` or ``destructiveHint`` here. Whether a tool
        is dangerous cannot be read off the catalog and must come from policy.
    """

    name: str | None = None
    description: str | None = None
    parameters: Mapping[str, Any] | None = None
    strict: bool | None = None


@dataclass(frozen=True)
class ToolIntent:
    """A tool call the model asked for, before the harness dispatched it.

    Attributes:
        id: Provider-assigned call id, echoed back by the harness as
            ``tool_call_id`` when it reports the result.
        name: Function name the model wants invoked.
        arguments_json: The raw argument string exactly as the model emitted
            it. Kept verbatim because the model may emit invalid JSON, and an
            authorizer must be able to see what was actually sent.
        arguments: ``arguments_json`` decoded into a mapping, or ``None`` when
            it was not a valid JSON object.
        choice_index: Index of the choice this call belongs to (``n`` > 1).
        index: Position of this call within its choice, for parallel calls.
    """

    id: str | None = None
    name: str | None = None
    arguments_json: str = ""
    arguments: Mapping[str, Any] | None = None
    choice_index: int = 0
    index: int = 0

    @property
    def arguments_valid(self) -> bool:
        """Whether :attr:`arguments_json` decoded into a JSON object."""
        return self.arguments is not None

    @classmethod
    def build(
        cls,
        *,
        id: str | None,
        name: str | None,
        arguments_json: str,
        choice_index: int = 0,
        index: int = 0,
    ) -> ToolIntent:
        """Build an intent, decoding the argument string without raising."""
        return cls(
            id=id,
            name=name,
            arguments_json=arguments_json,
            arguments=_decode_arguments(arguments_json),
            choice_index=choice_index,
            index=index,
        )


@dataclass(frozen=True)
class ToolResult:
    """The outcome of an earlier tool call, as the harness reported it back.

    These appear in the *request*: a ``role: "tool"`` message carrying the
    ``tool_call_id`` of the intent it answers. They are the proxy's only
    evidence of what actually happened when a tool ran, since execution itself
    happens inside the harness and never crosses this wire.
    """

    tool_call_id: str | None = None
    name: str | None = None
    content: str | None = None

    @property
    def content_size(self) -> int:
        """Length of the reported result content in characters."""
        return len(self.content) if self.content else 0


# -- requests ---------------------------------------------------------------


@dataclass(frozen=True)
class ChatRequest:
    """A parsed view of one Chat Completions request body.

    Attributes:
        model: Requested model name.
        messages: The full conversation history, verbatim.
        tools: The declared tool catalog. Present in-band on every request, so
            the proxy never needs to fetch or cache a catalog separately.
        tool_choice: The request's ``tool_choice`` directive, when set.
        stream: Whether the client asked for a streamed response.
        n: Number of choices requested (default 1).
        raw: The decoded body, for diagnostics only. Never re-sent.
        validation_error: Set when the body could not be read as a Chat
            Completions request. The body is still relayed unchanged.
    """

    model: str | None = None
    messages: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    tools: Sequence[ToolSpec] = field(default_factory=tuple)
    tool_choice: Any = None
    stream: bool = False
    n: int = 1
    raw: Any = None
    validation_error: str | None = None

    @property
    def message_count(self) -> int:
        """How many messages the conversation history carries."""
        return len(self.messages)

    @property
    def tool_results(self) -> tuple[ToolResult, ...]:
        """Results of previously dispatched tool calls, in order.

        This is the evidence loop: request *N+1* reports what happened when the
        harness ran the intents from response *N*.
        """
        results: list[ToolResult] = []
        for message in self.messages:
            if not isinstance(message, Mapping) or message.get("role") != ROLE_TOOL:
                continue
            content = message.get("content")
            results.append(
                ToolResult(
                    tool_call_id=_as_str(message.get("tool_call_id")),
                    name=_as_str(message.get("name")),
                    content=content if isinstance(content, str) else _dump(content),
                )
            )
        return tuple(results)

    @property
    def prior_tool_calls(self) -> tuple[ToolIntent, ...]:
        """Tool calls already present in the assistant turns of the history.

        Lets an observer correlate a result back to the intent that produced it
        without holding cross-request state.
        """
        intents: list[ToolIntent] = []
        for message in self.messages:
            if not isinstance(message, Mapping) or message.get("role") != ROLE_ASSISTANT:
                continue
            intents.extend(_intents_from_tool_calls(message.get("tool_calls"), choice_index=0))
        return tuple(intents)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Names of every declared tool, skipping malformed entries."""
        return tuple(spec.name for spec in self.tools if spec.name)


def parse_chat_request(body: bytes | None) -> ChatRequest | None:
    """Parse a request body into a :class:`ChatRequest` view.

    Args:
        body: The raw request body. ``None`` or empty yields ``None``, since a
            ``GET`` carries no completion request at all.

    Returns:
        A parsed view, or one with :attr:`ChatRequest.validation_error` set when
        the body was not a JSON object. Never raises: the relay must keep
        forwarding whatever the client sent.
    """
    if not body:
        return None

    try:
        decoded = json.loads(body)
    except (ValueError, RecursionError) as exc:
        # Not just JSONDecodeError: oversized int literals raise bare ValueError
        # and deep nesting raises RecursionError.
        logger.debug("Request body is not JSON (%s); relaying unchanged", exc)
        return ChatRequest(validation_error=f"not JSON: {exc}")

    if not isinstance(decoded, Mapping):
        return ChatRequest(raw=decoded, validation_error="request body must be a JSON object")

    raw_messages = decoded.get("messages")
    messages = tuple(m for m in raw_messages if isinstance(m, Mapping)) if isinstance(raw_messages, list) else ()

    raw_tools = decoded.get("tools")
    tools = tuple(_tool_spec(entry) for entry in raw_tools if isinstance(entry, Mapping)) if isinstance(raw_tools, list) else ()

    return ChatRequest(
        model=_as_str(decoded.get("model")),
        messages=messages,
        tools=tools,
        tool_choice=decoded.get("tool_choice"),
        stream=bool(decoded.get("stream")),
        n=decoded.get("n") if isinstance(decoded.get("n"), int) else 1,
        raw=decoded,
    )


def _tool_spec(entry: Mapping[str, Any]) -> ToolSpec:
    """Build a :class:`ToolSpec` from one entry of the request's ``tools``."""
    function = entry.get("function")
    if not isinstance(function, Mapping):
        # Some providers accept a flattened form without the `function` wrapper.
        function = entry
    parameters = function.get("parameters")
    strict = function.get("strict")
    return ToolSpec(
        name=_as_str(function.get("name")),
        description=_as_str(function.get("description")),
        parameters=parameters if isinstance(parameters, Mapping) else None,
        strict=strict if isinstance(strict, bool) else None,
    )


# -- responses --------------------------------------------------------------


@dataclass(frozen=True)
class ChatResponse:
    """A parsed view of one Chat Completions response, streamed or not.

    Attributes:
        id: Provider-assigned completion id.
        model: Model that actually served the request.
        tool_intents: Every tool call the model asked for, with streamed
            fragments already reassembled.
        finish_reasons: Finish reason per choice index, in index order.
        usage: Token accounting, when the provider reported it.
        content_sample: A bounded prefix of the assistant's text.
        is_stream: Whether the response arrived as ``text/event-stream``.
        complete: For a stream, whether the terminating ``[DONE]`` was seen.
            False means the stream was cut short, so :attr:`tool_intents` may
            be partial — treat it as unknown, not as "no tool calls".
        frame_count: Number of SSE frames observed.
        validation_error: Set when the body could not be read as a completion.
    """

    id: str | None = None
    model: str | None = None
    tool_intents: Sequence[ToolIntent] = field(default_factory=tuple)
    finish_reasons: Sequence[str | None] = field(default_factory=tuple)
    usage: Mapping[str, Any] | None = None
    content_sample: str = ""
    is_stream: bool = False
    complete: bool = True
    frame_count: int = 0
    validation_error: str | None = None

    @property
    def has_tool_intents(self) -> bool:
        """Whether the model asked for at least one tool call."""
        return bool(self.tool_intents)

    @property
    def requested_tool_names(self) -> tuple[str, ...]:
        """Names of every requested tool call, in order."""
        return tuple(intent.name for intent in self.tool_intents if intent.name)


def parse_chat_response(body: bytes) -> ChatResponse:
    """Parse a complete non-streaming Chat Completions response body."""
    try:
        decoded = json.loads(body)
    except (ValueError, RecursionError) as exc:
        return ChatResponse(validation_error=f"not JSON: {exc}")

    if not isinstance(decoded, Mapping):
        return ChatResponse(validation_error="response body must be a JSON object")

    intents: list[ToolIntent] = []
    finish_reasons: list[str | None] = []
    content_parts: list[str] = []

    choices = decoded.get("choices")
    for position, choice in enumerate(choices if isinstance(choices, list) else ()):
        if not isinstance(choice, Mapping):
            continue
        choice_index = choice.get("index")
        choice_index = choice_index if isinstance(choice_index, int) else position
        finish_reasons.append(_as_str(choice.get("finish_reason")))
        message = choice.get("message")
        if not isinstance(message, Mapping):
            continue
        intents.extend(_intents_from_tool_calls(message.get("tool_calls"), choice_index=choice_index))
        if isinstance(message.get("content"), str):
            content_parts.append(message["content"])

    usage = decoded.get("usage")
    return ChatResponse(
        id=_as_str(decoded.get("id")),
        model=_as_str(decoded.get("model")),
        tool_intents=tuple(intents),
        finish_reasons=tuple(finish_reasons),
        usage=usage if isinstance(usage, Mapping) else None,
        content_sample="".join(content_parts)[:DEFAULT_CONTENT_SAMPLE],
        is_stream=False,
    )


def parse_sse_frames(buffer: bytes) -> tuple[tuple[bytes, ...], bytes]:
    """Split a byte buffer into complete SSE frames plus an unfinished remainder.

    Args:
        buffer: Accumulated stream bytes, possibly ending mid-frame.

    Returns:
        A ``(frames, remainder)`` pair. Each frame is the joined ``data:``
        payload of one event, with comment lines (``:`` prefix, e.g. keep-alive
        pings) dropped. ``remainder`` is the trailing partial frame to carry
        into the next call.
    """
    normalized = buffer.replace(b"\r\n", b"\n")
    blocks = normalized.split(b"\n\n")
    remainder = blocks.pop()
    return tuple(payload for block in blocks for payload in _frame_payload(block)), remainder


def _frame_payload(block: bytes) -> Iterator[bytes]:
    """Yield the joined ``data:`` payload of one SSE event block, if any."""
    data_lines: list[bytes] = []
    for line in block.split(b"\n"):
        if line.startswith(b":"):
            continue
        name, _, value = line.partition(b":")
        if name.strip() != b"data":
            continue
        if value.startswith(b" "):
            value = value[1:]
        data_lines.append(value)
    if data_lines:
        yield b"\n".join(data_lines)


@dataclass
class _ToolCallSlot:
    """Fragments of one streamed tool call, accumulating across SSE frames."""

    id: str | None = None
    name: str | None = None
    arguments: str = ""


class StreamAccumulator:
    """Reassembles a streamed Chat Completions response as bytes pass by.

    Fed each chunk the relay forwards, so observation costs one pass over the
    stream and never holds the whole response. Only accumulator state and a
    bounded text sample are retained; the relay forwards every byte regardless.

    The reassembly that matters is ``function.arguments``: the provider emits it
    as a string split across frames, identified only by ``(choice index, tool
    call index)``. A parallel tool call interleaves its fragments with its
    siblings', so the fragments must be keyed, not appended in arrival order.
    """

    def __init__(self, *, content_sample_limit: int = DEFAULT_CONTENT_SAMPLE) -> None:
        self._carry = b""
        self._slots: dict[tuple[int, int], _ToolCallSlot] = {}
        self._finish_reasons: dict[int, str | None] = {}
        self._content: list[str] = []
        self._content_size = 0
        self._content_limit = content_sample_limit
        self._id: str | None = None
        self._model: str | None = None
        self._usage: Mapping[str, Any] | None = None
        self._frames = 0
        self._done = False
        self._errors: list[str] = []

    @property
    def saw_done(self) -> bool:
        """Whether the terminating ``data: [DONE]`` frame arrived."""
        return self._done

    @property
    def frame_count(self) -> int:
        """How many SSE data frames have been observed."""
        return self._frames

    @property
    def parse_errors(self) -> tuple[str, ...]:
        """Frames that could not be decoded, for diagnostics."""
        return tuple(self._errors)

    def feed(self, chunk: bytes) -> None:
        """Observe one chunk of the streamed response.

        Never raises: a malformed frame is recorded and skipped, because the
        bytes have already been forwarded and the relay cannot take them back.
        """
        if not chunk:
            return
        self._carry += chunk
        if len(self._carry) > MAX_SSE_CARRY:
            # No frame boundary in a megabyte: this is not an SSE stream we can
            # follow. Drop the buffer rather than grow it without bound.
            self._errors.append(f"no SSE frame boundary within {MAX_SSE_CARRY} bytes; observation dropped")
            self._carry = b""
            return
        frames, self._carry = parse_sse_frames(self._carry)
        for payload in frames:
            self._handle(payload)

    def close(self) -> None:
        """Flush any trailing frame left unterminated by the stream's end."""
        if not self._carry.strip():
            self._carry = b""
            return
        for payload in _frame_payload(self._carry.replace(b"\r\n", b"\n")):
            self._handle(payload)
        self._carry = b""

    def _handle(self, payload: bytes) -> None:
        """Process one frame's ``data:`` payload."""
        if payload.strip() == SSE_DONE:
            self._done = True
            return
        self._frames += 1
        try:
            decoded = json.loads(payload)
        except (ValueError, RecursionError) as exc:
            self._errors.append(f"frame is not JSON: {exc}")
            return
        if not isinstance(decoded, Mapping):
            self._errors.append("frame is not a JSON object")
            return

        self._id = self._id or _as_str(decoded.get("id"))
        self._model = self._model or _as_str(decoded.get("model"))
        usage = decoded.get("usage")
        if isinstance(usage, Mapping):
            self._usage = usage

        choices = decoded.get("choices")
        for position, choice in enumerate(choices if isinstance(choices, list) else ()):
            if isinstance(choice, Mapping):
                self._handle_choice(choice, position)

    def _handle_choice(self, choice: Mapping[str, Any], position: int) -> None:
        """Fold one choice's delta into the accumulated state."""
        raw_index = choice.get("index")
        choice_index = raw_index if isinstance(raw_index, int) else position

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None or choice_index not in self._finish_reasons:
            self._finish_reasons[choice_index] = _as_str(finish_reason)

        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            return

        content = delta.get("content")
        if isinstance(content, str) and self._content_size < self._content_limit:
            keep = content[: self._content_limit - self._content_size]
            self._content.append(keep)
            self._content_size += len(keep)

        tool_calls = delta.get("tool_calls")
        for tool_position, call in enumerate(tool_calls if isinstance(tool_calls, list) else ()):
            if isinstance(call, Mapping):
                self._handle_tool_call_delta(call, choice_index, tool_position)

    def _handle_tool_call_delta(self, call: Mapping[str, Any], choice_index: int, position: int) -> None:
        """Merge one ``delta.tool_calls[]`` fragment into its slot."""
        raw_index = call.get("index")
        index = raw_index if isinstance(raw_index, int) else position
        slot = self._slots.setdefault((choice_index, index), _ToolCallSlot())

        call_id = _as_str(call.get("id"))
        if call_id:
            slot.id = call_id

        function = call.get("function")
        if not isinstance(function, Mapping):
            return
        name = _as_str(function.get("name"))
        if name:
            # The provider sends the name once, complete, in the opening
            # fragment; only `arguments` is split. Overwriting rather than
            # concatenating keeps a provider that repeats it from doubling it.
            slot.name = name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            slot.arguments += arguments

    def result(self, *, content_type: str | None = None) -> ChatResponse:
        """Build the :class:`ChatResponse` observers are handed."""
        intents = tuple(
            ToolIntent.build(
                id=slot.id,
                name=slot.name,
                arguments_json=slot.arguments,
                choice_index=choice_index,
                index=index,
            )
            for (choice_index, index), slot in sorted(self._slots.items())
        )
        finish_reasons = tuple(self._finish_reasons[key] for key in sorted(self._finish_reasons))
        error = "; ".join(self._errors) if self._errors else None
        return ChatResponse(
            id=self._id,
            model=self._model,
            tool_intents=intents,
            finish_reasons=finish_reasons,
            usage=self._usage,
            content_sample="".join(self._content),
            is_stream=True,
            complete=self._done,
            frame_count=self._frames,
            validation_error=error,
        )


class ResponseObserver:
    """Observes a response body of either shape, streamed or whole.

    The relay does not want to know which shape it is relaying, so this picks
    the strategy off the content type: SSE bodies go to a
    :class:`StreamAccumulator` fed chunk by chunk, everything else is buffered
    up to a bound and parsed once at the end.
    """

    def __init__(
        self,
        *,
        content_type: str | None,
        content_sample_limit: int = DEFAULT_CONTENT_SAMPLE,
        json_buffer_limit: int = DEFAULT_JSON_BUFFER_LIMIT,
    ) -> None:
        self.content_type = content_type
        self.is_stream = bool(content_type and "text/event-stream" in content_type.lower())
        self._json_limit = json_buffer_limit
        self._buffer = bytearray()
        self._overflowed = False
        self._accumulator = StreamAccumulator(content_sample_limit=content_sample_limit) if self.is_stream else None
        self._observed = 0

    @property
    def observed_size(self) -> int:
        """Total response bytes seen, whether or not they were retained."""
        return self._observed

    def feed(self, chunk: bytes) -> None:
        """Observe one relayed chunk. Never raises."""
        if not chunk:
            return
        self._observed += len(chunk)
        if self._accumulator is not None:
            self._accumulator.feed(chunk)
            return
        if self._overflowed:
            return
        if len(self._buffer) + len(chunk) > self._json_limit:
            self._overflowed = True
            self._buffer = bytearray()
            return
        self._buffer += chunk

    def result(self) -> ChatResponse | None:
        """Finish observation and build the parsed view, or ``None`` if empty."""
        if self._accumulator is not None:
            self._accumulator.close()
            return self._accumulator.result(content_type=self.content_type)
        if self._overflowed:
            return ChatResponse(validation_error=f"response body exceeded {self._json_limit} bytes; not parsed")
        if not self._buffer:
            return None
        return parse_chat_response(bytes(self._buffer))


# -- helpers ----------------------------------------------------------------


def _intents_from_tool_calls(raw: Any, *, choice_index: int) -> tuple[ToolIntent, ...]:
    """Build intents from a complete (non-streamed) ``tool_calls`` array."""
    if not isinstance(raw, list):
        return ()
    intents: list[ToolIntent] = []
    for position, call in enumerate(raw):
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        function = function if isinstance(function, Mapping) else {}
        arguments = function.get("arguments")
        intents.append(
            ToolIntent.build(
                id=_as_str(call.get("id")),
                name=_as_str(function.get("name")),
                arguments_json=arguments if isinstance(arguments, str) else _dump(arguments),
                choice_index=choice_index,
                index=position,
            )
        )
    return tuple(intents)


def _decode_arguments(raw: str) -> Mapping[str, Any] | None:
    """Decode a tool-argument string, returning ``None`` if it is not an object.

    A model can and does emit invalid JSON here, and a truncated stream leaves a
    half-written object. Both must surface as "unparsed" rather than raise.
    """
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _as_str(value: Any) -> str | None:
    """Return ``value`` when it is a string, else ``None``."""
    return value if isinstance(value, str) else None


def _dump(value: Any) -> str:
    """Render a non-string payload as JSON text, falling back to ``repr``."""
    if value is None:
        return ""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(value)
