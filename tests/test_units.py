"""Unit tests: parsing, stream reassembly, header rules, config and hooks.

No servers and no sockets — everything here runs in-process.
"""

from __future__ import annotations

import json

import pytest

from llm_proxy.chat import (
    ChatRequest,
    ResponseObserver,
    StreamAccumulator,
    ToolIntent,
    parse_chat_request,
    parse_chat_response,
    parse_sse_frames,
)
from llm_proxy.config import (
    DEFAULT_MOUNT_PREFIX,
    ProxySettings,
    normalize_mount_prefix,
)
from llm_proxy.headers import (
    forward_request_headers,
    forward_response_headers,
    is_api_wire_header,
    redact,
)
from llm_proxy.hooks import (
    CompositeHooks,
    LoggingHooks,
    NullHooks,
    RequestContext,
    ResponseContext,
    normalize_hooks,
)

# -- request parsing --------------------------------------------------------


def _request_body(**overrides: object) -> bytes:
    payload: dict[str, object] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "what is the weather?"}],
    }
    payload.update(overrides)
    return json.dumps(payload).encode("utf-8")


def test_parse_request_extracts_model_and_history() -> None:
    """The basic shape of a completion request is surfaced to observers."""
    request = parse_chat_request(_request_body(stream=True, n=2))

    assert request is not None
    assert request.model == "gpt-4o-mini"
    assert request.message_count == 1
    assert request.stream is True
    assert request.n == 2
    assert request.validation_error is None


def test_parse_request_reads_declared_tool_catalog() -> None:
    """The tool catalog arrives in-band, so no separate catalog fetch is needed."""
    body = _request_body(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Run a shell command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                    "strict": True,
                },
            },
            {"type": "function", "function": {"name": "read_file"}},
        ],
        tool_choice="auto",
    )
    request = parse_chat_request(body)

    assert request is not None
    assert request.tool_names == ("run_shell", "read_file")
    assert request.tools[0].description == "Run a shell command"
    assert request.tools[0].parameters is not None
    assert request.tools[0].strict is True
    assert request.tool_choice == "auto"


def test_parse_request_surfaces_prior_tool_results() -> None:
    """`role: "tool"` messages are the only evidence of what a tool actually did."""
    body = _request_body(
        messages=[
            {"role": "user", "content": "delete the temp files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "run_shell", "arguments": '{"cmd":"rm -rf /tmp/x"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "name": "run_shell", "content": "removed 3 files"},
        ]
    )
    request = parse_chat_request(body)

    assert request is not None
    (result,) = request.tool_results
    assert result.tool_call_id == "call_a"
    assert result.name == "run_shell"
    assert result.content == "removed 3 files"
    assert result.content_size == len("removed 3 files")

    # The intent that produced it is recoverable from the same request.
    (prior,) = request.prior_tool_calls
    assert prior.id == "call_a"
    assert prior.name == "run_shell"
    assert prior.arguments == {"cmd": "rm -rf /tmp/x"}


def test_parse_request_tolerates_non_string_tool_result_content() -> None:
    """A structured tool result is rendered rather than dropped."""
    body = _request_body(
        messages=[{"role": "tool", "tool_call_id": "call_b", "content": [{"type": "text", "text": "ok"}]}]
    )
    request = parse_chat_request(body)

    assert request is not None
    assert request.tool_results[0].content is not None
    assert "ok" in request.tool_results[0].content


def test_parse_garbage_request_records_error_without_raising() -> None:
    """An unparseable body still relays; the failure is recorded, not raised."""
    request = parse_chat_request(b"{not json at all")

    assert request is not None
    assert request.validation_error is not None
    assert "not JSON" in request.validation_error
    assert request.model is None


def test_parse_non_object_request_body() -> None:
    """A JSON array is valid JSON but not a completion request."""
    request = parse_chat_request(b'["nope"]')

    assert request is not None
    assert request.validation_error == "request body must be a JSON object"


def test_parse_empty_body_observes_nothing() -> None:
    """A bodyless request (a GET) yields no chat view at all."""
    assert parse_chat_request(None) is None
    assert parse_chat_request(b"") is None


def test_request_defaults_are_inert() -> None:
    """A default-constructed view answers every accessor without a body."""
    request = ChatRequest()
    assert request.message_count == 0
    assert request.tool_results == ()
    assert request.prior_tool_calls == ()
    assert request.tool_names == ()


# -- non-streaming response parsing ----------------------------------------


def test_parse_response_extracts_tool_intents() -> None:
    """Tool-call intents come off the response, which is the interception point."""
    body = json.dumps(
        {
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "run_shell", "arguments": '{"cmd":"ls"}'}},
                            {"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"/etc/passwd"}'}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 42},
        }
    ).encode("utf-8")

    response = parse_chat_response(body)

    assert response.has_tool_intents
    assert response.requested_tool_names == ("run_shell", "read_file")
    assert response.tool_intents[1].arguments == {"path": "/etc/passwd"}
    assert response.tool_intents[1].index == 1
    assert response.finish_reasons == ("tool_calls",)
    assert response.usage == {"total_tokens": 42}
    assert response.is_stream is False


def test_parse_response_without_tool_calls() -> None:
    """A plain answer reports no intents and keeps a content sample."""
    body = json.dumps(
        {"id": "c", "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
    ).encode("utf-8")

    response = parse_chat_response(body)

    assert response.has_tool_intents is False
    assert response.content_sample == "hi"
    assert response.finish_reasons == ("stop",)


def test_invalid_tool_arguments_are_preserved_verbatim() -> None:
    """A model can emit invalid JSON; the raw string must survive for the authorizer."""
    intent = ToolIntent.build(id="call_1", name="run_shell", arguments_json='{"cmd": "rm -rf ')

    assert intent.arguments is None
    assert intent.arguments_valid is False
    assert intent.arguments_json == '{"cmd": "rm -rf '


def test_parse_garbage_response_records_error() -> None:
    """An unparseable response is reported, never raised."""
    response = parse_chat_response(b"<html>gateway error</html>")

    assert response.validation_error is not None
    assert response.has_tool_intents is False


# -- SSE framing ------------------------------------------------------------


def test_parse_sse_frames_splits_and_carries_remainder() -> None:
    """Complete frames come out; a partial trailing frame is carried forward."""
    frames, remainder = parse_sse_frames(b'data: {"a":1}\n\ndata: {"b":2}\n\ndata: {"c"')

    assert frames == (b'{"a":1}', b'{"b":2}')
    assert remainder == b'data: {"c"'


def test_parse_sse_frames_ignores_comments_and_crlf() -> None:
    """Keep-alive comments are skipped and CRLF framing is handled."""
    frames, remainder = parse_sse_frames(b': keep-alive\r\n\r\ndata: {"a":1}\r\n\r\n')

    assert frames == (b'{"a":1}',)
    assert remainder == b""


def test_parse_sse_frames_joins_multiline_data() -> None:
    """Multi-line `data:` fields join with newlines, per the SSE spec."""
    frames, _ = parse_sse_frames(b"data: line one\ndata: line two\n\n")

    assert frames == (b"line one\nline two",)


# -- stream accumulation ----------------------------------------------------


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> bytes:
    payload = {"id": "chatcmpl-s", "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _tool_stream() -> bytes:
    """A stream whose single tool call has its arguments split across frames."""
    return b"".join(
        [
            _chunk({"role": "assistant", "content": ""}),
            _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "run_shell", "arguments": ""}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"cm'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'd":"ls '}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '-la"}'}}]}),
            _chunk({}, finish_reason="tool_calls"),
            b"data: [DONE]\n\n",
        ]
    )


def test_stream_accumulator_reassembles_fragmented_arguments() -> None:
    """Argument fragments spread across frames rebuild into one valid object."""
    accumulator = StreamAccumulator()
    accumulator.feed(_tool_stream())
    accumulator.close()
    result = accumulator.result()

    (intent,) = result.tool_intents
    assert intent.id == "call_1"
    assert intent.name == "run_shell"
    assert intent.arguments == {"cmd": "ls -la"}
    assert result.complete is True
    assert result.finish_reasons == ("tool_calls",)
    assert result.is_stream is True


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
def test_stream_accumulator_survives_chunk_boundaries(chunk_size: int) -> None:
    """Frames split across TCP chunks must still reassemble.

    This is the failure mode a naive `split(b"\\n\\n")` per chunk would have:
    the boundary falls mid-frame and the fragment is lost.
    """
    body = _tool_stream()
    accumulator = StreamAccumulator()
    for start in range(0, len(body), chunk_size):
        accumulator.feed(body[start : start + chunk_size])
    accumulator.close()

    (intent,) = accumulator.result().tool_intents
    assert intent.arguments == {"cmd": "ls -la"}


def test_stream_accumulator_keys_parallel_calls_by_index() -> None:
    """Interleaved parallel calls must not have their fragments concatenated."""
    body = b"".join(
        [
            _chunk({"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "read_file", "arguments": ""}}]}),
            _chunk({"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "run_shell", "arguments": ""}}]}),
            # Fragments arrive round-robin, as a real provider emits them.
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path'}}]}),
            _chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"cmd'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '":"/etc/hosts"}'}}]}),
            _chunk({"tool_calls": [{"index": 1, "function": {"arguments": '":"whoami"}'}}]}),
            _chunk({}, finish_reason="tool_calls"),
            b"data: [DONE]\n\n",
        ]
    )
    accumulator = StreamAccumulator()
    accumulator.feed(body)
    accumulator.close()
    intents = accumulator.result().tool_intents

    assert [intent.name for intent in intents] == ["read_file", "run_shell"]
    assert intents[0].arguments == {"path": "/etc/hosts"}
    assert intents[1].arguments == {"cmd": "whoami"}


def test_stream_accumulator_separates_choices() -> None:
    """With n>1 the same tool index in different choices stays distinct."""
    def chunk_for(choice_index: int, fragment: str, name: str | None = None) -> bytes:
        call: dict[str, object] = {"index": 0, "function": {"arguments": fragment}}
        if name:
            call = {"index": 0, "id": f"call_{choice_index}", "function": {"name": name, "arguments": fragment}}
        payload = {"choices": [{"index": choice_index, "delta": {"tool_calls": [call]}, "finish_reason": None}]}
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

    accumulator = StreamAccumulator()
    accumulator.feed(chunk_for(0, "", "alpha") + chunk_for(1, "", "beta"))
    accumulator.feed(chunk_for(0, '{"x":1}') + chunk_for(1, '{"y":2}'))
    accumulator.close()
    intents = accumulator.result().tool_intents

    assert [(i.choice_index, i.name, i.arguments) for i in intents] == [
        (0, "alpha", {"x": 1}),
        (1, "beta", {"y": 2}),
    ]


def test_stream_accumulator_reports_incomplete_stream() -> None:
    """A stream cut before `[DONE]` is incomplete, so intents may be partial."""
    accumulator = StreamAccumulator()
    accumulator.feed(_chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "run_shell", "arguments": '{"cmd":"l'}}]}))
    accumulator.close()
    result = accumulator.result()

    assert result.complete is False
    assert result.tool_intents[0].arguments is None  # truncated JSON does not decode
    assert result.tool_intents[0].arguments_json == '{"cmd":"l'


def test_stream_accumulator_flushes_unterminated_trailing_frame() -> None:
    """A final frame with no blank line still counts on close()."""
    accumulator = StreamAccumulator()
    accumulator.feed(b'data: {"choices":[{"index":0,"delta":{"content":"tail"}}]}')
    accumulator.close()

    assert accumulator.result().content_sample == "tail"


def test_stream_accumulator_bounds_content_sample() -> None:
    """Assistant text is sampled, not accumulated without bound."""
    accumulator = StreamAccumulator(content_sample_limit=10)
    for _ in range(50):
        accumulator.feed(_chunk({"content": "0123456789"}))
    accumulator.close()

    assert len(accumulator.result().content_sample) == 10


def test_stream_accumulator_drops_a_runaway_buffer() -> None:
    """A stream with no frame boundary cannot grow the carry buffer forever."""
    accumulator = StreamAccumulator()
    accumulator.feed(b"x" * (2 * 1024 * 1024))
    accumulator.close()

    assert accumulator.result().validation_error is not None


def test_stream_accumulator_records_bad_frames_without_raising() -> None:
    """A malformed frame is noted; the bytes were already forwarded."""
    accumulator = StreamAccumulator()
    accumulator.feed(b"data: {broken\n\n" + _chunk({"content": "ok"}))
    accumulator.close()
    result = accumulator.result()

    assert result.validation_error is not None
    assert result.content_sample == "ok"


# -- response observer ------------------------------------------------------


def test_observer_dispatches_on_content_type() -> None:
    """SSE goes to the accumulator; JSON is buffered and parsed once."""
    stream_observer = ResponseObserver(content_type="text/event-stream; charset=utf-8")
    assert stream_observer.is_stream is True
    stream_observer.feed(_tool_stream())
    stream_result = stream_observer.result()
    assert stream_result is not None
    assert stream_result.requested_tool_names == ("run_shell",)

    json_observer = ResponseObserver(content_type="application/json")
    assert json_observer.is_stream is False
    json_observer.feed(b'{"id":"x","model":"m","choices":[{"index":0,"message":{"content":"hi"},"finish_reason":"stop"}]}')
    json_result = json_observer.result()
    assert json_result is not None
    assert json_result.content_sample == "hi"


def test_observer_reports_empty_body_as_nothing() -> None:
    """No bytes means no parsed view."""
    assert ResponseObserver(content_type="application/json").result() is None


def test_observer_refuses_to_buffer_an_oversized_body() -> None:
    """A huge non-streaming body is reported unparsed rather than held."""
    observer = ResponseObserver(content_type="application/json", json_buffer_limit=64)
    observer.feed(b"x" * 128)
    result = observer.result()

    assert result is not None
    assert result.validation_error is not None
    assert observer.observed_size == 128


# -- headers ----------------------------------------------------------------


def test_hop_by_hop_headers_are_not_relayed() -> None:
    """RFC 9110 §7.6.1 headers belong to one connection only."""
    relayed = forward_request_headers(
        [("content-type", "application/json"), ("connection", "keep-alive"), ("transfer-encoding", "chunked")],
        host="api.example.com",
        trust_client_headers=True,
    )

    assert "connection" not in relayed
    assert "transfer-encoding" not in relayed
    assert relayed["host"] == "api.example.com"


def test_connection_nominated_headers_are_dropped() -> None:
    """A `Connection:` value names further headers to drop."""
    relayed = forward_request_headers(
        [("connection", "x-custom, close"), ("x-custom", "secret"), ("content-type", "application/json")],
        trust_client_headers=True,
    )

    assert "x-custom" not in relayed
    assert relayed["content-type"] == "application/json"


def test_untrusted_mode_relays_only_api_wire_headers() -> None:
    """Cookies and unrelated headers are dropped; the API's own survive."""
    relayed = forward_request_headers(
        [
            ("authorization", "Bearer sk-test"),
            ("content-type", "application/json"),
            ("accept", "text/event-stream"),
            ("openai-organization", "org-1"),
            ("x-stainless-lang", "python"),
            ("cookie", "session=1"),
            ("x-random", "nope"),
        ],
        trust_client_headers=False,
    )

    assert relayed["authorization"] == "Bearer sk-test"
    assert relayed["openai-organization"] == "org-1"
    assert relayed["x-stainless-lang"] == "python"
    assert "cookie" not in relayed
    assert "x-random" not in relayed


def test_accept_encoding_is_forced_to_identity() -> None:
    """A compressed response would reach the observer as noise, hiding tool calls."""
    relayed = forward_request_headers(
        [("accept-encoding", "gzip, br"), ("content-type", "application/json")],
        trust_client_headers=True,
    )

    assert relayed["accept-encoding"] == "identity"


def test_client_credentials_can_be_withheld() -> None:
    """With forward_client_auth off, only the deployment's own key goes upstream."""
    relayed = forward_request_headers(
        [("authorization", "Bearer sk-agent"), ("content-type", "application/json")],
        extra={"Authorization": "Bearer sk-deployment"},
        forward_client_auth=False,
    )

    assert relayed["authorization"] == "Bearer sk-deployment"


def test_client_credentials_are_dropped_when_not_forwarded() -> None:
    """Without a replacement, the agent's key simply does not reach the provider."""
    relayed = forward_request_headers([("authorization", "Bearer sk-agent")], forward_client_auth=False)

    assert "authorization" not in relayed
    assert is_api_wire_header("authorization", forward_client_auth=False) is False
    assert is_api_wire_header("authorization", forward_client_auth=True) is True


def test_extra_headers_override_inbound() -> None:
    """A deployment credential wins over whatever the client sent."""
    relayed = forward_request_headers(
        [("authorization", "Bearer sk-agent")],
        extra={"Authorization": "Bearer sk-deployment"},
    )

    assert relayed["authorization"] == "Bearer sk-deployment"


def test_response_headers_relayed_as_asgi_pairs() -> None:
    """Rate-limit headers survive; content-length is left to the ASGI server."""
    pairs = forward_response_headers(
        [("content-type", "text/event-stream"), ("content-length", "12"), ("x-ratelimit-remaining", "42"), ("connection", "close")]
    )
    names = {name for name, _ in pairs}

    assert (b"x-ratelimit-remaining", b"42") in pairs
    assert b"content-length" not in names
    assert b"connection" not in names


def test_redact_masks_credentials() -> None:
    """No log line may ever carry a raw key."""
    masked = redact({"Authorization": "Bearer sk-secret", "content-type": "application/json"})

    assert masked["Authorization"] == "<redacted>"
    assert masked["content-type"] == "application/json"


# -- config -----------------------------------------------------------------


def test_settings_from_env() -> None:
    """Every setting can come from the environment."""
    settings = ProxySettings.from_env(
        {
            "LLM_PROXY_UPSTREAM_URL": "https://api.openai.com/v1",
            "LLM_PROXY_PORT": "9100",
            "LLM_PROXY_MOUNT_PREFIX": "/openai",
            "LLM_PROXY_TIMEOUT": "12.5",
        }
    )

    assert settings.upstream_url == "https://api.openai.com/v1"
    assert settings.port == 9100
    assert settings.mount_prefix == "/openai"
    assert settings.timeout == 12.5


def test_settings_strip_trailing_slash_from_upstream() -> None:
    """A trailing slash would produce a doubled separator once a suffix is joined."""
    assert ProxySettings(upstream_url="https://api.openai.com/v1/").upstream_url == "https://api.openai.com/v1"


def test_settings_reject_relative_upstream_url() -> None:
    """A relative upstream cannot be resolved by the relay."""
    with pytest.raises(ValueError, match="absolute HTTP"):
        ProxySettings(upstream_url="api.openai.com/v1")


def test_settings_from_env_requires_upstream() -> None:
    """The upstream URL is the one thing with no sensible default."""
    with pytest.raises(ValueError, match="LLM_PROXY_UPSTREAM_URL"):
        ProxySettings.from_env({})


def test_settings_reject_nonpositive_limits() -> None:
    """A zero timeout or body cap is a configuration error, not a disable switch."""
    with pytest.raises(ValueError, match="timeouts must be positive"):
        ProxySettings(upstream_url="https://x/v1", timeout=0)
    with pytest.raises(ValueError, match="max_body_size must be positive"):
        ProxySettings(upstream_url="https://x/v1", max_body_size=0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("/v1", "/v1"), ("v1", "/v1"), ("/v1/", "/v1"), ("*", None), ("/", None), ("", None), (None, None)],
)
def test_normalize_mount_prefix(raw: str | None, expected: str | None) -> None:
    """`*`, `/` and empty all mean "serve any path"."""
    assert normalize_mount_prefix(raw) == expected


def test_base_url_for_clients() -> None:
    """The proxy can tell an operator exactly what to point the harness at."""
    settings = ProxySettings(upstream_url="https://api.openai.com/v1", host="127.0.0.1", port=8900)

    assert settings.base_url_for_clients == f"http://127.0.0.1:8900{DEFAULT_MOUNT_PREFIX}"


# -- hooks ------------------------------------------------------------------


def _request_context(*, chat: ChatRequest | None = None, suffix: str = "/chat/completions") -> RequestContext:
    return RequestContext(
        http_method="POST",
        path=f"/v1{suffix}",
        upstream_suffix=suffix,
        query="",
        headers={},
        body_size=0,
        chat=chat if chat is not None else ChatRequest(model="m"),
    )


def test_response_context_reports_observed_intents() -> None:
    """A parsed response with calls is positively effectful."""
    context = ResponseContext(
        request=_request_context(),
        status_code=200,
        headers={},
        content_type="application/json",
        chat=parse_chat_response(
            json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"tool_calls": [{"id": "c", "type": "function", "function": {"name": "run_shell", "arguments": "{}"}}]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ).encode()
        ),
    )

    assert context.has_tool_intents is True
    assert context.intents_unknown is False
    assert context.may_dispatch_tools is True


def test_unparseable_response_is_conservatively_unknown() -> None:
    """The gate must not wave through exactly the responses it failed to read."""
    context = ResponseContext(
        request=_request_context(),
        status_code=200,
        headers={},
        content_type="application/json",
        chat=parse_chat_response(b"<html/>"),
    )

    assert context.has_tool_intents is False
    assert context.intents_unknown is True
    assert context.may_dispatch_tools is True


def test_truncated_stream_is_conservatively_unknown() -> None:
    """A stream cut before `[DONE]` may have been about to request a call."""
    accumulator = StreamAccumulator()
    accumulator.feed(_chunk({"content": "thinking"}))
    accumulator.close()

    context = ResponseContext(
        request=_request_context(),
        status_code=200,
        headers={},
        content_type="text/event-stream",
        chat=accumulator.result(),
    )

    assert context.intents_unknown is True
    assert context.may_dispatch_tools is True


def test_error_response_dispatches_nothing() -> None:
    """A provider error carries no intent, so it is not conservatively gated."""
    context = ResponseContext(
        request=_request_context(),
        status_code=429,
        headers={},
        content_type="application/json",
        chat=None,
    )

    assert context.is_error is True
    assert context.may_dispatch_tools is False


def test_non_completion_endpoint_is_not_gated() -> None:
    """`GET /v1/models` cannot produce a tool call."""
    context = ResponseContext(
        request=_request_context(chat=ChatRequest(), suffix="/models"),
        status_code=200,
        headers={},
        content_type="application/json",
        chat=None,
    )

    assert context.request.is_chat_completion is False
    assert context.may_dispatch_tools is False


async def test_composite_hooks_dispatch_to_all() -> None:
    """Every hook sees every callback."""
    seen: list[str] = []

    class Recorder:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        async def on_request(self, context: RequestContext) -> None:
            seen.append(f"{self.tag}-request")

        async def on_response(self, context: ResponseContext) -> None:
            seen.append(f"{self.tag}-response")

        async def on_relay_error(self, context: RequestContext, error: BaseException) -> None:
            seen.append(f"{self.tag}-error")

    hooks = CompositeHooks([Recorder("a"), Recorder("b")])
    request = _request_context()
    await hooks.on_request(request)
    await hooks.on_response(ResponseContext(request=request, status_code=200, headers={}, content_type=None))
    await hooks.on_relay_error(request, RuntimeError("x"))

    assert seen == ["a-request", "b-request", "a-response", "b-response", "a-error", "b-error"]


async def test_raising_hook_does_not_break_the_others() -> None:
    """An observer must never be able to break the transport."""
    seen: list[str] = []

    class Exploding:
        async def on_request(self, context: RequestContext) -> None:
            raise RuntimeError("hook is broken")

    class Working:
        async def on_request(self, context: RequestContext) -> None:
            seen.append("worked")

    await CompositeHooks([Exploding(), Working()]).on_request(_request_context())

    assert seen == ["worked"]


async def test_partial_hook_object_is_tolerated() -> None:
    """A duck-typed observer needs only the callbacks it cares about."""
    seen: list[str] = []

    class OnlyResponses:
        async def on_response(self, context: ResponseContext) -> None:
            seen.append("response")

    hooks = CompositeHooks([OnlyResponses()])
    request = _request_context()
    await hooks.on_request(request)
    await hooks.on_response(ResponseContext(request=request, status_code=200, headers={}, content_type=None))

    assert seen == ["response"]


def test_normalize_hooks() -> None:
    """One hook passes through; several compose; none becomes a no-op."""
    single = NullHooks()

    assert isinstance(normalize_hooks(None), NullHooks)
    assert isinstance(normalize_hooks([]), NullHooks)
    assert normalize_hooks(single) is single
    assert normalize_hooks([single]) is single
    assert isinstance(normalize_hooks([single, NullHooks()]), CompositeHooks)


async def test_null_and_logging_hooks_are_noops() -> None:
    """Both default observers run without touching the exchange."""
    request = _request_context()
    response = ResponseContext(request=request, status_code=200, headers={}, content_type=None)

    for hooks in (NullHooks(), LoggingHooks()):
        await hooks.on_request(request)
        await hooks.on_response(response)
        await hooks.on_relay_error(request, RuntimeError("x"))
