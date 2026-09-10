"""End-to-end tests: a real OpenAI client talking to a provider through the proxy.

These are the exercises that matter for a drop-in reverse proxy. An unmodified
`openai` client points at the proxy's base URL; the proxy relays to a fake
provider over TCP. Nothing here knows the proxy exists except by its URL —
which is precisely the deployment contract.

    tests:  openai.AsyncOpenAI ──TCP──► proxy (uvicorn) ──TCP──► fake provider
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI, APIStatusError

from tests.conftest import ProxyFixture, ToolCallScript

API_KEY = "sk-test-key"


def client_for(fixture: ProxyFixture, **kwargs: object) -> AsyncOpenAI:
    """An OpenAI client pointed at the proxy — the entire integration."""
    kwargs.setdefault("max_retries", 0)
    return AsyncOpenAI(base_url=fixture.proxy_base_url, api_key=API_KEY, **kwargs)  # type: ignore[arg-type]


# -- the basic contract -----------------------------------------------------


async def test_non_streaming_completion(proxy_pair: ProxyFixture) -> None:
    """A plain completion round-trips and is observed once."""
    async with client_for(proxy_pair) as client:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert completion.choices[0].message.content == proxy_pair.fake.content
    assert completion.model == proxy_pair.fake.model

    await proxy_pair.wait_for_responses()
    (request,) = proxy_pair.hooks.requests
    assert request.model == "gpt-4o-mini"
    assert request.is_chat_completion is True
    (response,) = proxy_pair.hooks.responses
    assert response.status_code == 200
    assert response.has_tool_intents is False
    assert response.may_dispatch_tools is False


async def test_streaming_completion_arrives_incrementally(proxy_pair: ProxyFixture) -> None:
    """The client sees many chunks, not one buffered blob at the end."""
    proxy_pair.fake.content = "streamed answer in several pieces"
    proxy_pair.fake.fragment_size = 4

    pieces: list[str] = []
    async with client_for(proxy_pair) as client:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                pieces.append(chunk.choices[0].delta.content)

    assert len(pieces) > 1, "a streamed response must not arrive as one piece"
    assert "".join(pieces) == proxy_pair.fake.content

    await proxy_pair.wait_for_responses()
    (response,) = proxy_pair.hooks.responses
    assert response.chat is not None
    assert response.chat.is_stream is True
    assert response.chat.complete is True


async def test_credential_reaches_the_provider(proxy_pair: ProxyFixture) -> None:
    """The proxy is on the credential path: without relaying it, nothing works."""
    async with client_for(proxy_pair) as client:
        await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    assert proxy_pair.fake.last_request["headers"]["authorization"] == f"Bearer {API_KEY}"


async def test_models_endpoint_passes_through(proxy_pair: ProxyFixture) -> None:
    """Every endpoint under the base URL routes, not just chat completions."""
    async with client_for(proxy_pair) as client:
        models = await client.models.list()

    assert [model.id for model in models.data] == [proxy_pair.fake.model]
    # A side endpoint is not a completion, so the conservative gate stays shut.
    await proxy_pair.wait_for_responses()
    (response,) = proxy_pair.hooks.responses
    assert response.request.is_chat_completion is False
    assert response.may_dispatch_tools is False


# -- tool calls, the reason this proxy exists -------------------------------


async def test_tool_intent_is_observed_in_the_response(proxy_pair: ProxyFixture) -> None:
    """The model's requested tool call is seen before the harness could run it."""
    proxy_pair.fake.content = ""
    proxy_pair.fake.tool_calls = [
        ToolCallScript(id="call_1", name="run_shell", arguments='{"cmd": "rm -rf /tmp/scratch"}')
    ]

    async with client_for(proxy_pair) as client:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "clean up"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "run_shell",
                        "description": "Run a shell command",
                        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                    },
                }
            ],
        )

    # The client sees a normal tool call...
    tool_call = completion.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "run_shell"

    # ...and so did the proxy, from the response side.
    await proxy_pair.wait_for_responses()
    (request,) = proxy_pair.hooks.requests
    assert request.declared_tools[0].name == "run_shell"
    (response,) = proxy_pair.hooks.responses
    (intent,) = response.tool_intents
    assert intent.name == "run_shell"
    assert intent.arguments == {"cmd": "rm -rf /tmp/scratch"}
    assert response.may_dispatch_tools is True


async def test_streamed_tool_arguments_are_reassembled(proxy_pair: ProxyFixture) -> None:
    """Arguments split across frames and chunks still rebuild into one object."""
    proxy_pair.fake.content = ""
    proxy_pair.fake.tool_calls = [
        ToolCallScript(id="call_1", name="write_file", arguments='{"path": "/etc/hosts", "text": "hello"}')
    ]
    proxy_pair.fake.fragment_size = 3
    proxy_pair.fake.chunk_size = 11  # also splits SSE frames across TCP chunks

    async with client_for(proxy_pair) as client:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "write it"}],
            stream=True,
        )
        async for _ in stream:
            pass

    await proxy_pair.wait_for_responses()
    (response,) = proxy_pair.hooks.responses
    (intent,) = response.tool_intents
    assert intent.name == "write_file"
    assert intent.arguments == {"path": "/etc/hosts", "text": "hello"}
    assert response.chat is not None
    assert response.chat.complete is True


async def test_parallel_streamed_tool_calls_stay_separate(proxy_pair: ProxyFixture) -> None:
    """Interleaved parallel calls must not have their fragments concatenated."""
    proxy_pair.fake.content = ""
    proxy_pair.fake.tool_calls = [
        ToolCallScript(id="call_a", name="read_file", arguments='{"path": "/etc/passwd"}'),
        ToolCallScript(id="call_b", name="run_shell", arguments='{"cmd": "whoami"}'),
    ]
    proxy_pair.fake.fragment_size = 4
    proxy_pair.fake.interleave = True

    async with client_for(proxy_pair) as client:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "inspect"}],
            stream=True,
        )
        async for _ in stream:
            pass

    await proxy_pair.wait_for_responses()
    (response,) = proxy_pair.hooks.responses
    intents = response.tool_intents
    assert [intent.name for intent in intents] == ["read_file", "run_shell"]
    assert intents[0].arguments == {"path": "/etc/passwd"}
    assert intents[1].arguments == {"cmd": "whoami"}


async def test_tool_result_is_observed_on_the_next_request(proxy_pair: ProxyFixture) -> None:
    """The evidence loop: request N+1 reports what response N's calls actually did.

    Tool execution happens inside the harness and never crosses this wire, so
    the `role: "tool"` message is the proxy's only evidence of the outcome.
    """
    proxy_pair.fake.content = ""
    proxy_pair.fake.tool_calls = [ToolCallScript(id="call_1", name="run_shell", arguments='{"cmd": "whoami"}')]

    async with client_for(proxy_pair) as client:
        first = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "who am i?"}],
        )
        call = first.choices[0].message.tool_calls[0]

        # The harness runs the tool locally, then reports the result back.
        proxy_pair.fake.tool_calls = []
        proxy_pair.fake.content = "You are root."
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "who am i?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": call.id, "type": "function", "function": {"name": "run_shell", "arguments": '{"cmd": "whoami"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": call.id, "content": "root"},
            ],
        )

    await proxy_pair.wait_for_responses(2)
    second_request = proxy_pair.hooks.requests[1]
    (result,) = second_request.tool_results
    assert result.tool_call_id == "call_1"
    assert result.content == "root"
    # And the intent that produced it is recoverable from the same request.
    assert second_request.chat is not None
    assert second_request.chat.prior_tool_calls[0].name == "run_shell"


# -- fidelity and failure ---------------------------------------------------


async def test_non_ascii_content_crosses_unchanged(proxy_pair: ProxyFixture) -> None:
    """Multi-byte characters must survive both hops intact."""
    proxy_pair.fake.content = "héllo — ünicode ✓ 日本語"

    async with client_for(proxy_pair) as client:
        completion = await client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "echo — ünicode ✓"}],
        )

    assert completion.choices[0].message.content == proxy_pair.fake.content
    assert proxy_pair.fake.last_request["body"]["messages"][0]["content"] == "echo — ünicode ✓"


async def test_proxied_response_is_byte_identical_to_direct(proxy_pair: ProxyFixture) -> None:
    """The relay is byte-exact: same request, same bytes, proxied or not."""
    payload = {"model": "m", "messages": [{"role": "user", "content": "compare — ünicode"}]}
    headers = {"authorization": f"Bearer {API_KEY}", "content-type": "application/json"}

    async with httpx.AsyncClient() as http_client:
        through_proxy = await http_client.post(
            f"{proxy_pair.proxy_base_url}/chat/completions", json=payload, headers=headers
        )
        direct = await http_client.post(
            f"{proxy_pair.upstream_base_url}/chat/completions", json=payload, headers=headers
        )

    assert through_proxy.status_code == direct.status_code
    assert through_proxy.content == direct.content


async def test_provider_error_surfaces_as_an_api_error(proxy_pair: ProxyFixture) -> None:
    """An upstream failure reaches the client as a typed error, not a parse failure."""
    proxy_pair.fake.status_code = 429
    proxy_pair.fake.error_body = {"error": {"message": "slow down", "type": "rate_limit_error"}}

    async with client_for(proxy_pair) as client:
        with pytest.raises(APIStatusError) as caught:
            await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    assert caught.value.status_code == 429
    await proxy_pair.wait_for_responses()
    (response,) = proxy_pair.hooks.responses
    assert response.is_error is True
    # An error response dispatches nothing, so it is not conservatively gated.
    assert response.may_dispatch_tools is False


async def test_health_endpoint_is_local(proxy_pair: ProxyFixture) -> None:
    """The proxy's health path is answered locally, never relayed."""
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(f"{proxy_pair.proxy.root_url}/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert proxy_pair.fake.requests == []


async def test_unknown_path_is_refused_locally(proxy_pair: ProxyFixture) -> None:
    """A path outside the mount prefix never reaches the provider."""
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(f"{proxy_pair.proxy.root_url}/not-the-api", json={})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_path"
    assert proxy_pair.fake.requests == []


async def test_concurrent_clients_do_not_cross_talk(proxy_pair: ProxyFixture) -> None:
    """Parallel sessions each get their own correct result."""
    proxy_pair.fake.content = "shared answer"

    async def one_call(index: int) -> str:
        async with client_for(proxy_pair) as client:
            completion = await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": f"request-{index}"}]
            )
            return completion.choices[0].message.content or ""

    results = await asyncio.gather(*[one_call(index) for index in range(8)])

    assert results == ["shared answer"] * 8
    await proxy_pair.wait_for_responses(8)
    sent = {entry["body"]["messages"][0]["content"] for entry in proxy_pair.fake.requests}
    assert sent == {f"request-{index}" for index in range(8)}
