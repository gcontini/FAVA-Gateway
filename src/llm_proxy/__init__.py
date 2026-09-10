"""FAVA's runtime gateway seam: a pass-through LLM API reverse proxy.

The proxy sits between an agent harness and the LLM provider, relaying
OpenAI-compatible Chat Completions traffic byte-for-byte while observing the
tool-call intents in the model's responses. Intercepting here rather than on an
MCP wire is what makes mediation universal: a harness's local tools (shell, file
edits) never touch MCP, but every tool it can run is first *requested* over this
connection.

Today the proxy only observes. The hooks are where FAVA's authorization stage
will attach — see ``docs/implementation_plan.md``.
"""

from llm_proxy.app import create_proxy_app
from llm_proxy.chat import (
    ChatRequest,
    ChatResponse,
    ResponseObserver,
    StreamAccumulator,
    ToolIntent,
    ToolResult,
    ToolSpec,
)
from llm_proxy.config import ProxySettings
from llm_proxy.hooks import (
    CompositeHooks,
    LoggingHooks,
    NullHooks,
    ProxyHooks,
    RequestContext,
    ResponseContext,
)
from llm_proxy.relay import ForwardingProxy

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "CompositeHooks",
    "ForwardingProxy",
    "LoggingHooks",
    "NullHooks",
    "ProxyHooks",
    "ProxySettings",
    "RequestContext",
    "ResponseContext",
    "ResponseObserver",
    "StreamAccumulator",
    "ToolIntent",
    "ToolResult",
    "ToolSpec",
    "create_proxy_app",
]
