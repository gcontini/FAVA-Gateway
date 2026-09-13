"""The Permission IR and the LLM that extracts it from the task.

Formal verifiers cannot read natural language and probabilistic models cannot be
trusted with access-control decisions, so the paper uses an LLM strictly as a
*semantic extractor*: it turns the task text into the five structured fields
below, and everything downstream reasons over the structure rather than over the
model's opinion.

The task text is free here. It arrives on every request as part of the message
array whether or not FAVA asks for it, so there is nothing to instrument — see
:func:`fava.state.events.task_text`.

Two things this module must never do:

* **Raise.** A failed extraction yields an IR with
  :attr:`RiskPosture.AMBIGUOUS`, which is the value the gateway fails closed on.
  An exception would instead take down an observer on the relay's path.
* **Call through the proxy.** The extractor talks to the provider directly. Sent
  through the gateway, its own request would be observed, minted as a run, and
  extracted from — recursively. Requests carry ``X-FAVA-Internal`` so a
  misconfiguration shows up in a log rather than as a runaway loop.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

ENV_BASE_URL: Final = "FAVA_EXTRACTOR_BASE_URL"
ENV_MODEL: Final = "FAVA_EXTRACTOR_MODEL"
ENV_API_KEY: Final = "FAVA_EXTRACTOR_API_KEY"
ENV_TIMEOUT: Final = "FAVA_EXTRACTOR_TIMEOUT"
ENV_ENABLED: Final = "FAVA_EXTRACTOR_ENABLED"

DEFAULT_EXTRACTOR_TIMEOUT: Final = 20.0

# Marks FAVA's own calls, so a deployment that accidentally points the extractor
# at the proxy can see it in one grep instead of one runaway bill.
INTERNAL_HEADER: Final = "X-FAVA-Internal"

# The task text is a system prompt plus one user turn; agent system prompts run
# long, and the tail matters less than the head for intent.
MAX_TASK_TEXT: Final = 16_000


class RiskPosture(str, Enum):
    """How dangerous the extracted task looks.

    The gateway fails closed on everything but :attr:`BENIGN`: the paper is
    explicit that an action whose posture is sensitive, dangerous or merely
    ambiguous is blocked rather than escalated to a human.
    """

    BENIGN = "benign"
    SENSITIVE = "sensitive"
    DANGEROUS = "dangerous"
    AMBIGUOUS = "ambiguous"

    @property
    def fails_closed(self) -> bool:
        """Whether this posture denies by default, absent an explicit allow."""
        return self is not RiskPosture.BENIGN


@dataclass(frozen=True)
class Evidence:
    """What grounds a claim: where it came from, and the span that says so.

    Attributes:
        ref: The source — ``"task"`` for the extracted task text, or an
            ``event_id`` for something observed on the wire.
        quote: The span of source text supporting the claim, when there is one.
    """

    ref: str
    quote: str = ""


@dataclass(frozen=True)
class Asset:
    """A resource the task operates on, e.g. a file, a secret, a table."""

    name: str
    labels: tuple[str, ...] = ()
    evidence: Evidence | None = None


@dataclass(frozen=True)
class Action:
    """An operation the task calls for, e.g. ``read`` on ``config.yaml``."""

    op: str
    target: str = ""
    evidence: Evidence | None = None


@dataclass(frozen=True)
class Obligation:
    """A temporal guard: ``requires`` must be established before ``before``.

    The paper's example is "require review before testing" — a prerequisite,
    not a permission. Lowering turns these into ``control`` edges, which is the
    only thing that makes sequence-dependent constraints expressible at all.
    """

    requires: str
    before: str = ""
    evidence: Evidence | None = None


@dataclass(frozen=True)
class Sink:
    """A destination the task may send data to, e.g. a channel or a host."""

    target: str
    labels: tuple[str, ...] = ()
    evidence: Evidence | None = None


@dataclass(frozen=True)
class PermissionIR:
    """The authorization boundary the task text describes.

    The paper's five fields, plus the risk posture the gateway fails closed on.
    Every element carries its own :class:`Evidence`, so a decision can be traced
    back to the sentence that justified it rather than to "the model said so".

    Attributes:
        intent: One-line summary of what the task is for.
        assets: Resources the task operates on.
        actions: Operations it calls for.
        obligations: Prerequisites that must hold first.
        sinks: Destinations data may leave through.
        risk_posture: How dangerous the task looks overall.
        extracted: False when no extraction has happened or it failed, in which
            case the posture is :attr:`RiskPosture.AMBIGUOUS`.
        error: Why extraction did not produce an IR, for diagnostics.
    """

    intent: str = ""
    assets: Sequence[Asset] = field(default_factory=tuple)
    actions: Sequence[Action] = field(default_factory=tuple)
    obligations: Sequence[Obligation] = field(default_factory=tuple)
    sinks: Sequence[Sink] = field(default_factory=tuple)
    risk_posture: RiskPosture = RiskPosture.AMBIGUOUS
    extracted: bool = False
    error: str | None = None

    @classmethod
    def unknown(cls, error: str | None = None) -> PermissionIR:
        """The IR a run holds before — or instead of — a successful extraction.

        Ambiguous by construction. A run whose task was never understood must
        not read as benign simply because nothing was extracted from it.
        """
        return cls(error=error)

    def as_payload(self) -> dict[str, Any]:
        """A JSON-safe rendering, for recording on the ``ir_extracted`` event."""
        return {
            "intent": self.intent,
            "risk_posture": self.risk_posture.value,
            "assets": [{"name": a.name, "labels": list(a.labels)} for a in self.assets],
            "actions": [{"op": a.op, "target": a.target} for a in self.actions],
            "obligations": [{"requires": o.requires, "before": o.before} for o in self.obligations],
            "sinks": [{"target": s.target, "labels": list(s.labels)} for s in self.sinks],
            "extracted": self.extracted,
            "error": self.error,
        }


# -- extraction -------------------------------------------------------------


EXTRACTION_SYSTEM_PROMPT: Final = """\
You are a security extractor. You do not make authorization decisions; you only \
restate an agent's task as structured data for a formal verifier.

Read the task below and return a single JSON object with exactly these keys:

  "intent":       string, one line, what the task is for.
  "risk_posture": one of "benign", "sensitive", "dangerous", "ambiguous".
  "assets":       array of {"name": string, "labels": array of string,
                            "evidence": string}
  "actions":      array of {"op": string, "target": string, "evidence": string}
  "obligations":  array of {"requires": string, "before": string,
                            "evidence": string}
  "sinks":        array of {"target": string, "labels": array of string,
                            "evidence": string}

Rules:
- "evidence" must be a short verbatim quote from the task text. If you cannot \
quote the task for a claim, do not make the claim.
- Useful labels include "secret", "pii", "credential", "public", "internal".
- "assets" are resources operated on; "sinks" are destinations data can leave \
through (a channel, a host, an address, an external service).
- "obligations" are prerequisites: "requires" must hold before "before" happens.
- Choose "benign" only if nothing in the task touches sensitive data, external \
destinations, or destructive operations. Choose "ambiguous" when the task is \
too vague to tell.
- Return only the JSON object."""


@dataclass(frozen=True)
class ExtractorSettings:
    """Where and how to reach the model that extracts the IR.

    Deliberately standalone: the model that extracts the IR must be a
    different model from the one under observation, reached with its own
    credential, never the monitored agent's. Reusing either would let the
    very traffic FAVA reasons about shape the extraction that reasons about
    it. So nothing here is inherited from the proxy's own settings or from a
    request in flight — set ``FAVA_EXTRACTOR_BASE_URL`` /
    ``FAVA_EXTRACTOR_MODEL`` / ``FAVA_EXTRACTOR_API_KEY`` explicitly (see
    :meth:`from_env`). Leaving any of the three unset — the default — leaves
    extraction unconfigured and every run ambiguous, same as
    ``FAVA_EXTRACTOR_ENABLED=0``.

    Attributes:
        base_url: OpenAI-compatible **base** URL for the extraction model's
            *own* provider, e.g. ``https://api.openai.com/v1``. It must never
            be the proxy's own URL — see the module docstring.
        model: Model to extract with. Required — this is not the model the
            harness itself asked for.
        api_key: Credential for the extraction call. Required — this is not
            the agent's own credential.
        timeout: Seconds for the whole extraction call.
        enabled: Set false to skip extraction entirely.
    """

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout: float = DEFAULT_EXTRACTOR_TIMEOUT
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.base_url:
            object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def configured(self) -> bool:
        """Whether base URL, model and credential are all present."""
        return bool(self.base_url and self.model and self.api_key)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ExtractorSettings:
        """Build settings from the environment.

        Reads only ``FAVA_EXTRACTOR_*`` — nothing here falls back to the
        proxy's own upstream, model or credential; see the class docstring
        for why.

        Args:
            environ: Mapping to read, defaulting to ``os.environ``.
        """
        env = os.environ if environ is None else environ
        raw_timeout = env.get(ENV_TIMEOUT)
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_EXTRACTOR_TIMEOUT
        except ValueError as exc:
            raise ValueError(f"{ENV_TIMEOUT} must be a number, got {raw_timeout!r}") from exc

        enabled = env.get(ENV_ENABLED, "1").strip().lower() not in ("0", "false", "no", "off")
        return cls(
            base_url=env.get(ENV_BASE_URL, ""),
            model=env.get(ENV_MODEL, ""),
            api_key=env.get(ENV_API_KEY, ""),
            timeout=timeout,
            enabled=enabled,
        )


@runtime_checkable
class IRExtractor(Protocol):
    """Turns task text into a :class:`PermissionIR`. Must never raise.

    ``model`` and ``authorization`` are the harness's own — what the
    monitored request is using. They are offered for extractors that have no
    configuration of their own, but :class:`LlmIRExtractor` deliberately
    ignores both: its model and credential must stay independent of the
    traffic it is reasoning about.
    """

    async def extract(
        self,
        task_text: str,
        *,
        model: str | None = None,
        authorization: str | None = None,
    ) -> PermissionIR:
        """Extract the IR, returning an ambiguous one on any failure."""


class NullIRExtractor:
    """The extractor used when no model is configured or reachable.

    Everything it returns is ambiguous, which is the honest answer: with no
    extraction, nothing about the task has been established.
    """

    def __init__(self, reason: str = "extraction disabled") -> None:
        self._reason = reason

    async def extract(
        self,
        task_text: str,
        *,
        model: str | None = None,
        authorization: str | None = None,
    ) -> PermissionIR:
        """Return an ambiguous IR carrying the reason extraction was skipped."""
        return PermissionIR.unknown(self._reason)


class LlmIRExtractor:
    """Extracts the IR with one Chat Completions call to the provider.

    Uses the OpenAI SDK's ``AsyncOpenAI`` client against
    :attr:`ExtractorSettings.base_url` — any OpenAI-compatible endpoint works,
    not just OpenAI's own. One call, one JSON object, no streaming, and no
    SDK-level retries: a slow or flaky provider must not compound its own
    backoff onto an observer that has to stay off the relay's critical path.

    Deliberately does not fall back to the harness's own model or credential
    (the ``model``/``authorization`` arguments to :meth:`extract`): see
    :class:`ExtractorSettings` for why, and for how to configure this.
    """

    def __init__(
        self,
        settings: ExtractorSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport

    async def extract(
        self,
        task_text: str,
        *,
        model: str | None = None,
        authorization: str | None = None,
    ) -> PermissionIR:
        """Extract the IR from task text.

        Args:
            task_text: The system prompt plus first user turn of the run.
            model: Ignored. The extraction model comes only from
                :attr:`ExtractorSettings.model` — see the class docstring.
            authorization: Ignored, for the same reason: the extraction
                credential comes only from :attr:`ExtractorSettings.api_key`.

        Returns:
            The extracted IR, or an ambiguous one carrying ``error``. Never
            raises: an observer on the relay's path must not be able to fail.
        """
        if not self.settings.enabled:
            return PermissionIR.unknown("extraction disabled")
        if not task_text.strip():
            return PermissionIR.unknown("no task text to extract from")
        if not self.settings.base_url:
            return PermissionIR.unknown("no base URL configured for extraction")
        if not self.settings.model:
            return PermissionIR.unknown("no model configured for extraction")
        if not self.settings.api_key:
            return PermissionIR.unknown("no credential configured for extraction")

        try:
            content = await self._call(task_text[:MAX_TASK_TEXT])
        except Exception as exc:  # noqa: BLE001 - the extractor is best-effort by contract
            logger.warning("IR extraction failed: %s: %s", type(exc).__name__, exc)
            return PermissionIR.unknown(f"{type(exc).__name__}: {exc}")

        return parse_ir(content)

    async def _call(self, task_text: str) -> str:
        """Run one completion and return the assistant's text."""
        transport_client = httpx.AsyncClient(transport=self._transport) if self._transport is not None else None
        async with AsyncOpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            http_client=transport_client,  # type: ignore[arg-type]
            timeout=self.settings.timeout,
            max_retries=0,
            default_headers={INTERNAL_HEADER: "extract"},
        ) as client:
            response = await client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": task_text},
                ],
                # `json_object` is supported far more widely than `json_schema`, and
                # the parse below tolerates a provider that ignores it anyway. One
                # code path beats a schema plus a fallback.
                response_format={"type": "json_object"},
                temperature=0,
                stream=False,
            )

        choices = response.choices
        if not choices:
            raise ValueError("extraction response carried no choices")
        content = choices[0].message.content
        if not content:
            raise ValueError("extraction response carried no text content")
        return content


def parse_ir(content: str) -> PermissionIR:
    """Parse an extractor's reply into a :class:`PermissionIR`.

    Tolerant on purpose: models wrap JSON in prose or fences even when asked
    not to, so the first balanced object in the text is used. A reply that
    yields nothing parseable returns an ambiguous IR rather than raising.
    """
    decoded = _first_json_object(content)
    if decoded is None:
        return PermissionIR.unknown("extractor reply contained no JSON object")

    posture = _posture(decoded.get("risk_posture"))
    return PermissionIR(
        intent=_text(decoded.get("intent")),
        assets=tuple(
            Asset(name=_text(e.get("name")), labels=_labels(e.get("labels")), evidence=_evidence(e))
            for e in _entries(decoded.get("assets"))
            if _text(e.get("name"))
        ),
        actions=tuple(
            Action(op=_text(e.get("op")), target=_text(e.get("target")), evidence=_evidence(e))
            for e in _entries(decoded.get("actions"))
            if _text(e.get("op"))
        ),
        obligations=tuple(
            Obligation(requires=_text(e.get("requires")), before=_text(e.get("before")), evidence=_evidence(e))
            for e in _entries(decoded.get("obligations"))
            if _text(e.get("requires"))
        ),
        sinks=tuple(
            Sink(target=_text(e.get("target")), labels=_labels(e.get("labels")), evidence=_evidence(e))
            for e in _entries(decoded.get("sinks"))
            if _text(e.get("target"))
        ),
        risk_posture=posture,
        extracted=True,
    )


def _first_json_object(text: str) -> Mapping[str, Any] | None:
    """Return the first balanced JSON object in ``text``, or ``None``."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        decoded = json.loads(text[start : position + 1])
                    except (ValueError, RecursionError):
                        break
                    if isinstance(decoded, Mapping):
                        return decoded
                    break
        start = text.find("{", start + 1)
    return None


def _entries(raw: Any) -> tuple[Mapping[str, Any], ...]:
    """The mapping entries of a list field, skipping anything malformed."""
    if not isinstance(raw, list):
        return ()
    return tuple(entry for entry in raw if isinstance(entry, Mapping))


def _labels(raw: Any) -> tuple[str, ...]:
    """Normalize a label list to lowercase strings."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return ()
    return tuple(item.strip().lower() for item in raw if isinstance(item, str) and item.strip())


def _evidence(entry: Mapping[str, Any]) -> Evidence | None:
    """Build the evidence span for one IR element, when it quoted the task."""
    quote = _text(entry.get("evidence"))
    return Evidence(ref="task", quote=quote) if quote else None


def _posture(raw: Any) -> RiskPosture:
    """Map a posture string onto the enum, defaulting to ambiguous."""
    if isinstance(raw, str):
        try:
            return RiskPosture(raw.strip().lower())
        except ValueError:
            return RiskPosture.AMBIGUOUS
    return RiskPosture.AMBIGUOUS


def _text(value: Any) -> str:
    """Return a stripped string for a text field, else empty."""
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "Action",
    "Asset",
    "Evidence",
    "ExtractorSettings",
    "IRExtractor",
    "LlmIRExtractor",
    "NullIRExtractor",
    "Obligation",
    "PermissionIR",
    "RiskPosture",
    "Sink",
    "parse_ir",
]
