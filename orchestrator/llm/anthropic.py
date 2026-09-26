"""Concrete Anthropic client with project-owned transport retry semantics."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types.message_count_tokens_tool_param import MessageCountTokensToolParam
from anthropic.types.text_block_param import TextBlockParam

from orchestrator.contracts import SCHEMA_MODELS, response_parses
from orchestrator.llm.base import LLMCall, LLMClient, Message, TransportRetry

API_MODEL = "claude-haiku-4-5-20251001"
MODEL_ID = f"anthropic:{API_MODEL}"
# Reproduce configured inputs:
# python -c "from orchestrator.llm.anthropic import *; print(MAX_TOKENS, REQUEST_TIMEOUT_S,
# MAX_TRANSPORT_RETRIES, RETRY_AFTER_LIMIT_S, BACKOFF_S)"
MAX_TOKENS = 2048
REQUEST_TIMEOUT_S = 60.0
MAX_TRANSPORT_RETRIES = 2
RETRY_AFTER_LIMIT_S = 60.0
BACKOFF_S = (1.0, 2.0)
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
# Provider-supported JSON Schema subset:
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "discriminator",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxLength",
        "maximum",
        "minLength",
        "minimum",
        "multipleOf",
        "propertyNames",
    }
)


class ProviderTransportError(RuntimeError):
    """Sanitized terminal provider failure with retry metadata only."""

    def __init__(
        self,
        category: str,
        status_code: int | None,
        request_id: str | None,
        retries: Sequence[TransportRetry],
    ) -> None:
        self.category = category
        self.status_code = status_code
        self.request_id = request_id
        self.retries = tuple(retries)
        super().__init__(
            f"provider transport failed: category={category}, status={status_code}, "
            f"request_id={request_id}"
        )


def _request_id(response: object) -> str | None:
    value = getattr(response, "_request_id", None)
    return value if isinstance(value, str) else None


def _response_request_id(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("request-id") or headers.get("x-request-id")
    return str(value) if value else None


def _status_code(error: BaseException) -> int | None:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _retryable(error: BaseException) -> bool:
    if isinstance(error, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    return (
        isinstance(error, anthropic.APIStatusError) and _status_code(error) in _RETRYABLE_STATUSES
    )


def _retry_after_seconds(error: BaseException, now: datetime) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - now).total_seconds()
    return max(0.0, seconds)


def _provider_schema(value: object) -> object:
    """Move unsupported constraints to descriptions while preserving local validation."""

    if isinstance(value, list):
        return [_provider_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    transformed: dict[str, object] = {}
    local_constraints: list[str] = []
    for key, item in value.items():
        if key == "oneOf":
            transformed["anyOf"] = _provider_schema(item)
            continue
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            local_constraints.append(
                f"{key}={json.dumps(item, ensure_ascii=False, sort_keys=True)}"
            )
            continue
        if key == "additionalProperties" and isinstance(item, dict):
            transformed[key] = False
            local_constraints.append(
                "additionalProperties=" + json.dumps(item, ensure_ascii=False, sort_keys=True)
            )
            continue
        transformed[key] = _provider_schema(item)

    if local_constraints:
        note = "Local validation constraints: " + ", ".join(local_constraints)
        existing = transformed.get("description")
        transformed["description"] = f"{existing}\n{note}" if existing else note
    return transformed


def _output_config(schema_name: str) -> dict[str, object]:
    model = SCHEMA_MODELS.get(schema_name)
    if model is None:
        raise ValueError(f"unknown schema: {schema_name}")
    return {
        "format": {
            "schema": _provider_schema(model.model_json_schema()),
            "type": "json_schema",
        }
    }


def build_request(
    messages: Sequence[Message],
    schema_name: str,
    *,
    model: str = API_MODEL,
    max_tokens: int = MAX_TOKENS,
    cache_prefix: bool = False,
) -> dict[str, object]:
    """Map internal messages to the Anthropic system/messages request shape."""

    system_messages = [message for message in messages if message["role"] == "system"]
    conversation = [message for message in messages if message["role"] != "system"]
    system: list[TextBlockParam] = []
    for index, message in enumerate(system_messages):
        block: TextBlockParam = {"type": "text", "text": message["content"]}
        if cache_prefix and index == len(system_messages) - 1:
            block["cache_control"] = {"type": "ephemeral"}
        system.append(block)
    return {
        "max_tokens": max_tokens,
        "messages": [
            {"role": message["role"], "content": message["content"]} for message in conversation
        ],
        "model": model,
        "output_config": _output_config(schema_name),
        "system": system,
    }


class AnthropicClient(LLMClient):
    """One concrete provider client; the SDK handles auth, types, and HTTP."""

    def __init__(
        self,
        *,
        sdk_client: anthropic.AsyncAnthropic | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        cache_stages: frozenset[str] = frozenset(),
        before_transport: Callable[[Sequence[Message], str, int], Awaitable[object]] | None = None,
        transport_succeeded: Callable[[object, LLMCall], None] | None = None,
    ) -> None:
        if sdk_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic runs")
            sdk_client = anthropic.AsyncAnthropic(
                api_key=api_key,
                max_retries=0,
                timeout=REQUEST_TIMEOUT_S,
            )
        self._sdk = sdk_client
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._cache_stages = cache_stages
        self._before_transport = before_transport
        self._transport_succeeded = transport_succeeded
        self._calls: list[LLMCall] = []

    @property
    def model(self) -> str:
        return MODEL_ID

    @property
    def calls(self) -> Sequence[LLMCall]:
        return tuple(self._calls)

    async def complete(
        self,
        messages: Sequence[Message],
        schema_name: str,
        *,
        stage: str,
        agent: str,
        attempt: int,
    ) -> str:
        elapsed_s = 0.0
        retries: list[TransportRetry] = []
        request = build_request(
            messages,
            schema_name,
            cache_prefix=stage in self._cache_stages,
        )
        for transport_attempt in range(MAX_TRANSPORT_RETRIES + 1):
            reservation: object = None
            if self._before_transport is not None:
                reservation = await self._before_transport(messages, schema_name, transport_attempt)
            attempt_started = self._clock()
            try:
                response = cast(
                    AnthropicMessage,
                    await self._sdk.messages.create(**cast(Any, request)),
                )
            except (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.APIStatusError,
            ) as error:
                elapsed_s += self._clock() - attempt_started
                status = _status_code(error)
                request_id = _response_request_id(error)
                category = type(error).__name__
                if not _retryable(error) or transport_attempt == MAX_TRANSPORT_RETRIES:
                    raise ProviderTransportError(category, status, request_id, retries) from error
                wait_s = _retry_after_seconds(error, self._now())
                if wait_s is None:
                    wait_s = BACKOFF_S[transport_attempt]
                if wait_s > RETRY_AFTER_LIMIT_S:
                    raise ProviderTransportError(category, status, request_id, retries) from error
                retry = TransportRetry(
                    error_category=category,
                    status_code=status,
                    transport_attempt=transport_attempt,
                    wait_s=wait_s,
                    request_id=request_id,
                )
                retries.append(retry)
                sleep_started = self._clock()
                await self._sleep(wait_s)
                elapsed_s += self._clock() - sleep_started
                continue

            elapsed_s += self._clock() - attempt_started
            raw_response = "".join(block.text for block in response.content if block.type == "text")
            usage = response.usage
            call = LLMCall(
                stage=stage,
                agent=agent,
                attempt=attempt,
                latency_s=elapsed_s,
                raw_response=raw_response,
                parsed_ok=response_parses(schema_name, raw_response),
                model=self.model,
                request_id=_request_id(response),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                transport_retries=tuple(retries),
            )
            self._calls.append(call)
            if self._transport_succeeded is not None:
                self._transport_succeeded(reservation, call)
            return raw_response
        raise AssertionError("transport retry loop exited unexpectedly")  # pragma: no cover

    async def count_prefix_tokens(
        self,
        messages: Sequence[Message],
        schema_name: str,
        *,
        tools: Sequence[MessageCountTokensToolParam] = (),
    ) -> int:
        request = build_request(messages, schema_name)
        kwargs: dict[str, object] = {
            "model": API_MODEL,
            "messages": request["messages"],
            "output_config": request["output_config"],
            "system": request["system"],
        }
        if tools:
            kwargs["tools"] = tools
        result = await self._sdk.messages.count_tokens(**cast(Any, kwargs))
        return result.input_tokens
