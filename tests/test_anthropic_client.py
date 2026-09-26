from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from orchestrator.contracts import TelemetryReport
from orchestrator.llm.anthropic import (
    API_MODEL,
    AnthropicClient,
    ProviderTransportError,
    build_request,
)
from orchestrator.repair import complete_with_repair


def _response(
    text_blocks: list[str],
    *,
    request_id: str = "req_test",
) -> httpx2.Response:
    return httpx2.Response(
        200,
        headers={"request-id": request_id},
        json={
            "content": [{"text": text, "type": "text"} for text in text_blocks],
            "id": "msg_test",
            "model": API_MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 5,
                "input_tokens": 23,
                "output_tokens": 11,
            },
        },
    )


def _sdk(handler: Callable[[httpx2.Request], httpx2.Response]) -> anthropic.AsyncAnthropic:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return anthropic.AsyncAnthropic(
        api_key="sentinel-api-key",
        http_client=http_client,
        max_retries=0,
        timeout=60.0,
    )


def test_production_client_reads_only_environment_and_disables_sdk_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class StubSDK:
        """Constructor sentinel; no network method is used in this test."""

    def construct(**kwargs: object) -> StubSDK:
        captured.update(kwargs)
        return StubSDK()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key-sentinel")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", construct)
    client = AnthropicClient()

    assert client.model == "anthropic:claude-haiku-4-5-20251001"
    assert captured == {
        "api_key": "environment-key-sentinel",
        "max_retries": 0,
        "timeout": 60.0,
    }


def test_production_client_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


async def test_request_mapping_usage_and_request_id() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(json.loads(request.content))
        return _response(['{"summary":', '"joined","findings":[]}'])

    client = AnthropicClient(sdk_client=_sdk(handler))
    raw = await client.complete(
        [
            {"role": "system", "content": "role"},
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "dynamic"},
            {"role": "assistant", "content": "repair-source"},
            {"role": "user", "content": "repair-feedback"},
        ],
        "TelemetryReport",
        stage="telemetry",
        agent="telemetry_analyst",
        attempt=1,
    )

    assert raw == '{"summary":"joined","findings":[]}'
    assert len(captured) == 1
    output_config = captured[0].pop("output_config")
    assert captured[0] == {
        "max_tokens": 2048,
        "messages": [
            {"content": "dynamic", "role": "user"},
            {"content": "repair-source", "role": "assistant"},
            {"content": "repair-feedback", "role": "user"},
        ],
        "model": API_MODEL,
        "system": [
            {"text": "role", "type": "text"},
            {"text": "tools", "type": "text"},
        ],
    }
    assert output_config["format"]["type"] == "json_schema"
    call = client.calls[0]
    assert call.request_id == "req_test"
    assert call.input_tokens == 23
    assert call.output_tokens == 11
    assert call.cache_creation_input_tokens == 7
    assert call.cache_read_input_tokens == 5
    assert call.transport_retries == ()


async def test_contract_schema_is_sent_as_structured_output() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(json.loads(request.content))
        return _response(['{"findings":[],"summary":"ok"}'])

    client = AnthropicClient(sdk_client=_sdk(handler))
    await client.complete(
        [{"role": "user", "content": "payload"}],
        "TelemetryReport",
        stage="telemetry",
        agent="telemetry_analyst",
        attempt=0,
    )

    output_format = captured[0]["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["additionalProperties"] is False
    assert set(output_format["schema"]["required"]) == {"findings", "summary"}


@pytest.mark.parametrize(
    "schema_name",
    ["TelemetryReport", "PolicyPlan", "ValidationVerdict", "ExecutionResult"],
)
def test_all_provider_schemas_use_supported_union_keyword(schema_name: str) -> None:
    request = build_request([], schema_name)
    output_config = json.dumps(request["output_config"], sort_keys=True)
    assert '"oneOf"' not in output_config
    assert '"discriminator"' not in output_config


def test_cache_breakpoint_is_only_on_last_stable_system_block() -> None:
    request = build_request(
        [
            {"role": "system", "content": "role"},
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "dynamic"},
        ],
        "ExecutionResult",
        cache_prefix=True,
    )
    assert request["system"] == [
        {"text": "role", "type": "text"},
        {
            "cache_control": {"type": "ephemeral"},
            "text": "tools",
            "type": "text",
        },
    ]


async def test_cache_is_sent_only_for_measured_stage() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(json.loads(request.content))
        if len(captured) == 1:
            return _response(['{"findings":[],"summary":"ok"}'])
        return _response(['{"applied":[],"skipped":[],"post_metrics":{}}'])

    client = AnthropicClient(sdk_client=_sdk(handler), cache_stages=frozenset({"execution"}))
    messages = [
        {"role": "system", "content": "role"},
        {"role": "system", "content": "tools"},
        {"role": "user", "content": "dynamic"},
    ]
    await client.complete(
        messages,
        "TelemetryReport",
        stage="telemetry",
        agent="telemetry_analyst",
        attempt=0,
    )
    await client.complete(
        messages,
        "ExecutionResult",
        stage="execution",
        agent="action_executor",
        attempt=0,
    )

    assert all("cache_control" not in block for block in captured[0]["system"])
    assert captured[1]["system"][-1]["cache_control"] == {"type": "ephemeral"}


async def test_retry_after_and_backoff_are_project_owned() -> None:
    responses = [
        httpx2.Response(
            429,
            headers={"request-id": "req_rate", "retry-after": "0.25"},
            json={"error": {"message": "secret provider body", "type": "rate_limit_error"}},
        ),
        httpx2.Response(
            503,
            headers={"request-id": "req_busy"},
            json={"error": {"message": "still secret", "type": "api_error"}},
        ),
        _response(['{"findings":[],"summary":"ok"}']),
    ]
    sleeps: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        return responses.pop(0)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = AnthropicClient(sdk_client=_sdk(handler), sleep=sleep)
    await client.complete(
        [{"role": "user", "content": "payload"}],
        "TelemetryReport",
        stage="telemetry",
        agent="telemetry_analyst",
        attempt=0,
    )

    assert sleeps == [0.25, 2.0]
    retries = client.calls[0].transport_retries
    assert [retry.transport_attempt for retry in retries] == [0, 1]
    assert [retry.status_code for retry in retries] == [429, 503]
    assert [retry.request_id for retry in retries] == ["req_rate", "req_busy"]


@pytest.mark.parametrize("failure", ["connection", "timeout"])
async def test_connection_and_timeout_are_retryable(failure: str) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "connection":
                raise httpx2.ConnectError("offline", request=request)
            raise httpx2.ReadTimeout("slow", request=request)
        return _response(['{"findings":[],"summary":"ok"}'])

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = AnthropicClient(sdk_client=_sdk(handler), sleep=sleep)
    await client.complete(
        [{"role": "user", "content": "payload"}],
        "TelemetryReport",
        stage="telemetry",
        agent="telemetry_analyst",
        attempt=0,
    )

    assert attempts == 2
    assert sleeps == [1.0]
    assert client.calls[0].transport_retries[0].status_code is None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_non_retryable_status_fails_without_sleep(status: int) -> None:
    sleeps: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            headers={"request-id": "req_bad"},
            json={"error": {"message": "body-must-not-escape", "type": "invalid_request_error"}},
        )

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = AnthropicClient(sdk_client=_sdk(handler), sleep=sleep)
    with pytest.raises(ProviderTransportError) as captured:
        await client.complete(
            [{"role": "user", "content": "payload"}],
            "TelemetryReport",
            stage="telemetry",
            agent="telemetry_analyst",
            attempt=0,
        )
    assert sleeps == []
    assert captured.value.status_code == status
    assert "body-must-not-escape" not in str(captured.value)


async def test_retry_after_over_limit_stops_without_sleep() -> None:
    sleeps: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429,
            headers={"request-id": "req_later", "retry-after": "61"},
            json={"error": {"message": "later", "type": "rate_limit_error"}},
        )

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = AnthropicClient(sdk_client=_sdk(handler), sleep=sleep)
    with pytest.raises(ProviderTransportError):
        await client.complete(
            [{"role": "user", "content": "payload"}],
            "TelemetryReport",
            stage="telemetry",
            agent="telemetry_analyst",
            attempt=0,
        )
    assert sleeps == []


async def test_transport_retry_does_not_consume_schema_attempt() -> None:
    responses = [
        httpx2.Response(
            503,
            json={"error": {"message": "busy", "type": "api_error"}},
        ),
        _response(["{"]),
        _response(['{"findings":[],"summary":"repaired"}']),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return responses.pop(0)

    async def sleep(delay: float) -> None:
        assert delay == 1.0

    client = AnthropicClient(sdk_client=_sdk(handler), sleep=sleep)
    parsed, calls = await complete_with_repair(
        client,
        [{"role": "user", "content": "payload"}],
        "TelemetryReport",
        TelemetryReport,
        stage="telemetry",
        agent="telemetry_analyst",
    )
    assert parsed.summary == "repaired"
    assert [call.attempt for call in calls] == [0, 1]
    assert [len(call.transport_retries) for call in calls] == [1, 0]


async def test_count_tokens_uses_same_provider_mapping_without_generation_fields() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/messages/count_tokens"
        captured.append(json.loads(request.content))
        return httpx2.Response(200, json={"input_tokens": 123})

    client = AnthropicClient(sdk_client=_sdk(handler))
    count = await client.count_prefix_tokens(
        [
            {"role": "system", "content": "role"},
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "dynamic"},
        ],
        "TelemetryReport",
    )
    assert count == 123
    assert len(captured) == 1
    output_config = captured[0].pop("output_config")
    assert captured[0] == {
        "messages": [{"content": "dynamic", "role": "user"}],
        "model": API_MODEL,
        "system": [
            {"text": "role", "type": "text"},
            {"text": "tools", "type": "text"},
        ],
    }
    assert output_config["format"]["type"] == "json_schema"
