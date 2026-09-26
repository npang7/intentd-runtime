from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx2

from orchestrator.contracts import TelemetryReport
from orchestrator.graph import RunContext, _journal_llm_calls
from orchestrator.journal import Journal
from orchestrator.llm.anthropic import API_MODEL, AnthropicClient
from sim.scenarios import SCENARIOS_BY_ID


async def test_usage_request_id_latency_and_retry_enter_journal_without_key(
    tmp_path: Path,
) -> None:
    responses = [
        httpx2.Response(
            429,
            headers={"request-id": "req_retry", "retry-after": "0"},
            json={"error": {"message": "provider-body-sentinel", "type": "rate_limit_error"}},
        ),
        httpx2.Response(
            200,
            headers={"request-id": "req_success"},
            json={
                "content": [{"text": '{"findings":[],"summary":"ok"}', "type": "text"}],
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
        ),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["x-api-key"] == "journal-key-sentinel"
        return responses.pop(0)

    sdk = anthropic.AsyncAnthropic(
        api_key="journal-key-sentinel",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
        timeout=60.0,
    )

    async def no_sleep(delay: float) -> None:
        assert delay == 0.0

    client = AnthropicClient(sdk_client=sdk, sleep=no_sleep)
    await client.complete(
        [{"role": "user", "content": "payload"}],
        "TelemetryReport",
        stage="telemetry",
        agent="telemetry_analyst",
        attempt=0,
    )

    database = tmp_path / "journal.sqlite"
    scenario = SCENARIOS_BY_ID["false_alarm_01"]
    with Journal(database) as journal:
        journal.create_run("provider-run", scenario.id, scenario.initial_state, model=client.model)
        context = RunContext(
            run_id="provider-run",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=client,
            llm_factory=lambda: client,
        )
        _journal_llm_calls(context, "TelemetryReport", TelemetryReport, list(client.calls))
        events = journal.events("provider-run")

    assert [kind for _, kind, _ in events] == ["transport_retry", "llm_call"]
    retry = json.loads(events[0][2])
    assert retry == {
        "error_category": "RateLimitError",
        "request_id": "req_retry",
        "stage": "telemetry",
        "status_code": 429,
        "transport_attempt": 0,
        "wait_s": 0.0,
    }
    call = json.loads(events[1][2])
    assert call["request_id"] == "req_success"
    assert call["input_tokens"] == 23
    assert call["output_tokens"] == 11
    assert call["cache_creation_input_tokens"] == 7
    assert call["cache_read_input_tokens"] == 5
    assert call["transport_retry_count"] == 1
    assert isinstance(call["latency_s"], float)
    raw = database.read_bytes()
    assert b"journal-key-sentinel" not in raw
    assert b"provider-body-sentinel" not in raw
