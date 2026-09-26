from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from orchestrator.contracts import (
    Action,
    ExecutionResult,
    PolicyPlan,
    TelemetryReport,
    ValidationVerdict,
    response_parses,
)
from orchestrator.llm.base import Message
from orchestrator.llm.fake import FakeModel, cassette_key

ROOT = Path(__file__).resolve().parent.parent
ACTION_ADAPTER = TypeAdapter(Action)


VALID_ACTIONS = [
    {"tool": "get_metrics", "cell_ids": ["cell-1"], "window_s": 60},
    {"tool": "set_tx_power", "cell_id": "cell-1", "tx_power_dbm": 42.0},
    {"tool": "set_a3_offset", "cell_id": "cell-1", "a3_offset_db": 2.0},
    {"tool": "set_tilt", "cell_id": "cell-1", "tilt_deg": 8.0},
    {"tool": "set_admission_threshold", "cell_id": "cell-1", "threshold": 0.8},
    {
        "tool": "report_non_radio_issue",
        "cell_id": "cell-1",
        "category": "backhaul",
        "detail": "Congestion detected.",
    },
    {"tool": "no_op", "reason": "All metrics are within policy."},
]


@pytest.mark.parametrize("action", VALID_ACTIONS)
def test_every_action_variant_parses(action: dict[str, object]) -> None:
    assert ACTION_ADAPTER.validate_python(action).tool == action["tool"]


def test_top_level_contracts_parse() -> None:
    report = TelemetryReport.model_validate(
        {
            "findings": [{"cell_id": "cell-1", "metric": "rtt", "detail": "Above policy."}],
            "summary": "One issue found.",
        }
    )
    plan = PolicyPlan.model_validate(
        {
            "actions": VALID_ACTIONS,
            "rationale": "Act on the reported condition.",
            "expected_effect": "The condition is corrected.",
        }
    )
    verdict = ValidationVerdict.model_validate(
        {"approved": True, "violations": [], "revised_actions": []}
    )
    result = ExecutionResult.model_validate(
        {"applied": [VALID_ACTIONS[-1]], "skipped": [], "post_metrics": {"rtt": 12.5}}
    )
    assert report.findings[0].cell_id == "cell-1"
    assert len(plan.actions) == len(VALID_ACTIONS)
    assert verdict.approved is True
    assert result.post_metrics["rtt"] == 12.5


def test_findings_may_be_empty_but_actions_may_not() -> None:
    assert TelemetryReport.model_validate({"findings": [], "summary": "No issue."}).findings == []
    with pytest.raises(ValidationError):
        PolicyPlan.model_validate(
            {"actions": [], "rationale": "No issue.", "expected_effect": "No change."}
        )


def test_post_metrics_reject_numeric_strings() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(
            {"applied": [], "skipped": [], "post_metrics": {"rtt": "12.5"}}
        )


def test_float_action_field_accepts_integer() -> None:
    action = {**VALID_ACTIONS[1], "tx_power_dbm": 42}
    assert ACTION_ADAPTER.validate_python(action).tx_power_dbm == 42.0


def test_integer_action_field_rejects_float() -> None:
    action = {**VALID_ACTIONS[0], "window_s": 60.0}
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python(action)


def test_float_action_field_rejects_bool() -> None:
    action = {**VALID_ACTIONS[1], "tx_power_dbm": True}
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python(action)


@pytest.mark.parametrize(
    ("action", "field", "integer_value"),
    [
        (VALID_ACTIONS[0], "window_s", 60),
        (VALID_ACTIONS[1], "tx_power_dbm", 42),
        (VALID_ACTIONS[2], "a3_offset_db", 2),
        (VALID_ACTIONS[3], "tilt_deg", 8),
        (VALID_ACTIONS[4], "threshold", 1),
    ],
)
def test_every_action_numeric_field_accepts_valid_integer(
    action: dict[str, object], field: str, integer_value: int
) -> None:
    parsed = ACTION_ADAPTER.validate_python({**action, field: integer_value})
    assert getattr(parsed, field) == integer_value


@pytest.mark.parametrize(
    ("action", "field", "bad_value"),
    [
        (VALID_ACTIONS[0], "window_s", 0),
        (VALID_ACTIONS[0], "window_s", 3601),
        (VALID_ACTIONS[1], "tx_power_dbm", 19.9),
        (VALID_ACTIONS[1], "tx_power_dbm", 46.1),
        (VALID_ACTIONS[2], "a3_offset_db", -10.1),
        (VALID_ACTIONS[2], "a3_offset_db", 10.1),
        (VALID_ACTIONS[3], "tilt_deg", -0.1),
        (VALID_ACTIONS[3], "tilt_deg", 15.1),
        (VALID_ACTIONS[4], "threshold", -0.1),
        (VALID_ACTIONS[4], "threshold", 1.1),
    ],
)
def test_action_numeric_bounds(
    action: dict[str, object], field: str, bad_value: int | float
) -> None:
    invalid = {**action, field: bad_value}
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python(invalid)


@pytest.mark.parametrize(
    ("action", "field", "string_value"),
    [
        (VALID_ACTIONS[0], "window_s", "60"),
        (VALID_ACTIONS[1], "tx_power_dbm", "42"),
        (VALID_ACTIONS[2], "a3_offset_db", "2"),
        (VALID_ACTIONS[3], "tilt_deg", "8"),
        (VALID_ACTIONS[4], "threshold", "0.8"),
    ],
)
def test_action_numeric_strings_are_rejected(
    action: dict[str, object], field: str, string_value: str
) -> None:
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({**action, field: string_value})


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_ACTIONS[1], "unexpected": True},
        {"tool": "unknown_tool", "cell_id": "cell-1"},
    ],
)
def test_action_rejects_extra_fields_and_unknown_tools(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python(payload)


def test_report_non_radio_issue_rejects_unknown_category() -> None:
    action = {**VALID_ACTIONS[5], "category": "wifi"}
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python(action)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (TelemetryReport, {"findings": [], "summary": "Fine.", "unexpected": True}),
        (
            PolicyPlan,
            {
                "actions": [VALID_ACTIONS[-1]],
                "rationale": "Fine.",
                "expected_effect": "No change.",
                "unexpected": True,
            },
        ),
        (
            ValidationVerdict,
            {"approved": True, "violations": [], "revised_actions": [], "unexpected": True},
        ),
        (
            ExecutionResult,
            {"applied": [], "skipped": [], "post_metrics": {}, "unexpected": True},
        ),
    ],
)
def test_top_level_contracts_reject_extra_fields(
    model: type[TelemetryReport | PolicyPlan | ValidationVerdict | ExecutionResult],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def _messages(index: int) -> list[Message]:
    return [{"role": "user", "content": f"request-{index}"}]


async def _complete(model: FakeModel, index: int) -> str:
    return await model.complete(
        _messages(index),
        "PolicyPlan",
        stage="policy",
        agent="planner",
        attempt=0,
    )


@pytest.mark.asyncio
async def test_zero_invalid_rates_produce_only_valid_responses() -> None:
    model = FakeModel(modes=("scripted", "adversarial"), seed="zero")
    responses = [await _complete(model, index) for index in range(1000)]
    assert all(response_parses("PolicyPlan", response) for response in responses)


@pytest.mark.asyncio
async def test_malformed_rate_one_produces_only_malformed_json() -> None:
    model = FakeModel(modes=("scripted", "adversarial"), seed="malformed-all", malformed_rate=1.0)
    responses = [await _complete(model, index) for index in range(1000)]
    for response in responses:
        with pytest.raises(json.JSONDecodeError):
            json.loads(response)


@pytest.mark.asyncio
async def test_schema_violation_rate_one_produces_only_schema_invalid_json() -> None:
    model = FakeModel(
        modes=("scripted", "adversarial"),
        seed="schema-all",
        schema_violation_rate=1.0,
    )
    responses = [await _complete(model, index) for index in range(1000)]
    assert all(isinstance(json.loads(response), dict) for response in responses)
    assert not any(response_parses("PolicyPlan", response) for response in responses)


@pytest.mark.parametrize("kind", ["malformed", "schema"])
@pytest.mark.asyncio
async def test_invalid_rates_are_calibrated(kind: str) -> None:
    model = FakeModel(
        modes=("scripted", "adversarial"),
        seed=f"calibration-{kind}",
        malformed_rate=0.3 if kind == "malformed" else 0.0,
        schema_violation_rate=0.3 if kind == "schema" else 0.0,
    )
    responses = [await _complete(model, index) for index in range(1000)]
    if kind == "malformed":
        invalid = sum(1 for response in responses if response == "{")
    else:
        invalid = sum(1 for response in responses if not response_parses("PolicyPlan", response))
    empirical_rate = invalid / len(responses)
    assert 0.27 <= empirical_rate <= 0.33


@pytest.mark.asyncio
async def test_cassette_replays_by_sha256_prompt_key(tmp_path: Path) -> None:
    messages = _messages(7)
    response = json.dumps(
        {
            "actions": [{"tool": "no_op", "reason": "Recorded response."}],
            "rationale": "Recorded response.",
            "expected_effect": "No change.",
        }
    )
    path = tmp_path / "cassette.json"
    path.write_text(json.dumps({cassette_key(messages, "PolicyPlan"): response}), encoding="utf-8")
    model = FakeModel(modes=("cassette",), cassette_path=path)
    actual = await model.complete(
        messages,
        "PolicyPlan",
        stage="policy",
        agent="planner",
        attempt=0,
    )
    assert actual == response
    assert model.calls[0].parsed_ok is True


@pytest.mark.asyncio
async def test_cassette_missing_internal_data_raises_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    path.write_text("{}", encoding="utf-8")
    model = FakeModel(modes=("cassette",), cassette_path=path)
    model._cassette = None
    with pytest.raises(RuntimeError, match="no loaded cassette data"):
        await _complete(model, 0)


@pytest.mark.asyncio
async def test_modes_compose_and_call_record_is_filled() -> None:
    model = FakeModel(
        modes=("latency", "adversarial", "scripted"),
        seed="composed",
        latency_s=0.0,
    )
    raw = await model.complete(
        [{"role": "user", "content": "Keep this cell unchanged."}],
        "PolicyPlan",
        stage="policy",
        agent="planner",
        attempt=2,
    )
    call = model.calls[0]
    assert model.model == "fake:scripted+adversarial+latency"
    assert call.stage == "policy"
    assert call.agent == "planner"
    assert call.attempt == 2
    assert call.raw_response == raw
    assert call.parsed_ok is True
    assert call.model == model.model


@pytest.mark.asyncio
async def test_latency_mode_requests_configured_sleep() -> None:
    requested: list[float] = []

    async def recorder(delay: float) -> None:
        requested.append(delay)

    model = FakeModel(modes=("scripted", "latency"), latency_s=0.25, sleep=recorder)
    await _complete(model, 0)
    assert requested == [0.25]


@pytest.mark.asyncio
async def test_without_latency_mode_no_sleep_is_requested() -> None:
    requested: list[float] = []

    async def recorder(delay: float) -> None:
        requested.append(delay)

    model = FakeModel(modes=("scripted",), sleep=recorder)
    await _complete(model, 0)
    assert requested == []


@pytest.mark.asyncio
async def test_draw_ignores_instance_call_history() -> None:
    """Use only logical-call identity when drawing adversarial outcomes.

    The former call_index input made recovery produce a different response for
    the same prompt because a new process resets instance history.
    """
    kwargs = {
        "modes": ("scripted", "adversarial"),
        "seed": "history",
        "malformed_rate": 0.3,
        "schema_violation_rate": 0.3,
    }
    fresh = FakeModel(**kwargs)
    warmed = FakeModel(**kwargs)
    for index in range(10):
        await _complete(warmed, index)
    for index in range(100, 120):
        assert await _complete(fresh, index) == await _complete(warmed, index)


@pytest.mark.asyncio
async def test_draw_depends_on_attempt() -> None:
    model = FakeModel(
        modes=("scripted", "adversarial"),
        seed="attempt",
        malformed_rate=0.3,
        schema_violation_rate=0.3,
    )
    messages = _messages(1)
    outputs = {
        await model.complete(
            messages,
            "PolicyPlan",
            stage="policy",
            agent="planner",
            attempt=attempt,
        )
        for attempt in range(20)
    }
    assert len(outputs) > 1


@pytest.mark.asyncio
async def test_adversarial_draw_depends_on_attempt_not_instance_history() -> None:
    used_model = FakeModel(modes=("scripted", "adversarial"), seed="m4", malformed_rate=0.5)
    fresh_model = FakeModel(modes=("scripted", "adversarial"), seed="m4", malformed_rate=0.5)
    retry_model = FakeModel(modes=("scripted", "adversarial"), seed="m4", malformed_rate=0.5)
    await _complete(used_model, 999)

    messages: list[Message] = [{"role": "user", "content": "logical-call"}]
    used_response = await used_model.complete(
        messages,
        "PolicyPlan",
        stage="policy",
        agent="planner",
        attempt=0,
    )
    fresh_response = await fresh_model.complete(
        messages,
        "PolicyPlan",
        stage="policy",
        agent="planner",
        attempt=0,
    )
    retry_response = await retry_model.complete(
        messages,
        "PolicyPlan",
        stage="policy",
        agent="planner",
        attempt=1,
    )

    assert used_response == fresh_response
    assert retry_response != fresh_response


def test_same_seed_is_byte_identical_across_processes() -> None:
    program = """
import asyncio
import json
from orchestrator.llm.fake import FakeModel

async def main():
    model = FakeModel(
        modes=("scripted", "adversarial"),
        seed="cross-process",
        malformed_rate=0.2,
        schema_violation_rate=0.3,
    )
    outputs = []
    for index in range(64):
        outputs.append(await model.complete(
            [{"role": "user", "content": f"request-{index}"}],
            "PolicyPlan",
            stage="policy",
            agent="planner",
            attempt=0,
        ))
    print(json.dumps(outputs, ensure_ascii=False, separators=(",", ":")))

asyncio.run(main())
"""

    def run(hash_seed: str) -> bytes:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    assert run("1") == run("987654")
