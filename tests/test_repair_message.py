from __future__ import annotations

import json

import pytest

from orchestrator.contracts import PolicyPlan
from orchestrator.repair import build_repair_feedback


def _policy(action: dict[str, object], **extra: object) -> str:
    return json.dumps(
        {
            "actions": [action],
            "expected_effect": "Correct the condition.",
            "rationale": "Act on telemetry.",
            **extra,
        }
    )


def test_malformed_json_feedback_has_position_and_no_json_path() -> None:
    feedback = build_repair_feedback("{", PolicyPlan)
    assert feedback.failure_kind == "malformed"
    assert "not valid JSON" in feedback.message
    assert "line 1" in feedback.message
    assert "column 2" in feedback.message
    assert "character 1" in feedback.message
    assert "JSON path" not in feedback.message


def test_out_of_range_feedback_has_path_full_range_and_actual_value() -> None:
    raw = _policy({"tool": "set_tx_power", "cell_id": "cell-1", "tx_power_dbm": 47.0})
    feedback = build_repair_feedback(raw, PolicyPlan)
    assert feedback.failure_kind == "schema"
    assert "actions.0.tx_power_dbm" in feedback.message
    assert "range 20..46" in feedback.message
    assert "47.0" in feedback.message


def test_unknown_action_feedback_identifies_discriminator_field() -> None:
    raw = _policy({"tool": "unknown_tool", "cell_id": "cell-1"})
    feedback = build_repair_feedback(raw, PolicyPlan)
    assert "actions.0.tool" in feedback.message
    assert "discriminator" in feedback.message
    assert "unknown_tool" in feedback.message


def test_extra_field_feedback_names_the_field() -> None:
    raw = _policy({"tool": "no_op", "reason": "No change.", "unexpected": True})
    feedback = build_repair_feedback(raw, PolicyPlan)
    assert "actions.0.unexpected" in feedback.message
    assert "no extra field" in feedback.message
    assert "true" in feedback.message


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        _policy({"tool": "set_tx_power", "cell_id": "cell-1", "tx_power_dbm": 47.0}),
        _policy({"tool": "unknown_tool", "cell_id": "cell-1"}),
        _policy({"tool": "no_op", "reason": "No change.", "unexpected": True}),
    ],
)
def test_feedback_never_resends_full_schema(raw: str) -> None:
    feedback = build_repair_feedback(raw, PolicyPlan)
    schema = json.dumps(PolicyPlan.model_json_schema(), sort_keys=True)
    assert schema not in feedback.message


def test_feedback_is_much_shorter_than_full_schema() -> None:
    feedback = build_repair_feedback(
        _policy({"tool": "set_tx_power", "cell_id": "cell-1", "tx_power_dbm": 47.0}),
        PolicyPlan,
    )
    schema = json.dumps(PolicyPlan.model_json_schema(), sort_keys=True)
    # The one-half ratio is an input guard, reproduced by this test itself:
    # .\.venv\Scripts\python.exe -m pytest tests/test_repair_message.py
    assert len(feedback.message) * 2 < len(schema)
