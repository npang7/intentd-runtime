from __future__ import annotations

import argparse
import json

import pytest

from orchestrator.contracts import PolicyPlan
from orchestrator.llm.base import Message
from orchestrator.llm.fake import FakeModel
from orchestrator.repair import UnrepairableStageError, complete_with_repair
from orchestrator.run import _run


def _messages(label: str = "repair") -> list[Message]:
    return [{"role": "user", "content": label}]


async def _complete(model: FakeModel, *, max_repair: int = 2):
    return await complete_with_repair(
        model,
        _messages(),
        "PolicyPlan",
        PolicyPlan,
        stage="policy",
        agent="planner",
        max_repair=max_repair,
    )


@pytest.mark.parametrize(
    "model",
    [
        FakeModel(modes=("scripted", "adversarial"), malformed_rate=1.0),
        FakeModel(modes=("scripted", "adversarial"), schema_violation_rate=1.0),
    ],
    ids=["malformed", "schema"],
)
async def test_permanently_invalid_output_exhausts_exact_attempt_limit(
    model: FakeModel,
) -> None:
    with pytest.raises(UnrepairableStageError) as raised:
        await _complete(model)
    assert raised.value.stage == "policy"
    assert raised.value.attempts == 3
    assert raised.value.last_error
    assert [call.attempt for call in model.calls] == [0, 1, 2]
    assert [call.parsed_ok for call in model.calls] == [False, False, False]


async def test_valid_output_calls_model_once() -> None:
    model = FakeModel(modes=("scripted",))
    parsed, calls = await _complete(model)
    assert isinstance(parsed, PolicyPlan)
    assert [call.attempt for call in calls] == [0]
    assert [call.parsed_ok for call in calls] == [True]


async def test_zero_repairs_raises_after_first_invalid_output() -> None:
    model = FakeModel(modes=("scripted", "adversarial"), malformed_rate=1.0)
    with pytest.raises(UnrepairableStageError) as raised:
        await _complete(model, max_repair=0)
    assert raised.value.attempts == 1
    assert [call.attempt for call in model.calls] == [0]


async def test_repair_messages_are_appended_after_the_raw_assistant_reply() -> None:
    seen: list[list[Message]] = []

    class InspectingFake(FakeModel):
        async def complete(self, messages, schema_name, **kwargs):
            seen.append([dict(message) for message in messages])
            return await super().complete(messages, schema_name, **kwargs)

    model = InspectingFake(modes=("scripted", "adversarial"), malformed_rate=1.0)
    with pytest.raises(UnrepairableStageError):
        await _complete(model, max_repair=1)
    assert [message["role"] for message in seen[1][-2:]] == ["assistant", "user"]
    assert seen[1][-2]["content"] == "{"
    assert "not valid JSON" in seen[1][-1]["content"]


async def test_negative_repair_limit_is_rejected_before_calling_model() -> None:
    model = FakeModel(modes=("scripted",))
    with pytest.raises(ValueError, match="nonnegative"):
        await _complete(model, max_repair=-1)
    assert model.calls == ()


async def test_cli_records_unrepairable_without_grade(tmp_path, monkeypatch, capsys) -> None:
    model = FakeModel(modes=("scripted", "adversarial"), malformed_rate=1.0)
    monkeypatch.setattr("orchestrator.run._model", lambda name: model)
    database = tmp_path / "unrepairable.sqlite"
    exit_code = await _run(
        argparse.Namespace(
            scenario="false_alarm_01",
            model="fake:scripted",
            db=database,
            run_id="unrepairable-run",
        )
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "unrepairable"
    assert output["stage"] == "telemetry"
    assert output["attempts"] == 3
    assert "grade" not in output

    import sqlite3

    with sqlite3.connect(database) as connection:
        kind, payload = connection.execute(
            "SELECT kind, payload FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
    assert kind == "run_finished"
    assert json.loads(payload)["status"] == "unrepairable"
