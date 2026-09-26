from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

import orchestrator.resume as resume_module
from orchestrator.graph import RunContext, run_graph
from orchestrator.journal import Journal
from orchestrator.llm.anthropic import MODEL_ID
from orchestrator.llm.base import LLMCall, LLMClient, Message
from orchestrator.resume import resume_run
from sim.grader import grade
from sim.network import apply, metrics, state_hash
from sim.oracle_model import OracleModel
from sim.oracles import ORACLES
from sim.scenarios import SCENARIOS_BY_ID


class InterruptingOracle(LLMClient):
    def __init__(self, fail_stage: str | None) -> None:
        self.fail_stage = fail_stage
        self.delegate = OracleModel("standard")

    @property
    def model(self) -> str:
        return self.delegate.model

    @property
    def calls(self) -> Sequence[LLMCall]:
        return self.delegate.calls

    async def complete(
        self,
        messages: Sequence[Message],
        schema_name: str,
        *,
        stage: str,
        agent: str,
        attempt: int,
    ) -> str:
        if stage == self.fail_stage:
            raise RuntimeError(f"stop before {stage}")
        return await self.delegate.complete(
            messages,
            schema_name,
            stage=stage,
            agent=agent,
            attempt=attempt,
        )


class RecordingOracle(OracleModel):
    def __init__(self) -> None:
        super().__init__("standard")
        self.stages: list[str] = []

    async def complete(self, messages, schema_name, *, stage, agent, attempt):
        self.stages.append(stage)
        return await super().complete(
            messages,
            schema_name,
            stage=stage,
            agent=agent,
            attempt=attempt,
        )


async def _partial_run(path: Path, cut_after: str) -> None:
    scenario = SCENARIOS_BY_ID["coverage_hole_01"]
    fail_stage = {
        "telemetry": "policy",
        "policy": "validation",
        "validation": "execution",
        "execution": None,
    }[cut_after]
    client = InterruptingOracle(fail_stage)

    def grade_or_stop(state, actions, current_scenario):
        if cut_after == "execution":
            raise RuntimeError("stop after execution")
        return grade(state, actions, current_scenario)

    with Journal(path) as journal:
        context = RunContext(
            run_id="resume-run",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=client,
            llm_factory=lambda: InterruptingOracle(fail_stage),
        )
        with pytest.raises(RuntimeError, match="stop"):
            await run_graph(
                context,
                apply_action=apply,
                read_metrics=metrics,
                grade_run=grade_or_stop,
            )


@pytest.mark.parametrize(
    ("cut_after", "expected_calls"),
    [("telemetry", 3), ("policy", 2), ("validation", 1), ("execution", 0)],
)
def test_resume_from_each_stage_uses_only_missing_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cut_after: str,
    expected_calls: int,
) -> None:
    database = tmp_path / f"{cut_after}.sqlite"
    asyncio.run(_partial_run(database, cut_after))
    recorder = RecordingOracle()
    monkeypatch.setattr(resume_module, "_model_from_name", lambda _: recorder)

    result = resume_run(str(database), "resume-run")
    expected = SCENARIOS_BY_ID["coverage_hole_01"].initial_state
    for action in ORACLES["coverage_hole_01"].standard:
        expected = apply(expected, action)
    assert state_hash(result.final_state) == state_hash(expected)
    assert result.grade.success
    assert len(recorder.stages) == expected_calls
    expected_stages = ["policy", "validation", "execution"]
    assert recorder.stages == (expected_stages[-expected_calls:] if expected_calls else [])


def test_resume_replays_execution_but_skips_already_committed_effects(tmp_path: Path) -> None:
    database = tmp_path / "effects.sqlite"
    scenario = SCENARIOS_BY_ID["coverage_hole_01"]
    with Journal(database) as journal:
        journal.create_run(
            "effect-run", scenario.id, scenario.initial_state, model="oracle:standard"
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                CREATE TRIGGER fail_execution_finished BEFORE INSERT ON events
                WHEN NEW.kind = 'stage_finished'
                     AND NEW.payload LIKE '%\"stage\":\"execution\"%'
                BEGIN SELECT RAISE(ABORT, 'stop before execution finish'); END
                """
            )
        context = RunContext(
            run_id="effect-run",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=OracleModel("standard"),
            llm_factory=lambda: OracleModel("standard"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="stop before execution finish"):
            asyncio.run(
                run_graph(
                    context,
                    apply_action=apply,
                    read_metrics=metrics,
                    grade_run=grade,
                    start_new=False,
                )
            )
    with sqlite3.connect(database) as connection:
        before = int(connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0])
        connection.execute("DROP TRIGGER fail_execution_finished")

    result = resume_run(str(database), "effect-run")
    with sqlite3.connect(database) as connection:
        after = int(connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0])
        tool_events = int(
            connection.execute("SELECT COUNT(*) FROM events WHERE kind = 'tool_called'").fetchone()[
                0
            ]
        )
    assert before == after == 2
    assert tool_events == after
    assert result.grade.success


def test_resume_of_finished_run_is_database_no_op(tmp_path: Path) -> None:
    database = tmp_path / "finished.sqlite"
    scenario = SCENARIOS_BY_ID["false_alarm_01"]
    with Journal(database) as journal:
        context = RunContext(
            run_id="finished-run",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=OracleModel("standard"),
            llm_factory=lambda: OracleModel("standard"),
        )
        asyncio.run(
            run_graph(
                context,
                apply_action=apply,
                read_metrics=metrics,
                grade_run=grade,
            )
        )
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        runs_row = connection.execute(
            "SELECT state_json, model FROM runs WHERE run_id = ?", ("finished-run",)
        ).fetchone()
    assert runs_row is not None
    assert "_model" not in runs_row["state_json"]
    assert runs_row["model"] == "oracle:standard"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = resume_run(str(database), "finished-run")
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert before == after
    assert result.status == "finished"
    assert result.grade.success


def test_real_provider_run_resumes_from_persisted_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "provider.sqlite"
    asyncio.run(_partial_run(database, "telemetry"))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE runs SET model = ? WHERE run_id = ?",
            (MODEL_ID, "resume-run"),
        )
    recorder = RecordingOracle()
    constructed = 0

    def provider_factory() -> RecordingOracle:
        nonlocal constructed
        constructed += 1
        return recorder

    monkeypatch.setattr(resume_module, "AnthropicClient", provider_factory)
    result = resume_run(str(database), "resume-run")

    assert constructed >= 1
    assert recorder.stages == ["policy", "validation", "execution"]
    assert result.grade.success
