"""Resume a durable orchestration run without repeating completed model stages.

An unfinished stage may call the model again. This is deterministic for ``FakeModel``
because its draw is a function of ``(seed, canonical_prompt, attempt)`` and contains no
instance call index or other process-local history.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from orchestrator.contracts import (
    Action,
    ExecutionResult,
    PolicyPlan,
    StrictContract,
    TelemetryReport,
    ValidationVerdict,
)
from orchestrator.graph import RejectedGrade, RunContext, RunResult, run_graph
from orchestrator.journal import Journal, canonical_json, network_state_from_json
from orchestrator.llm.anthropic import MODEL_ID, AnthropicClient
from orchestrator.llm.base import LLMClient
from orchestrator.llm.fake import FakeModel
from sim.grader import GradeResult, grade
from sim.network import apply, metrics
from sim.oracle_model import OracleModel
from sim.scenarios import SCENARIOS_BY_ID

_ACTION_ADAPTER: TypeAdapter[Any] = TypeAdapter(Action)
_OUTPUT_MODELS: dict[str, type[StrictContract]] = {
    "telemetry": TelemetryReport,
    "policy": PolicyPlan,
    "validation": ValidationVerdict,
    "execution": ExecutionResult,
}


def _model_from_name(name: str) -> LLMClient:
    if name == "fake:scripted":
        return FakeModel(modes=("scripted",))
    if name == "oracle:standard":
        return OracleModel("standard")
    if name == "oracle:trap":
        return OracleModel("trap")
    if name == MODEL_ID:
        return AnthropicClient()
    raise ValueError(f"run uses an unsupported resumable model: {name}")


def _load(
    db_path: str | Path, run_id: str
) -> tuple[str, str, str, str, dict[str, StrictContract], list[Any]]:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT scenario_id, status, state_json, model FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        scenario_id, status, state_json, model_name = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
        )

        completed: dict[str, StrictContract] = {}
        event_rows = connection.execute(
            "SELECT payload FROM events WHERE run_id = ? AND kind = 'stage_finished' ORDER BY seq",
            (run_id,),
        ).fetchall()
        for (payload_text,) in event_rows:
            payload: Any = json.loads(str(payload_text))
            if not isinstance(payload, dict) or not isinstance(payload.get("stage"), str):
                raise ValueError("stage_finished payload is incomplete")
            stage = str(payload["stage"])
            try:
                model = _OUTPUT_MODELS[stage]
            except KeyError as exc:
                raise ValueError(f"unknown completed stage: {stage}") from exc
            completed[stage] = model.model_validate(payload.get("output"))

        action_rows = connection.execute(
            """
            SELECT args_json FROM effects
            WHERE run_id = ? AND stage = 'execution'
            ORDER BY step
            """,
            (run_id,),
        ).fetchall()
        actions = [_ACTION_ADAPTER.validate_json(str(args_json)) for (args_json,) in action_rows]
    return scenario_id, status, state_json, model_name, completed, actions


def _completed_result(
    status: str,
    state_json: str,
    scenario_id: str,
    completed: dict[str, StrictContract],
    actions: list[Any],
) -> RunResult:
    state = network_state_from_json(state_json)
    scenario = SCENARIOS_BY_ID[scenario_id]
    telemetry = cast(TelemetryReport, completed["telemetry"])
    policy = cast(PolicyPlan, completed["policy"])
    validation = cast(ValidationVerdict, completed["validation"])
    execution = cast(ExecutionResult | None, completed.get("execution"))
    if status == "rejected":
        result_grade: GradeResult | RejectedGrade = RejectedGrade(
            success=False,
            reasons=[f"validation_rejected: {item}" for item in validation.violations]
            or ["validation_rejected: no revised actions"],
        )
    else:
        result_grade = grade(state, actions, scenario)
    return RunResult(
        status=status,
        grade=result_grade,
        final_state=state,
        action_log=tuple(actions),
        telemetry=telemetry,
        policy=policy,
        validation=validation,
        execution=execution,
    )


async def _resume_run(db_path: str, run_id: str) -> RunResult:
    scenario_id, status, state_json, model_name, completed, actions = _load(db_path, run_id)
    if status in {"finished", "rejected"}:
        return _completed_result(status, state_json, scenario_id, completed, actions)
    if status != "running":
        raise ValueError(f"run status is not resumable: {status}")

    scenario = SCENARIOS_BY_ID[scenario_id]

    def model_factory() -> LLMClient:
        return _model_from_name(model_name)

    with Journal(db_path) as journal:
        context = RunContext(
            run_id=run_id,
            scenario=scenario,
            state=network_state_from_json(state_json),
            action_log=list(actions),
            journal=journal,
            llm=model_factory(),
            llm_factory=model_factory,
        )
        return await run_graph(
            context,
            apply_action=apply,
            read_metrics=metrics,
            grade_run=grade,
            recovered=completed,
            start_new=False,
        )


def resume_run(db_path: str, run_id: str) -> RunResult:
    """Resume ``run_id`` from its first unfinished stage and return its final result."""

    return asyncio.run(_resume_run(db_path, run_id))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = resume_run(args.db, args.run_id)
    print(
        canonical_json(
            {
                "action_count": len(result.action_log),
                "status": result.status,
                "success": result.grade.success,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
