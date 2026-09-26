"""Validate result records using local runs and synthetic provider usage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.analyze import render_summary
from bench.run_eval import _result_row
from orchestrator.graph import RunContext, run_graph
from orchestrator.journal import Journal
from sim.grader import grade
from sim.network import apply, metrics, state_hash
from sim.oracle_model import OracleModel
from sim.scenarios import SCENARIOS_BY_ID


@pytest.mark.parametrize("mode", ["serial", "parallel"])
async def test_result_record_preserves_raw_usage_and_execution_fields(
    tmp_path: Path, mode: str
) -> None:
    scenario = SCENARIOS_BY_ID["coverage_hole_01"]
    database = tmp_path / "result.sqlite"
    with Journal(database) as journal:
        context = RunContext(
            run_id="record-test",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=OracleModel("standard"),
            llm_factory=lambda: OracleModel("standard"),
            parallel_telemetry=mode == "parallel",
        )
        result = await run_graph(context, apply_action=apply, read_metrics=metrics, grade_run=grade)
        journal.append(
            "record-test",
            "llm_call",
            {
                "stage": "policy",
                "input_tokens": 23,
                "output_tokens": 11,
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 5,
            },
        )
        journal.append("record-test", "repair_attempted", {"stage": "policy"})
        journal.append("record-test", "transport_retry", {"stage": "policy"})
    row = _result_row(
        commit_sha="synthetic-checkout",
        database=database,
        elapsed_s=1.25,
        mode=mode,
        model="synthetic-provider",
        result=result,
        run_id="record-test",
        scenario_id=scenario.id,
    )
    assert json.loads(json.dumps(row)) == row
    assert row["record_type"] == "run"
    assert row["commit_sha"] == "synthetic-checkout"
    assert row["mode"] == mode
    assert row["status"] == "finished"
    assert row["grader"] == {"success": True, "reasons": []}
    assert row["state_hash"] == state_hash(result.final_state)
    assert row["action_log"] == [action.model_dump(mode="json") for action in result.action_log]
    assert row["usage"] == {
        "input_tokens": 23,
        "output_tokens": 11,
        "cache_creation_input_tokens": 7,
        "cache_read_input_tokens": 5,
    }
    assert set(row["stage_latency_s"]) == {"telemetry", "policy", "validation", "execution"}
    assert row["schema_repair_count"] == 1
    assert row["transport_retry_count"] == 1
    assert "cost_usd" not in row
    summary = render_summary([row])
    assert scenario.id in summary
    assert "| 23 | 11 | 7 | 5 |" in summary
