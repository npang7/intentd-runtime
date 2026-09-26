from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from examples.inspect_journal import inspect
from examples.recovery_demo import run_demo
from orchestrator.graph import RunContext, run_graph
from orchestrator.journal import Journal
from sim.grader import grade
from sim.network import apply, metrics
from sim.oracle_model import OracleModel
from sim.scenarios import SCENARIOS_BY_ID


def test_inspection_reads_actions_without_changing_database(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite"
    scenario = SCENARIOS_BY_ID["coverage_hole_01"]
    with Journal(database) as journal:
        context = RunContext(
            run_id="demo",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=OracleModel("standard"),
            llm_factory=lambda: OracleModel("standard"),
        )
        result = asyncio.run(
            run_graph(context, apply_action=apply, read_metrics=metrics, grade_run=grade)
        )
        assert result.grade.success
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    summary = inspect(database, "demo")
    assert summary["status"] == "finished"
    assert summary["effects"] == 2
    assert [action["tool"] for action in summary["actions"]] == ["set_tx_power", "set_tilt"]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        inspect(database, "demo")
    assert not database.exists()


def test_unknown_run_does_not_change_database(tmp_path: Path) -> None:
    database = tmp_path / "empty.sqlite"
    with Journal(database):
        pass
    with pytest.raises(KeyError, match="unknown run"):
        inspect(database, "missing")


def test_recovery_demo_checks_committed_action_and_repeated_resume() -> None:
    result = run_demo()
    assert result["exit_status"] != 0
    assert result["committed_before_resume"] == 1
    assert result["effects_after_resume"] == 2
    assert result["state_matches_reference"] is True
    assert result["repeat_resume_added_effects"] == 0
