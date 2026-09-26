from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from bench.analyze import Usage
from bench.run_eval import (
    MODEL_ID,
    SCENARIO_ORDER,
    BudgetExceeded,
    SpendGuard,
    _database_path,
    _eligible_cache_stages,
    scenarios_to_run,
)
from orchestrator.llm.base import LLMCall, Message
from sim.scenarios import ALL_SCENARIOS


def test_representative_scenarios_are_first_and_order_has_no_duplicates() -> None:
    assert SCENARIO_ORDER[:3] == (
        "overload_dense_01",
        "conflicting_sla_01",
        "false_alarm_01",
    )
    assert len(SCENARIO_ORDER) == len(ALL_SCENARIOS)
    assert len(set(SCENARIO_ORDER)) == len(SCENARIO_ORDER)


def test_existing_same_model_and_mode_runs_are_skipped() -> None:
    rows = [
        {
            "commit_sha": "current",
            "mode": "parallel",
            "model": MODEL_ID,
            "record_type": "run",
            "scenario_id": "overload_dense_01",
            "status": "finished",
        },
        {
            "commit_sha": "current",
            "mode": "serial",
            "model": MODEL_ID,
            "record_type": "run",
            "scenario_id": "conflicting_sla_01",
            "status": "finished",
        },
    ]
    assert scenarios_to_run(rows, MODEL_ID, "parallel", 3) == [
        "conflicting_sla_01",
        "false_alarm_01",
    ]


def test_previous_commit_runs_do_not_block_a_fixed_commit() -> None:
    rows = [
        {
            "commit_sha": "previous",
            "mode": "parallel",
            "model": MODEL_ID,
            "record_type": "run",
            "scenario_id": "overload_dense_01",
            "status": "unrepairable",
        }
    ]
    assert scenarios_to_run(rows, MODEL_ID, "parallel", 3) == list(SCENARIO_ORDER[:3])


def test_finished_previous_commit_is_not_repeated_after_evidence_commit() -> None:
    rows = [
        {
            "commit_sha": "provider-run",
            "mode": "parallel",
            "model": MODEL_ID,
            "record_type": "run",
            "scenario_id": "overload_dense_01",
            "status": "finished",
        }
    ]
    assert scenarios_to_run(rows, MODEL_ID, "parallel", 3) == [
        "conflicting_sla_01",
        "false_alarm_01",
    ]


def test_only_current_commit_measured_execution_enables_cache() -> None:
    rows = [
        {
            "cache_eligible": True,
            "commit_sha": "current",
            "model": MODEL_ID,
            "record_type": "prefix_measurement",
            "stage": "execution",
        },
        {
            "cache_eligible": False,
            "commit_sha": "current",
            "model": MODEL_ID,
            "record_type": "prefix_measurement",
            "stage": "policy",
        },
        {
            "cache_eligible": True,
            "commit_sha": "old",
            "model": MODEL_ID,
            "record_type": "prefix_measurement",
            "stage": "policy",
        },
    ]
    assert _eligible_cache_stages(rows, MODEL_ID, "current") == frozenset({"execution"})


def test_database_parent_is_created_before_journal_open(tmp_path: Path) -> None:
    output = tmp_path / "results" / "runs.jsonl"
    database = _database_path(output, "run-id")
    assert database == tmp_path / "results" / "runs" / "run-id.sqlite"
    assert database.parent.is_dir()


class _CountingClient:
    def __init__(self, input_tokens: int) -> None:
        self.input_tokens = input_tokens

    async def count_prefix_tokens(self, messages: Sequence[Message], schema_name: str) -> int:
        return self.input_tokens


async def test_spend_guard_reserves_before_attempt_and_reconciles_success() -> None:
    guard = SpendGuard(1.0)
    reservation = await guard.reserve(
        _CountingClient(1_000),  # type: ignore[arg-type]
        [{"role": "user", "content": "payload"}],
        "TelemetryReport",
        0,
    )
    assert guard.unknown_attempt_reserve_usd == reservation
    call = LLMCall(
        "telemetry",
        "agent",
        0,
        0.1,
        "{}",
        True,
        MODEL_ID,
        input_tokens=10,
        output_tokens=20,
    )
    guard.success(reservation, call)
    assert guard.unknown_attempt_reserve_usd == pytest.approx(0.0)
    assert guard.known_usage == Usage(input_tokens=10, output_tokens=20)


async def test_spend_guard_rejects_worst_case_attempt_over_limit() -> None:
    guard = SpendGuard(0.000001)
    with pytest.raises(BudgetExceeded):
        await guard.reserve(
            _CountingClient(1_000),  # type: ignore[arg-type]
            [{"role": "user", "content": "payload"}],
            "TelemetryReport",
            0,
        )
