from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orchestrator.contracts import SetTxPower
from orchestrator.journal import Journal
from orchestrator.tools.ran_tools import run_tool
from sim.network import apply, metrics
from sim.scenarios import SCENARIOS_BY_ID


def _counts(path: Path) -> tuple[str, int, int]:
    with sqlite3.connect(path) as connection:
        state_json = str(connection.execute("SELECT state_json FROM runs").fetchone()[0])
        effects = int(connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0])
        events = int(
            connection.execute("SELECT COUNT(*) FROM events WHERE kind = 'tool_called'").fetchone()[
                0
            ]
        )
    return state_json, effects, events


def test_state_effect_and_event_all_roll_back_when_event_write_fails(tmp_path: Path) -> None:
    database = tmp_path / "atomic.sqlite"
    initial = SCENARIOS_BY_ID["coverage_hole_01"].initial_state
    action = SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0)
    with Journal(database) as journal:
        journal.create_run("atomic", "coverage_hole_01", initial, model="oracle:standard")
        before = _counts(database)
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                CREATE TRIGGER fail_tool_event BEFORE INSERT ON events
                WHEN NEW.kind = 'tool_called'
                BEGIN SELECT RAISE(ABORT, 'injected event failure'); END
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="injected event failure"):
            run_tool(
                initial,
                action,
                [],
                apply_action=apply,
                read_metrics=metrics,
                journal=journal,
                run_id="atomic",
                step=0,
            )
        after = _counts(database)
    assert after == before


def test_successful_tool_call_persists_all_three_records(tmp_path: Path) -> None:
    database = tmp_path / "committed.sqlite"
    initial = SCENARIOS_BY_ID["coverage_hole_01"].initial_state
    action = SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0)
    with Journal(database) as journal:
        journal.create_run("committed", "coverage_hole_01", initial, model="oracle:standard")
        before = _counts(database)
        run_tool(
            initial,
            action,
            [],
            apply_action=apply,
            read_metrics=metrics,
            journal=journal,
            run_id="committed",
            step=0,
        )
        after = _counts(database)
    assert after[0] != before[0]
    assert after[1:] == (1, 1)
