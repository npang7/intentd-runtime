from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.contracts import SetTxPower
from orchestrator.journal import Journal, idem_key
from orchestrator.tools.ran_tools import run_tool
from sim.network import apply, metrics
from sim.scenarios import SCENARIOS_BY_ID

BASE = ("run", "execution", 0, "set_tx_power", {"cell_id": "cell-1", "value": 40.0})


def test_idem_key_is_stable() -> None:
    assert idem_key(*BASE) == idem_key(*BASE)


@pytest.mark.parametrize(
    "changed",
    [
        ("other", BASE[1], BASE[2], BASE[3], BASE[4]),
        (BASE[0], "policy", BASE[2], BASE[3], BASE[4]),
        (BASE[0], BASE[1], 1, BASE[3], BASE[4]),
        (BASE[0], BASE[1], BASE[2], "set_tilt", BASE[4]),
        (BASE[0], BASE[1], BASE[2], BASE[3], {"cell_id": "cell-1", "value": 41.0}),
    ],
)
def test_changing_any_identity_field_changes_key(changed: tuple[object, ...]) -> None:
    assert idem_key(*BASE) != idem_key(*changed)


def test_argument_key_order_does_not_change_key() -> None:
    first = {"cell_id": "cell-1", "value": 40.0}
    second = {"value": 40.0, "cell_id": "cell-1"}
    assert idem_key("run", "execution", 0, "set_tx_power", first) == idem_key(
        "run", "execution", 0, "set_tx_power", second
    )


def test_idem_key_is_stable_across_process_hash_seeds() -> None:
    script = """
from orchestrator.journal import idem_key
print(idem_key('run', 'execution', 0, 'set_tx_power', {'value': 40.0, 'cell_id': 'cell-1'}))
"""
    outputs = []
    for seed in ("1", "98765"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1]


def test_same_action_twice_has_one_effect_and_changes_state_once(tmp_path: Path) -> None:
    initial = SCENARIOS_BY_ID["coverage_hole_01"].initial_state
    action = SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0)
    action_log = []
    database = tmp_path / "idem.sqlite"
    with Journal(database) as journal:
        first = run_tool(
            initial,
            action,
            action_log,
            apply_action=apply,
            read_metrics=metrics,
            journal=journal,
            run_id="idem-run",
            step=0,
        )
        second = run_tool(
            first.state,
            action,
            action_log,
            apply_action=apply,
            read_metrics=metrics,
            journal=journal,
            run_id="idem-run",
            step=0,
        )
    with sqlite3.connect(database) as connection:
        effects = connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0]
        tool_events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'tool_called'"
        ).fetchone()[0]
    assert effects == 1
    assert tool_events == 1
    assert action_log == [action]
    assert first.state == second.state
