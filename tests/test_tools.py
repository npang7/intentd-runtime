from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.contracts import (
    ActionValue,
    GetMetrics,
    NoOp,
    ReportNonRadioIssue,
    SetA3Offset,
    SetAdmissionThreshold,
    SetTilt,
    SetTxPower,
)
from orchestrator.journal import Journal
from orchestrator.tools.ran_tools import run_tool
from sim.network import apply, metrics, state_hash
from sim.scenarios import SCENARIOS_BY_ID

ACTIONS: list[ActionValue] = [
    GetMetrics(tool="get_metrics", cell_ids=["cell-1"], window_s=60),
    SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0),
    SetA3Offset(tool="set_a3_offset", cell_id="cell-1", a3_offset_db=2.0),
    SetTilt(tool="set_tilt", cell_id="cell-1", tilt_deg=2.0),
    SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=1.0),
    ReportNonRadioIssue(
        tool="report_non_radio_issue",
        cell_id="cell-1",
        category="backhaul",
        detail="Inspect backhaul.",
    ),
    NoOp(tool="no_op", reason="No change required."),
]


@pytest.mark.parametrize("action", ACTIONS, ids=lambda action: action.tool)
def test_each_tool_changes_only_its_declared_state(tmp_path: Path, action: ActionValue) -> None:
    before = SCENARIOS_BY_ID["coverage_hole_01"].initial_state
    log: list[ActionValue] = []
    with Journal(tmp_path / f"{action.tool}.sqlite") as journal:
        result = run_tool(
            before,
            action,
            log,
            apply_action=apply,
            read_metrics=metrics,
            journal=journal,
            run_id=action.tool,
        )

    assert log == [action]
    assert result.state.seed == before.seed
    assert result.state.backhaul_penalty_ms == before.backhaul_penalty_ms
    if isinstance(action, SetTxPower):
        assert result.state.cells["cell-1"].tx_power_dbm == 40.0
        assert result.state.cells["cell-1"].tilt_deg == before.cells["cell-1"].tilt_deg
    elif isinstance(action, SetA3Offset):
        assert result.state.cells["cell-1"].a3_offset_db == 2.0
        assert result.state.cells["cell-1"].tx_power_dbm == before.cells["cell-1"].tx_power_dbm
    elif isinstance(action, SetTilt):
        assert result.state.cells["cell-1"].tilt_deg == 2.0
        assert result.state.cells["cell-1"].tx_power_dbm == before.cells["cell-1"].tx_power_dbm
    elif isinstance(action, SetAdmissionThreshold):
        assert result.state.cells["cell-1"].admission_threshold == 1.0
        assert result.state.cells["cell-1"].tx_power_dbm == before.cells["cell-1"].tx_power_dbm
    else:
        assert result.state == before


def test_get_metrics_does_not_change_state_hash(tmp_path: Path) -> None:
    state = SCENARIOS_BY_ID["coverage_hole_01"].initial_state
    with Journal(tmp_path / "metrics.sqlite") as journal:
        result = run_tool(
            state,
            ACTIONS[0],
            [],
            apply_action=apply,
            read_metrics=metrics,
            journal=journal,
            run_id="metrics",
        )
    assert state_hash(result.state) == state_hash(state)


def test_all_tool_calls_are_appended_to_action_log(tmp_path: Path) -> None:
    state = SCENARIOS_BY_ID["coverage_hole_01"].initial_state
    log: list[ActionValue] = []
    with Journal(tmp_path / "all.sqlite") as journal:
        for action in ACTIONS:
            result = run_tool(
                state,
                action,
                log,
                apply_action=apply,
                read_metrics=metrics,
                journal=journal,
                run_id="all-tools",
            )
            state = result.state
    assert log == ACTIONS
