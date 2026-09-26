from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from orchestrator.contracts import NoOp, SetA3Offset, SetAdmissionThreshold, SetTilt, SetTxPower
from sim.network import CellState, NetworkState, apply, metrics, state_hash

ROOT = Path(__file__).resolve().parent.parent


def _state(seed: int = 11) -> NetworkState:
    return NetworkState(
        cells={
            "cell-a": CellState(
                tx_power_dbm=39.0,
                tilt_deg=4.0,
                a3_offset_db=-2.0,
                admission_threshold=0.8,
                neighbors=("cell-b", "cell-c"),
                load=40.0,
            ),
            "cell-b": CellState(
                tx_power_dbm=40.0,
                tilt_deg=5.0,
                a3_offset_db=1.0,
                admission_threshold=0.9,
                neighbors=("cell-a", "cell-c"),
                load=25.0,
            ),
            "cell-c": CellState(
                tx_power_dbm=38.0,
                tilt_deg=3.0,
                a3_offset_db=0.0,
                admission_threshold=0.9,
                neighbors=("cell-a", "cell-b"),
                load=20.0,
            ),
        },
        backhaul_penalty_ms={"cell-a": 0.0, "cell-b": 0.0, "cell-c": 0.0},
        seed=seed,
    )


def _run(state: NetworkState) -> NetworkState:
    actions = (
        SetTxPower(tool="set_tx_power", cell_id="cell-a", tx_power_dbm=42.0),
        SetTilt(tool="set_tilt", cell_id="cell-a", tilt_deg=2.0),
    )
    for action in actions:
        state = apply(state, action)
    return state


def test_same_seed_and_actions_have_same_hash() -> None:
    assert state_hash(_run(_state())) == state_hash(_run(_state()))


def test_different_seed_has_different_hash() -> None:
    assert state_hash(_run(_state(1))) != state_hash(_run(_state(2)))


def test_same_state_hash_is_stable_across_process_hash_seeds() -> None:
    program = """
from orchestrator.contracts import SetTxPower
from sim.network import CellState, NetworkState, apply, state_hash
state = NetworkState(
    cells={
        "cell-a": CellState(
            tx_power_dbm=39.0,
            tilt_deg=4.0,
            a3_offset_db=-2.0,
            admission_threshold=0.8,
            neighbors=("cell-b", "cell-c"),
            load=40.0,
        ),
        "cell-b": CellState(
            tx_power_dbm=40.0,
            tilt_deg=5.0,
            a3_offset_db=1.0,
            admission_threshold=0.9,
            neighbors=("cell-a", "cell-c"),
            load=25.0,
        ),
        "cell-c": CellState(
            tx_power_dbm=38.0,
            tilt_deg=3.0,
            a3_offset_db=0.0,
            admission_threshold=0.9,
            neighbors=("cell-a", "cell-b"),
            load=20.0,
        ),
    },
    backhaul_penalty_ms={"cell-a": 0.0, "cell-b": 0.0, "cell-c": 0.0},
    seed=11,
)
action = SetTxPower(tool="set_tx_power", cell_id="cell-a", tx_power_dbm=42.0)
print(state_hash(apply(state, action)))
"""

    def run(hash_seed: str) -> bytes:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        return subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
        ).stdout

    assert run("1") == run("987654")


def test_higher_neighbor_power_reduces_edge_throughput() -> None:
    state = _state()
    changed = apply(state, SetTxPower(tool="set_tx_power", cell_id="cell-b", tx_power_dbm=46.0))
    assert metrics(changed)["cell-a"].edge_throughput < metrics(state)["cell-a"].edge_throughput


def test_higher_own_power_improves_capacity_and_costs_energy() -> None:
    state = _state()
    changed = apply(state, SetTxPower(tool="set_tx_power", cell_id="cell-a", tx_power_dbm=44.0))
    assert metrics(changed)["cell-a"].capacity > metrics(state)["cell-a"].capacity
    assert metrics(changed)["cell-a"].energy_w > metrics(state)["cell-a"].energy_w


def test_higher_tilt_reduces_sinr_and_edge_throughput() -> None:
    state = _state()
    changed = apply(state, SetTilt(tool="set_tilt", cell_id="cell-a", tilt_deg=12.0))
    assert metrics(changed)["cell-a"].sinr_proxy < metrics(state)["cell-a"].sinr_proxy
    assert metrics(changed)["cell-a"].edge_throughput < metrics(state)["cell-a"].edge_throughput


def test_higher_admission_increases_capacity_and_reduces_utilization() -> None:
    state = _state()
    changed = apply(
        state,
        SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-a", threshold=1.0),
    )
    assert metrics(changed)["cell-a"].capacity > metrics(state)["cell-a"].capacity
    assert metrics(changed)["cell-a"].prb_utilization < metrics(state)["cell-a"].prb_utilization


def test_higher_a3_offset_reduces_ping_pong_rate() -> None:
    state = _state()
    changed = apply(state, SetA3Offset(tool="set_a3_offset", cell_id="cell-a", a3_offset_db=6.0))
    assert metrics(changed)["cell-a"].ping_pong_rate < metrics(state)["cell-a"].ping_pong_rate


def test_higher_load_increases_utilization_and_rtt_and_reduces_edge() -> None:
    state = _state()
    cells = dict(state.cells)
    cells["cell-a"] = replace(cells["cell-a"], load=70.0)
    changed = replace(state, cells=cells)
    before = metrics(state)["cell-a"]
    after = metrics(changed)["cell-a"]
    assert after.prb_utilization > before.prb_utilization
    assert after.rtt_ms > before.rtt_ms
    assert after.edge_throughput < before.edge_throughput


def test_backhaul_penalty_changes_only_rtt() -> None:
    state = _state()
    changed = replace(state, backhaul_penalty_ms={"cell-a": 50.0, "cell-b": 0.0, "cell-c": 0.0})
    before = metrics(state)["cell-a"]
    after = metrics(changed)["cell-a"]
    assert after.rtt_ms > before.rtt_ms
    assert replace(after, rtt_ms=before.rtt_ms) == before


def test_average_throughput_is_edge_throughput_times_factor() -> None:
    value = metrics(_state())["cell-a"]
    assert value.avg_throughput == value.edge_throughput * 2.5


def test_apply_is_pure_and_radio_action_preserves_backhaul() -> None:
    state = replace(_state(), backhaul_penalty_ms={"cell-a": 75.0, "cell-b": 0.0, "cell-c": 0.0})
    original_hash = state_hash(state)
    changed = apply(state, SetTxPower(tool="set_tx_power", cell_id="cell-a", tx_power_dbm=43.0))
    assert state_hash(state) == original_hash
    assert changed.backhaul_penalty_ms == state.backhaul_penalty_ms
    assert changed.cells["cell-a"].tx_power_dbm == 43.0


def test_no_op_returns_same_state() -> None:
    state = _state()
    assert apply(state, NoOp(tool="no_op", reason="No change needed.")) is state


def test_state_dataclass_is_frozen() -> None:
    state = _state()
    with pytest.raises(FrozenInstanceError):
        state.seed = 12  # type: ignore[misc]
