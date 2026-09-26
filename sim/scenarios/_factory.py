"""Scenario constructors; family modules are the registry source of truth."""

from __future__ import annotations

from sim.network import CellState, NetworkState
from sim.scenarios.model import (
    ActionMatcher,
    ActionRequirement,
    MetricRequirement,
    Scenario,
    SuccessSpec,
    Threshold,
)

RADIO_TOOLS = ("set_tx_power", "set_tilt", "set_a3_offset", "set_admission_threshold")

# Scenario loads, faults, SLA thresholds, and step budgets below are fixture
# inputs. Reproduce their integrity with:
# python -m pytest tests/test_scenarios_integrity.py


def _state(
    seed: int, target: CellState, *, neighbor_power: float = 40.0, penalty: float = 0.0
) -> NetworkState:
    return NetworkState(
        cells={
            "cell-1": target,
            "cell-2": CellState(
                tx_power_dbm=neighbor_power,
                tilt_deg=4.0,
                a3_offset_db=1.0,
                admission_threshold=0.9,
                neighbors=("cell-1", "cell-3"),
                load=25.0,
            ),
            "cell-3": CellState(
                tx_power_dbm=39.0,
                tilt_deg=5.0,
                a3_offset_db=1.0,
                admission_threshold=0.9,
                neighbors=("cell-1", "cell-2"),
                load=22.0,
            ),
        },
        backhaul_penalty_ms={"cell-1": penalty, "cell-2": 0.0, "cell-3": 0.0},
        seed=seed,
    )


def overload(scenario_id: str, seed: int, load: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=36.0,
            tilt_deg=5.0,
            a3_offset_db=1.0,
            admission_threshold=0.65,
            neighbors=("cell-2", "cell-3"),
            load=load,
        ),
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {"edge_throughput": Threshold(min_value=2.0)},
        SuccessSpec(
            (MetricRequirement("cell-1", "edge_throughput"),),
            (
                MetricRequirement("cell-2", "edge_throughput"),
                MetricRequirement("cell-3", "edge_throughput"),
            ),
            ActionRequirement((ActionMatcher(("set_admission_threshold",), "cell-1"),)),
            (),
            2,
        ),
        2,
    )


def coverage(scenario_id: str, seed: int, tilt: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=24.0,
            tilt_deg=tilt,
            a3_offset_db=1.0,
            admission_threshold=0.9,
            neighbors=("cell-2", "cell-3"),
            load=18.0,
        ),
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {"edge_throughput": Threshold(min_value=3.0), "handover_fail": Threshold(max_value=0.2)},
        SuccessSpec(
            (
                MetricRequirement("cell-1", "edge_throughput"),
                MetricRequirement("cell-1", "handover_fail"),
            ),
            (
                MetricRequirement("cell-2", "edge_throughput"),
                MetricRequirement("cell-3", "edge_throughput"),
            ),
            ActionRequirement((ActionMatcher(("set_tx_power",), "cell-1"),)),
            (),
            2,
        ),
        2,
    )


def ping_pong(scenario_id: str, seed: int, offset: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=40.0,
            tilt_deg=4.0,
            a3_offset_db=offset,
            admission_threshold=0.9,
            neighbors=("cell-2", "cell-3"),
            load=25.0,
        ),
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {"ping_pong_rate": Threshold(max_value=0.08), "handover_fail": Threshold(max_value=0.2)},
        SuccessSpec(
            (MetricRequirement("cell-1", "ping_pong_rate"),),
            (MetricRequirement("cell-1", "handover_fail"),),
            ActionRequirement((ActionMatcher(("set_a3_offset",), "cell-1"),)),
            (),
            1,
        ),
        1,
    )


def interference(scenario_id: str, seed: int, aggressor_power: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=38.0,
            tilt_deg=4.0,
            a3_offset_db=1.0,
            admission_threshold=0.9,
            neighbors=("cell-2", "cell-3"),
            load=60.0,
        ),
        neighbor_power=aggressor_power,
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {"edge_throughput": Threshold(min_value=4.0)},
        SuccessSpec(
            (MetricRequirement("cell-1", "edge_throughput"),),
            (
                MetricRequirement("cell-2", "edge_throughput"),
                MetricRequirement("cell-3", "edge_throughput"),
            ),
            ActionRequirement((ActionMatcher(("set_tx_power",), "cell-2"),)),
            (),
            1,
        ),
        1,
    )


def energy(scenario_id: str, seed: int, power: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=power,
            tilt_deg=4.0,
            a3_offset_db=1.0,
            admission_threshold=0.9,
            neighbors=("cell-2", "cell-3"),
            load=12.0,
        ),
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {"energy_w": Threshold(max_value=150.0), "edge_throughput": Threshold(min_value=3.0)},
        SuccessSpec(
            (MetricRequirement("cell-1", "energy_w"),),
            (MetricRequirement("cell-1", "edge_throughput"),),
            ActionRequirement((ActionMatcher(("set_tx_power",), "cell-1"),)),
            (),
            1,
        ),
        1,
    )


def conflicting(scenario_id: str, seed: int, load: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=33.0,
            tilt_deg=5.0,
            a3_offset_db=1.0,
            admission_threshold=0.65,
            neighbors=("cell-2", "cell-3"),
            load=load,
        ),
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {"edge_throughput": Threshold(min_value=2.5), "energy_w": Threshold(max_value=250.0)},
        SuccessSpec(
            (MetricRequirement("cell-1", "edge_throughput"),),
            (
                MetricRequirement("cell-2", "edge_throughput"),
                MetricRequirement("cell-1", "energy_w"),
            ),
            ActionRequirement(
                (
                    ActionMatcher(("set_tx_power",), "cell-1"),
                    ActionMatcher(("set_admission_threshold",), "cell-1"),
                )
            ),
            (),
            2,
        ),
        2,
    )


def backhaul(scenario_id: str, seed: int, penalty: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=40.0,
            tilt_deg=4.0,
            a3_offset_db=1.0,
            admission_threshold=0.9,
            neighbors=("cell-2", "cell-3"),
            load=18.0,
        ),
        penalty=penalty,
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {"rtt_ms": Threshold(max_value=100.0), "prb_utilization": Threshold(max_value=0.8)},
        SuccessSpec(
            (),
            (MetricRequirement("cell-1", "prb_utilization"),),
            ActionRequirement((ActionMatcher(("report_non_radio_issue",), "cell-1", "backhaul"),)),
            (ActionMatcher(RADIO_TOOLS),),
            2,
        ),
        2,
    )


def false_alarm(scenario_id: str, seed: int, load: float) -> Scenario:
    state = _state(
        seed,
        CellState(
            tx_power_dbm=40.0,
            tilt_deg=4.0,
            a3_offset_db=1.0,
            admission_threshold=0.9,
            neighbors=("cell-2", "cell-3"),
            load=load,
        ),
    )
    return Scenario(
        scenario_id,
        seed,
        state,
        {
            "rtt_ms": Threshold(max_value=100.0),
            "prb_utilization": Threshold(max_value=0.8),
            "edge_throughput": Threshold(min_value=3.0),
            "ping_pong_rate": Threshold(max_value=0.2),
            "handover_fail": Threshold(max_value=0.2),
        },
        SuccessSpec(
            (),
            tuple(
                MetricRequirement(cell_id, metric)
                for cell_id in state.cells
                for metric in (
                    "rtt_ms",
                    "prb_utilization",
                    "edge_throughput",
                    "ping_pong_rate",
                    "handover_fail",
                )
            ),
            ActionRequirement((ActionMatcher(("no_op",)),), exact=True),
            (),
            1,
        ),
        1,
    )
