from __future__ import annotations

from orchestrator.contracts import SetTxPower
from sim.network import apply, metrics
from sim.oracles import ORACLES
from sim.scenarios import ALL_SCENARIOS, SCENARIOS_BY_FAMILY
from sim.scenarios.model import Scenario, Threshold

# Ten percent is the selected input guard band for injected SLA violations.
# Reproduce: .\.venv\Scripts\python.exe -m pytest tests/test_scenarios_integrity.py
INITIAL_VIOLATION_MARGIN_RATIO = 0.10


def _violates_any_sla(scenario_index: int) -> bool:
    scenario = ALL_SCENARIOS[scenario_index]
    derived = metrics(scenario.initial_state)
    return any(
        not threshold.contains(float(getattr(cell_metrics, metric_name)))
        for metric_name, threshold in scenario.sla.items()
        for cell_metrics in derived.values()
    )


def _margin_base(threshold: Threshold, boundary: float) -> float:
    if threshold.min_value is not None and threshold.max_value is not None:
        return threshold.max_value - threshold.min_value
    return abs(boundary)


def _initial_violations(scenario: Scenario) -> list[tuple[str, str, float, float, float]]:
    derived = metrics(scenario.initial_state)
    violations: list[tuple[str, str, float, float, float]] = []
    for cell_id, cell_metrics in derived.items():
        for metric_name, threshold in scenario.sla.items():
            actual = float(getattr(cell_metrics, metric_name))
            if threshold.min_value is not None and actual < threshold.min_value:
                violations.append(
                    (
                        cell_id,
                        metric_name,
                        actual,
                        threshold.min_value - actual,
                        _margin_base(threshold, threshold.min_value)
                        * INITIAL_VIOLATION_MARGIN_RATIO,
                    )
                )
            if threshold.max_value is not None and actual > threshold.max_value:
                violations.append(
                    (
                        cell_id,
                        metric_name,
                        actual,
                        actual - threshold.max_value,
                        _margin_base(threshold, threshold.max_value)
                        * INITIAL_VIOLATION_MARGIN_RATIO,
                    )
                )
    return violations


def test_exactly_eighteen_unique_scenarios_cover_all_families() -> None:
    assert len(ALL_SCENARIOS) == 18
    assert len({scenario.id for scenario in ALL_SCENARIOS}) == 18
    assert set(SCENARIOS_BY_FAMILY) == {
        "cell_overload",
        "coverage_hole",
        "ping_pong_handover",
        "neighbor_interference",
        "energy_waste",
        "conflicting_sla",
        "backhaul_congestion",
        "false_alarm",
    }


def test_every_scenario_has_exactly_one_standard_and_trap_pair() -> None:
    assert set(ORACLES) == {scenario.id for scenario in ALL_SCENARIOS}
    assert all(oracle.standard and oracle.trap for oracle in ORACLES.values())


def test_all_non_false_alarms_initially_violate_their_own_sla() -> None:
    indices = [
        index
        for index, scenario in enumerate(ALL_SCENARIOS)
        if scenario not in SCENARIOS_BY_FAMILY["false_alarm"]
    ]
    assert len(indices) == 16
    assert all(_violates_any_sla(index) for index in indices)
    for index in indices:
        scenario = ALL_SCENARIOS[index]
        violations = _initial_violations(scenario)
        assert violations
        for cell_id, metric_name, actual, actual_margin, required_margin in violations:
            assert actual_margin > required_margin, (
                f"{scenario.id} {cell_id}.{metric_name}: actual={actual}, "
                f"threshold={scenario.sla[metric_name]}, violation_margin={actual_margin}, "
                f"required_margin={required_margin}"
            )


def test_false_alarms_initially_satisfy_every_sla() -> None:
    for scenario in SCENARIOS_BY_FAMILY["false_alarm"]:
        derived = metrics(scenario.initial_state)
        assert all(
            threshold.contains(float(getattr(cell_metrics, metric_name)))
            for metric_name, threshold in scenario.sla.items()
            for cell_metrics in derived.values()
        )
        assert not scenario.success.forbidden_actions


def test_fourteen_restorable_scenarios_start_with_a_restore_violation() -> None:
    restorable = [scenario for scenario in ALL_SCENARIOS if scenario.success.must_restore]
    assert len(restorable) == 14
    for scenario in restorable:
        derived = metrics(scenario.initial_state)
        assert any(
            not scenario.sla[requirement.metric_name].contains(
                float(getattr(derived[requirement.cell_id], requirement.metric_name))
            )
            for requirement in scenario.success.must_restore
        )


def test_backhaul_fault_has_high_rtt_and_normal_prb() -> None:
    for scenario in SCENARIOS_BY_FAMILY["backhaul_congestion"]:
        value = metrics(scenario.initial_state)["cell-1"]
        assert not scenario.sla["rtt_ms"].contains(value.rtt_ms)
        assert scenario.sla["prb_utilization"].contains(value.prb_utilization)
        assert not scenario.success.must_restore


def test_radio_action_cannot_change_backhaul_penalty() -> None:
    for scenario in SCENARIOS_BY_FAMILY["backhaul_congestion"]:
        changed = apply(
            scenario.initial_state,
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=46.0),
        )
        assert changed.backhaul_penalty_ms == scenario.initial_state.backhaul_penalty_ms
        assert metrics(changed)["cell-1"].rtt_ms > scenario.sla["rtt_ms"].max_value  # type: ignore[operator]


def test_scenario_and_success_step_budgets_agree() -> None:
    assert all(scenario.max_steps == scenario.success.max_steps for scenario in ALL_SCENARIOS)
