from __future__ import annotations

import pytest

from sim.grader import grade
from sim.network import ActionValue, NetworkState, apply, metrics
from sim.oracles import ORACLES
from sim.scenarios import ALL_SCENARIOS
from sim.scenarios.model import MetricRequirement, Scenario, Threshold

# Ten percent is the selected input guard band against standard solutions sitting on a boundary.
# Reproduce: .\.venv\Scripts\python.exe -m pytest tests/test_grader_oracles.py
STANDARD_MARGIN_RATIO = 0.10


def _run(state: NetworkState, actions: tuple[ActionValue, ...]) -> NetworkState:
    for action in actions:
        state = apply(state, action)
    return state


def _assert_margin(
    scenario: Scenario, final_state: NetworkState, requirement: MetricRequirement
) -> None:
    threshold = scenario.sla[requirement.metric_name]
    actual = float(getattr(metrics(final_state)[requirement.cell_id], requirement.metric_name))
    margins: list[tuple[float, float]] = []
    if threshold.min_value is not None:
        base = _margin_base(threshold, threshold.min_value)
        margins.append((actual - threshold.min_value, base * STANDARD_MARGIN_RATIO))
    if threshold.max_value is not None:
        base = _margin_base(threshold, threshold.max_value)
        margins.append((threshold.max_value - actual, base * STANDARD_MARGIN_RATIO))
    assert all(actual_margin > required_margin for actual_margin, required_margin in margins), (
        f"{scenario.id} {requirement.cell_id}.{requirement.metric_name}: "
        f"actual={actual}, threshold={threshold}, margins={margins}"
    )


def _assert_standard_restores_injected_fault(scenario: Scenario, final_state: NetworkState) -> None:
    if not scenario.success.must_restore:
        return
    initial = metrics(scenario.initial_state)
    final = metrics(final_state)
    restored = []
    for requirement in scenario.success.must_restore:
        threshold = scenario.sla[requirement.metric_name]
        initial_value = float(getattr(initial[requirement.cell_id], requirement.metric_name))
        final_value = float(getattr(final[requirement.cell_id], requirement.metric_name))
        if not threshold.contains(initial_value):
            restored.append(threshold.contains(final_value))
    assert restored, f"{scenario.id}: standard oracle has no injected fault to restore"
    assert all(restored), f"{scenario.id}: standard oracle did not restore every injected fault"


def _margin_base(threshold: Threshold, boundary: float) -> float:
    if threshold.min_value is not None and threshold.max_value is not None:
        return threshold.max_value - threshold.min_value
    return abs(boundary)


def _assert_trap_violation_margins(
    scenario: Scenario, final_state: NetworkState, requirements: tuple[MetricRequirement, ...]
) -> None:
    derived = metrics(final_state)
    violations: list[tuple[MetricRequirement, float, float, float]] = []
    for requirement in requirements:
        threshold = scenario.sla[requirement.metric_name]
        actual = float(getattr(derived[requirement.cell_id], requirement.metric_name))
        if threshold.min_value is not None and actual < threshold.min_value:
            violations.append(
                (
                    requirement,
                    actual,
                    threshold.min_value - actual,
                    _margin_base(threshold, threshold.min_value) * STANDARD_MARGIN_RATIO,
                )
            )
        if threshold.max_value is not None and actual > threshold.max_value:
            violations.append(
                (
                    requirement,
                    actual,
                    actual - threshold.max_value,
                    _margin_base(threshold, threshold.max_value) * STANDARD_MARGIN_RATIO,
                )
            )

    assert violations, f"{scenario.id}: trap has no target metric violation"
    for requirement, actual, actual_margin, required_margin in violations:
        assert actual_margin > required_margin, (
            f"{scenario.id} {requirement.cell_id}.{requirement.metric_name}: "
            f"actual={actual}, threshold={scenario.sla[requirement.metric_name]}, "
            f"violation_margin={actual_margin}, required_margin={required_margin}"
        )


CASES = [
    pytest.param(scenario, "standard", id=f"{scenario.id}-standard") for scenario in ALL_SCENARIOS
] + [pytest.param(scenario, "trap", id=f"{scenario.id}-trap") for scenario in ALL_SCENARIOS]


@pytest.mark.parametrize(("scenario", "kind"), CASES)
def test_oracle_is_graded_for_the_expected_reason(scenario: Scenario, kind: str) -> None:
    oracle = ORACLES[scenario.id]
    actions = oracle.standard if kind == "standard" else oracle.trap
    final_state = _run(scenario.initial_state, actions)
    result = grade(final_state, actions, scenario)

    if kind == "standard":
        assert result.success, result.reasons
        assert not result.reasons
        _assert_standard_restores_injected_fault(scenario, final_state)
        for requirement in scenario.success.must_restore + scenario.success.must_not_break:
            _assert_margin(scenario, final_state, requirement)
    else:
        assert not result.success
        assert any(reason.startswith(f"{oracle.trap_expected_check}:") for reason in result.reasons)
        if oracle.trap_expected_check == "must_restore":
            _assert_trap_violation_margins(scenario, final_state, scenario.success.must_restore)
        elif oracle.trap_expected_check == "must_not_break":
            _assert_trap_violation_margins(scenario, final_state, scenario.success.must_not_break)
