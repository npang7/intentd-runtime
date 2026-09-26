from __future__ import annotations

import pytest

from sim.grader import CHECKS, GradeContext, run_checks
from sim.network import apply, metrics
from sim.oracles import ORACLES, SINGLE_FAILURE_CASES
from sim.scenarios import SCENARIOS_BY_ID


@pytest.mark.parametrize("check_name", [name for name, _ in CHECKS])
def test_each_check_is_necessary_for_its_single_failure_trap(check_name: str) -> None:
    scenario = SCENARIOS_BY_ID[SINGLE_FAILURE_CASES[check_name]]
    actions = ORACLES[scenario.id].trap
    final_state = scenario.initial_state
    for action in actions:
        final_state = apply(final_state, action)

    context = GradeContext(final_state, actions, scenario, metrics(final_state))
    failures_by_check = {name: check(context) for name, check in CHECKS}
    assert failures_by_check[check_name]
    assert all(not reasons for name, reasons in failures_by_check.items() if name != check_name)

    remaining = [(name, check) for name, check in CHECKS if name != check_name]
    result = run_checks(remaining, final_state, actions, scenario)
    assert result.success, result.reasons
