from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.graph import RunContext, run_graph
from orchestrator.journal import Journal
from sim.grader import grade
from sim.network import apply, metrics
from sim.oracle_model import OracleKind, OracleModel
from sim.oracles import ORACLES
from sim.scenarios import ALL_SCENARIOS
from sim.scenarios.model import Scenario

CASES = [
    pytest.param(scenario, "standard", id=f"{scenario.id}-standard") for scenario in ALL_SCENARIOS
] + [pytest.param(scenario, "trap", id=f"{scenario.id}-trap") for scenario in ALL_SCENARIOS]


@pytest.mark.parametrize(("scenario", "kind"), CASES)
async def test_oracle_actions_cross_the_complete_pipeline_unchanged(
    tmp_path: Path, scenario: Scenario, kind: OracleKind
) -> None:
    model = OracleModel(kind)
    with Journal(tmp_path / f"{scenario.id}-{kind}.sqlite") as journal:
        context = RunContext(
            run_id=f"{scenario.id}-{kind}",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=model,
            llm_factory=lambda: OracleModel(kind),
        )
        result = await run_graph(
            context,
            apply_action=apply,
            read_metrics=metrics,
            grade_run=grade,
        )

    oracle = ORACLES[scenario.id]
    expected = oracle.standard if kind == "standard" else oracle.trap
    assert result.validation.approved is True
    assert result.action_log == expected
    if kind == "standard":
        assert result.grade.success, result.grade.reasons
        assert result.grade.reasons == []
    else:
        assert not result.grade.success
        assert any(
            reason.startswith(f"{oracle.trap_expected_check}:") for reason in result.grade.reasons
        )
