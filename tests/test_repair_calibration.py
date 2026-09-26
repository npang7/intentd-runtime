from __future__ import annotations

import math
from contextlib import suppress

from orchestrator.contracts import PolicyPlan
from orchestrator.llm.base import Message
from orchestrator.llm.fake import FakeModel
from orchestrator.repair import UnrepairableStageError, complete_with_repair, validity_stats

# These are statistical input choices, not measured system performance. The sample size makes
# the expected unrepairable tail observable, while the confidence level limits deterministic
# false alarms. Reproduce: .\.venv\Scripts\python.exe -m pytest tests/test_repair_calibration.py
SAMPLE_COUNT = 1000
MALFORMED_RATE = 0.15
SCHEMA_VIOLATION_RATE = 0.15
CONFIDENCE = 0.99


def _binomial_interval(n: int, probability: float, confidence: float) -> tuple[int, int]:
    lower_tail = (1.0 - confidence) / 2.0
    upper_target = 1.0 - lower_tail
    probability_mass = (1.0 - probability) ** n
    cumulative = 0.0
    lower = 0
    lower_found = False
    upper = n
    for successes in range(n + 1):
        cumulative += probability_mass
        if not lower_found and cumulative >= lower_tail:
            lower = successes
            lower_found = True
        if cumulative >= upper_target:
            upper = successes
            break
        if successes < n:
            probability_mass *= (
                (n - successes) / (successes + 1) * probability / (1.0 - probability)
            )
    return lower, upper


async def test_invalid_and_unrepairable_counts_match_binomial_prediction() -> None:
    model = FakeModel(
        modes=("scripted", "adversarial"),
        seed="m3-calibration",
        malformed_rate=MALFORMED_RATE,
        schema_violation_rate=SCHEMA_VIOLATION_RATE,
    )
    for index in range(SAMPLE_COUNT):
        messages: list[Message] = [{"role": "user", "content": f"calibration-{index}"}]
        with suppress(UnrepairableStageError):
            await complete_with_repair(
                model,
                messages,
                "PolicyPlan",
                PolicyPlan,
                stage="policy",
                agent="planner",
            )

    stats = validity_stats(list(model.calls))
    invalid_probability = MALFORMED_RATE + SCHEMA_VIOLATION_RATE
    unrepairable_probability = invalid_probability ** (1 + 2)
    invalid_interval = _binomial_interval(SAMPLE_COUNT, invalid_probability, CONFIDENCE)
    unrepairable_interval = _binomial_interval(SAMPLE_COUNT, unrepairable_probability, CONFIDENCE)
    assert stats.total_stage_calls == SAMPLE_COUNT
    assert invalid_interval[0] <= stats.invalid_first_count <= invalid_interval[1]
    assert unrepairable_interval[0] <= stats.unrepairable_count <= unrepairable_interval[1]
    assert math.isclose(
        stats.invalid_first_emission or 0.0,
        stats.invalid_first_count / SAMPLE_COUNT,
    )
