from __future__ import annotations

import pytest

from orchestrator.llm.base import LLMCall
from orchestrator.repair import ValidityStats, validity_stats


def _call(attempt: int, parsed_ok: bool, stage: str = "policy") -> LLMCall:
    return LLMCall(stage, "agent", attempt, 0.0, "{}", parsed_ok, "fake")


def test_mixed_calls_produce_exact_counts_and_ratios() -> None:
    stats = validity_stats(
        [
            _call(0, True, "telemetry"),
            _call(0, False, "policy"),
            _call(1, True, "policy"),
            _call(0, False, "validation"),
            _call(1, False, "validation"),
        ]
    )
    assert stats == ValidityStats(3, 2, 1, 1)
    assert stats.invalid_first_emission == pytest.approx(2 / 3)
    assert stats.repaired == pytest.approx(1 / 2)
    assert stats.unrepairable == pytest.approx(1 / 3)


def test_all_success_has_no_repair_denominator() -> None:
    stats = validity_stats([_call(0, True), _call(0, True)])
    assert stats == ValidityStats(2, 0, 0, 0)
    assert stats.invalid_first_emission == 0.0
    assert stats.repaired is None
    assert stats.unrepairable == 0.0


def test_empty_calls_have_none_for_every_ratio() -> None:
    stats = validity_stats([])
    assert stats == ValidityStats(0, 0, 0, 0)
    assert stats.invalid_first_emission is None
    assert stats.repaired is None
    assert stats.unrepairable is None


def test_all_failed_and_second_attempt_success_boundaries() -> None:
    failed = validity_stats([_call(0, False), _call(1, False), _call(2, False)])
    repaired = validity_stats([_call(0, False), _call(1, True)])
    assert failed == ValidityStats(1, 1, 0, 1)
    assert failed.repaired == 0.0
    assert repaired == ValidityStats(1, 1, 1, 0)
    assert repaired.repaired == 1.0


def test_aggregation_sums_counts_before_dividing() -> None:
    small = ValidityStats(1, 1, 0, 1)
    large = ValidityStats(9, 1, 1, 0)
    combined = ValidityStats(
        small.total_stage_calls + large.total_stage_calls,
        small.invalid_first_count + large.invalid_first_count,
        small.repaired_count + large.repaired_count,
        small.unrepairable_count + large.unrepairable_count,
    )
    averaged_ratios = (small.invalid_first_emission + large.invalid_first_emission) / 2
    assert combined.invalid_first_emission == pytest.approx(2 / 10)
    assert averaged_ratios != pytest.approx(combined.invalid_first_emission)


def test_fanout_cell_logs_remain_contiguous_for_validity_grouping() -> None:
    stats = validity_stats(
        [
            _call(0, False, "telemetry"),
            _call(1, True, "telemetry"),
            _call(0, True, "telemetry"),
            _call(0, False, "telemetry"),
            _call(1, False, "telemetry"),
        ]
    )
    assert stats == ValidityStats(3, 2, 1, 1)


@pytest.mark.parametrize(
    "calls",
    [
        [_call(1, False)],
        [_call(0, False), _call(2, True)],
    ],
)
def test_noncontiguous_attempts_are_rejected(calls: list[LLMCall]) -> None:
    with pytest.raises(ValueError, match="contiguously"):
        validity_stats(calls)
