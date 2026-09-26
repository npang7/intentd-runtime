from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.analyze import Usage, analyze, render_summary, usage_cost_usd


def test_cost_uses_all_four_raw_usage_fields() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert usage_cost_usd(usage) == pytest.approx(7.35)


def test_analyzer_generates_scenario_usage_latency_retry_and_cost(tmp_path: Path) -> None:
    rows = [
        {
            "cache_eligible": False,
            "cache_min_tokens": 4096,
            "input_tokens": 321,
            "model": "anthropic:claude-haiku-4-5-20251001",
            "record_type": "prefix_measurement",
            "stage": "telemetry",
        },
        {
            "grader": {"reasons": [], "success": True},
            "latency_s": 1.25,
            "mode": "parallel",
            "model": "anthropic:claude-haiku-4-5-20251001",
            "record_type": "run",
            "scenario_id": "false_alarm_01",
            "schema_repair_count": 2,
            "stage_latency_s": {"policy": 0.5, "telemetry": 0.75},
            "status": "finished",
            "transport_retry_count": 3,
            "usage": {
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 5,
                "input_tokens": 23,
                "output_tokens": 11,
            },
        },
    ]
    input_path = tmp_path / "runs.jsonl"
    output_path = tmp_path / "summary.md"
    input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    analyze(input_path, output_path)

    summary = output_path.read_text(encoding="utf-8")
    assert summary == render_summary(rows)
    assert "false_alarm_01" in summary
    assert "telemetry=0.750s, policy=0.500s" in summary
    assert "| 23 | 11 | 7 | 5 | 2 | 3 |" in summary
    assert "Calculated cost" in summary
    assert "provider work that completed" in summary


def test_analyzer_rejects_negative_usage() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        render_summary(
            [
                {
                    "record_type": "run",
                    "usage": {"input_tokens": -1},
                }
            ]
        )
