from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.run import _parser
from tests.fanout_helpers import run_fanout_fixture

# Fixed timing input and ratio. Reproduce them with:
# .\.venv\Scripts\python.exe -m pytest tests/test_fanout_concurrency_limit.py -vv
LATENCY_S = 0.5
MAX_SERIAL_RATIO = 1.5


async def test_parallel_mode_with_one_slot_is_serialized(tmp_path: Path) -> None:
    serial = await run_fanout_fixture(
        tmp_path / "serial.sqlite",
        cell_count=5,
        parallel=False,
        telemetry_latency_s=LATENCY_S,
    )
    limited = await run_fanout_fixture(
        tmp_path / "limited.sqlite",
        cell_count=5,
        parallel=True,
        max_concurrency=1,
        telemetry_latency_s=LATENCY_S,
    )

    assert limited.telemetry_elapsed_s / serial.telemetry_elapsed_s < MAX_SERIAL_RATIO


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_concurrency(value: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--scenario",
                "false_alarm_01",
                "--model",
                "fake:scripted",
                "--db",
                "run.sqlite",
                "--run-id",
                "cli-run",
                "--max-concurrency",
                value,
            ]
        )


def test_cli_defaults_to_parallel_and_honors_serial_switch() -> None:
    common = [
        "--scenario",
        "false_alarm_01",
        "--model",
        "fake:scripted",
        "--db",
        "run.sqlite",
        "--run-id",
        "cli-run",
    ]
    assert _parser().parse_args(common).parallel is True
    assert _parser().parse_args([*common, "--serial"]).parallel is False
    assert _parser().parse_args([*common, "--parallel"]).parallel is True
