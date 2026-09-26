from __future__ import annotations

from pathlib import Path

from tests.fanout_helpers import run_fanout_fixture

# Fixed timing input and acceptance ratios. Reproduce them with:
# .\.venv\Scripts\python.exe -m pytest tests/test_fanout_speedup.py -vv
LATENCY_S = 0.5
TELEMETRY_MIN_SPEEDUP = 4.0
RUN_MIN_SPEEDUP = 2.0


async def test_parallel_fanout_meets_stage_and_run_speedup_thresholds(tmp_path: Path) -> None:
    serial = await run_fanout_fixture(
        tmp_path / "serial.sqlite",
        cell_count=5,
        parallel=False,
        telemetry_latency_s=LATENCY_S,
    )
    parallel = await run_fanout_fixture(
        tmp_path / "parallel.sqlite",
        cell_count=5,
        parallel=True,
        telemetry_latency_s=LATENCY_S,
    )

    telemetry_speedup = serial.telemetry_elapsed_s / parallel.telemetry_elapsed_s
    run_speedup = serial.run_elapsed_s / parallel.run_elapsed_s
    assert telemetry_speedup >= TELEMETRY_MIN_SPEEDUP
    assert run_speedup >= RUN_MIN_SPEEDUP
