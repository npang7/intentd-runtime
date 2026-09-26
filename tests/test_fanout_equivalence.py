from __future__ import annotations

from pathlib import Path

import pytest

from tests.fanout_helpers import run_fanout_fixture


@pytest.mark.parametrize("cell_count", [1, 3], ids=["single-cell", "multi-cell"])
async def test_serial_and_parallel_runs_are_observably_identical(
    tmp_path: Path,
    cell_count: int,
) -> None:
    serial = await run_fanout_fixture(
        tmp_path / f"serial-{cell_count}.sqlite",
        cell_count=cell_count,
        parallel=False,
    )
    parallel = await run_fanout_fixture(
        tmp_path / f"parallel-{cell_count}.sqlite",
        cell_count=cell_count,
        parallel=True,
    )

    assert parallel.state_hash == serial.state_hash
    assert parallel.events == serial.events
    assert parallel.result.action_log == serial.result.action_log


async def test_parallel_telemetry_events_are_sorted_and_cell_local(tmp_path: Path) -> None:
    outcome = await run_fanout_fixture(
        tmp_path / "ordered.sqlite",
        cell_count=3,
        parallel=True,
    )
    telemetry_calls = [
        payload
        for kind, payload in outcome.events
        if kind == "llm_call" and '"stage":"telemetry"' in payload
    ]
    assert ['"cell_id":"cell-1"' in payload for payload in telemetry_calls] == [
        True,
        False,
        False,
    ]
    assert ['"cell_id":"cell-2"' in payload for payload in telemetry_calls] == [
        False,
        True,
        False,
    ]
    assert ['"cell_id":"cell-3"' in payload for payload in telemetry_calls] == [
        False,
        False,
        True,
    ]
