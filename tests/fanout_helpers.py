from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path

from orchestrator.graph import RunContext, RunResult, run_graph
from orchestrator.journal import Journal
from orchestrator.llm.fake import FakeModel
from sim.grader import GradeResult
from sim.network import CellState, NetworkState, apply, state_hash
from sim.scenarios import SCENARIOS_BY_ID


@dataclass(frozen=True)
class FanoutOutcome:
    result: RunResult
    state_hash: str
    events: list[tuple[str, str]]
    telemetry_elapsed_s: float
    run_elapsed_s: float


def _state(cell_count: int) -> NetworkState:
    cell_ids = [f"cell-{index + 1}" for index in range(cell_count)]
    cells = {
        cell_id: CellState(
            tx_power_dbm=40.0,
            tilt_deg=5.0,
            a3_offset_db=0.0,
            admission_threshold=0.8,
            neighbors=tuple(other for other in cell_ids if other != cell_id),
            load=10.0 + index,
        )
        for index, cell_id in enumerate(cell_ids)
    }
    return NetworkState(
        cells=cells,
        backhaul_penalty_ms={cell_id: 0.0 for cell_id in cell_ids},
        seed=17,
    )


def _metrics(state: NetworkState) -> dict[str, dict[str, float]]:
    return {
        cell_id: {"load": cell.load, "tx_power_dbm": cell.tx_power_dbm}
        for cell_id, cell in state.cells.items()
    }


def _grade(*_args: object) -> GradeResult:
    return GradeResult(success=True, reasons=[])


async def run_fanout_fixture(
    database: Path,
    *,
    cell_count: int,
    parallel: bool,
    max_concurrency: int = 8,
    telemetry_latency_s: float = 0.0,
) -> FanoutOutcome:
    state = _state(cell_count)
    scenario = replace(SCENARIOS_BY_ID["false_alarm_01"], initial_state=state)

    def telemetry_model() -> FakeModel:
        if telemetry_latency_s:
            return FakeModel(
                modes=("scripted", "latency"),
                latency_s=telemetry_latency_s,
            )
        return FakeModel(modes=("scripted",))

    with Journal(database) as journal:
        context = RunContext(
            run_id="fanout-run",
            scenario=scenario,
            state=state,
            action_log=[],
            journal=journal,
            llm=FakeModel(modes=("scripted",)),
            llm_factory=telemetry_model,
            parallel_telemetry=parallel,
            max_concurrency=max_concurrency,
        )
        started = time.perf_counter()
        result = await run_graph(
            context,
            apply_action=apply,
            read_metrics=_metrics,
            grade_run=_grade,
        )
        run_elapsed_s = time.perf_counter() - started
        events = [(kind, payload) for _, kind, payload in journal.events("fanout-run")]

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT ts, kind, payload FROM events WHERE run_id = ? ORDER BY seq",
            ("fanout-run",),
        ).fetchall()
    stage_started = next(
        float(ts)
        for ts, kind, payload in rows
        if kind == "stage_started" and json.loads(str(payload))["stage"] == "telemetry"
    )
    stage_finished = next(
        float(ts)
        for ts, kind, payload in rows
        if kind == "stage_finished" and json.loads(str(payload))["stage"] == "telemetry"
    )
    return FanoutOutcome(
        result=result,
        state_hash=state_hash(result.final_state),
        events=events,
        telemetry_elapsed_s=stage_finished - stage_started,
        run_elapsed_s=run_elapsed_s,
    )
