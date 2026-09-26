from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from orchestrator.contracts import response_parses
from orchestrator.graph import RunContext, run_graph
from orchestrator.journal import Journal, canonical_json
from orchestrator.llm.base import LLMCall, LLMClient, Message
from sim.grader import grade
from sim.network import apply, metrics
from sim.oracle_model import OracleModel
from sim.scenarios import SCENARIOS_BY_ID


class RepairOnceClient(LLMClient):
    def __init__(self, failure_kind: str) -> None:
        self.failure_kind = failure_kind
        self._calls: list[LLMCall] = []

    @property
    def model(self) -> str:
        return "repair-once"

    @property
    def calls(self) -> Sequence[LLMCall]:
        return tuple(self._calls)

    async def complete(
        self,
        messages: Sequence[Message],
        schema_name: str,
        *,
        stage: str,
        agent: str,
        attempt: int,
    ) -> str:
        started = time.perf_counter()
        if stage == "telemetry" and attempt == 0:
            raw = (
                "{"
                if self.failure_kind == "malformed"
                else json.dumps({"findings": [], "summary": "invalid", "unexpected": True})
            )
        elif stage == "telemetry":
            raw = json.dumps({"findings": [], "summary": "repaired"})
        elif stage == "policy":
            raw = json.dumps(
                {
                    "actions": [{"tool": "no_op", "reason": "No change."}],
                    "expected_effect": "No change.",
                    "rationale": "Metrics are healthy.",
                }
            )
        elif stage == "validation":
            raw = json.dumps({"approved": True, "revised_actions": [], "violations": []})
        else:
            raw = json.dumps(
                {
                    "applied": [{"tool": "no_op", "reason": "No change."}],
                    "post_metrics": {},
                    "skipped": [],
                }
            )
        self._calls.append(
            LLMCall(
                stage=stage,
                agent=agent,
                attempt=attempt,
                latency_s=time.perf_counter() - started,
                raw_response=raw,
                parsed_ok=response_parses(schema_name, raw),
                model=self.model,
            )
        )
        return raw


async def _journaled_run(path: Path) -> list[tuple[int, str, str]]:
    scenario = SCENARIOS_BY_ID["false_alarm_01"]
    with Journal(path) as journal:
        context = RunContext(
            run_id="repeatable-run",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=OracleModel("standard"),
            llm_factory=lambda: OracleModel("standard"),
        )
        await run_graph(
            context,
            apply_action=apply,
            read_metrics=metrics,
            grade_run=grade,
        )
        return journal.events("repeatable-run")


async def test_event_sequence_is_ordered_and_has_contiguous_sequence_numbers(
    tmp_path: Path,
) -> None:
    events = await _journaled_run(tmp_path / "events.sqlite")
    assert [seq for seq, _, _ in events] == list(range(len(events)))
    assert [kind for _, kind, _ in events] == [
        "run_started",
        "stage_started",
        "llm_call",
        "llm_call",
        "llm_call",
        "stage_finished",
        "stage_started",
        "llm_call",
        "stage_finished",
        "stage_started",
        "llm_call",
        "stage_finished",
        "stage_started",
        "llm_call",
        "tool_called",
        "stage_finished",
        "run_finished",
    ]


async def test_payloads_are_canonical_and_repeat_byte_for_byte(tmp_path: Path) -> None:
    first = await _journaled_run(tmp_path / "first.sqlite")
    second = await _journaled_run(tmp_path / "second.sqlite")
    assert first == second
    for _, _, payload in first:
        assert canonical_json(__import__("json").loads(payload)) == payload


@pytest.mark.parametrize("failure_kind", ["malformed", "schema"])
async def test_repair_events_follow_failed_call_and_precede_retry(
    tmp_path: Path, failure_kind: str
) -> None:
    scenario = SCENARIOS_BY_ID["false_alarm_01"]
    with Journal(tmp_path / f"{failure_kind}.sqlite") as journal:
        context = RunContext(
            run_id="repair-run",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=RepairOnceClient(failure_kind),
            llm_factory=lambda: RepairOnceClient(failure_kind),
        )
        await run_graph(
            context,
            apply_action=apply,
            read_metrics=metrics,
            grade_run=grade,
        )
        events = journal.events("repair-run")

    telemetry_events = events[1:11]
    assert [kind for _, kind, _ in telemetry_events] == [
        "stage_started",
        "llm_call",
        "repair_attempted",
        "llm_call",
        "llm_call",
        "repair_attempted",
        "llm_call",
        "llm_call",
        "repair_attempted",
        "llm_call",
    ]
    repair_payload = json.loads(telemetry_events[2][2])
    assert repair_payload["failure_kind"] == failure_kind
    assert repair_payload["attempt"] == 1
    assert repair_payload["stage"] == "telemetry"
    cell_ids = [
        json.loads(payload)["cell_id"]
        for _, kind, payload in telemetry_events
        if kind == "llm_call"
    ]
    assert cell_ids == ["cell-1", "cell-1", "cell-2", "cell-2", "cell-3", "cell-3"]
