from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

from orchestrator.graph import RunContext, build_messages, run_graph
from orchestrator.journal import Journal
from orchestrator.llm.base import LLMCall, LLMClient, Message
from sim.grader import grade
from sim.network import apply, metrics
from sim.scenarios import SCENARIOS_BY_ID


class RecordingClient(LLMClient):
    def __init__(
        self,
        verdict: dict[str, object] | None = None,
        invocations: list[tuple[str, list[Message]]] | None = None,
    ) -> None:
        self.invocations = invocations if invocations is not None else []
        self._calls: list[LLMCall] = []
        self._verdict = verdict or {"approved": True, "revised_actions": [], "violations": []}

    def spawn(self) -> RecordingClient:
        return RecordingClient(self._verdict, self.invocations)

    @property
    def model(self) -> str:
        return "recording"

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
        copied = [dict(message) for message in messages]
        self.invocations.append((stage, copied))
        dynamic = json.loads(messages[-1]["content"])
        if stage == "telemetry":
            payload: object = {"findings": [], "summary": "telemetry-output-marker"}
        elif stage == "policy":
            payload = {
                "actions": [{"reason": "policy-action-marker", "tool": "no_op"}],
                "expected_effect": "policy-effect-marker",
                "rationale": "policy-rationale-marker",
            }
        elif stage == "validation":
            payload = self._verdict
        elif stage == "execution":
            payload = {
                "applied": dynamic["selected_actions"],
                "post_metrics": {},
                "skipped": [],
            }
        else:  # pragma: no cover - test double guard
            raise AssertionError(stage)
        raw_response = json.dumps(payload)
        self._calls.append(
            LLMCall(
                stage=stage,
                agent=agent,
                attempt=attempt,
                latency_s=time.perf_counter() - started,
                raw_response=raw_response,
                parsed_ok=True,
                model=self.model,
            )
        )
        return raw_response


async def _run(tmp_path: Path, client: RecordingClient):
    scenario = SCENARIOS_BY_ID["false_alarm_01"]
    with Journal(tmp_path / "events.sqlite") as journal:
        context = RunContext(
            run_id="graph-run",
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=client,
            llm_factory=client.spawn,
        )
        return await run_graph(
            context,
            apply_action=apply,
            read_metrics=metrics,
            grade_run=grade,
        )


async def test_four_stages_run_once_in_order_and_receive_upstream_output(tmp_path: Path) -> None:
    client = RecordingClient()
    await _run(tmp_path, client)

    assert [stage for stage, _ in client.invocations] == [
        "telemetry",
        "telemetry",
        "telemetry",
        "policy",
        "validation",
        "execution",
    ]
    prompts = {stage: messages[-1]["content"] for stage, messages in client.invocations}
    assert "telemetry-output-marker" in prompts["policy"]
    assert "policy-rationale-marker" in prompts["validation"]
    assert '"approved":true' in prompts["execution"]


async def test_validation_approved_executes_policy_actions(tmp_path: Path) -> None:
    result = await _run(tmp_path, RecordingClient())
    assert result.status == "finished"
    assert [action.tool for action in result.action_log] == ["no_op"]


async def test_validation_revisions_replace_policy_actions(tmp_path: Path) -> None:
    client = RecordingClient(
        {
            "approved": False,
            "revised_actions": [
                {"tool": "set_a3_offset", "cell_id": "cell-1", "a3_offset_db": 2.0}
            ],
            "violations": ["revise"],
        }
    )
    result = await _run(tmp_path, client)
    assert result.status == "finished"
    assert [action.tool for action in result.action_log] == ["set_a3_offset"]
    assert result.final_state.cells["cell-1"].a3_offset_db == 2.0


async def test_validation_rejection_stops_before_execution(tmp_path: Path) -> None:
    client = RecordingClient({"approved": False, "revised_actions": [], "violations": ["unsafe"]})
    result = await _run(tmp_path, client)
    assert result.status == "rejected"
    assert result.action_log == ()
    assert [stage for stage, _ in client.invocations] == [
        "telemetry",
        "telemetry",
        "telemetry",
        "policy",
        "validation",
    ]
    assert result.grade.reasons == ["validation_rejected: unsafe"]


def test_prompt_prefix_is_stable_and_dynamic_message_changes() -> None:
    first = build_messages("policy", {"scenario_id": "first", "telemetry": "one"})
    second = build_messages("policy", {"scenario_id": "second", "telemetry": "two"})
    assert first[:2] == second[:2]
    assert first[2] != second[2]
    assert [message["role"] for message in first] == ["system", "system", "user"]
