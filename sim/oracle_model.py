"""Deterministic pipeline fixture backed by the simulator's oracle sequences."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from orchestrator.contracts import response_parses
from orchestrator.llm.base import LLMCall, LLMClient, Message
from sim.oracles import ORACLES

OracleKind = Literal["standard", "trap"]


class OracleModel(LLMClient):
    """Emit stage contracts without adding operational judgment to the fixture."""

    def __init__(self, kind: OracleKind) -> None:
        self._kind = kind
        self._calls: list[LLMCall] = []

    @property
    def model(self) -> str:
        return f"oracle:{self._kind}"

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
        dynamic = json.loads(messages[-1]["content"])
        scenario_id = dynamic.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in ORACLES:
            raise ValueError("oracle prompt must contain a known scenario_id")
        pair = ORACLES[scenario_id]
        actions = pair.standard if self._kind == "standard" else pair.trap

        if stage == "telemetry" and schema_name == "TelemetryReport":
            payload: object = {
                "findings": [],
                "summary": f"Deterministic telemetry for {scenario_id}.",
            }
        elif stage == "policy" and schema_name == "PolicyPlan":
            payload = {
                "actions": [action.model_dump(mode="json") for action in actions],
                "expected_effect": "Apply the selected oracle action sequence unchanged.",
                "rationale": f"Use the {self._kind} fixture for {scenario_id}.",
            }
        elif stage == "validation" and schema_name == "ValidationVerdict":
            # Validation is not the oracle: both action fixtures pass through unchanged.
            payload = {"approved": True, "revised_actions": [], "violations": []}
        elif stage == "execution" and schema_name == "ExecutionResult":
            payload = {
                "applied": [action.model_dump(mode="json") for action in actions],
                "post_metrics": {},
                "skipped": [],
            }
        else:
            raise ValueError(f"unsupported oracle stage/schema pair: {stage}/{schema_name}")

        raw_response = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        self._calls.append(
            LLMCall(
                stage=stage,
                agent=agent,
                attempt=attempt,
                latency_s=0.0,
                raw_response=raw_response,
                parsed_ok=response_parses(schema_name, raw_response),
                model=self.model,
            )
        )
        return raw_response
