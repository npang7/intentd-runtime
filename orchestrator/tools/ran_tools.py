"""Execute registered actions with SQLite-backed idempotency."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestrator.contracts import ActionValue, GetMetrics, NoOp, ReportNonRadioIssue
from orchestrator.journal import Journal, idem_key

if TYPE_CHECKING:
    from sim.network import NetworkState


@dataclass(frozen=True)
class ToolResult:
    state: NetworkState
    output: object


def run_tool(
    state: NetworkState,
    action: ActionValue,
    action_log: list[ActionValue],
    *,
    apply_action: Callable[[NetworkState, ActionValue], NetworkState],
    read_metrics: Callable[[NetworkState], object],
    journal: Journal,
    run_id: str,
    stage: str = "execution",
    step: int | None = None,
) -> ToolResult:
    """Run one logical action once and atomically persist its effect and event."""

    logical_step = len(action_log) if step is None else step
    args = action.model_dump(mode="json")
    key = idem_key(run_id, stage, logical_step, action.tool, args)
    journal.ensure_tool_run(run_id, state)
    # Idempotency has two layers: this lookup and the effects.idem_key primary
    # key. Without the lookup, a repeated call raises IntegrityError and rolls
    # back rather than silently applying the state change twice.
    stored = journal.effect_result(key)
    if stored is not None:
        return ToolResult(state=journal.run_state(run_id), output=json.loads(stored))

    if isinstance(action, GetMetrics):
        all_metrics = read_metrics(state)
        if not isinstance(all_metrics, dict):
            raise TypeError("read_metrics must return a dictionary")
        output: object = {
            cell_id: all_metrics[cell_id] for cell_id in action.cell_ids if cell_id in all_metrics
        }
        new_state = state
    elif isinstance(action, (ReportNonRadioIssue, NoOp)):
        output = {"recorded": True}
        new_state = state
    else:
        new_state = apply_action(state, action)
        output = {"applied": True}

    journal.commit_effect(
        key=key,
        run_id=run_id,
        stage=stage,
        step=logical_step,
        tool=action.tool,
        args=args,
        result=output,
        state=new_state,
    )
    action_log.append(action)
    return ToolResult(state=new_state, output=output)
