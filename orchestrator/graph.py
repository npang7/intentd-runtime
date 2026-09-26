"""Deterministic orchestration graph with telemetry fan-out."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from orchestrator.contracts import (
    ActionValue,
    ExecutionResult,
    PolicyPlan,
    StrictContract,
    TelemetryReport,
    ValidationVerdict,
)
from orchestrator.journal import Journal, canonical_json
from orchestrator.llm.anthropic import ProviderTransportError
from orchestrator.llm.base import LLMCall, LLMClient, Message, TransportRetry
from orchestrator.repair import (
    UnrepairableStageError,
    build_repair_feedback,
    complete_with_repair,
)
from orchestrator.tools.ran_tools import run_tool
from orchestrator.tools.registry import export_tool_definitions

if TYPE_CHECKING:
    from sim.grader import GradeResult
    from sim.network import NetworkState
    from sim.scenarios.model import Scenario

ContractT = TypeVar("ContractT", bound=StrictContract)


@dataclass(frozen=True)
class RejectedGrade:
    success: bool
    reasons: list[str]


@dataclass
class RunContext:
    run_id: str
    scenario: Scenario
    state: NetworkState
    action_log: list[ActionValue]
    journal: Journal
    llm: LLMClient
    llm_factory: Callable[[], LLMClient]
    parallel_telemetry: bool = True
    # Reproduce the configured input: python -m orchestrator.run --help
    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")


@dataclass(frozen=True)
class RunResult:
    status: str
    grade: GradeResult | RejectedGrade
    final_state: NetworkState
    action_log: tuple[ActionValue, ...]
    telemetry: TelemetryReport
    policy: PolicyPlan
    validation: ValidationVerdict
    execution: ExecutionResult | None


_STAGE_CONFIG: dict[str, tuple[str, str, type[StrictContract]]] = {
    "telemetry": ("telemetry_analyst", "TelemetryReport", TelemetryReport),
    "policy": ("policy_planner", "PolicyPlan", PolicyPlan),
    "validation": ("policy_validator", "ValidationVerdict", ValidationVerdict),
    "execution": ("action_executor", "ExecutionResult", ExecutionResult),
}


@dataclass(frozen=True)
class _TelemetryCellResult:
    cell_id: str
    report: TelemetryReport | None
    calls: tuple[LLMCall, ...]
    error: UnrepairableStageError | ProviderTransportError | None


def build_messages(stage: str, dynamic: Mapping[str, object]) -> list[Message]:
    """Place cacheable static content before all run-varying content."""

    try:
        agent, schema_name, model = _STAGE_CONFIG[stage]
    except KeyError as exc:
        raise ValueError(f"unknown stage: {stage}") from exc
    # Keep the cacheable prefix free of run IDs, scenario IDs, telemetry, and
    # upstream stage output. Provider cache eligibility is measured separately.
    system = (
        f"Role: {agent}. Perform the {stage} stage. "
        f"Return only JSON satisfying {schema_name}: "
        f"{canonical_json(model.model_json_schema())}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "system", "content": export_tool_definitions()},
        {"role": "user", "content": canonical_json(dynamic)},
    ]


async def _call_stage(
    context: RunContext,
    stage: str,
    dynamic: Mapping[str, object],
) -> StrictContract:
    agent, schema_name, model = _STAGE_CONFIG[stage]
    messages = build_messages(stage, dynamic)
    context.journal.append(context.run_id, "stage_started", {"stage": stage})
    started_at = len(context.llm.calls)
    try:
        parsed, calls = await complete_with_repair(
            context.llm,
            messages,
            schema_name,
            model,
            stage=stage,
            agent=agent,
        )
    except UnrepairableStageError:
        calls = list(context.llm.calls[started_at:])
        _journal_llm_calls(context, schema_name, model, calls)
        raise
    except ProviderTransportError as error:
        calls = list(context.llm.calls[started_at:])
        _journal_llm_calls(context, schema_name, model, calls)
        for retry in error.retries:
            _journal_transport_retry(context, stage, retry)
        raise
    _journal_llm_calls(context, schema_name, model, calls)
    if stage != "execution":
        context.journal.append(
            context.run_id,
            "stage_finished",
            {"output": parsed, "stage": stage},
        )
    return cast(StrictContract, parsed)


def _journal_llm_calls(
    context: RunContext,
    schema_name: str,
    model: type[StrictContract],
    calls: list[LLMCall],
    *,
    cell_id: str | None = None,
) -> None:
    for index, call in enumerate(calls):
        for retry in call.transport_retries:
            _journal_transport_retry(context, call.stage, retry, cell_id=cell_id)
        payload: dict[str, object] = {
            "agent": call.agent,
            "attempt": call.attempt,
            "model": call.model,
            "parsed_ok": call.parsed_ok,
            "raw_response": call.raw_response,
            "schema_name": schema_name,
            "stage": call.stage,
            "latency_s": call.latency_s,
            "request_id": call.request_id,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cache_creation_input_tokens": call.cache_creation_input_tokens,
            "cache_read_input_tokens": call.cache_read_input_tokens,
            "transport_retry_count": len(call.transport_retries),
        }
        if cell_id is not None:
            payload["cell_id"] = cell_id
        context.journal.append(
            context.run_id,
            "llm_call",
            payload,
        )
        if not call.parsed_ok and index + 1 < len(calls):
            feedback = build_repair_feedback(call.raw_response, model)
            repair_payload: dict[str, object] = {
                "attempt": calls[index + 1].attempt,
                "error": feedback.summary,
                "failure_kind": feedback.failure_kind,
                "stage": call.stage,
            }
            if cell_id is not None:
                repair_payload["cell_id"] = cell_id
            context.journal.append(
                context.run_id,
                "repair_attempted",
                repair_payload,
            )


def _journal_transport_retry(
    context: RunContext,
    stage: str,
    retry: TransportRetry,
    *,
    cell_id: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "error_category": retry.error_category,
        "request_id": retry.request_id,
        "stage": stage,
        "status_code": retry.status_code,
        "transport_attempt": retry.transport_attempt,
        "wait_s": retry.wait_s,
    }
    if cell_id is not None:
        payload["cell_id"] = cell_id
    context.journal.append(context.run_id, "transport_retry", payload)


async def _call_telemetry_stage(
    context: RunContext,
    initial_metrics: object,
    scenario_id: str,
) -> TelemetryReport:
    if not isinstance(initial_metrics, Mapping):
        raise TypeError("read_metrics must return a mapping")
    cells: list[tuple[str, object]] = []
    for cell_id, cell_metrics in initial_metrics.items():
        if not isinstance(cell_id, str):
            raise TypeError("metric cell IDs must be strings")
        cells.append((cell_id, cell_metrics))
    cells.sort(key=lambda item: item[0])
    if not cells:
        raise ValueError("telemetry requires at least one cell")

    context.journal.append(context.run_id, "stage_started", {"stage": "telemetry"})
    semaphore = asyncio.Semaphore(context.max_concurrency)
    agent, schema_name, model = _STAGE_CONFIG["telemetry"]

    async def analyze_cell(cell_id: str, cell_metrics: object) -> _TelemetryCellResult:
        # Each fan-out task may use its own client because fixture draws depend on
        # seed, canonical prompt, and attempt, not client history. Otherwise,
        # scheduling could change the response sequence across tasks.
        client = context.llm_factory()
        messages = build_messages(
            "telemetry",
            {
                "cell_id": cell_id,
                "current_metrics": {cell_id: cell_metrics},
                "scenario_id": scenario_id,
            },
        )
        try:
            async with semaphore:
                parsed, calls = await complete_with_repair(
                    client,
                    messages,
                    schema_name,
                    model,
                    stage="telemetry",
                    agent=agent,
                )
        except (UnrepairableStageError, ProviderTransportError) as error:
            return _TelemetryCellResult(cell_id, None, tuple(client.calls), error)
        return _TelemetryCellResult(
            cell_id,
            cast(TelemetryReport, parsed),
            tuple(calls),
            None,
        )

    jobs = [analyze_cell(cell_id, cell_metrics) for cell_id, cell_metrics in cells]
    if context.parallel_telemetry:
        # gather preserves input order. Fan-out tasks perform no blocking I/O and never
        # touch SQLite; future blocking work belongs behind asyncio.to_thread.
        results = list(await asyncio.gather(*jobs))
    else:
        results = [await job for job in jobs]

    # Buffering trades away mid-fan-out call records: a crash can lose these model-only
    # events, so recovery reruns the unfinished stage. The model call has no side effect,
    # and FakeModel deterministically reproduces it. Only the parent writes the journal,
    # in sorted cell order with each cell's attempts kept contiguous.
    results.sort(key=lambda result: result.cell_id)
    for result in results:
        _journal_llm_calls(
            context,
            schema_name,
            model,
            list(result.calls),
            cell_id=result.cell_id,
        )
    for result in results:
        if result.error is not None:
            if isinstance(result.error, ProviderTransportError):
                for retry in result.error.retries:
                    _journal_transport_retry(
                        context,
                        "telemetry",
                        retry,
                        cell_id=result.cell_id,
                    )
            raise result.error

    reports = [cast(TelemetryReport, result.report) for result in results]
    merged = TelemetryReport(
        findings=[finding for report in reports for finding in report.findings],
        summary="\n".join(
            f"{result.cell_id}: {report.summary}"
            for result, report in zip(results, reports, strict=True)
        ),
    )
    context.journal.append(
        context.run_id,
        "stage_finished",
        {"output": merged, "stage": "telemetry"},
    )
    return merged


def _post_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("read_metrics must return a mapping")
    flattened: dict[str, float] = {}
    for cell_id, metrics in value.items():
        if not isinstance(cell_id, str):
            raise TypeError("metric cell IDs must be strings")
        fields: Mapping[Any, Any]
        if isinstance(metrics, BaseModel):
            fields = metrics.model_dump(mode="python")
        elif is_dataclass(metrics) and not isinstance(metrics, type):
            fields = asdict(metrics)
        elif isinstance(metrics, Mapping):
            fields = metrics
        else:
            raise TypeError("cell metrics must be a model, dataclass, or mapping")
        for name, metric in fields.items():
            if (
                not isinstance(name, str)
                or isinstance(metric, bool)
                or not isinstance(metric, (int, float))
            ):
                raise TypeError("metric fields must have string names and numeric values")
            flattened[f"{cell_id}.{name}"] = float(metric)
    return flattened


async def run_graph(
    context: RunContext,
    *,
    apply_action: Callable[[NetworkState, ActionValue], NetworkState],
    read_metrics: Callable[[NetworkState], object],
    grade_run: Callable[[NetworkState, Sequence[ActionValue], Scenario], GradeResult],
    recovered: Mapping[str, StrictContract] | None = None,
    start_new: bool = True,
) -> RunResult:
    """Run missing stages once, then grade the resulting action sequence."""

    scenario_id = context.scenario.id
    completed = dict(recovered or {})
    if start_new:
        context.journal.create_run(
            context.run_id,
            scenario_id,
            context.state,
            model=context.llm.model,
        )
        context.journal.append(
            context.run_id,
            "run_started",
            {"model": context.llm.model, "scenario_id": scenario_id},
        )

    initial_metrics = read_metrics(context.state)
    recovered_telemetry = completed.get("telemetry")
    if recovered_telemetry is None:
        telemetry = await _call_telemetry_stage(
            context,
            initial_metrics,
            scenario_id,
        )
    else:
        telemetry = cast(TelemetryReport, recovered_telemetry)
    recovered_policy = completed.get("policy")
    if recovered_policy is None:
        policy = cast(
            PolicyPlan,
            await _call_stage(
                context,
                "policy",
                {
                    "current_metrics": initial_metrics,
                    "scenario_id": scenario_id,
                    "telemetry": telemetry,
                },
            ),
        )
    else:
        policy = cast(PolicyPlan, recovered_policy)
    recovered_validation = completed.get("validation")
    if recovered_validation is None:
        # Validation stays serial: the pure-code SLA check is microsecond-scale beside a
        # model call, so concurrency has no measurable benefit and adds a failure mode.
        validation = cast(
            ValidationVerdict,
            await _call_stage(
                context,
                "validation",
                {
                    "policy": policy,
                    "scenario_id": scenario_id,
                    "telemetry": telemetry,
                },
            ),
        )
    else:
        validation = cast(ValidationVerdict, recovered_validation)

    if validation.approved:
        selected_actions = list(policy.actions)
    elif validation.revised_actions:
        selected_actions = list(validation.revised_actions)
    else:
        grade = RejectedGrade(
            success=False,
            reasons=[f"validation_rejected: {violation}" for violation in validation.violations]
            or ["validation_rejected: no revised actions"],
        )
        context.journal.append(
            context.run_id,
            "run_finished",
            {"grade": grade, "status": "rejected"},
            run_status="rejected",
        )
        return RunResult(
            status="rejected",
            grade=grade,
            final_state=context.state,
            action_log=tuple(context.action_log),
            telemetry=telemetry,
            policy=policy,
            validation=validation,
            execution=None,
        )

    execution = cast(ExecutionResult | None, completed.get("execution"))
    if execution is None:
        await _call_stage(
            context,
            "execution",
            {
                "current_metrics": read_metrics(context.state),
                "scenario_id": scenario_id,
                "selected_actions": selected_actions,
                "validation": validation,
            },
        )
        for step, action in enumerate(selected_actions):
            result = run_tool(
                context.state,
                action,
                context.action_log,
                apply_action=apply_action,
                read_metrics=read_metrics,
                journal=context.journal,
                run_id=context.run_id,
                stage="execution",
                step=step,
            )
            context.state = result.state

        execution = ExecutionResult(
            applied=selected_actions,
            skipped=[],
            post_metrics=_post_metrics(read_metrics(context.state)),
        )
        context.journal.append(
            context.run_id,
            "stage_finished",
            {"output": execution, "stage": "execution"},
        )
    final_grade = grade_run(
        context.state,
        tuple(context.action_log),
        context.scenario,
    )
    context.journal.append(
        context.run_id,
        "run_finished",
        {"grade": final_grade, "status": "finished"},
        run_status="finished",
    )
    return RunResult(
        status="finished",
        grade=final_grade,
        final_state=context.state,
        action_log=tuple(context.action_log),
        telemetry=telemetry,
        policy=policy,
        validation=validation,
        execution=execution,
    )
