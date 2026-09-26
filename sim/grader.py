"""Pure-code semantic grader for simulator action sequences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sim.network import ActionValue, CellMetrics, NetworkState, metrics
from sim.scenarios.model import MetricRequirement, Scenario


@dataclass(frozen=True)
class GradeResult:
    success: bool
    reasons: list[str]


@dataclass(frozen=True)
class GradeContext:
    final_state: NetworkState
    action_log: Sequence[ActionValue]
    scenario: Scenario
    final_metrics: dict[str, CellMetrics]


def _metric_failure(
    check_name: str,
    requirement: MetricRequirement,
    context: GradeContext,
) -> str | None:
    threshold = context.scenario.sla[requirement.metric_name]
    actual = float(getattr(context.final_metrics[requirement.cell_id], requirement.metric_name))
    if threshold.contains(actual):
        return None
    return (
        f"{check_name}: cell={requirement.cell_id} metric={requirement.metric_name} "
        f"expected={threshold.describe()} actual={actual:.6f}"
    )


def check_must_restore(context: GradeContext) -> list[str]:
    return [
        failure
        for requirement in context.scenario.success.must_restore
        if (failure := _metric_failure("must_restore", requirement, context)) is not None
    ]


def check_must_not_break(context: GradeContext) -> list[str]:
    return [
        failure
        for requirement in context.scenario.success.must_not_break
        if (failure := _metric_failure("must_not_break", requirement, context)) is not None
    ]


def check_required_actions(context: GradeContext) -> list[str]:
    requirement = context.scenario.success.required_actions
    if requirement.exact:
        if len(requirement.matchers) != len(context.action_log):
            return [
                "required_actions: exact sequence length "
                f"expected={len(requirement.matchers)} actual={len(context.action_log)}"
            ]
        return [
            f"required_actions: position={index} expected={matcher.describe()} actual={action.tool}"
            for index, (matcher, action) in enumerate(
                zip(requirement.matchers, context.action_log, strict=True)
            )
            if not matcher.matches(action)
        ]

    reasons: list[str] = []
    for matcher in requirement.matchers:
        if not any(matcher.matches(action) for action in context.action_log):
            reasons.append(f"required_actions: missing expected={matcher.describe()}")
    return reasons


def check_forbidden_actions(context: GradeContext) -> list[str]:
    reasons: list[str] = []
    for index, action in enumerate(context.action_log):
        for matcher in context.scenario.success.forbidden_actions:
            if matcher.matches(action):
                reasons.append(
                    f"forbidden_actions: position={index} expected=not({matcher.describe()}) "
                    f"actual={action.tool}"
                )
    return reasons


def check_max_steps(context: GradeContext) -> list[str]:
    maximum = context.scenario.success.max_steps
    if len(context.action_log) <= maximum:
        return []
    return [f"max_steps: expected<= {maximum} actual={len(context.action_log)}"]


Check = tuple[str, Callable[[GradeContext], list[str]]]
CHECKS: list[Check] = [
    ("must_restore", check_must_restore),
    ("must_not_break", check_must_not_break),
    ("required_actions", check_required_actions),
    ("forbidden_actions", check_forbidden_actions),
    ("max_steps", check_max_steps),
]


def run_checks(
    checks: Sequence[Check],
    final_state: NetworkState,
    action_log: Sequence[ActionValue],
    scenario: Scenario,
) -> GradeResult:
    context = GradeContext(final_state, action_log, scenario, metrics(final_state))
    reasons = [reason for _, check in checks for reason in check(context)]
    return GradeResult(success=not reasons, reasons=reasons)


def grade(
    final_state: NetworkState,
    action_log: Sequence[ActionValue],
    scenario: Scenario,
) -> GradeResult:
    return run_checks(CHECKS, final_state, action_log, scenario)
