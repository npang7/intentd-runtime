"""Bounded schema repair for provider responses.

The adversarial fixture emits invalid output independently of repair feedback.
Its tests cover retry bounds, feedback, validity accounting, and the expected
probability pattern. They do not establish repair effectiveness with a live provider.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ValidationError

from orchestrator.contracts import (
    GetMetrics,
    NoOp,
    ReportNonRadioIssue,
    SetA3Offset,
    SetAdmissionThreshold,
    SetTilt,
    SetTxPower,
    StrictContract,
)
from orchestrator.llm.base import LLMCall, LLMClient, Message

FailureKind = Literal["malformed", "schema"]

_ACTION_MODELS: dict[str, type[StrictContract]] = {
    "get_metrics": GetMetrics,
    "set_tx_power": SetTxPower,
    "set_a3_offset": SetA3Offset,
    "set_tilt": SetTilt,
    "set_admission_threshold": SetAdmissionThreshold,
    "report_non_radio_issue": ReportNonRadioIssue,
    "no_op": NoOp,
}


@dataclass(frozen=True, slots=True)
class RepairFeedback:
    failure_kind: FailureKind
    summary: str
    message: str


class UnrepairableStageError(RuntimeError):
    def __init__(self, stage: str, attempts: int, last_error: str) -> None:
        self.stage = stage
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"stage {stage!r} remained invalid after {attempts} attempts: {last_error}"
        )


def _json_value(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return repr(value)


def _normalized_path(loc: tuple[object, ...], parsed: object) -> str:
    parts = list(loc)
    if (
        len(parts) >= 3
        and parts[0] == "actions"
        and isinstance(parts[1], int)
        and isinstance(parsed, dict)
    ):
        actions = parsed.get("actions")
        if isinstance(actions, list) and parts[1] < len(actions):
            action = actions[parts[1]]
            if isinstance(action, dict) and parts[2] == action.get("tool"):
                del parts[2]
    return ".".join(str(part) for part in parts)


def _field_bounds(
    model_cls: type[BaseModel], loc: tuple[object, ...], parsed: object
) -> tuple[object | None, object | None]:
    field = None
    if (
        len(loc) >= 4
        and loc[0] == "actions"
        and isinstance(loc[1], int)
        and isinstance(parsed, dict)
    ):
        actions = parsed.get("actions")
        if isinstance(actions, list) and loc[1] < len(actions):
            action = actions[loc[1]]
            if isinstance(action, dict):
                action_tool = action.get("tool")
                action_model = (
                    _ACTION_MODELS.get(action_tool) if isinstance(action_tool, str) else None
                )
                if action_model is not None and isinstance(loc[-1], str):
                    field = action_model.model_fields.get(loc[-1])
    elif len(loc) == 1 and isinstance(loc[0], str):
        field = model_cls.model_fields.get(loc[0])

    if field is None:
        return None, None
    lower = None
    upper = None
    for item in field.metadata:
        lower = getattr(item, "ge", lower)
        upper = getattr(item, "le", upper)
    return lower, upper


def _schema_feedback(
    model_cls: type[BaseModel], parsed: object, error: ValidationError
) -> RepairFeedback:
    lines: list[str] = []
    for detail in error.errors(include_url=False):
        loc = cast(tuple[object, ...], detail["loc"])
        path = _normalized_path(loc, parsed)
        error_type = detail["type"]
        actual = detail.get("input")
        context = detail.get("ctx") or {}
        if error_type == "union_tag_invalid":
            discriminator = str(context.get("discriminator", "tool")).strip("'")
            path = f"{path}.{discriminator}" if path else discriminator
            expected = f"discriminator one of {context.get('expected_tags', 'the declared tags')}"
            actual = context.get("tag", actual)
        elif error_type in {"less_than_equal", "greater_than_equal"}:
            lower, upper = _field_bounds(model_cls, loc, parsed)
            if lower is not None and upper is not None:
                expected = f"range {lower:g}..{upper:g}"
            elif "le" in context:
                expected = f"value <= {context['le']:g}"
            else:
                expected = f"value >= {context['ge']:g}"
        elif error_type == "extra_forbidden":
            expected = "no extra field"
        else:
            expected = str(detail["msg"])
        lines.append(f"{path}: expected {expected}; received {_json_value(actual)}")

    summary = "; ".join(lines)
    # Never resend the full schema: it increases token cost and encourages rewriting the
    # entire response instead of correcting the specific invalid locations.
    message = "Correct only these schema errors and return JSON only:\n" + "\n".join(lines)
    return RepairFeedback("schema", summary, message)


def build_repair_feedback(raw_response: str, model_cls: type[BaseModel]) -> RepairFeedback:
    """Build compact feedback for malformed JSON or a schema-invalid JSON value."""

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as error:
        summary = (
            "output is not valid JSON: "
            f"line {error.lineno}, column {error.colno}, character {error.pos}"
        )
        return RepairFeedback("malformed", summary, summary + ". Return valid JSON only.")

    try:
        model_cls.model_validate(parsed)
    except ValidationError as error:
        return _schema_feedback(model_cls, parsed, error)
    raise ValueError("repair feedback requested for a valid response")


async def complete_with_repair(
    llm: LLMClient,
    messages: Sequence[Message],
    schema_name: str,
    model_cls: type[BaseModel],
    *,
    stage: str,
    agent: str,
    max_repair: int = 2,
) -> tuple[BaseModel, list[LLMCall]]:
    """Complete one logical stage with a bounded number of schema repair attempts."""

    if max_repair < 0:
        raise ValueError("max_repair must be nonnegative")
    started_at = len(llm.calls)
    working_messages = [
        Message(role=message["role"], content=message["content"]) for message in messages
    ]
    last_error = ""
    for attempt in range(max_repair + 1):
        raw_response = await llm.complete(
            working_messages,
            schema_name,
            stage=stage,
            agent=agent,
            attempt=attempt,
        )
        try:
            parsed_json = json.loads(raw_response)
            parsed = model_cls.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError):
            feedback = build_repair_feedback(raw_response, model_cls)
            last_error = feedback.summary
            working_messages.append(Message(role="assistant", content=raw_response))
            working_messages.append(Message(role="user", content=feedback.message))
            if attempt == max_repair:
                raise UnrepairableStageError(stage, attempt + 1, last_error) from None
        else:
            return parsed, list(llm.calls[started_at:])
    raise AssertionError("bounded repair loop exited unexpectedly")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class ValidityStats:
    total_stage_calls: int
    invalid_first_count: int
    repaired_count: int
    unrepairable_count: int

    @property
    def invalid_first_emission(self) -> float | None:
        if self.total_stage_calls == 0:
            return None
        return self.invalid_first_count / self.total_stage_calls

    @property
    def repaired(self) -> float | None:
        if self.invalid_first_count == 0:
            return None
        return self.repaired_count / self.invalid_first_count

    @property
    def unrepairable(self) -> float | None:
        if self.total_stage_calls == 0:
            return None
        return self.unrepairable_count / self.total_stage_calls


def validity_stats(calls: list[LLMCall]) -> ValidityStats:
    """Derive validity counts and ratios from ordered calls.

    When aggregating runs, sum the counts before dividing; never average ratios. An
    unrepairable run stops early and fan-out makes stage-call counts scenario-dependent, so
    averaging ratios would give runs with smaller denominators excessive weight.
    """

    groups: list[list[LLMCall]] = []
    for call in calls:
        if call.attempt == 0:
            groups.append([call])
            continue
        if not groups or call.attempt != groups[-1][-1].attempt + 1:
            raise ValueError("LLMCall attempts must start at zero and increase contiguously")
        groups[-1].append(call)

    invalid_first = sum(not group[0].parsed_ok for group in groups)
    repaired = sum(not group[0].parsed_ok and group[-1].parsed_ok for group in groups)
    unrepairable = sum(not group[-1].parsed_ok for group in groups)
    return ValidityStats(len(groups), invalid_first, repaired, unrepairable)
