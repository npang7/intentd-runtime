"""Strict contracts at each orchestration stage boundary."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class StrictContract(BaseModel):
    """Reject coercion and undeclared fields so invalid-output metrics stay meaningful."""

    model_config = ConfigDict(extra="forbid", strict=True)


NonEmptyStr = Annotated[str, Field(min_length=1)]
# Window duration is intentionally integer-only. A float is a real contract violation that
# should be recorded as initially invalid and corrected by one repair attempt.
WindowSeconds = Annotated[int, Field(ge=1, le=3600)]
TxPowerDbm = Annotated[float, Field(ge=20, le=46)]
A3OffsetDb = Annotated[float, Field(ge=-10, le=10)]
TiltDegrees = Annotated[float, Field(ge=0, le=15)]
AdmissionThreshold = Annotated[float, Field(ge=0, le=1)]
MetricValue = Annotated[float, Field(strict=True)]


class GetMetrics(StrictContract):
    tool: Literal["get_metrics"]
    cell_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    window_s: WindowSeconds


class SetTxPower(StrictContract):
    tool: Literal["set_tx_power"]
    cell_id: NonEmptyStr
    tx_power_dbm: TxPowerDbm


class SetA3Offset(StrictContract):
    tool: Literal["set_a3_offset"]
    cell_id: NonEmptyStr
    a3_offset_db: A3OffsetDb


class SetTilt(StrictContract):
    tool: Literal["set_tilt"]
    cell_id: NonEmptyStr
    tilt_deg: TiltDegrees


class SetAdmissionThreshold(StrictContract):
    tool: Literal["set_admission_threshold"]
    cell_id: NonEmptyStr
    threshold: AdmissionThreshold


class ReportNonRadioIssue(StrictContract):
    tool: Literal["report_non_radio_issue"]
    cell_id: NonEmptyStr
    category: Literal["backhaul", "core", "transport", "unknown"]
    detail: NonEmptyStr


class NoOp(StrictContract):
    tool: Literal["no_op"]
    reason: NonEmptyStr


ActionValue: TypeAlias = (
    GetMetrics
    | SetTxPower
    | SetA3Offset
    | SetTilt
    | SetAdmissionThreshold
    | ReportNonRadioIssue
    | NoOp
)

Action = Annotated[
    ActionValue,
    Field(discriminator="tool"),
]


class TelemetryFinding(StrictContract):
    cell_id: NonEmptyStr
    metric: NonEmptyStr
    detail: NonEmptyStr


class TelemetryReport(StrictContract):
    findings: list[TelemetryFinding]
    summary: NonEmptyStr


class PolicyPlan(StrictContract):
    actions: Annotated[list[Action], Field(min_length=1)]
    rationale: NonEmptyStr
    expected_effect: NonEmptyStr


class ValidationVerdict(StrictContract):
    approved: bool
    violations: list[NonEmptyStr]
    revised_actions: list[Action]


class ExecutionResult(StrictContract):
    applied: list[Action]
    skipped: list[Action]
    post_metrics: dict[NonEmptyStr, MetricValue]


SCHEMA_MODELS: dict[str, type[StrictContract]] = {
    "TelemetryReport": TelemetryReport,
    "PolicyPlan": PolicyPlan,
    "ValidationVerdict": ValidationVerdict,
    "ExecutionResult": ExecutionResult,
}


def response_parses(schema_name: str, raw_response: str) -> bool:
    """Return whether raw text satisfies the named stage contract."""

    model = SCHEMA_MODELS.get(schema_name)
    if model is None:
        raise ValueError(f"unknown schema: {schema_name}")
    try:
        model.model_validate_json(raw_response)
    except ValueError:
        return False
    return True
