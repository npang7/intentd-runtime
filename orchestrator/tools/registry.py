"""Stable registry of the action contracts exposed to orchestration stages."""

from __future__ import annotations

import json
from collections.abc import Iterator

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

TOOL_MODELS: tuple[type[StrictContract], ...] = (
    GetMetrics,
    SetTxPower,
    SetA3Offset,
    SetTilt,
    SetAdmissionThreshold,
    ReportNonRadioIssue,
    NoOp,
)


def iter_tool_schemas() -> Iterator[tuple[str, dict[str, object]]]:
    """Yield tool names and schemas in registry order."""

    for model in TOOL_MODELS:
        schema = model.model_json_schema()
        properties = schema.get("properties")
        if not isinstance(properties, dict):  # pragma: no cover - Pydantic schema guard
            raise TypeError(f"{model.__name__} has no properties schema")
        tool_schema = properties.get("tool")
        if not isinstance(tool_schema, dict):  # pragma: no cover - Pydantic schema guard
            raise TypeError(f"{model.__name__}.tool has no schema")
        tool_name = tool_schema.get("const")
        if not isinstance(tool_name, str):  # pragma: no cover - registry construction guard
            raise TypeError(f"{model.__name__}.tool must have a string literal")
        yield tool_name, schema


def export_tool_definitions() -> str:
    """Return byte-stable tool definitions for the static prompt prefix."""

    definitions = [{"name": name, "parameters": schema} for name, schema in iter_tool_schemas()]
    return json.dumps(definitions, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
