"""Run each selected scenario once and append raw provider evidence to JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import anthropic

from bench.analyze import Usage, usage_cost_usd, usage_from_mapping
from orchestrator.graph import RunContext, RunResult, run_graph
from orchestrator.journal import Journal, canonical_json
from orchestrator.llm.anthropic import (
    MAX_TOKENS,
    MODEL_ID,
    AnthropicClient,
    ProviderTransportError,
)
from orchestrator.llm.base import LLMCall, LLMClient, Message
from orchestrator.repair import UnrepairableStageError
from sim.grader import grade
from sim.network import apply, metrics, state_hash
from sim.scenarios import ALL_SCENARIOS, SCENARIOS_BY_ID

# Reproduce runner inputs: python -m bench.run_eval --help
DEFAULT_MAX_SPEND_USD = 5.0
_FIRST_SCENARIOS = ("overload_dense_01", "conflicting_sla_01", "false_alarm_01")
SCENARIO_ORDER = _FIRST_SCENARIOS + tuple(
    scenario.id for scenario in ALL_SCENARIOS if scenario.id not in _FIRST_SCENARIOS
)


class BudgetExceeded(RuntimeError):
    """The next transport attempt would exceed the configured local cap."""


class SpendGuard:
    """Conservatively reserve worst-case cost before each Messages attempt."""

    def __init__(self, max_spend_usd: float, initial_usage: Usage | None = None) -> None:
        self.max_spend_usd = max_spend_usd
        self.known_usage = initial_usage or Usage()
        self.unknown_attempt_reserve_usd = 0.0

    async def reserve(
        self,
        client: AnthropicClient,
        messages: Sequence[Message],
        schema_name: str,
        transport_attempt: int,
    ) -> object:
        input_tokens = await client.count_prefix_tokens(messages, schema_name)
        worst_case = usage_cost_usd(
            Usage(
                cache_creation_input_tokens=input_tokens,
                output_tokens=MAX_TOKENS,
            )
        )
        projected = usage_cost_usd(self.known_usage) + self.unknown_attempt_reserve_usd + worst_case
        if projected > self.max_spend_usd:
            raise BudgetExceeded(
                "local spend guard rejected provider attempt "
                f"{transport_attempt}: projected={projected:.6f}, "
                f"limit={self.max_spend_usd:.6f}"
            )
        self.unknown_attempt_reserve_usd += worst_case
        return worst_case

    def success(self, reservation: object, call: LLMCall) -> None:
        if not isinstance(reservation, float):
            raise TypeError("budget reservation must be a float")
        self.unknown_attempt_reserve_usd -= reservation
        self.known_usage += Usage(
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cache_creation_input_tokens=call.cache_creation_input_tokens,
            cache_read_input_tokens=call.cache_read_input_tokens,
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _spend_cap(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed > DEFAULT_MAX_SPEND_USD:
        raise argparse.ArgumentTypeError(
            f"must be positive and no greater than {DEFAULT_MAX_SPEND_USD:g}"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(MODEL_ID,))
    parser.add_argument("--limit", required=True, type=_positive_int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-spend-usd",
        type=_spend_cap,
        default=DEFAULT_MAX_SPEND_USD,
    )
    parser.add_argument("--parallel", action="store_true")
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value: Any = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("results JSONL rows must be objects")
        rows.append(value)
    return rows


def _existing_usage(rows: Sequence[Mapping[str, object]], model: str, mode: str) -> Usage:
    total = Usage()
    for row in rows:
        if (
            row.get("record_type") == "run"
            and row.get("model") == model
            and row.get("mode") == mode
        ):
            raw = row.get("usage")
            if isinstance(raw, Mapping):
                total += usage_from_mapping(raw)
    return total


def scenarios_to_run(
    rows: Sequence[Mapping[str, object]],
    model: str,
    mode: str,
    limit: int,
) -> list[str]:
    selected = SCENARIO_ORDER[:limit]
    completed = {
        str(row.get("scenario_id"))
        for row in rows
        if row.get("record_type") == "run"
        and row.get("model") == model
        and row.get("mode") == mode
        and row.get("status") == "finished"
    }
    return [scenario_id for scenario_id in selected if scenario_id not in completed]


def _eligible_cache_stages(
    rows: Sequence[Mapping[str, object]], model: str, commit_sha: str
) -> frozenset[str]:
    return frozenset(
        str(row["stage"])
        for row in rows
        if row.get("record_type") == "prefix_measurement"
        and row.get("model") == model
        and row.get("commit_sha") == commit_sha
        and row.get("cache_eligible") is True
        and isinstance(row.get("stage"), str)
    )


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _journal_facts(
    database: Path, run_id: str
) -> tuple[dict[str, float], dict[str, Usage], int, int]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT ts, kind, payload FROM events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()

    starts: dict[str, float] = {}
    stage_latency: dict[str, float] = {}
    usage_by_stage: dict[str, Usage] = {}
    schema_repairs = 0
    transport_retries = 0
    for ts_value, kind_value, payload_text in rows:
        ts = float(ts_value)
        kind = str(kind_value)
        payload: Any = json.loads(str(payload_text))
        if not isinstance(payload, dict):
            continue
        stage_value = payload.get("stage")
        stage = stage_value if isinstance(stage_value, str) else ""
        if kind == "stage_started":
            starts.setdefault(stage, ts)
        elif kind == "stage_finished" and stage in starts:
            stage_latency[stage] = ts - starts[stage]
        elif kind == "llm_call":
            usage = usage_from_mapping(payload)
            usage_by_stage[stage] = usage_by_stage.get(stage, Usage()) + usage
        elif kind == "repair_attempted":
            schema_repairs += 1
        elif kind == "transport_retry":
            transport_retries += 1
    return stage_latency, usage_by_stage, schema_repairs, transport_retries


def _usage_mapping(usage: Usage) -> dict[str, int]:
    return {
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _result_row(
    *,
    commit_sha: str,
    database: Path,
    elapsed_s: float,
    mode: str,
    model: str,
    result: RunResult,
    run_id: str,
    scenario_id: str,
) -> dict[str, object]:
    stage_latency, usage_by_stage, schema_repairs, transport_retries = _journal_facts(
        database, run_id
    )
    total = Usage()
    for usage in usage_by_stage.values():
        total += usage
    return {
        "action_log": [action.model_dump(mode="json") for action in result.action_log],
        "commit_sha": commit_sha,
        "grader": {"reasons": result.grade.reasons, "success": result.grade.success},
        "latency_s": elapsed_s,
        "mode": mode,
        "model": model,
        "record_type": "run",
        "run_id": run_id,
        "scenario_id": scenario_id,
        "schema_repair_count": schema_repairs,
        "stage_latency_s": stage_latency,
        "status": result.status,
        "state_hash": state_hash(result.final_state),
        "transport_retry_count": transport_retries,
        "usage": _usage_mapping(total),
        "usage_by_stage": {
            stage: _usage_mapping(usage) for stage, usage in sorted(usage_by_stage.items())
        },
    }


def _append_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def _database_path(output: Path, run_id: str) -> Path:
    database = output.parent / "runs" / f"{run_id}.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    return database


async def _run(args: argparse.Namespace) -> int:
    model_name = cast(str, args.model)
    output = cast(Path, args.output)
    mode = "parallel" if cast(bool, args.parallel) else "serial"
    rows = _load_rows(output)
    commit_sha = _git_sha()
    selected = scenarios_to_run(rows, model_name, mode, cast(int, args.limit))
    guard = SpendGuard(
        cast(float, args.max_spend_usd),
        _existing_usage(rows, model_name, mode),
    )
    cache_stages = _eligible_cache_stages(rows, model_name, commit_sha)

    def model_factory() -> LLMClient:
        holder: dict[str, AnthropicClient] = {}

        async def reserve(
            messages: Sequence[Message], schema_name: str, transport_attempt: int
        ) -> object:
            return await guard.reserve(holder["client"], messages, schema_name, transport_attempt)

        client = AnthropicClient(
            before_transport=reserve,
            transport_succeeded=guard.success,
            cache_stages=cache_stages,
        )
        holder["client"] = client
        return client

    for scenario_id in selected:
        scenario = SCENARIOS_BY_ID[scenario_id]
        run_id = f"{scenario_id}-{uuid.uuid4().hex}"
        database = _database_path(output, run_id)
        client = model_factory()
        started = time.perf_counter()
        with Journal(database) as journal:
            context = RunContext(
                run_id=run_id,
                scenario=scenario,
                state=scenario.initial_state,
                action_log=[],
                journal=journal,
                llm=client,
                llm_factory=model_factory,
                parallel_telemetry=cast(bool, args.parallel),
            )
            try:
                result = await run_graph(
                    context,
                    apply_action=apply,
                    read_metrics=metrics,
                    grade_run=grade,
                )
            except UnrepairableStageError as error:
                journal.append(
                    run_id,
                    "run_finished",
                    {
                        "attempts": error.attempts,
                        "error": error.last_error,
                        "stage": error.stage,
                        "status": "unrepairable",
                    },
                    run_status="unrepairable",
                )
                elapsed_s = time.perf_counter() - started
                stage_latency, usage_by_stage, schema_repairs, transport_retries = _journal_facts(
                    database, run_id
                )
                total = Usage()
                for usage in usage_by_stage.values():
                    total += usage
                row: dict[str, object] = {
                    "action_log": [action.model_dump(mode="json") for action in context.action_log],
                    "commit_sha": commit_sha,
                    "grader": {
                        "reasons": [f"unrepairable {error.stage}: {error.last_error}"],
                        "success": False,
                    },
                    "latency_s": elapsed_s,
                    "mode": mode,
                    "model": model_name,
                    "record_type": "run",
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "schema_repair_count": schema_repairs,
                    "stage_latency_s": stage_latency,
                    "status": "unrepairable",
                    "state_hash": state_hash(context.state),
                    "transport_retry_count": transport_retries,
                    "usage": _usage_mapping(total),
                    "usage_by_stage": {
                        stage: _usage_mapping(usage)
                        for stage, usage in sorted(usage_by_stage.items())
                    },
                }
                _append_row(output, row)
                print(canonical_json(row))
                continue
        row = _result_row(
            commit_sha=commit_sha,
            database=database,
            elapsed_s=time.perf_counter() - started,
            mode=mode,
            model=model_name,
            result=result,
            run_id=run_id,
            scenario_id=scenario_id,
        )
        _append_row(output, row)
        print(canonical_json(row))
    return 0


def main() -> int:
    try:
        return asyncio.run(_run(_parser().parse_args()))
    except BudgetExceeded as error:
        print(canonical_json({"error": str(error), "status": "budget_exceeded"}), file=sys.stderr)
        return 2
    except ProviderTransportError as error:
        print(
            canonical_json(
                {
                    "error_category": error.category,
                    "request_id": error.request_id,
                    "status": "provider_transport_error",
                    "status_code": error.status_code,
                }
            ),
            file=sys.stderr,
        )
        return 2
    except anthropic.APIError as error:
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {})
        request_id = headers.get("request-id") or headers.get("x-request-id")
        print(
            canonical_json(
                {
                    "error_category": type(error).__name__,
                    "request_id": request_id,
                    "status": "provider_error",
                    "status_code": status_code,
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
