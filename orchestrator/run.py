"""Command-line composition root for the orchestration pipeline."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import cast

from orchestrator.graph import RunContext, run_graph
from orchestrator.journal import Journal, canonical_json
from orchestrator.llm.anthropic import MODEL_ID, AnthropicClient, ProviderTransportError
from orchestrator.llm.base import LLMClient
from orchestrator.llm.fake import FakeModel
from orchestrator.repair import UnrepairableStageError
from sim.grader import grade
from sim.network import apply, metrics
from sim.oracle_model import OracleModel
from sim.scenarios import SCENARIOS_BY_ID


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS_BY_ID))
    parser.add_argument(
        "--model",
        required=True,
        choices=("fake:scripted", "oracle:standard", "oracle:trap", MODEL_ID),
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    scheduling = parser.add_mutually_exclusive_group()
    scheduling.add_argument("--serial", dest="parallel", action="store_false")
    scheduling.add_argument("--parallel", dest="parallel", action="store_true")
    parser.set_defaults(parallel=True)
    # Reproduce this configured input: python -m orchestrator.run --help
    parser.add_argument("--max-concurrency", type=_positive_int, default=8)
    return parser


def _model(name: str) -> LLMClient:
    if name == "fake:scripted":
        return FakeModel(modes=("scripted",))
    if name == "oracle:standard":
        return OracleModel("standard")
    if name == "oracle:trap":
        return OracleModel("trap")
    if name == MODEL_ID:
        return AnthropicClient()
    raise ValueError(f"unknown model: {name}")


async def _run(args: argparse.Namespace) -> int:
    scenario = SCENARIOS_BY_ID[cast(str, args.scenario)]
    database = cast(Path, args.db)
    database.parent.mkdir(parents=True, exist_ok=True)
    model_name = cast(str, args.model)

    def model_factory() -> LLMClient:
        return _model(model_name)

    with Journal(database) as journal:
        context = RunContext(
            run_id=cast(str, args.run_id),
            scenario=scenario,
            state=scenario.initial_state,
            action_log=[],
            journal=journal,
            llm=model_factory(),
            llm_factory=model_factory,
            parallel_telemetry=cast(bool, getattr(args, "parallel", True)),
            max_concurrency=cast(int, getattr(args, "max_concurrency", 8)),
        )
        try:
            result = await run_graph(
                context,
                apply_action=apply,
                read_metrics=metrics,
                grade_run=grade,
            )
        except UnrepairableStageError as error:
            # Retain unrepairable runs as graded failures, not missing observations.
            journal.append(
                context.run_id,
                "run_finished",
                {
                    "attempts": error.attempts,
                    "error": error.last_error,
                    "stage": error.stage,
                    "status": "unrepairable",
                },
                run_status="unrepairable",
            )
            output = {
                "action_log": [action.model_dump(mode="json") for action in context.action_log],
                "attempts": error.attempts,
                "error": error.last_error,
                "journal": str(database.resolve()),
                "stage": error.stage,
                "status": "unrepairable",
            }
        except ProviderTransportError as error:
            payload = {
                "error_category": error.category,
                "request_id": error.request_id,
                "status": "provider_transport_error",
                "status_code": error.status_code,
            }
            journal.append(
                context.run_id,
                "run_finished",
                payload,
                run_status="provider_error",
            )
            print(canonical_json(payload))
            return 2
        else:
            output = {
                "action_log": [action.model_dump(mode="json") for action in result.action_log],
                "grade": {
                    "reasons": result.grade.reasons,
                    "success": result.grade.success,
                },
                "journal": str(database.resolve()),
                "status": result.status,
            }

    print(canonical_json(output))
    return 0


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
