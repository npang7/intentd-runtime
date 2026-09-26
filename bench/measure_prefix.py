"""Measure stable provider-formatted stage prefixes with the token-count API."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from orchestrator.graph import build_messages
from orchestrator.journal import canonical_json
from orchestrator.llm.anthropic import API_MODEL, MODEL_ID, AnthropicClient

# Reproduce configured input: python -m bench.measure_prefix --help
CACHE_MIN_TOKENS = 4096
CACHE_SOURCE = "https://platform.claude.com/docs/en/build-with-claude/prompt-caching"
_STAGES = {
    "telemetry": "TelemetryReport",
    "policy": "PolicyPlan",
    "validation": "ValidationVerdict",
    "execution": "ExecutionResult",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(API_MODEL,))
    parser.add_argument("--output", type=Path, default=Path("results/runs.jsonl"))
    parser.add_argument("--cache-min-tokens", type=int, default=CACHE_MIN_TOKENS)
    return parser


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value: Any = json.loads(line)
        if isinstance(value, dict) and value.get("record_type") == "prefix_measurement":
            keys.add(
                (
                    str(value.get("commit_sha")),
                    str(value.get("model")),
                    str(value.get("stage")),
                )
            )
    return keys


async def _run(args: argparse.Namespace) -> int:
    output = cast(Path, args.output)
    threshold = cast(int, args.cache_min_tokens)
    if threshold <= 0:
        raise ValueError("cache-min-tokens must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    commit_sha = _git_sha()
    existing = _existing_keys(output)
    client = AnthropicClient()
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for stage, schema_name in _STAGES.items():
            key = (commit_sha, MODEL_ID, stage)
            if key in existing:
                continue
            messages = build_messages(stage, {})
            input_tokens = await client.count_prefix_tokens(messages, schema_name)
            row = {
                "cache_eligible": input_tokens >= threshold,
                "cache_min_tokens": threshold,
                "cache_source": CACHE_SOURCE,
                "commit_sha": commit_sha,
                "input_tokens": input_tokens,
                "model": MODEL_ID,
                "record_type": "prefix_measurement",
                "stage": stage,
            }
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json(row))
    return 0


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
