"""Hard-kill subprocess runs and verify durable recovery invariants."""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from orchestrator.journal import network_state_from_json
from sim.network import state_hash
from sim.scenarios import ALL_SCENARIOS


@dataclass(frozen=True)
class RunSnapshot:
    state_hash: str
    effects: int
    tool_events: int
    events: int


@dataclass(frozen=True)
class ChaosSummary:
    passed: int
    checkpoint_a: int
    checkpoint_b: int
    elapsed_s: float


def _snapshot(path: Path, run_id: str) -> RunSnapshot:
    with sqlite3.connect(path) as connection:
        state_row = connection.execute(
            "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if state_row is None:
            raise AssertionError(f"missing run row for {run_id}")
        effects = int(
            connection.execute(
                "SELECT COUNT(*) FROM effects WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        )
        tool_events = int(
            connection.execute(
                "SELECT COUNT(*) FROM events WHERE run_id = ? AND kind = 'tool_called'",
                (run_id,),
            ).fetchone()[0]
        )
        events = int(
            connection.execute(
                "SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        )
    return RunSnapshot(
        state_hash=state_hash(network_state_from_json(str(state_row[0]))),
        effects=effects,
        tool_events=tool_events,
        events=events,
    )


def _run_command(
    arguments: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_chaos(n: int, seed_start: int = 0) -> ChaosSummary:
    if n < 1:
        raise ValueError("n must be positive")
    started = time.perf_counter()
    checkpoint_a = 0
    checkpoint_b = 0

    with tempfile.TemporaryDirectory(
        prefix="intentd-chaos-", ignore_cleanup_errors=True
    ) as raw_directory:
        directory = Path(raw_directory)
        for seed in range(seed_start, seed_start + n):
            scenario = ALL_SCENARIOS[seed % len(ALL_SCENARIOS)]
            clean_db = directory / f"clean-{seed}.sqlite"
            crash_db = directory / f"crash-{seed}.sqlite"
            clean_id = f"clean-{seed}"
            crash_id = f"crash-{seed}"
            base_arguments = [
                "-m",
                "orchestrator.run",
                "--scenario",
                scenario.id,
                "--model",
                "oracle:standard",
            ]

            clean = _run_command([*base_arguments, "--db", str(clean_db), "--run-id", clean_id])
            if clean.returncode != 0:
                raise AssertionError(f"clean seed={seed} failed: {clean.stderr or clean.stdout}")
            clean_snapshot = _snapshot(clean_db, clean_id)
            if clean_snapshot.tool_events != clean_snapshot.effects:
                raise AssertionError(
                    f"clean seed={seed}: tool_called={clean_snapshot.tool_events} "
                    f"effects={clean_snapshot.effects}"
                )

            total_checkpoints = clean_snapshot.events * 2
            die_at = random.Random(seed).randint(1, total_checkpoints)
            checkpoint = "A" if die_at % 2 else "B"
            if checkpoint == "A":
                checkpoint_a += 1
            else:
                checkpoint_b += 1

            crash_env = dict(os.environ)
            crash_env["CHAOS_DIE_AT"] = str(die_at)
            crashed = _run_command(
                [*base_arguments, "--db", str(crash_db), "--run-id", crash_id],
                env=crash_env,
            )
            if crashed.returncode == 0:
                raise AssertionError(
                    f"seed={seed} CHAOS_DIE_AT={die_at} checkpoint={checkpoint} did not exit"
                )

            resumed = _run_command(
                ["-m", "orchestrator.resume", "--db", str(crash_db), "--run-id", crash_id]
            )
            if resumed.returncode != 0:
                raise AssertionError(
                    f"resume seed={seed} CHAOS_DIE_AT={die_at} checkpoint={checkpoint} "
                    f"failed: {resumed.stderr or resumed.stdout}"
                )
            recovered = _snapshot(crash_db, crash_id)
            if recovered.state_hash != clean_snapshot.state_hash:
                raise AssertionError(
                    f"seed={seed} CHAOS_DIE_AT={die_at} checkpoint={checkpoint}: state mismatch"
                )
            if recovered.effects != clean_snapshot.effects:
                raise AssertionError(
                    f"seed={seed} CHAOS_DIE_AT={die_at} checkpoint={checkpoint}: "
                    f"effects clean={clean_snapshot.effects} recovered={recovered.effects}"
                )
            if recovered.tool_events != recovered.effects:
                raise AssertionError(
                    f"seed={seed} CHAOS_DIE_AT={die_at} checkpoint={checkpoint}: "
                    f"tool_called={recovered.tool_events} effects={recovered.effects}"
                )
            print(f"seed={seed} CHAOS_DIE_AT={die_at} checkpoint={checkpoint} passed")

    elapsed = time.perf_counter() - started
    if n > 1 and (checkpoint_a == 0 or checkpoint_b == 0):
        raise AssertionError(f"checkpoint coverage invalid: A={checkpoint_a} B={checkpoint_b}")
    return ChaosSummary(n, checkpoint_a, checkpoint_b, elapsed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--seed-start", type=int, default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = run_chaos(args.n, args.seed_start)
    except (AssertionError, ValueError) as error:
        print(f"FAILED: {error}")
        return 1
    print(f"{summary.passed}/{summary.passed} passed")
    print(f"checkpoint A: {summary.checkpoint_a}")
    print(f"checkpoint B: {summary.checkpoint_b}")
    print(f"wall clock: {summary.elapsed_s:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
