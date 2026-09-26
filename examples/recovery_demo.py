"""Verify recovery after a process exits immediately following a committed action."""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from orchestrator.journal import network_state_from_json
from sim.network import state_hash

ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _snapshot(database: Path) -> tuple[str, int, int]:
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT state_json FROM runs WHERE run_id = 'demo'").fetchone()
        if row is None:
            raise AssertionError("demo run was not persisted")
        effects = int(connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0])
        events = int(
            connection.execute("SELECT COUNT(*) FROM events WHERE kind = 'tool_called'").fetchone()[
                0
            ]
        )
    return state_hash(network_state_from_json(str(row[0]))), effects, events


def _expect_success(process: subprocess.CompletedProcess[str]) -> None:
    if process.returncode != 0:
        raise AssertionError(process.stderr or process.stdout)


def run_demo() -> dict[str, int | bool]:
    env = dict(os.environ)
    env.pop("CHAOS_DIE_AT", None)
    with tempfile.TemporaryDirectory(
        prefix="intentd-recovery-demo-", ignore_cleanup_errors=True
    ) as directory:
        clean = Path(directory) / "clean.sqlite"
        interrupted = Path(directory) / "interrupted.sqlite"
        base = [
            "-m",
            "orchestrator.run",
            "--scenario",
            "coverage_hole_01",
            "--model",
            "oracle:standard",
            "--run-id",
            "demo",
        ]
        _expect_success(_run([*base, "--db", str(clean)], env))
        reference = _snapshot(clean)
        with sqlite3.connect(clean) as connection:
            sequence = int(
                connection.execute(
                    "SELECT MIN(seq) FROM events WHERE kind = 'tool_called'"
                ).fetchone()[0]
            )
        # Event sequences start at zero; checkpoint numbers start at one.
        # The second checkpoint for this event runs immediately after its commit.
        crash_env = {**env, "CHAOS_DIE_AT": str((sequence + 1) * 2)}
        crashed = _run([*base, "--db", str(interrupted)], crash_env)
        before = _snapshot(interrupted)
        if crashed.returncode == 0 or before[1] != 1 or before[2] != 1:
            raise AssertionError("process did not exit after its first committed action")
        resume = ["-m", "orchestrator.resume", "--db", str(interrupted), "--run-id", "demo"]
        _expect_success(_run(resume, env))
        recovered = _snapshot(interrupted)
        if recovered != reference:
            raise AssertionError("recovered state, effects, or tool events differ from reference")
        _expect_success(_run(resume, env))
        repeated = _snapshot(interrupted)
        if repeated != recovered:
            raise AssertionError("resuming a completed run changed its persisted effects")
    return {
        "exit_status": crashed.returncode,
        "committed_before_resume": before[1],
        "effects_after_resume": recovered[1],
        "state_matches_reference": recovered[0] == reference[0],
        "repeat_resume_added_effects": repeated[1] - recovered[1],
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = run_demo()
    print("Scenario: coverage_hole_01 (deterministic demo)")
    print(f"Process exited after committed action: exit={result['exit_status']}")
    print(f"Committed effects before recovery: {result['committed_before_resume']}")
    print(f"Recovered state matches reference: {result['state_matches_reference']}")
    print(f"Effects after recovery: {result['effects_after_resume']}")
    print(f"Repeated resume added effects: {result['repeat_resume_added_effects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
