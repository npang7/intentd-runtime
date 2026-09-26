"""Inspect one run without changing its SQLite journal."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def inspect(database: Path, run_id: str) -> dict[str, object]:
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        actions = connection.execute(
            "SELECT step, tool, args_json FROM effects WHERE run_id = ? ORDER BY stage, step",
            (run_id,),
        ).fetchall()
    return {
        "run_id": run_id,
        "status": str(row[0]),
        "effects": len(actions),
        "actions": [
            {"step": step, "tool": tool, "args": json.loads(args)} for step, tool, args in actions
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.db, args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
