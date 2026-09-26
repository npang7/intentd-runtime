"""Concrete SQLite journal, durable run state, and idempotent effects."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sim.network import CellState, NetworkState


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON does not support non-finite floats")
    if value == 0.0:
        return "0.0"
    text = format(value, ".17g").lower()
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        exponent = str(int(exponent))
        return f"{mantissa}e{exponent}"
    if "." not in text:
        text += ".0"
    return text


def _canonical(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float_text(value)
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        items = sorted(value.items())
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{_canonical(item)}" for key, item in items
            )
            + "}"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize JSON with sorted keys and deterministic float spellings."""

    return _canonical(value)


def network_state_json(state: NetworkState) -> str:
    payload: dict[str, object] = {
        "backhaul_penalty_ms": state.backhaul_penalty_ms,
        "cells": {cell_id: asdict(cell) for cell_id, cell in state.cells.items()},
        "seed": state.seed,
    }
    return canonical_json(payload)


def network_state_from_json(text: str) -> NetworkState:
    value: Any = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("state_json must contain an object")
    raw_cells = value.get("cells")
    raw_penalties = value.get("backhaul_penalty_ms")
    seed = value.get("seed")
    if not isinstance(raw_cells, dict) or not isinstance(raw_penalties, dict):
        raise TypeError("state_json has invalid cells or backhaul penalties")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("state_json has an invalid seed")
    cells: dict[str, CellState] = {}
    for cell_id, raw_cell in raw_cells.items():
        if not isinstance(cell_id, str) or not isinstance(raw_cell, dict):
            raise TypeError("state_json has an invalid cell")
        neighbors = raw_cell.get("neighbors")
        if not isinstance(neighbors, list) or not all(isinstance(item, str) for item in neighbors):
            raise TypeError("state_json has invalid neighbors")
        cells[cell_id] = CellState(
            tx_power_dbm=float(raw_cell["tx_power_dbm"]),
            tilt_deg=float(raw_cell["tilt_deg"]),
            a3_offset_db=float(raw_cell["a3_offset_db"]),
            admission_threshold=float(raw_cell["admission_threshold"]),
            neighbors=tuple(neighbors),
            load=float(raw_cell["load"]),
        )
    return NetworkState(
        cells=cells,
        backhaul_penalty_ms={str(key): float(item) for key, item in raw_penalties.items()},
        seed=seed,
    )


def idem_key(run_id: str, stage: str, step: int, tool: str, args: object) -> str:
    material = f"{run_id}|{stage}|{step}|{tool}|{canonical_json(args)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Journal:
    """Own one concrete SQLite connection; no storage abstraction is used."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self._connection = sqlite3.connect(self.path, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        # FULL ensures a COMMIT that returns before os._exit has reached durable storage.
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (run_id, seq)
            );
            CREATE TABLE IF NOT EXISTS effects (
                idem_key TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                step INTEGER NOT NULL,
                tool TEXT NOT NULL,
                args_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                committed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                scenario_id TEXT NOT NULL,
                status TEXT NOT NULL,
                state_json TEXT NOT NULL,
                model TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        raw_target = os.environ.get("CHAOS_DIE_AT")
        self._chaos_target = int(raw_target) if raw_target is not None else None
        if self._chaos_target is not None and self._chaos_target < 1:
            raise ValueError("CHAOS_DIE_AT must be positive")
        self._checkpoint_count = 0

    @property
    def checkpoint_count(self) -> int:
        return self._checkpoint_count

    def _checkpoint(self) -> None:
        self._checkpoint_count += 1
        if self._checkpoint_count == self._chaos_target:
            os._exit(137)

    def _next_seq(self, run_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(seq) + 1, 0) FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("failed to allocate journal sequence")
        return int(row[0])

    def _insert_event(self, run_id: str, kind: str, payload: object) -> int:
        seq = self._next_seq(run_id)
        self._connection.execute(
            "INSERT INTO events(run_id, seq, ts, kind, payload) VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, time.time(), kind, canonical_json(payload)),
        )
        return seq

    def create_run(self, run_id: str, scenario_id: str, state: NetworkState, *, model: str) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO runs(
                run_id, scenario_id, status, state_json, model, updated_at
            ) VALUES (?, ?, 'running', ?, ?, ?)
            """,
            (run_id, scenario_id, network_state_json(state), model, time.time()),
        )

    def ensure_tool_run(self, run_id: str, state: NetworkState) -> None:
        self.create_run(run_id, "", state, model="fake:scripted")

    def append(
        self,
        run_id: str,
        kind: str,
        payload: object,
        *,
        run_status: str | None = None,
    ) -> int:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            seq = self._insert_event(run_id, kind, payload)
            if run_status is not None:
                self._connection.execute(
                    "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                    (run_status, time.time(), run_id),
                )
            self._checkpoint()  # A: INSERT is pending in the open transaction.
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        self._checkpoint()  # B: COMMIT returned before the journal call returns.
        return seq

    def effect_result(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT result_json FROM effects WHERE idem_key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def run_state(self, run_id: str) -> NetworkState:
        row = self._connection.execute(
            "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return network_state_from_json(str(row[0]))

    def commit_effect(
        self,
        *,
        key: str,
        run_id: str,
        stage: str,
        step: int,
        tool: str,
        args: object,
        result: object,
        state: NetworkState,
    ) -> None:
        if (
            self._connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            is None
        ):
            raise KeyError(f"unknown run: {run_id}")

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            now = time.time()
            # Simulator state and journal share one SQLite file, so the effect
            # and its record can commit atomically. An external network device
            # would need its own idempotency support or read-modify-verify.
            self._connection.execute(
                "UPDATE runs SET state_json = ?, updated_at = ? WHERE run_id = ?",
                (network_state_json(state), now, run_id),
            )
            self._connection.execute(
                """
                INSERT INTO effects(
                    idem_key, run_id, stage, step, tool, args_json, result_json, committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    run_id,
                    stage,
                    step,
                    tool,
                    canonical_json(args),
                    canonical_json(result),
                    now,
                ),
            )
            self._insert_event(
                run_id,
                "tool_called",
                {"action": args, "idem_key": key, "output": result, "stage": stage, "step": step},
            )
            self._checkpoint()  # A: all three writes are pending together.
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        self._checkpoint()  # B: durable effect exists before caller memory refresh.

    def events(self, run_id: str) -> list[tuple[int, str, str]]:
        rows = self._connection.execute(
            "SELECT seq, kind, payload FROM events WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [(int(seq), str(kind), str(payload)) for seq, kind, payload in rows]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
