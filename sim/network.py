"""Deterministic, single-transform network model used by the simulator."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import TypeAlias

from orchestrator.contracts import (
    GetMetrics,
    NoOp,
    ReportNonRadioIssue,
    SetA3Offset,
    SetAdmissionThreshold,
    SetTilt,
    SetTxPower,
)

ActionValue: TypeAlias = (
    GetMetrics
    | SetTxPower
    | SetA3Offset
    | SetTilt
    | SetAdmissionThreshold
    | ReportNonRadioIssue
    | NoOp
)


@dataclass(frozen=True)
class CellState:
    # These schema bounds are inputs, not measured performance.
    # Reproduce: python -m pytest tests/test_sim_determinism.py
    tx_power_dbm: float
    tilt_deg: float
    a3_offset_db: float
    admission_threshold: float
    neighbors: tuple[str, ...]
    load: float

    def __post_init__(self) -> None:
        ranges = {
            "tx_power_dbm": (self.tx_power_dbm, 20.0, 46.0),
            "tilt_deg": (self.tilt_deg, 0.0, 15.0),
            "a3_offset_db": (self.a3_offset_db, -10.0, 10.0),
            "admission_threshold": (self.admission_threshold, 0.0, 1.0),
        }
        for name, (value, minimum, maximum) in ranges.items():
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if self.load < 0.0:
            raise ValueError("load must be nonnegative")


@dataclass(frozen=True)
class NetworkState:
    cells: dict[str, CellState]
    backhaul_penalty_ms: dict[str, float]
    seed: int

    def __post_init__(self) -> None:
        if set(self.cells) != set(self.backhaul_penalty_ms):
            raise ValueError("cells and backhaul_penalty_ms must have identical keys")
        for cell_id, cell in self.cells.items():
            missing = set(cell.neighbors) - self.cells.keys()
            if missing:
                raise ValueError(f"{cell_id} has unknown neighbors: {sorted(missing)}")
            if cell_id in cell.neighbors:
                raise ValueError(f"{cell_id} cannot be its own neighbor")


@dataclass(frozen=True)
class CellMetrics:
    interference: float
    sinr_proxy: float
    capacity: float
    prb_utilization: float
    edge_throughput: float
    avg_throughput: float
    rtt_ms: float
    ping_pong_rate: float
    handover_fail: float
    energy_w: float


def _noise(seed: int, cell_id: str, metric_name: str) -> float:
    material = "\x00".join((str(seed), cell_id, metric_name)).encode()
    digest = hashlib.sha256(material).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return 2.0 * unit - 1.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def metrics(state: NetworkState) -> dict[str, CellMetrics]:
    """Derive deterministic metrics from a state without mutating it."""

    # Formula coefficients are deterministic model inputs. Reproduce their
    # monotonic behavior: python -m pytest tests/test_sim_determinism.py
    result: dict[str, CellMetrics] = {}
    for cell_id, cell in state.cells.items():
        interference = sum(
            0.001
            * (1.0 + 0.15 * _noise(state.seed, f"{cell_id}->{neighbor}", "coupling"))
            * 10 ** (state.cells[neighbor].tx_power_dbm / 10.0)
            for neighbor in cell.neighbors
        )
        sinr = cell.tx_power_dbm - 10.0 * math.log10(interference) - 0.45 * cell.tilt_deg
        capacity = max(1.0, 10.0 + 4.0 * max(sinr, 0.0)) * cell.admission_threshold
        utilization = _clamp(cell.load / max(capacity, 0.001), 0.0, 1.0)
        quality = _clamp((sinr + 5.0) / 40.0, 0.05, 1.0)
        edge = 20.0 * quality * (1.0 - utilization)
        edge *= 1.0 + 0.005 * _noise(state.seed, cell_id, "edge_throughput")
        rtt = (
            10.0
            + 80.0 * utilization**2 / max(1.05 - utilization, 0.05)
            + state.backhaul_penalty_ms[cell_id]
            + 0.1 * _noise(state.seed, cell_id, "rtt_ms")
        )
        ping_pong = _clamp(
            0.25 * math.exp(-0.18 * (cell.a3_offset_db + 10.0))
            + 0.002 * _noise(state.seed, cell_id, "ping_pong_rate"),
            0.0,
            1.0,
        )
        handover_fail = _clamp(
            0.02
            + 0.01 * max(0.0, 8.0 - sinr)
            + 0.008 * max(0.0, cell.a3_offset_db)
            + 0.002 * _noise(state.seed, cell_id, "handover_fail"),
            0.0,
            1.0,
        )
        result[cell_id] = CellMetrics(
            interference=interference,
            sinr_proxy=sinr,
            capacity=capacity,
            prb_utilization=utilization,
            edge_throughput=edge,
            avg_throughput=edge * 2.5,
            rtt_ms=rtt,
            ping_pong_rate=ping_pong,
            handover_fail=handover_fail,
            energy_w=0.01 * 10 ** (cell.tx_power_dbm / 10.0),
        )
    return result


def apply(state: NetworkState, action: ActionValue) -> NetworkState:
    """Apply one schema-valid action and return a new immutable state value."""

    if isinstance(action, (GetMetrics, ReportNonRadioIssue, NoOp)):
        return state
    try:
        cell = state.cells[action.cell_id]
    except KeyError as exc:
        raise KeyError(f"unknown cell: {action.cell_id}") from exc

    if isinstance(action, SetTxPower):
        changed = replace(cell, tx_power_dbm=action.tx_power_dbm)
    elif isinstance(action, SetA3Offset):
        changed = replace(cell, a3_offset_db=action.a3_offset_db)
    elif isinstance(action, SetTilt):
        changed = replace(cell, tilt_deg=action.tilt_deg)
    elif isinstance(action, SetAdmissionThreshold):
        changed = replace(cell, admission_threshold=action.threshold)
    else:  # pragma: no cover - union exhaustiveness guard
        raise TypeError(f"unsupported action: {type(action).__name__}")

    cells = dict(state.cells)
    cells[action.cell_id] = changed
    return replace(state, cells=cells)


def state_hash(state: NetworkState) -> str:
    """Hash canonical JSON for every state field, including the seed."""

    payload = {
        "backhaul_penalty_ms": state.backhaul_penalty_ms,
        "cells": {cell_id: asdict(cell) for cell_id, cell in state.cells.items()},
        "seed": state.seed,
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
