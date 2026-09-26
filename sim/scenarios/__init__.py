"""Imported scenario registry; Python definitions cannot drift as unused config."""

from sim.scenarios import (
    backhaul_congestion,
    cell_overload,
    conflicting_sla,
    coverage_hole,
    energy_waste,
    false_alarm,
    neighbor_interference,
    ping_pong_handover,
)
from sim.scenarios.model import Scenario

SCENARIOS_BY_FAMILY: dict[str, tuple[Scenario, ...]] = {
    "cell_overload": cell_overload.SCENARIOS,
    "coverage_hole": coverage_hole.SCENARIOS,
    "ping_pong_handover": ping_pong_handover.SCENARIOS,
    "neighbor_interference": neighbor_interference.SCENARIOS,
    "energy_waste": energy_waste.SCENARIOS,
    "conflicting_sla": conflicting_sla.SCENARIOS,
    "backhaul_congestion": backhaul_congestion.SCENARIOS,
    "false_alarm": false_alarm.SCENARIOS,
}

ALL_SCENARIOS = tuple(scenario for family in SCENARIOS_BY_FAMILY.values() for scenario in family)
SCENARIOS_BY_ID = {scenario.id: scenario for scenario in ALL_SCENARIOS}

__all__ = ["ALL_SCENARIOS", "SCENARIOS_BY_FAMILY", "SCENARIOS_BY_ID"]
