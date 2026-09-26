"""Backhaul congestion scenarios whose fault is unreachable by radio actions."""

from sim.scenarios._factory import backhaul

SCENARIOS = (
    backhaul("backhaul_congestion_01", 701, 130.0),
    backhaul("backhaul_congestion_02", 702, 160.0),
)
