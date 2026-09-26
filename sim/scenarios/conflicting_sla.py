"""Scenarios requiring a balance between throughput and energy SLAs."""

from sim.scenarios._factory import conflicting

SCENARIOS = (
    conflicting("conflicting_sla_01", 601, 65.0),
    conflicting("conflicting_sla_02", 602, 68.0),
)
