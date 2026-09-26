"""Neighbor-interference scenarios."""

from sim.scenarios._factory import interference

SCENARIOS = (
    interference("neighbor_interference_01", 401, 46.0),
    interference("neighbor_interference_02", 402, 45.5),
)
