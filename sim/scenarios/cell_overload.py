"""Cell-overload scenarios."""

from sim.scenarios._factory import overload

SCENARIOS = (
    overload("overload_dense_01", 101, 78.0),
    overload("overload_dense_02", 102, 80.0),
    overload("overload_dense_03", 103, 82.0),
)
