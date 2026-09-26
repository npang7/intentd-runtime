"""Coverage-hole scenarios."""

from sim.scenarios._factory import coverage

SCENARIOS = (
    coverage("coverage_hole_01", 201, 13.0),
    coverage("coverage_hole_02", 202, 14.0),
    coverage("coverage_hole_03", 203, 15.0),
)
