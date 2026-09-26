"""Energy-waste scenarios."""

from sim.scenarios._factory import energy

SCENARIOS = (
    energy("energy_waste_01", 501, 46.0),
    energy("energy_waste_02", 502, 45.0),
)
