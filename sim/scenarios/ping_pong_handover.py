"""Ping-pong handover scenarios."""

from sim.scenarios._factory import ping_pong

SCENARIOS = (
    ping_pong("ping_pong_handover_01", 301, -8.0),
    ping_pong("ping_pong_handover_02", 302, -7.0),
)
