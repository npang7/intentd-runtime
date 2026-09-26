"""False alarms whose metrics remain inside every SLA."""

from sim.scenarios._factory import false_alarm

SCENARIOS = (
    false_alarm("false_alarm_01", 801, 18.0),
    false_alarm("false_alarm_02", 802, 20.0),
)
