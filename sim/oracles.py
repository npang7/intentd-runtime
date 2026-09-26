"""Standard and plausible trap solutions for every scenario.

Standard sequences are not guaranteed to be minimal. An action included in a
successful sequence is not necessarily required for success, so blast-radius
predictions must not treat "included" as "required." The grader tests instead
check that each standard sequence succeeds and each trap sequence fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.contracts import (
    NoOp,
    ReportNonRadioIssue,
    SetA3Offset,
    SetAdmissionThreshold,
    SetTilt,
    SetTxPower,
)
from sim.network import ActionValue


@dataclass(frozen=True)
class OraclePair:
    standard: tuple[ActionValue, ...]
    trap: tuple[ActionValue, ...]
    trap_expected_check: str


ORACLES: dict[str, OraclePair] = {
    "overload_dense_01": OraclePair(
        standard=(
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=1.0),
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=44.0),
        ),
        trap=(
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=0.7),
        ),
        trap_expected_check="must_restore",
    ),
    "overload_dense_02": OraclePair(
        standard=(
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=1.0),
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=44.0),
        ),
        trap=(
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=0.8),
        ),
        trap_expected_check="must_restore",
    ),
    "overload_dense_03": OraclePair(
        standard=(
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=1.0),
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=45.0),
        ),
        trap=(
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=0.8),
        ),
        trap_expected_check="must_restore",
    ),
    "coverage_hole_01": OraclePair(
        standard=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0),
            SetTilt(tool="set_tilt", cell_id="cell-1", tilt_deg=2.0),
        ),
        trap=(SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=24.5),),
        trap_expected_check="must_restore",
    ),
    "coverage_hole_02": OraclePair(
        standard=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0),
            SetTilt(tool="set_tilt", cell_id="cell-1", tilt_deg=2.0),
        ),
        trap=(SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=25.0),),
        trap_expected_check="must_restore",
    ),
    "coverage_hole_03": OraclePair(
        standard=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0),
            SetTilt(tool="set_tilt", cell_id="cell-1", tilt_deg=2.0),
        ),
        trap=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0),
            SetTilt(tool="set_tilt", cell_id="cell-1", tilt_deg=2.0),
            NoOp(tool="no_op", reason="Observe the stable post-change metrics."),
        ),
        trap_expected_check="max_steps",
    ),
    "ping_pong_handover_01": OraclePair(
        standard=(SetA3Offset(tool="set_a3_offset", cell_id="cell-1", a3_offset_db=2.0),),
        trap=(SetA3Offset(tool="set_a3_offset", cell_id="cell-1", a3_offset_db=-5.0),),
        trap_expected_check="must_restore",
    ),
    "ping_pong_handover_02": OraclePair(
        standard=(SetA3Offset(tool="set_a3_offset", cell_id="cell-1", a3_offset_db=1.0),),
        trap=(SetA3Offset(tool="set_a3_offset", cell_id="cell-1", a3_offset_db=-5.0),),
        trap_expected_check="must_restore",
    ),
    "neighbor_interference_01": OraclePair(
        standard=(SetTxPower(tool="set_tx_power", cell_id="cell-2", tx_power_dbm=34.0),),
        trap=(SetTxPower(tool="set_tx_power", cell_id="cell-2", tx_power_dbm=20.0),),
        trap_expected_check="must_not_break",
    ),
    "neighbor_interference_02": OraclePair(
        standard=(SetTxPower(tool="set_tx_power", cell_id="cell-2", tx_power_dbm=34.0),),
        trap=(SetTxPower(tool="set_tx_power", cell_id="cell-2", tx_power_dbm=22.0),),
        trap_expected_check="must_not_break",
    ),
    "energy_waste_01": OraclePair(
        standard=(SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0),),
        trap=(SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=44.0),),
        trap_expected_check="must_restore",
    ),
    "energy_waste_02": OraclePair(
        standard=(SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=40.0),),
        trap=(SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=43.0),),
        trap_expected_check="must_restore",
    ),
    "conflicting_sla_01": OraclePair(
        standard=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=42.0),
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=1.0),
        ),
        trap=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=36.0),
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=0.7),
        ),
        trap_expected_check="must_restore",
    ),
    "conflicting_sla_02": OraclePair(
        standard=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=42.0),
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=1.0),
        ),
        trap=(
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=36.0),
            SetAdmissionThreshold(tool="set_admission_threshold", cell_id="cell-1", threshold=0.7),
        ),
        trap_expected_check="must_restore",
    ),
    "backhaul_congestion_01": OraclePair(
        standard=(
            ReportNonRadioIssue(
                tool="report_non_radio_issue",
                cell_id="cell-1",
                category="backhaul",
                detail="Radio metrics are healthy; inspect backhaul latency.",
            ),
        ),
        trap=(
            ReportNonRadioIssue(
                tool="report_non_radio_issue",
                cell_id="cell-1",
                category="backhaul",
                detail="Backhaul latency is the primary fault.",
            ),
            SetTxPower(tool="set_tx_power", cell_id="cell-1", tx_power_dbm=42.0),
        ),
        trap_expected_check="forbidden_actions",
    ),
    "backhaul_congestion_02": OraclePair(
        standard=(
            ReportNonRadioIssue(
                tool="report_non_radio_issue",
                cell_id="cell-1",
                category="backhaul",
                detail="Escalate the backhaul latency fault.",
            ),
        ),
        trap=(
            ReportNonRadioIssue(
                tool="report_non_radio_issue",
                cell_id="cell-1",
                category="backhaul",
                detail="Backhaul latency is elevated.",
            ),
            SetTilt(tool="set_tilt", cell_id="cell-1", tilt_deg=2.0),
        ),
        trap_expected_check="forbidden_actions",
    ),
    "false_alarm_01": OraclePair(
        standard=(NoOp(tool="no_op", reason="All observed metrics remain inside SLA."),),
        trap=(SetA3Offset(tool="set_a3_offset", cell_id="cell-1", a3_offset_db=2.0),),
        trap_expected_check="required_actions",
    ),
    "false_alarm_02": OraclePair(
        standard=(NoOp(tool="no_op", reason="Noise has not produced an SLA violation."),),
        trap=(SetA3Offset(tool="set_a3_offset", cell_id="cell-1", a3_offset_db=2.0),),
        trap_expected_check="required_actions",
    ),
}

SINGLE_FAILURE_CASES = {
    "must_restore": "overload_dense_01",
    "must_not_break": "neighbor_interference_01",
    "required_actions": "false_alarm_01",
    "forbidden_actions": "backhaul_congestion_01",
    "max_steps": "coverage_hole_03",
}
