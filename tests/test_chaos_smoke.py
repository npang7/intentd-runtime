from __future__ import annotations

from bench.chaos import run_chaos


def test_ten_real_subprocess_crashes_recover_consistently() -> None:
    summary = run_chaos(10)
    assert summary.passed == 10
    assert summary.checkpoint_a > 0
    assert summary.checkpoint_b > 0
