"""
阈值判定逻辑：根据 baseline 和实际奖励判定 pass/fail。
"""

from benchmark.schema import ThresholdConfig


def check_threshold(
    actual_value: float,
    baseline_value: float,
    threshold: ThresholdConfig,
) -> tuple:
    if baseline_value <= 0:
        ratio = 1.0 if actual_value >= 0 else 0.0
    else:
        ratio = actual_value / baseline_value
    passed = ratio >= threshold.min_ratio
    return passed, ratio


def compute_effective_pass(passed: bool, expect_fail: bool) -> bool:
    if expect_fail:
        return not passed
    return passed
