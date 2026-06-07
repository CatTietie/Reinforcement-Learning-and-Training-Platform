"""
基准测试套件 YAML 模式验证与解析。
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ThresholdConfig:
    metric: str = "avg_final_10"
    min_ratio: float = 0.90


@dataclass
class BenchmarkConfig:
    name: str
    description: str
    config_path: str
    baseline_reward: float
    threshold: ThresholdConfig
    expect_fail: bool = False


@dataclass
class SuiteSettings:
    seed: int = 42
    repeat: int = 1
    timeout_seconds: int = 600
    chart_format: str = "png"
    report_dir: str = "reports"


@dataclass
class BenchmarkSuite:
    name: str
    description: str
    version: str
    defaults: dict = field(default_factory=dict)
    benchmarks: list = field(default_factory=list)
    settings: SuiteSettings = field(default_factory=SuiteSettings)


def load_suite(suite_path: str) -> BenchmarkSuite:
    with open(suite_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "suite" not in data:
        raise ValueError(f"Invalid suite YAML: missing 'suite' section in {suite_path}")

    suite_data = data["suite"]
    if "name" not in suite_data:
        raise ValueError("Suite YAML missing required field: suite.name")

    benchmarks_data = data.get("benchmarks", [])
    if not benchmarks_data:
        raise ValueError("Suite YAML must contain at least one benchmark in 'benchmarks' section")

    settings_data = data.get("settings", {})
    settings = SuiteSettings(
        seed=settings_data.get("seed", 42),
        repeat=settings_data.get("repeat", 1),
        timeout_seconds=settings_data.get("timeout_seconds", 600),
        chart_format=settings_data.get("chart_format", "png"),
        report_dir=settings_data.get("report_dir", "reports"),
    )

    benchmarks = []
    for b in benchmarks_data:
        if "name" not in b or "config_path" not in b or "baseline_reward" not in b:
            raise ValueError(
                f"Benchmark entry missing required fields (name, config_path, baseline_reward): {b}"
            )
        threshold_data = b.get("threshold", {})
        threshold = ThresholdConfig(
            metric=threshold_data.get("metric", "avg_final_10"),
            min_ratio=threshold_data.get("min_ratio", 0.90),
        )
        benchmarks.append(
            BenchmarkConfig(
                name=b["name"],
                description=b.get("description", ""),
                config_path=b["config_path"],
                baseline_reward=b["baseline_reward"],
                threshold=threshold,
                expect_fail=b.get("expect_fail", False),
            )
        )

    defaults = data.get("defaults", {})

    return BenchmarkSuite(
        name=suite_data["name"],
        description=suite_data.get("description", ""),
        version=suite_data.get("version", "1.0"),
        defaults=defaults,
        benchmarks=benchmarks,
        settings=settings,
    )


def merge_defaults(base_config: dict, defaults: dict) -> dict:
    result = copy.deepcopy(base_config)
    for key, value in defaults.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_defaults(result[key], value)
        elif key not in result:
            result[key] = copy.deepcopy(value)
    return result
