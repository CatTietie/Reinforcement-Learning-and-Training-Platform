"""
基准测试运行器：顺序执行套件中的每个 benchmark，收集结果。
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import yaml

from benchmark.schema import BenchmarkSuite, BenchmarkConfig, merge_defaults
from benchmark.threshold import check_threshold, compute_effective_pass


@dataclass
class BenchmarkResult:
    name: str
    description: str
    baseline_reward: float
    actual_reward: float
    metric_used: str
    ratio: float
    threshold_ratio: float
    passed: bool
    expect_fail: bool
    effective_pass: bool
    total_episodes: int
    error: Optional[str] = None
    duration_seconds: float = 0.0


class BenchmarkRunner:
    def __init__(self, suite: BenchmarkSuite, verbose: bool = False):
        self.suite = suite
        self.verbose = verbose

    def run_all(self) -> list:
        results = []
        for i, benchmark in enumerate(self.suite.benchmarks, 1):
            print(f"  [{i}/{len(self.suite.benchmarks)}] Running: {benchmark.name}...", end=" ", flush=True)
            result = self._run_single(benchmark)
            status_str = "PASS" if result.passed else "FAIL"
            if result.expect_fail and not result.passed:
                status_str = "FAIL (expected)"
            if result.error:
                status_str = f"ERROR: {result.error}"
            print(f"{status_str} (reward={result.actual_reward:.1f}, ratio={result.ratio:.1%})")
            results.append(result)
        return results

    def _run_single(self, benchmark: BenchmarkConfig) -> BenchmarkResult:
        try:
            config = self._load_config(benchmark.config_path)
            config = merge_defaults(config, self.suite.defaults)

            config.setdefault("experiment", {})["seed"] = self.suite.settings.seed
            config.setdefault("storage", {})["db_connection"] = None

            if not self.verbose:
                config.setdefault("logging", {})["level"] = "CRITICAL"

            self._suppress_logging()
            start_time = time.time()

            from train import Trainer

            trainer = Trainer(
                config=config,
                experiment_id=f"benchmark_{benchmark.name}",
            )
            result = trainer.train()
            duration = time.time() - start_time
            self._restore_logging()

            actual_value = result.get(benchmark.threshold.metric, result.get("avg_final_10", 0.0))
            passed, ratio = check_threshold(actual_value, benchmark.baseline_reward, benchmark.threshold)
            effective = compute_effective_pass(passed, benchmark.expect_fail)

            return BenchmarkResult(
                name=benchmark.name,
                description=benchmark.description,
                baseline_reward=benchmark.baseline_reward,
                actual_reward=actual_value,
                metric_used=benchmark.threshold.metric,
                ratio=ratio,
                threshold_ratio=benchmark.threshold.min_ratio,
                passed=passed,
                expect_fail=benchmark.expect_fail,
                effective_pass=effective,
                total_episodes=result.get("total_episodes", 0),
                duration_seconds=duration,
            )

        except Exception as e:
            self._restore_logging()
            return BenchmarkResult(
                name=benchmark.name,
                description=benchmark.description,
                baseline_reward=benchmark.baseline_reward,
                actual_reward=0.0,
                metric_used=benchmark.threshold.metric,
                ratio=0.0,
                threshold_ratio=benchmark.threshold.min_ratio,
                passed=False,
                expect_fail=benchmark.expect_fail,
                effective_pass=False,
                total_episodes=0,
                error=str(e),
                duration_seconds=0.0,
            )

    def _load_config(self, config_path: str) -> dict:
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.getcwd(), config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _suppress_logging(self):
        if not self.verbose:
            logging.getLogger("rl_trainer").setLevel(logging.CRITICAL)
            logging.getLogger().setLevel(logging.WARNING)

    def _restore_logging(self):
        logging.getLogger("rl_trainer").setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)
