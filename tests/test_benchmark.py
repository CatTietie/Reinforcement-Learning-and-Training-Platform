"""
基准测试模块的单元与集成测试。
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest
import yaml

from benchmark.schema import (
    BenchmarkConfig,
    BenchmarkSuite,
    SuiteSettings,
    ThresholdConfig,
    load_suite,
    merge_defaults,
)
from benchmark.threshold import check_threshold, compute_effective_pass
from benchmark.runner import BenchmarkResult, BenchmarkRunner
from benchmark.reporter import BenchmarkReporter
from db.database import Database


class TestThresholdJudgment:
    """阈值判定逻辑单元测试。"""

    def test_threshold_pass_when_above_ratio(self):
        """验证：实际奖励高于 baseline * min_ratio 时判定为 PASS。"""
        threshold = ThresholdConfig(metric="avg_final_10", min_ratio=0.90)
        passed, ratio = check_threshold(
            actual_value=320.0, baseline_value=350.0, threshold=threshold
        )
        assert passed is True
        assert abs(ratio - 320.0 / 350.0) < 1e-6

    def test_threshold_fail_when_below_ratio(self):
        """验证：实际奖励低于 baseline * min_ratio 时判定为 FAIL。"""
        threshold = ThresholdConfig(metric="avg_final_10", min_ratio=0.90)
        passed, ratio = check_threshold(
            actual_value=100.0, baseline_value=350.0, threshold=threshold
        )
        assert passed is False
        assert abs(ratio - 100.0 / 350.0) < 1e-6

    def test_effective_pass_with_expect_fail(self):
        """验证：expect_fail=True 时，判定逻辑取反（FAIL 变为有效 PASS）。"""
        assert compute_effective_pass(passed=False, expect_fail=True) is True
        assert compute_effective_pass(passed=True, expect_fail=True) is False
        assert compute_effective_pass(passed=True, expect_fail=False) is True
        assert compute_effective_pass(passed=False, expect_fail=False) is False


class TestReportGeneration:
    """报告生成器测试。"""

    def _make_results(self):
        return [
            BenchmarkResult(
                name="benchmark_pass",
                description="A passing benchmark",
                baseline_reward=350.0,
                actual_reward=340.0,
                metric_used="avg_final_10",
                ratio=340.0 / 350.0,
                threshold_ratio=0.90,
                passed=True,
                expect_fail=False,
                effective_pass=True,
                total_episodes=200,
                duration_seconds=10.0,
            ),
            BenchmarkResult(
                name="benchmark_fail_expected",
                description="Expected failure",
                baseline_reward=350.0,
                actual_reward=85.0,
                metric_used="avg_final_10",
                ratio=85.0 / 350.0,
                threshold_ratio=0.90,
                passed=False,
                expect_fail=True,
                effective_pass=True,
                total_episodes=200,
                duration_seconds=8.0,
            ),
        ]

    def test_markdown_table_format(self):
        """验证：生成的 Markdown 包含正确的表格结构和列标题。"""
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmp_dir:
            reporter = BenchmarkReporter(
                results=results, suite_name="Test Suite", report_dir=tmp_dir
            )
            report_path = reporter.generate()
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "| # | Benchmark | Baseline | Actual | Ratio | Threshold | Status |" in content
            assert "benchmark_pass" in content
            assert "benchmark_fail_expected" in content
            assert "PASS" in content

    def test_chart_file_created(self):
        """验证：调用 generate() 后在指定目录生成 PNG 图表文件。"""
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmp_dir:
            reporter = BenchmarkReporter(
                results=results, suite_name="Test Suite", report_dir=tmp_dir
            )
            reporter.generate()
            png_files = [f for f in os.listdir(tmp_dir) if f.endswith(".png")]
            assert len(png_files) == 1
            assert os.path.getsize(os.path.join(tmp_dir, png_files[0])) > 0

    def test_overall_status_pass(self):
        """验证：所有 benchmark 的 effective_pass 为 True 时总结为 PASS。"""
        results = self._make_results()
        reporter = BenchmarkReporter(results=results, suite_name="Test", report_dir=".")
        assert reporter.get_overall_status() == "PASS"

    def test_overall_status_regression(self):
        """验证：任一 benchmark 的 effective_pass 为 False 时总结为 REGRESSION DETECTED。"""
        results = self._make_results()
        results[0] = BenchmarkResult(
            name="unexpected_fail",
            description="Unexpected failure",
            baseline_reward=350.0,
            actual_reward=100.0,
            metric_used="avg_final_10",
            ratio=100.0 / 350.0,
            threshold_ratio=0.90,
            passed=False,
            expect_fail=False,
            effective_pass=False,
            total_episodes=200,
            duration_seconds=5.0,
        )
        reporter = BenchmarkReporter(results=results, suite_name="Test", report_dir=".")
        assert reporter.get_overall_status() == "REGRESSION DETECTED"


class TestBenchmarkRunnerIntegration:
    """基准测试运行器集成测试。"""

    def _make_minimal_config(self, tmp_dir, clip_epsilon=0.2):
        config = {
            "experiment": {"name": "test_benchmark", "seed": 42},
            "env": {"name": "CartPoleStandard-v0", "max_steps": 200},
            "network": {
                "hidden_sizes": [32],
                "activation": "tanh",
                "lstm_hidden_size": 32,
                "lstm_num_layers": 1,
                "use_lstm": False,
            },
            "algorithm": {
                "type": "ppo",
                "lr": 3.0e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": clip_epsilon,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "update_epochs": 2,
                "max_grad_norm": 0.5,
            },
            "training": {
                "num_episodes": 5,
                "batch_size": 32,
                "log_interval": 10,
                "save_interval": 100,
            },
            "storage": {"model_dir": os.path.join(tmp_dir, "models")},
            "logging": {"level": "CRITICAL", "log_dir": os.path.join(tmp_dir, "logs")},
        }
        config_path = os.path.join(tmp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_runner_executes_single_benchmark(self):
        """验证：Runner 能正确执行单个 benchmark 并返回 BenchmarkResult。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._make_minimal_config(tmp_dir)
            suite = BenchmarkSuite(
                name="Test",
                description="",
                version="1.0",
                defaults={},
                benchmarks=[
                    BenchmarkConfig(
                        name="test_run",
                        description="Minimal test",
                        config_path=config_path,
                        baseline_reward=50.0,
                        threshold=ThresholdConfig(metric="avg_final_10", min_ratio=0.01),
                    )
                ],
                settings=SuiteSettings(seed=42, report_dir=tmp_dir),
            )
            runner = BenchmarkRunner(suite=suite, verbose=False)
            results = runner.run_all()
            assert len(results) == 1
            assert results[0].name == "test_run"
            assert results[0].total_episodes == 5
            assert results[0].actual_reward > 0
            assert results[0].error is None

    def test_runner_suppresses_logging(self, capsys):
        """验证：运行期间不产生 INFO 级别控制台输出。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._make_minimal_config(tmp_dir)
            suite = BenchmarkSuite(
                name="Test",
                description="",
                version="1.0",
                defaults={},
                benchmarks=[
                    BenchmarkConfig(
                        name="quiet_run",
                        description="",
                        config_path=config_path,
                        baseline_reward=50.0,
                        threshold=ThresholdConfig(metric="avg_final_10", min_ratio=0.01),
                    )
                ],
                settings=SuiteSettings(seed=42, report_dir=tmp_dir),
            )
            runner = BenchmarkRunner(suite=suite, verbose=False)
            runner.run_all()
            captured = capsys.readouterr()
            assert "Episode" not in captured.out
            assert "Training" not in captured.out


class TestSuiteYAMLParsing:
    """套件 YAML 加载与验证测试。"""

    def test_load_valid_suite(self):
        """验证：有效的套件 YAML 被正确解析为 BenchmarkSuite 对象。"""
        suite_data = {
            "suite": {
                "name": "Test Suite",
                "description": "Test description",
                "version": "1.0",
            },
            "defaults": {"logging": {"level": "CRITICAL"}},
            "benchmarks": [
                {
                    "name": "bench1",
                    "description": "First benchmark",
                    "config_path": "configs/example.yaml",
                    "baseline_reward": 300.0,
                    "threshold": {"metric": "avg_final_10", "min_ratio": 0.85},
                }
            ],
            "settings": {"seed": 123, "report_dir": "output"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(suite_data, f)
            f_path = f.name

        try:
            suite = load_suite(f_path)
            assert suite.name == "Test Suite"
            assert suite.version == "1.0"
            assert len(suite.benchmarks) == 1
            assert suite.benchmarks[0].name == "bench1"
            assert suite.benchmarks[0].baseline_reward == 300.0
            assert suite.benchmarks[0].threshold.min_ratio == 0.85
            assert suite.settings.seed == 123
            assert suite.settings.report_dir == "output"
        finally:
            os.unlink(f_path)

    def test_load_invalid_suite_missing_fields(self):
        """验证：缺少必需字段时抛出 ValueError。"""
        invalid_data = {"suite": {"name": "No Benchmarks"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(invalid_data, f)
            f_path = f.name

        try:
            with pytest.raises(ValueError, match="at least one benchmark"):
                load_suite(f_path)
        finally:
            os.unlink(f_path)


class TestBenchmarkRunPersistence:
    """BenchmarkRun 持久化与历史查询测试。"""

    def _make_db(self):
        return Database("sqlite:///:memory:")

    def test_create_benchmark_run_stores_record(self):
        """验证：create_benchmark_run 正确写入 BenchmarkRun 及关联结果。"""
        db = self._make_db()
        result_records = [
            {
                "benchmark_name": "bench_a",
                "baseline_reward": 100.0,
                "actual_reward": 95.0,
                "ratio": 0.95,
                "threshold_ratio": 0.90,
                "passed": True,
            },
            {
                "benchmark_name": "bench_b",
                "baseline_reward": 100.0,
                "actual_reward": 50.0,
                "ratio": 0.50,
                "threshold_ratio": 0.90,
                "passed": False,
            },
        ]
        run_id = db.create_benchmark_run(
            suite_name="TestSuite",
            overall_status="REGRESSION DETECTED",
            passed_count=1,
            failed_count=1,
            result_records=result_records,
        )
        assert run_id is not None
        runs = db.list_benchmark_runs("TestSuite", limit=10)
        assert len(runs) == 1
        assert runs[0].suite_name == "TestSuite"
        assert runs[0].overall_status == "REGRESSION DETECTED"
        assert runs[0].passed_count == 1
        assert runs[0].failed_count == 1
        assert len(runs[0].results) == 2

    def test_list_benchmark_runs_respects_limit_and_order(self):
        """验证：list_benchmark_runs 按时间倒序返回且遵守 limit 参数。"""
        db = self._make_db()
        for i in range(5):
            db.create_benchmark_run(
                suite_name="MySuite",
                overall_status="PASS",
                passed_count=3,
                failed_count=0,
                result_records=[
                    {
                        "benchmark_name": f"bench_{i}",
                        "baseline_reward": 100.0,
                        "actual_reward": 90.0 + i,
                        "ratio": (90.0 + i) / 100.0,
                        "threshold_ratio": 0.90,
                        "passed": True,
                    }
                ],
            )
        runs = db.list_benchmark_runs("MySuite", limit=3)
        assert len(runs) == 3
        assert runs[0].results[0].actual_reward >= runs[1].results[0].actual_reward

    def test_list_benchmark_runs_filters_by_suite_name(self):
        """验证：list_benchmark_runs 仅返回匹配 suite_name 的记录。"""
        db = self._make_db()
        db.create_benchmark_run(
            suite_name="SuiteA",
            overall_status="PASS",
            passed_count=1,
            failed_count=0,
            result_records=[
                {"benchmark_name": "a", "baseline_reward": 50.0,
                 "actual_reward": 50.0, "ratio": 1.0, "threshold_ratio": 0.9, "passed": True}
            ],
        )
        db.create_benchmark_run(
            suite_name="SuiteB",
            overall_status="PASS",
            passed_count=1,
            failed_count=0,
            result_records=[
                {"benchmark_name": "b", "baseline_reward": 50.0,
                 "actual_reward": 50.0, "ratio": 1.0, "threshold_ratio": 0.9, "passed": True}
            ],
        )
        runs_a = db.list_benchmark_runs("SuiteA", limit=10)
        runs_b = db.list_benchmark_runs("SuiteB", limit=10)
        assert len(runs_a) == 1
        assert runs_a[0].results[0].benchmark_name == "a"
        assert len(runs_b) == 1
        assert runs_b[0].results[0].benchmark_name == "b"

    def test_benchmark_result_fields_persisted(self):
        """验证：BenchmarkResultRecord 各字段正确持久化。"""
        db = self._make_db()
        db.create_benchmark_run(
            suite_name="FieldTest",
            overall_status="PASS",
            passed_count=1,
            failed_count=0,
            result_records=[
                {
                    "benchmark_name": "precise_bench",
                    "baseline_reward": 200.0,
                    "actual_reward": 185.5,
                    "ratio": 0.9275,
                    "threshold_ratio": 0.90,
                    "passed": True,
                }
            ],
        )
        runs = db.list_benchmark_runs("FieldTest")
        rec = runs[0].results[0]
        assert rec.benchmark_name == "precise_bench"
        assert abs(rec.baseline_reward - 200.0) < 1e-6
        assert abs(rec.actual_reward - 185.5) < 1e-6
        assert abs(rec.ratio - 0.9275) < 1e-4
        assert abs(rec.threshold_ratio - 0.90) < 1e-6
        assert rec.passed == 1


class TestReportHistorySection:
    """报告历史趋势部分测试。"""

    def _make_db_with_history(self):
        db = Database("sqlite:///:memory:")
        for i in range(3):
            db.create_benchmark_run(
                suite_name="HistSuite",
                overall_status="PASS" if i < 2 else "REGRESSION DETECTED",
                passed_count=2 if i < 2 else 1,
                failed_count=0 if i < 2 else 1,
                result_records=[
                    {
                        "benchmark_name": "bench_x",
                        "baseline_reward": 100.0,
                        "actual_reward": 90.0 + i * 5,
                        "ratio": (90.0 + i * 5) / 100.0,
                        "threshold_ratio": 0.90,
                        "passed": True,
                    },
                    {
                        "benchmark_name": "bench_y",
                        "baseline_reward": 100.0,
                        "actual_reward": 80.0 + i * 10,
                        "ratio": (80.0 + i * 10) / 100.0,
                        "threshold_ratio": 0.90,
                        "passed": i >= 1,
                    },
                ],
            )
        return db

    def test_history_table_rendered_when_db_available(self):
        """验证：当 db 可用时报告包含 History Trend 表格。"""
        db = self._make_db_with_history()
        results = [
            BenchmarkResult(
                name="bench_x", description="", baseline_reward=100.0,
                actual_reward=100.0, metric_used="avg_final_10", ratio=1.0,
                threshold_ratio=0.90, passed=True, expect_fail=False,
                effective_pass=True, total_episodes=10, duration_seconds=1.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            reporter = BenchmarkReporter(
                results=results, suite_name="HistSuite", report_dir=tmp_dir, db=db
            )
            report_path = reporter.generate()
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "## History Trend" in content
            assert "bench_x" in content
            assert "bench_y" in content
            assert "PASS" in content

    def test_no_history_when_db_is_none(self):
        """验证：当 db 为 None 时报告不包含 History Trend 部分。"""
        results = [
            BenchmarkResult(
                name="bench_x", description="", baseline_reward=100.0,
                actual_reward=100.0, metric_used="avg_final_10", ratio=1.0,
                threshold_ratio=0.90, passed=True, expect_fail=False,
                effective_pass=True, total_episodes=10, duration_seconds=1.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            reporter = BenchmarkReporter(
                results=results, suite_name="NoDbSuite", report_dir=tmp_dir, db=None
            )
            report_path = reporter.generate()
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "## History Trend" not in content
