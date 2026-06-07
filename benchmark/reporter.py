"""
基准测试报告生成器：输出 Markdown 表格 + matplotlib 对比图表。
"""

import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.runner import BenchmarkResult


class BenchmarkReporter:
    def __init__(self, results: list, suite_name: str = "", report_dir: str = "reports",
                 db=None):
        self.results = results
        self.suite_name = suite_name
        self.report_dir = report_dir
        self.db = db

    def generate(self) -> str:
        os.makedirs(self.report_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        chart_filename = f"benchmark_chart_{timestamp}.png"
        chart_path = os.path.join(self.report_dir, chart_filename)
        self._generate_chart(chart_path)

        markdown = self._render_markdown(chart_filename)
        report_path = os.path.join(self.report_dir, "benchmark_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return report_path

    def get_overall_status(self) -> str:
        if all(r.effective_pass for r in self.results):
            return "PASS"
        return "REGRESSION DETECTED"

    def _render_markdown(self, chart_filename: str) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        overall = self.get_overall_status()

        total = len(self.results)
        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = total - passed_count
        expected_fails = sum(1 for r in self.results if r.expect_fail)

        lines = []
        lines.append(f"# Benchmark Report\n")
        lines.append(f"**Suite:** {self.suite_name}  ")
        lines.append(f"**Date:** {now}  ")
        lines.append(f"**Overall Status:** {overall}\n")

        lines.append("## Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Benchmarks | {total} |")
        lines.append(f"| Passed | {passed_count} |")
        lines.append(f"| Failed | {failed_count} |")
        lines.append(f"| Expected Failures | {expected_fails} |")
        lines.append("")

        lines.append("## Results\n")
        lines.append("| # | Benchmark | Baseline | Actual | Ratio | Threshold | Status |")
        lines.append("|---|-----------|----------|--------|-------|-----------|--------|")
        for i, r in enumerate(self.results, 1):
            if r.error:
                status = "ERROR"
            elif r.passed:
                status = "PASS"
            elif r.expect_fail:
                status = "FAIL (expected)"
            else:
                status = "**FAIL - REGRESSION**"
            lines.append(
                f"| {i} | {r.name} | {r.baseline_reward:.1f} | "
                f"{r.actual_reward:.1f} | {r.ratio:.1%} | "
                f">= {r.threshold_ratio:.0%} | {status} |"
            )
        lines.append("")

        lines.append("## Chart\n")
        lines.append(f"![Benchmark Comparison](./{chart_filename})\n")

        lines.append("## Conclusion\n")
        if overall == "PASS":
            lines.append(
                f"**PASS** - All benchmarks meet their expected outcomes "
                f"({passed_count} passed, {failed_count} failed as expected).\n"
            )
        else:
            failed_names = [r.name for r in self.results if not r.effective_pass]
            lines.append(
                f"**REGRESSION DETECTED** - The following benchmarks did not meet expectations: "
                f"{', '.join(failed_names)}\n"
            )

        history_section = self._render_history()
        if history_section:
            lines.append(history_section)

        return "\n".join(lines)

    def _render_history(self) -> str:
        if not self.db:
            return ""
        try:
            runs = self.db.list_benchmark_runs(self.suite_name, limit=3)
        except Exception:
            return ""
        if not runs:
            return ""

        lines = []
        lines.append("## History Trend\n")

        benchmark_names = sorted(
            {rec.benchmark_name for run in runs for rec in run.results}
        )

        header = "| Date | Status |"
        separator = "|------|--------|"
        for name in benchmark_names:
            header += f" {name} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        for run in runs:
            date_str = run.run_at.strftime("%Y-%m-%d %H:%M")
            reward_map = {rec.benchmark_name: rec.actual_reward for rec in run.results}
            row = f"| {date_str} | {run.overall_status} |"
            for name in benchmark_names:
                reward = reward_map.get(name)
                row += f" {reward:.1f} |" if reward is not None else " - |"
            lines.append(row)

        lines.append("")
        return "\n".join(lines)

    def _generate_chart(self, chart_path: str):
        names = [r.name for r in self.results]
        baselines = [r.baseline_reward for r in self.results]
        actuals = [r.actual_reward for r in self.results]
        thresholds = [r.baseline_reward * r.threshold_ratio for r in self.results]

        x = range(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            [i - width / 2 for i in x],
            baselines,
            width,
            label="Baseline",
            color="steelblue",
        )
        colors = ["green" if r.effective_pass else "red" for r in self.results]
        ax.bar(
            [i + width / 2 for i in x],
            actuals,
            width,
            label="Actual",
            color=colors,
        )

        for i, thresh in enumerate(thresholds):
            ax.hlines(
                thresh,
                i - 0.4,
                i + 0.4,
                colors="orange",
                linestyles="dashed",
                linewidth=1.5,
                label="Threshold" if i == 0 else None,
            )

        ax.set_xlabel("Benchmark")
        ax.set_ylabel("Reward")
        ax.set_title("Benchmark Results: Baseline vs Actual")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
