"""
基准测试模块：自动化训练回归检测。
"""

from benchmark.runner import BenchmarkRunner, BenchmarkResult
from benchmark.reporter import BenchmarkReporter
from benchmark.schema import load_suite, BenchmarkSuite
from benchmark.threshold import check_threshold, compute_effective_pass

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkReporter",
    "load_suite",
    "BenchmarkSuite",
    "check_threshold",
    "compute_effective_pass",
]
