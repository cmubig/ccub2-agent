"""
Dataset Splits

Standardized splits for CultureBench-Global.
"""

from .benchmark_splits import (
    BenchmarkSplit,
    load_benchmark_split,
    save_benchmark_split,
    get_default_splits,
)

__all__ = [
    "BenchmarkSplit",
    "load_benchmark_split",
    "save_benchmark_split",
    "get_default_splits",
]
