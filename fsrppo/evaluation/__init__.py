"""
Evaluation Module for FSRPPO

This module contains  evaluation and backtesting components:
- Backtesting framework with realistic market simulation
- Performance metrics and risk analysis
- Benchmark comparisons
- Statistical significance testing
- Visualization and reporting tools
"""

from .backtester import Backtester
from .metrics import PerformanceMetrics

__all__ = [
    "Backtester",
    "PerformanceMetrics",
]