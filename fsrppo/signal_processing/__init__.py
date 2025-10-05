"""
Signal Processing Module for FSRPPO

This module contains advanced signal processing techniques for financial data
according to Wang & Wang (2024) FSRPPO paper:

- CEESMDAN: Complete Ensemble Extreme-point Symmetric Mode Decomposition with Adaptive Noise
- ESMD: Extreme-Point Symmetric Mode Decomposition (Algorithm 2)
- MRS: Modified Rescaled Range Analysis for Hurst exponent (Algorithm 4)
- FSR: Financial Signal Representation pipeline (Section 2.1.4)
"""

from .ceesmdan import CEESMDAN
from .esmd import ESMD
from .hurst import hurst_exponent, HurstAnalyzer
from .fsr import FinancialSignalRepresentation

__all__ = [
    "CEESMDAN",
    "ESMD", 
    "hurst_exponent",
    "HurstAnalyzer",
    "FinancialSignalRepresentation"
]