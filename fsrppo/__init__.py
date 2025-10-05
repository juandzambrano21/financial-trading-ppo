"""
FSRPPO: Financial Signal Representation Proximal Policy Optimization

A  reinforcement learning framework for financial trading that combines:
- Advanced signal processing (CEEMDAN, ESMD, Hurst exponent)
- State-of-the-art PPO implementation
- Real market data integration
- Sophisticated risk management

Authors: Inspired by Wang & Wang's FSRPPO paper
Implementation: Robust Python package
"""

__version__ = "1.0.0"
__author__ = "FSRPPO Development Team"
__email__ = "fsrppo@example.com"

# Core imports - only import what exists
try:
    from .signal_processing.ceemdan import CEEMDAN
    from .signal_processing.esmd import ESMD
    from .signal_processing.hurst import hurst_exponent
    from .signal_processing.fsr import FinancialSignalRepresentation
    from .data.preprocessor import DataPreprocessor
    from .data.features import FeatureEngineer
    from .core.networks import ActorNetwork, CriticNetwork
    from .evaluation.metrics import PerformanceMetrics
    
    __all__ = [
        "CEEMDAN",
        "ESMD", 
        "hurst_exponent",
        "FinancialSignalRepresentation",
        "DataPreprocessor",
        "FeatureEngineer",
        "ActorNetwork",
        "CriticNetwork",
        "PerformanceMetrics"
    ]
    
except ImportError as e:
    # Graceful degradation - only export what's available
    __all__ = []
    print(f"Warning: Some FSRPPO modules could not be imported: {e}")