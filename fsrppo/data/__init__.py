"""
Data Module for FSRPPO

This module handles financial data acquisition, preprocessing, and management:
- Yahoo Finance integration for real market data
- Data preprocessing and cleaning
- Feature engineering pipelines
- Data validation and quality checks
"""

from .yahoo_finance import YahooFinanceDataProvider
from .preprocessor import DataPreprocessor
from .features import FeatureEngineer

__all__ = [
    "YahooFinanceDataProvider",
    "DataPreprocessor", 
    "FeatureEngineer"
]