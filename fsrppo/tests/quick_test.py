#!/usr/bin/env python3
"""
Quick FSRPPO Test

A fast test to verify core functionality without heavy computations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from fsrppo.signal_processing.hurst import hurst_exponent
        print("      Hurst exponent imported")
        
        from fsrppo.data.preprocessor import DataPreprocessor
        print("      Data preprocessor imported")
        
        from fsrppo.data.features import FeatureEngineer
        print("      Feature engineer imported")
        
        from fsrppo.core.networks import ActorNetwork, CriticNetwork
        print("      Neural networks imported")
        
        from fsrppo.evaluation.metrics import PerformanceMetrics
        print("      Performance metrics imported")
        
        return True
        
    except Exception as e:
        print(f"      Import failed: {e}")
        return False

def test_hurst_exponent():
    """Test Hurst exponent calculation"""
    print("\n    Testing Hurst exponent...")
    
    try:
        from fsrppo.signal_processing.hurst import hurst_exponent
        
        # Generate test signal
        np.random.seed(42)
        signal = np.random.randn(100)
        
        # Calculate Hurst exponent
        h = hurst_exponent(signal)
        
        assert 0 < h < 1, f"Hurst exponent should be between 0 and 1, got {h}"
        print(f"      Hurst exponent: {h:.4f}")
        
        return True
        
    except Exception as e:
        print(f"      Hurst test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering"""
    print("\n🔧 Testing feature engineering...")
    
    try:
        from fsrppo.data.features import FeatureEngineer
        
        # Create synthetic OHLCV data
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        N = len(dates)
        
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, N)
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'Open': np.array(prices[:-1]) * (1 + np.random.normal(0, 0.001, N)),
            'High': np.array(prices[:-1]) * (1 + np.abs(np.random.normal(0, 0.005, N))),
            'Low': np.array(prices[:-1]) * (1 - np.abs(np.random.normal(0, 0.005, N))),
            'Close': prices[:-1],
            'Volume': np.random.randint(1000000, 10000000, N)
        }, index=dates)
        
        # Create feature engineer (without FSR to avoid slow computation)
        feature_engineer = FeatureEngineer(
            fsr_processor=None,
            include_technical=True,
            include_volume=True,
            include_time=True,
            lookback_periods=[5, 10]
        )
        
        # Create features
        features_data = feature_engineer.create_features(data)
        
        original_cols = len(data.columns)
        feature_cols = len(features_data.columns)
        new_features = feature_cols - original_cols
        
        assert new_features > 0, "Should create new features"
        print(f"      Created {new_features} new features")
        
        return True
        
    except Exception as e:
        print(f"      Feature engineering test failed: {e}")
        return False

def test_neural_networks():
    """Test neural network creation"""
    print("\n🧠 Testing neural networks...")
    
    try:
        from fsrppo.core.networks import ActorNetwork, CriticNetwork
        import torch
        
        # Test Actor network
        actor = ActorNetwork(state_dim=50, action_dim=2, hidden_dim=64)
        test_state = torch.randn(1, 50)
        actions = actor(test_state)
        
        assert actions.shape == (1, 2), "Actor output shape incorrect"
        print("      Actor network working")
        
        # Test Critic network
        critic = CriticNetwork(state_dim=50, hidden_dim=64)
        value = critic(test_state)
        
        assert value.shape == (1, 1), "Critic output shape incorrect"
        print("      Critic network working")
        
        return True
        
    except Exception as e:
        print(f"      Neural network test failed: {e}")
        return False

def test_performance_metrics():
    """Test performance metrics calculation"""
    print("\n    Testing performance metrics...")
    
    try:
        from fsrppo.evaluation.metrics import PerformanceMetrics
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        
        # Portfolio values
        portfolio_values = [10000]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # Calculate metrics
        metrics_calc = PerformanceMetrics()
        metrics = metrics_calc.calculate_all_metrics(
            returns=returns,
            portfolio_values=portfolio_values
        )
        
        # Check key metrics exist
        required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        print(f"      Calculated {len(metrics)} metrics")
        print(f"    Total Return: {metrics['total_return']:.2%}")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"      Performance metrics test failed: {e}")
        return False

def main():
    """Run quick tests"""
    print("    FSRPPO Quick Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_hurst_exponent,
        test_feature_engineering,
        test_neural_networks,
        test_performance_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"    Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n    FSRPPO Core Components Working:")
        print("      Signal Processing:    ")
        print("  🔧 Feature Engineering:    ")
        print("  🧠 Neural Networks:    ")
        print("      Performance Metrics:    ")
        print("\n    FSRPPO is ready for use!")
        return True
    else:
        print(f"    {total - passed} tests failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)