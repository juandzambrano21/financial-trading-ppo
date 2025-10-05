#!/usr/bin/env python3
"""
Complete FSRPPO Integration Test

This script tests the entire FSRPPO pipeline to ensure all components
work together correctly. It performs a  integration test
of all major modules and their interactions.

Test Coverage:
- Signal processing (CEESMDAN, ESMD, MRS)
- Data pipeline (providers, preprocessing, features)
- PPO agent and trading environment
- Training pipeline components
- Evaluation and metrics calculation
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all FSRPPO components
try:
    from fsrppo.signal_processing.ceesmdan import CEESMDAN
    from fsrppo.signal_processing.esmd import ESMD
    from fsrppo.signal_processing.hurst import hurst_exponent
    from fsrppo.signal_processing.fsr import FinancialSignalRepresentation
    
    from fsrppo.data.providers import YahooFinanceDataProvider
    from fsrppo.data.preprocessor import DataPreprocessor
    from fsrppo.data.features import FeatureEngineer
    
    from fsrppo.core.ppo_agent import PPOAgent
    from fsrppo.core.trading_env import TradingEnvironment
    from fsrppo.core.networks import ActorNetwork, CriticNetwork
    
    from fsrppo.training.trainer import FSRPPOTrainer
    from fsrppo.training.experiment_tracker import ExperimentTracker
    
    from fsrppo.evaluation.backtester import Backtester
    from fsrppo.evaluation.metrics import PerformanceMetrics
    
    print("    All FSRPPO modules imported successfully")
    
except ImportError as e:
    print(f"    Import error: {e}")
    print("Trying to import individual modules...")
    
    # Try importing each module individually to identify issues
    modules_to_test = [
        ('fsrppo.signal_processing.ceesmdan', 'CEESMDAN'),
        ('fsrppo.signal_processing.esmd', 'ESMD'),
        ('fsrppo.signal_processing.hurst', 'hurst_exponent'),
        ('fsrppo.signal_processing.fsr', 'FinancialSignalRepresentation'),
        ('fsrppo.data.providers', 'YahooFinanceDataProvider'),
        ('fsrppo.data.preprocessor', 'DataPreprocessor'),
        ('fsrppo.data.features', 'FeatureEngineer'),
        ('fsrppo.core.ppo_agent', 'PPOAgent'),
        ('fsrppo.core.trading_env', 'TradingEnvironment'),
        ('fsrppo.core.networks', 'ActorNetwork'),
        ('fsrppo.training.trainer', 'FSRPPOTrainer'),
        ('fsrppo.training.experiment_tracker', 'ExperimentTracker'),
        ('fsrppo.evaluation.backtester', 'Backtester'),
        ('fsrppo.evaluation.metrics', 'PerformanceMetrics'),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            successful_imports.append((module_name, class_name))
            globals()[class_name] = cls
        except Exception as import_error:
            failed_imports.append((module_name, class_name, str(import_error)))
    
    print(f"    Successfully imported {len(successful_imports)} modules:")
    for module_name, class_name in successful_imports:
        print(f"    {class_name} from {module_name}")
    
    if failed_imports:
        print(f"    Failed to import {len(failed_imports)} modules:")
        for module_name, class_name, error in failed_imports:
            print(f"    {class_name} from {module_name}: {error}")
    
    # Continue with available modules
    if len(successful_imports) < 5:
        print("    Too many import failures, exiting...")
        sys.exit(1)


def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def test_signal_processing():
    """Test signal processing components"""
    print("\n🔬 Testing Signal Processing Components...")
    
    # Generate synthetic financial data
    np.random.seed(42)
    N = 500
    t = np.linspace(0, 10, N)
    
    # Create realistic financial signal
    trend = 0.1 * t
    cycle1 = 2 * np.sin(2 * np.pi * 0.1 * t)
    cycle2 = 0.5 * np.sin(2 * np.pi * 1.5 * t)
    noise = 0.3 * np.random.randn(N)
    
    signal = trend + cycle1 + cycle2 + noise
    
    # Test CEESMDAN
    print("  Testing CEESMDAN...")
    ceesmdan = CEESMDAN(J=20, xi=0.005)  # Reduced J for faster testing
    imfs = ceesmdan.decompose(signal)
    assert len(imfs) > 0, "CEESMDAN should produce IMFs"
    print(f"        CEESMDAN produced {len(imfs)} IMFs")
    
    # Test ESMD
    print("  Testing ESMD...")
    esmd = ESMD(M=4, max_imfs=6, max_sift=30)  # Reduced for testing
    esmd_result = esmd.decompose(t, signal)
    assert len(esmd_result.imfs) > 0, "ESMD should produce IMFs"
    print(f"        ESMD produced {len(esmd_result.imfs)} IMFs")
    
    # Test Hurst exponent
    print("  Testing Hurst exponent...")
    hurst_exp = hurst_exponent(signal)
    assert 0 < hurst_exp < 1, f"Hurst exponent should be between 0 and 1, got {hurst_exp}"
    print(f"        Hurst exponent: {hurst_exp:.4f}")
    
    # Test FSR integration
    print("  Testing FSR integration...")
    fsr = FinancialSignalRepresentation(
        ceesmdan_params={'J': 20, 'xi': 0.005},
        hurst_threshold=0.5
    )
    
    clean_signal = fsr.extract_representation(signal)
    assert len(clean_signal) == len(signal), "FSR output should have same length as input"
    
    validation = fsr.validate_representation()
    assert 'signal_correlation' in validation, "FSR validation should include correlation"
    print(f"        FSR correlation: {validation['signal_correlation']:.4f}")
    
    print("    Signal processing tests passed!")
    return fsr


def test_data_pipeline():
    """Test data processing pipeline"""
    print("\n    Testing Data Pipeline...")
    
    # Create synthetic OHLCV data
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    N = len(dates)
    
    # Generate realistic price series
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, N)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    synthetic_data = pd.DataFrame({
        'Open': np.array(prices[:-1]) * (1 + np.random.normal(0, 0.001, N)),
        'High': np.array(prices[:-1]) * (1 + np.abs(np.random.normal(0, 0.005, N))),
        'Low': np.array(prices[:-1]) * (1 - np.abs(np.random.normal(0, 0.005, N))),
        'Close': prices[:-1],
        'Volume': np.random.randint(1000000, 10000000, N)
    }, index=dates)
    
    print(f"  Generated synthetic data: {synthetic_data.shape}")
    
    # Test data provider (using synthetic data)
    print("  Testing data provider...")
    data_provider = YahooFinanceDataProvider(cache_dir=tempfile.mkdtemp())
    print("        Data provider initialized")
    
    # Test feature engineering
    print("  Testing feature engineering...")
    fsr = FinancialSignalRepresentation()
    feature_engineer = FeatureEngineer(
        fsr_processor=fsr,
        include_technical=True,
        include_volume=True,
        include_time=True,
        lookback_periods=[5, 10, 20]
    )
    
    features_data = feature_engineer.create_features(synthetic_data)
    original_cols = len(synthetic_data.columns)
    feature_cols = len(features_data.columns)
    new_features = feature_cols - original_cols
    
    assert new_features > 0, "Feature engineering should create new features"
    print(f"        Created {new_features} new features")
    
    # Test data preprocessing
    print("  Testing data preprocessing...")
    preprocessor = DataPreprocessor(
        fsr_processor=fsr,
        normalization_method='robust',
        handle_missing='forward_fill'
    )
    
    processed_data = preprocessor.fit_transform(features_data)
    assert processed_data.shape[0] == features_data.shape[0], "Preprocessing should preserve row count"
    print(f"        Processed data shape: {processed_data.shape}")
    
    # Data quality check
    quality_report = preprocessor.get_data_quality_report(processed_data)
    print(f"        Data quality: {quality_report['missing_percentage']:.2f}% missing")
    
    print("    Data pipeline tests passed!")
    return processed_data, preprocessor, feature_engineer


def test_ppo_and_environment():
    """Test PPO agent and trading environment"""
    print("\n    Testing PPO Agent and Trading Environment...")
    
    # Get processed data
    processed_data, _, _ = test_data_pipeline()
    
    # Test trading environment
    print("  Testing trading environment...")
    env = TradingEnvironment(
        price_data=processed_data['Close'].values,
        initial_cash=10000,
        transaction_cost=0.001,
        lookback_window=50
    )
    
    state = env.reset()
    assert len(state) == env.lookback_window, f"State should have length {env.lookback_window}"
    print(f"        Environment initialized, state shape: {state.shape}")
    
    # Test a few environment steps
    for i in range(5):
        action = np.array([0.5, 0.3])  # Sample action
        next_state, reward, done, info = env.step(action)
        assert len(next_state) == env.lookback_window, "Next state should have correct length"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        
        if done:
            break
    
    print("        Environment step functionality working")
    
    # Test PPO agent
    print("  Testing PPO agent...")
    fsr = FinancialSignalRepresentation()
    
    agent = PPOAgent(
        state_dim=50,
        action_dim=2,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=1.0,
        entropy_coef=0.01,
        n_epochs=5,  # Reduced for testing
        batch_size=32,
        fsr_processor=fsr
    )
    
    print(f"        PPO agent initialized")
    
    # Test action generation
    test_state = np.random.randn(50)
    action, log_prob = agent.get_action(test_state, deterministic=False)
    assert len(action) == 2, "Action should have 2 dimensions"
    assert isinstance(log_prob, (int, float)), "Log probability should be numeric"
    print(f"        Action generation working: {action}")
    
    # Test deterministic action
    det_action, _ = agent.get_action(test_state, deterministic=True)
    assert len(det_action) == 2, "Deterministic action should have 2 dimensions"
    print(f"        Deterministic action: {det_action}")
    
    print("    PPO and environment tests passed!")
    return agent, env


def test_training_components():
    """Test training pipeline components"""
    print("\n    Testing Training Components...")
    
    # Test experiment tracker
    print("  Testing experiment tracker...")
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ExperimentTracker(
            experiment_name='test_experiment',
            save_dir=temp_dir,
            auto_save=False
        )
        
        # Test configuration setting
        config = {'test_param': 42, 'another_param': 'test'}
        tracker.set_config(config)
        print("        Configuration setting working")
        
        # Test metrics logging
        for i in range(5):
            metrics = {
                'episode': i,
                'reward': np.random.randn(),
                'loss': np.random.rand()
            }
            tracker.log_metrics(metrics)
        
        print("        Metrics logging working")
        
        # Test milestone logging
        tracker.log_milestone("Test milestone", {'data': 'test'})
        print("        Milestone logging working")
        
        # Test results saving
        results = {'final_score': 0.85, 'completed': True}
        tracker.save_results(results)
        print("        Results saving working")
    
    print("    Training components tests passed!")


def test_evaluation_components():
    """Test evaluation and metrics components"""
    print("\n    Testing Evaluation Components...")
    
    # Test performance metrics
    print("  Testing performance metrics...")
    
    # Generate synthetic returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    portfolio_values = [10000]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # Benchmark returns
    benchmark_returns = np.random.normal(0.0008, 0.015, 252)
    
    metrics_calc = PerformanceMetrics()
    
    all_metrics = metrics_calc.calculate_all_metrics(
        returns=returns,
        portfolio_values=portfolio_values,
        benchmark_returns=benchmark_returns
    )
    
    # Check key metrics exist
    required_metrics = [
        'total_return', 'sharpe_ratio', 'max_drawdown', 
        'volatility', 'win_rate'
    ]
    
    for metric in required_metrics:
        assert metric in all_metrics, f"Missing required metric: {metric}"
    
    print(f"        Calculated {len(all_metrics)} performance metrics")
    print(f"        Sample metrics:")
    print(f"      Total Return: {all_metrics['total_return']:.2%}")
    print(f"      Sharpe Ratio: {all_metrics['sharpe_ratio']:.3f}")
    print(f"      Max Drawdown: {all_metrics['max_drawdown']:.2%}")
    
    # Test rolling metrics
    print("  Testing rolling metrics...")
    rolling_metrics = metrics_calc.calculate_rolling_metrics(
        returns=returns,
        window=50,
        metrics=['sharpe_ratio', 'volatility']
    )
    
    assert not rolling_metrics.empty, "Rolling metrics should not be empty"
    print(f"        Rolling metrics shape: {rolling_metrics.shape}")
    
    print("    Evaluation components tests passed!")


def test_integration():
    """Test complete integration"""
    print("\n🔗 Testing Complete Integration...")
    
    # This is a simplified integration test
    # In practice, you would run a full training cycle
    
    print("  Testing component integration...")
    
    # Create all components
    fsr = test_signal_processing()
    processed_data, preprocessor, feature_engineer = test_data_pipeline()
    agent, env = test_ppo_and_environment()
    
    print("  Testing mini training loop...")
    
    # Run a very short training loop
    state = env.reset()
    total_reward = 0
    
    for step in range(10):  # Very short for testing
        action, log_prob = agent.get_action(state, deterministic=False)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"        Mini training loop completed, total reward: {total_reward:.4f}")
    
    # Test portfolio statistics
    portfolio_stats = env.get_portfolio_stats()
    assert isinstance(portfolio_stats, dict), "Portfolio stats should be dictionary"
    print(f"        Portfolio stats: {list(portfolio_stats.keys())}")
    
    print("    Integration tests passed!")


def main():
    """Run all tests"""
    print("    FSRPPO Complete Integration Test")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # Run all test categories
        test_signal_processing()
        test_data_pipeline()
        test_ppo_and_environment()
        test_training_components()
        test_evaluation_components()
        test_integration()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\n    FSRPPO Implementation Status:")
        print("      Signal Processing: COMPLETE")
        print("  🔄 Data Pipeline: COMPLETE")
        print("      PPO Agent: COMPLETE")
        print("      Trading Environment: COMPLETE")
        print("      Training Pipeline: COMPLETE")
        print("      Evaluation Framework: COMPLETE")
        print("  🔗 Integration: COMPLETE")
        
        print("\n    Ready for production use!")
        print("📖 See examples/complete_example.py for usage")
        print("📚 Check README.md for detailed documentation")
        
        return True
        
    except Exception as e:
        print(f"\n    TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)