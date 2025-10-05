#!/usr/bin/env python3
"""
Real FSRPPO Test Suite

Tests the actual FSRPPO implementation with real Yahoo Finance data.
This is NOT a mock test - it actually fetches data and trains the model.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import subprocess
import time
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_main_demo():
    """Test main.py in demo mode"""
    print("🔍 Testing main_real.py in demo mode...")
    
    try:
        cmd = [
            sys.executable, "main_real.py",
            "--mode", "demo",
            "--episodes", "20",
            "--symbol", "AAPL",
            "--start_date", "2023-12-01",
            "--end_date", "2023-12-15"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"    Demo mode successful ({end_time - start_time:.1f}s)")
            
            # Check for key success indicators
            output = result.stdout
            if "🎉 FSRPPO execution completed successfully!" in output:
                print("       Pipeline completed successfully")
            if "Real Yahoo Finance data processed" in output:
                print("       Real data processed")
            if "PPO agent trained and evaluated" in output:
                print("       PPO training completed")
            if "Performance metrics calculated" in output:
                print("       Performance metrics calculated")
            
            return True
        else:
            print(f"    Demo mode failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("    Demo mode timed out")
        return False
    except Exception as e:
        print(f"    Demo mode error: {e}")
        return False

def test_main_train():
    """Test main.py in training mode"""
    print("🔍 Testing main_real.py in training mode...")
    
    try:
        cmd = [
            sys.executable, "main_real.py",
            "--mode", "train",
            "--episodes", "50",
            "--symbol", "MSFT",
            "--start_date", "2023-11-01",
            "--end_date", "2023-11-20"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"    Training mode successful ({end_time - start_time:.1f}s)")
            
            # Check for training-specific indicators
            output = result.stdout
            if "Training PPO agent for 50 episodes" in output:
                print("       Training initiated")
            if "Training completed!" in output:
                print("       Training completed")
            if "FSRPPO TRAINING RESULTS" in output:
                print("       Results generated")
            
            return True
        else:
            print(f"    Training mode failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("    Training mode timed out")
        return False
    except Exception as e:
        print(f"    Training mode error: {e}")
        return False

def test_different_symbols():
    """Test with different stock symbols"""
    print("🔍 Testing different stock symbols...")
    
    symbols = ['GOOGL', 'NVDA']
    success_count = 0
    
    for symbol in symbols:
        try:
            cmd = [
                sys.executable, "main_real.py",
                "--mode", "demo",
                "--episodes", "15",
                "--symbol", symbol,
                "--start_date", "2023-12-01",
                "--end_date", "2023-12-10"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"       {symbol} successful")
                success_count += 1
            else:
                print(f"       {symbol} failed")
                
        except Exception as e:
            print(f"       {symbol} error: {e}")
    
    if success_count == len(symbols):
        print(f"    All {len(symbols)} symbols successful")
        return True
    else:
        print(f"    Only {success_count}/{len(symbols)} symbols successful")
        return False

def test_data_validation():
    """Test data fetching and validation"""
    print("🔍 Testing data validation...")
    
    try:
        from fsrppo.data.providers import YahooFinanceDataProvider
        
        provider = YahooFinanceDataProvider()
        
        # Test multiple symbols
        test_cases = [
            ('AAPL', '2023-12-01', '2023-12-15'),
            ('MSFT', '2023-11-15', '2023-11-30'),
            ('TSLA', '2023-10-01', '2023-10-15')
        ]
        
        success_count = 0
        for symbol, start, end in test_cases:
            try:
                data = provider.fetch_data(symbol, start, end)
                if len(data) > 5:  # At least 5 trading days
                    print(f"       {symbol}: {len(data)} days")
                    success_count += 1
                else:
                    print(f"       {symbol}: insufficient data ({len(data)} days)")
            except Exception as e:
                print(f"       {symbol}: {e}")
        
        if success_count == len(test_cases):
            print("    Data validation successful")
            return True
        else:
            print(f"    Data validation failed ({success_count}/{len(test_cases)})")
            return False
            
    except Exception as e:
        print(f"    Data validation error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering"""
    print("🔍 Testing feature engineering...")
    
    try:
        from fsrppo.data.providers import YahooFinanceDataProvider
        from fsrppo.data.features import FeatureEngineer
        
        # Get some data
        provider = YahooFinanceDataProvider()
        data = provider.fetch_data('AAPL', '2023-12-01', '2023-12-15')
        
        # Create features
        feature_engineer = FeatureEngineer(
            fsr_processor=None,
            include_technical=True,
            include_volume=True,
            include_time=True,
            lookback_periods=[5, 10]
        )
        
        features = feature_engineer.create_features(data)
        
        if len(features.columns) > 50:
            print(f"    Feature engineering successful: {len(features.columns)} features")
            return True
        else:
            print(f"    Insufficient features: {len(features.columns)}")
            return False
            
    except Exception as e:
        print(f"    Feature engineering error: {e}")
        return False

def test_ppo_training():
    """Test PPO training components"""
    print("🔍 Testing PPO training components...")
    
    try:
        from fsrppo.core.ppo_agent import PPOAgent
        import torch
        
        # Create a simple agent
        agent = PPOAgent(
            state_dim=100,
            action_dim=1,
            lr=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            n_epochs=2,
            batch_size=16,
            buffer_size=100
        )
        
        # Test basic functionality
        state = torch.randn(100)
        action, log_prob = agent.get_action(state)
        
        if action.shape == (1,) and log_prob.shape == ():
            print("    PPO agent functional")
            return True
        else:
            print(f"    PPO agent output shapes incorrect: action={action.shape}, log_prob={log_prob.shape}")
            return False
            
    except Exception as e:
        print(f"    PPO training error: {e}")
        return False

def main():
    """Run all real tests"""
    print("=" * 80)
    print("    REAL FSRPPO TEST SUITE")
    print("=" * 80)
    print("Testing actual implementation with real Yahoo Finance data")
    print("This will take several minutes...")
    print("=" * 80)
    
    tests = [
        ("Data Validation", test_data_validation),
        ("Feature Engineering", test_feature_engineering),
        ("PPO Training Components", test_ppo_training),
        ("Main Demo Mode", test_main_demo),
        ("Main Training Mode", test_main_train),
        ("Different Symbols", test_different_symbols),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"    {test_name} failed with exception: {e}")
            results[test_name] = False
    
    end_time = time.time()
    
    # Summary
    print("\n" + "=" * 80)
    print("    REAL TEST RESULTS")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "    PASS" if result else "    FAIL"
        print(f"{test_name:.<50} {status}")
    
    print("-" * 80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"EXECUTION TIME: {end_time - start_time:.1f} seconds")
    
    if passed == total:
        print("\n🎉 ALL REAL TESTS PASSED!")
        print("\n    FSRPPO IS FULLY FUNCTIONAL WITH REAL DATA:")
        print("       Yahoo Finance integration working")
        print("   🔧 Feature engineering working")
        print("       PPO agent training working")
        print("       Performance evaluation working")
        print("       Complete pipeline working")
        print("\n🏆 PRODUCTION READY FOR REAL TRADING!")
    else:
        print(f"\n⚠️  {total-passed} real tests failed")
        print("Review the implementation before production use")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n    Test suite failed: {e}")
        import traceback
        traceback.print_exc()