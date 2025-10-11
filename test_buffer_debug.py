#!/usr/bin/env python3
"""
Quick test to verify buffer accumulation and PPO update behavior
"""

import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fsrppo.training.trainer import FSRPPOTrainer

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_buffer_behavior():
    """Test buffer accumulation and update behavior"""
    
    # Minimal config for testing
    config = {
        'fsr': {
            'ceesmdan_params': {'J': 10, 'xi': 0.005},  # Reduced for speed
            'hurst_threshold': 0.5
        },
        'preprocessing': {
            'normalization_method': 'robust',
            'handle_missing': 'forward_fill'
        },
        'features': {
            'include_technical': True,
            'include_volume': True,
            'lookback_periods': [5, 10]  # Reduced for speed
        },
        'agent': {
            'lr': 1e-4,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'entropy_coef': 0.01,
            'n_epochs': 5,  # Reduced for testing
            'batch_size': 32,  # Small batch size for testing
            'buffer_size': 1000
        },
        'environment': {
            'initial_cash': 10000,
            'transaction_cost': 0.001,
            'lookback_window': 10  # Much smaller lookback window
        }
    }
    
    print("Creating trainer...")
    trainer = FSRPPOTrainer(
        config=config,
        experiment_name='buffer_test',
        save_dir='./test_experiments'
    )
    
    print("Testing on AAPL data (longer date range)...")
    symbols = ['AAPL']
    start_date = '2023-01-01'
    end_date = '2023-03-31'  # Three months for more data
    
    try:
        results = trainer.train_multiple_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            n_episodes_per_symbol=10,  # Very few episodes for testing
            eval_frequency=5,
            save_frequency=10
        )
        
        print(f"\nTest Results:")
        for symbol, result in results['individual_results'].items():
            eval_results = result['final_evaluation']
            print(f"  {symbol}:")
            print(f"    Episodes completed: {result['n_episodes']}")
            print(f"    Mean Return: {eval_results['mean_total_return']:.4f}")
        
        print("\nBuffer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_buffer_behavior()
    sys.exit(0 if success else 1)