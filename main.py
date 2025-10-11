#!/usr/bin/env python3
"""
FSRPPO Main Application

Complete implementation using ALL components as specified in the paper:
- CEESMDAN signal decomposition
- ESMD (Extreme-Point Symmetric Mode Decomposition)  
- Hurst exponent analysis
- FSR (Financial Signal Representation)
- PPO training with FSRPPOTrainer
- Experiment tracking
- Real Yahoo Finance data

Usage:
    python main.py --mode train --symbol AAPL --episodes 1000
    python main.py --mode backtest --symbol MSFT --model_path models/fsrppo_model.pth
    python main.py --mode demo --symbol GOOGL
"""

import argparse
import logging
import sys
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fsrppo.training.trainer import FSRPPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_experiment_config(args):
    """Create experiment configuration"""
    return {
        'experiment_name': f'FSRPPO_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'symbol': args.symbol,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'episodes': args.episodes,
        'mode': args.mode,
        'ceesmdan_trials': args.ceesmdan_trials,
        'esmd_M': args.esmd_M,
        'include_hurst': True,
        'ppo_lr': args.ppo_lr,
        'ppo_gamma': args.ppo_gamma,
        'ppo_clip_epsilon': args.ppo_clip_epsilon,
        'batch_size': args.batch_size,
        'buffer_size': args.buffer_size,
        'initial_cash': args.initial_cash,
        'transaction_cost': args.transaction_cost,
        'max_position_size': args.max_position_size
    }


def main():
    parser = argparse.ArgumentParser(description='FSRPPO - Complete Implementation')
    
    parser.add_argument('--mode', choices=['train', 'backtest', 'demo'], default='demo',
                       help='Run mode')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol to trade')
    parser.add_argument('--start_date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    
    parser.add_argument('--ceesmdan_trials', type=int, default=100, 
                       help='Number of CEESMDAN trials')
    parser.add_argument('--esmd_M', type=int, default=4, 
                       help='ESMD parameter M (number of inner curves)')
    
    parser.add_argument('--ppo_lr', type=float, default=1e-5, help='PPO learning rate')
    parser.add_argument('--ppo_gamma', type=float, default=0.99, help='PPO discount factor')
    parser.add_argument('--ppo_clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=2048, help='Buffer size')
    
    parser.add_argument('--initial_cash', type=float, default=100000.0, help='Initial cash')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost')
    parser.add_argument('--max_position_size', type=float, default=0.3, help='Max position size')
    
    parser.add_argument('--model_path', help='Path to saved model for backtesting')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("    FSRPPO: COMPLETE IMPLEMENTATION WITH ALL COMPONENTS")
    print("=" * 100)
    print(f"Mode: {args.mode}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Episodes: {args.episodes}")
    print(f"CEESMDAN Trials: {args.ceesmdan_trials}")
    print(f"ESMD M Parameter: {args.esmd_M}")
    print(f"Include Hurst: True")
    print("=" * 100)
    
    try:
        # 1. Create comprehensive configuration for FSRPPOTrainer
        logger.info("🔧 Creating configuration...")
        config = create_experiment_config(args)
        
        # Build trainer configuration with all parameters
        trainer_config = {
            'fsr': {
                'ceesmdan_params': {
                    'J': min(args.ceesmdan_trials, 100),  # Use full trials as per paper
                    'xi': 0.005,
                    'varpi': args.esmd_M,
                    'D': 8
                },
                'hurst_params': {'min_window': 10},
                'hurst_threshold': 0.5
            },
            'preprocessing': {
                'normalization_method': 'robust',
                'handle_missing': 'forward_fill'
            },
            'features': {
                'include_technical': True,
                'include_volume': True,
                'include_time': True,
                'lookback_periods': [5, 10, 20]
            },
            'agent': {
                'lr': args.ppo_lr,
                'gamma': args.ppo_gamma,
                'gae_lambda': 1.0,
                'entropy_coef': 0.01,
                'n_epochs': 200,  # Updated to match paper specification
                'clip_epsilon': args.ppo_clip_epsilon,
                'batch_size': args.batch_size,
                'buffer_size': args.buffer_size,
                'device': 'cpu'  # Add device parameter
            },
            'environment': {
                'initial_cash': args.initial_cash,
                'transaction_cost': args.transaction_cost,
                'slippage': 0.0005,
                'max_position_size': args.max_position_size,
                'lookback_window': 10,  # Reduced for short test periods
                'min_periods': 100
            }
        }
        
        logger.info(f"Configuration created for {args.mode} mode")
        logger.info(f"Symbol: {args.symbol}, Period: {args.start_date} to {args.end_date}")
        logger.info(f"CEESMDAN trials: {args.ceesmdan_trials}, ESMD M: {args.esmd_M}")
        
        # 2. Initialize FSRPPOTrainer (handles ALL data processing internally)
        if args.mode in ['train', 'demo']:
            logger.info("🚀 Initializing FSRPPOTrainer...")
            trainer = FSRPPOTrainer(
                config=trainer_config,
                experiment_name=f"fsrppo_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            logger.info(f"🎯 Starting {args.mode} for {args.episodes} episodes...")
            
            # Train the agent using the trainer's method
            training_results = trainer.train_multiple_symbols(
                symbols=[args.symbol],
                start_date=args.start_date,
                end_date=args.end_date,
                n_episodes_per_symbol=args.episodes,
                eval_frequency=50,
                save_frequency=100
            )
            
            # Log results
            logger.info("✅ Training completed!")
            for symbol, result in training_results['individual_results'].items():
                eval_results = result['final_evaluation']
                logger.info(f"📊 {symbol} Results:")
                logger.info(f"   Mean Return: {eval_results['mean_total_return']:.4f}")
                logger.info(f"   Mean Sharpe: {eval_results['mean_sharpe_ratio']:.4f}")
                logger.info(f"   Mean Drawdown: {eval_results['mean_max_drawdown']:.4f}")
            
            # Save model if requested
            if args.save_model:
                logger.info("💾 Model saved automatically by trainer")
                
        elif args.mode == 'backtest':
            logger.info("📈 Starting backtesting mode...")
            
            # Check if model path is provided
            if not args.model_path:
                logger.error("❌ Model path is required for backtesting mode")
                logger.info("   Use --model_path to specify the trained model")
                return False
            
            # Verify model file exists
            if not os.path.exists(args.model_path):
                logger.error(f"❌ Model file not found: {args.model_path}")
                return False
            
            logger.info(f"📂 Loading model from: {args.model_path}")
            
            # Initialize components for backtesting
            from fsrppo.data import YahooFinanceDataProvider, DataPreprocessor, FeatureEngineer
            from fsrppo.signal_processing import FinancialSignalRepresentation
            from fsrppo.core.ppo_agent import PPOAgent
            from fsrppo.evaluation import Backtester
            
            # Initialize data provider and processors
            data_provider = YahooFinanceDataProvider(cache_dir='./data_cache')
            
            # Initialize FSR processor
            fsr_processor = FinancialSignalRepresentation(
                ceesmdan_params=trainer_config['fsr']['ceesmdan_params'],
                hurst_params=trainer_config['fsr']['hurst_params'],
                hurst_threshold=trainer_config['fsr']['hurst_threshold']
            )
            
            # Initialize preprocessor
            preprocessor = DataPreprocessor(
                fsr_processor=fsr_processor,
                normalization_method=trainer_config['preprocessing']['normalization_method'],
                handle_missing=trainer_config['preprocessing']['handle_missing']
            )
            
            # Initialize feature engineer
            feature_engineer = FeatureEngineer(
                fsr_processor=fsr_processor,
                include_technical=trainer_config['features']['include_technical'],
                include_volume=trainer_config['features']['include_volume']
            )
            
            # Load the trained agent
            try:
                # Get sample data to determine state dimensions
                sample_data = data_provider.get_historical_data(
                    symbol=args.symbol,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                
                if sample_data.empty:
                    logger.error(f"❌ No data available for {args.symbol}")
                    return False
                
                # Process sample data to get state dimensions
                features_data = feature_engineer.create_features(sample_data)
                
                # Fit the preprocessor with the sample data
                preprocessor.fit(features_data)
                processed_data = preprocessor.transform(features_data)
                
                # Create trading environment to get the correct state dimension
                from fsrppo.core.trading_env import TradingEnvironment, TradingConfig
                
                env_config = trainer_config.get('environment', {})
                trading_config = TradingConfig(
                    initial_cash=env_config.get('initial_cash', 10000),
                    transaction_cost=env_config.get('transaction_cost', 0.001),
                    lookback_window=env_config.get('lookback_window', 50),
                    max_position_size=env_config.get('max_position_size', 1.0)
                )
                
                # Create environment with processed data
                trading_data = {args.symbol: processed_data}
                env = TradingEnvironment(data=trading_data, config=trading_config)
                
                # Get the correct state dimension from the environment
                state_dim = env.observation_space.shape[0]
                
                # Initialize agent with correct dimensions
                agent = PPOAgent(
                    state_dim=state_dim,
                    action_dim=2,  # [direction, amount]
                    lr=trainer_config['agent']['lr'],
                    gamma=trainer_config['agent']['gamma'],
                    gae_lambda=trainer_config['agent']['gae_lambda'],
                    clip_epsilon=trainer_config['agent']['clip_epsilon'],
                    entropy_coef=trainer_config['agent']['entropy_coef'],
                    n_epochs=trainer_config['agent']['n_epochs'],
                    batch_size=trainer_config['agent']['batch_size'],
                    device=trainer_config['agent']['device']
                )
                
                # Load the trained model
                agent.load(args.model_path)
                logger.info("✅ Model loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                return False
            
            # Initialize backtester
            backtester = Backtester(
                agent=agent,
                data_provider=data_provider,
                preprocessor=preprocessor,
                feature_engineer=feature_engineer,
                transaction_cost=trainer_config['environment']['transaction_cost'],
                slippage_model='linear',
                slippage_rate=trainer_config['environment'].get('slippage_rate', 0.0005)
            )
            
            logger.info(f"🔍 Running backtest for {args.symbol} from {args.start_date} to {args.end_date}")
            
            try:
                # Run comprehensive backtest
                backtest_results = backtester.run_backtest(
                    symbol=args.symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    initial_cash=trainer_config['environment']['initial_cash'],
                    lookback_window=trainer_config['environment']['lookback_window']
                )
                
                # Display results
                logger.info("✅ Backtest completed successfully!")
                logger.info("=" * 60)
                logger.info("📊 BACKTEST RESULTS")
                logger.info("=" * 60)
                
                metrics = backtest_results['performance_metrics']
                metadata = backtest_results['metadata']
                
                logger.info(f"Symbol: {args.symbol}")
                logger.info(f"Period: {args.start_date} to {args.end_date}")
                logger.info(f"Initial Cash: ${args.initial_cash:,.2f}")
                logger.info(f"Trading Days: {metadata['trading_days']}")
                logger.info("")
                logger.info("📈 Performance Metrics:")
                logger.info(f"  Total Return: {metrics['total_return']:.2%}")
                logger.info(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
                logger.info(f"  Volatility: {metrics['volatility']:.2%}")
                logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
                logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
                
                backtest_data = backtest_results['backtest_data']
                logger.info("")
                logger.info("💼 Trading Statistics:")
                logger.info(f"  Total Trades: {backtest_data['total_trades']}")
                logger.info(f"  Final Portfolio Value: ${backtest_data['final_portfolio_value']:,.2f}")
                logger.info(f"  Total Transaction Costs: ${backtest_data['total_transaction_costs']:,.2f}")
                
                # Benchmark comparison
                if backtest_data['benchmark_returns']:
                    benchmark_return = (backtest_data['benchmark_values'][-1] / backtest_data['benchmark_values'][0] - 1)
                    logger.info(f"  Benchmark Return (Buy & Hold): {benchmark_return:.2%}")
                    excess_return = metrics['total_return'] - benchmark_return
                    logger.info(f"  Excess Return: {excess_return:.2%}")
                
                logger.info("=" * 60)
                
                # Optional: Run walk-forward analysis if requested
                if hasattr(args, 'walk_forward') and args.walk_forward:
                    logger.info("🔄 Running walk-forward analysis...")
                    
                    wf_results = backtester.run_walk_forward_analysis(
                        symbol=args.symbol,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        train_window_months=12,
                        test_window_months=3,
                        step_months=1,
                        initial_cash=args.initial_cash
                    )
                    
                    logger.info("📊 Walk-Forward Analysis Results:")
                    agg_results = wf_results['aggregated_results']
                    logger.info(f"  Mean Window Return: {agg_results['mean_window_total_return']:.2%}")
                    logger.info(f"  Mean Window Sharpe: {agg_results['mean_window_sharpe']:.3f}")
                    logger.info(f"  Consistency Ratio: {agg_results['consistency_ratio']:.1%}")
                    logger.info(f"  Best Window: {agg_results['best_window_return']:.2%}")
                    logger.info(f"  Worst Window: {agg_results['worst_window_return']:.2%}")
                
            except Exception as e:
                logger.error(f"❌ Backtest failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
        logger.info("🎉 FSRPPO execution completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"    FSRPPO execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()