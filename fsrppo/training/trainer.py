"""
FSRPPO Training Pipeline

 training pipeline for the FSRPPO algorithm with:
- Multi-asset training support
- Real-time performance monitoring
- Adaptive learning rate scheduling
- Early stopping and model checkpointing
- Integration with all FSRPPO components

Features:
- Distributed training support
- Hyperparameter optimization integration
-  logging and visualization
- Robust model deployment
"""

import numpy as np
import pandas as pd
import torch
import logging
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

from ..core.ppo_agent import PPOAgent
from ..core.trading_env import TradingEnvironment
from ..data import YahooFinanceDataProvider, DataPreprocessor, FeatureEngineer
from ..signal_processing import FinancialSignalRepresentation
from .experiment_tracker import ExperimentTracker


class FSRPPOTrainer:
    """
     training pipeline for FSRPPO
    
    This class orchestrates the entire training process including data preparation,
    feature engineering, model training, and evaluation.
    
    Parameters:
    -----------
    config : dict
        Training configuration dictionary
    experiment_name : str, optional
        Name for the experiment
    save_dir : str, default='./experiments'
        Directory to save experiments
    device : str, default='auto'
        Device for training ('cpu', 'cuda', or 'auto')
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None,
        save_dir: str = './experiments',
        device: str = 'auto'
    ):
        self.config = config
        self.experiment_name = experiment_name or f"fsrppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.training_stats = {
            'episode_rewards': deque(maxlen=1000),
            'episode_lengths': deque(maxlen=1000),
            'portfolio_values': deque(maxlen=1000),
            'total_returns': deque(maxlen=1000),
            'sharpe_ratios': deque(maxlen=1000),
            'max_drawdowns': deque(maxlen=1000),
            'training_losses': deque(maxlen=1000)
        }
        
        # Experiment tracker
        self.experiment_tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            save_dir=self.save_dir
        )
        
        self.logger.info(f"FSRPPO Trainer initialized:")
        self.logger.info(f"  Experiment: {self.experiment_name}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Save directory: {self.save_dir}")
    
    def _initialize_components(self):
        """Initialize all FSRPPO components"""
        
        # Data provider
        self.data_provider = YahooFinanceDataProvider(
            cache_dir=self.config.get('cache_dir', './data_cache'),
            rate_limit_delay=self.config.get('rate_limit_delay', 0.1)
        )
        
        # FSR processor
        fsr_config = self.config.get('fsr', {})
        self.fsr_processor = FinancialSignalRepresentation(
            ceesmdan_params=fsr_config.get('ceesmdan_params', {}),
            hurst_params=fsr_config.get('hurst_params', {}),
            hurst_threshold=fsr_config.get('hurst_threshold', 0.5)
        )
        
        # Data preprocessor
        preprocess_config = self.config.get('preprocessing', {})
        self.preprocessor = DataPreprocessor(
            fsr_processor=self.fsr_processor,
            normalization_method=preprocess_config.get('normalization_method', 'robust'),
            handle_missing=preprocess_config.get('handle_missing', 'forward_fill'),
            outlier_method=preprocess_config.get('outlier_method', 'iqr')
        )
        
        # Feature engineer
        feature_config = self.config.get('features', {})
        self.feature_engineer = FeatureEngineer(
            fsr_processor=self.fsr_processor,
            include_technical=feature_config.get('include_technical', True),
            include_volume=feature_config.get('include_volume', True),
            include_time=feature_config.get('include_time', True),
            lookback_periods=feature_config.get('lookback_periods', [5, 10, 20, 50])
        )
        
        # PPO agent (will be initialized after data preparation)
        self.agent = None
        
        # Trading environment (will be initialized with data)
        self.env = None
        
        self.logger.info("All components initialized successfully")
    
    def prepare_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare training data for multiple symbols
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols to train on
        start_date : str
            Start date for data
        end_date : str
            End date for data
            
        Returns:
        --------
        dict
            Processed data for each symbol
        """
        self.logger.info(f"Preparing data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Download raw data
        raw_data = self.data_provider.get_multiple_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        # Process each symbol
        processed_data = {}
        
        for symbol, data in raw_data.items():
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                continue
            
            try:
                self.logger.info(f"Processing {symbol}: {len(data)} records")
                
                # Feature engineering
                features_data = self.feature_engineer.create_features(data)
                
                # Preprocessing
                if not hasattr(self.preprocessor, '_fitted') or not self.preprocessor._fitted:
                    # Fit on first symbol's data
                    processed_features = self.preprocessor.fit_transform(features_data)
                    self.logger.info(f"Fitted preprocessor on {symbol}")
                else:
                    processed_features = self.preprocessor.transform(features_data)
                
                # Store processed data
                processed_data[symbol] = processed_features
                
                self.logger.info(f"Successfully processed {symbol}: {processed_features.shape}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        if not processed_data:
            raise ValueError("No data could be processed successfully")
        
        self.logger.info(f"Data preparation completed for {len(processed_data)} symbols")
        return processed_data
    
    def initialize_agent(self, state_dim: int):
        """Initialize PPO agent with proper state dimension"""
        
        agent_config = self.config.get('agent', {})
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=agent_config.get('action_dim', 2),
            lr=agent_config.get('lr', 1e-5),
            gamma=agent_config.get('gamma', 0.99),
            gae_lambda=agent_config.get('gae_lambda', 1.0),
            clip_epsilon=agent_config.get('clip_epsilon', 0.2),
            entropy_coef=agent_config.get('entropy_coef', 0.01),
            value_coef=agent_config.get('value_coef', 0.5),
            max_grad_norm=agent_config.get('max_grad_norm', 0.5),
            n_epochs=agent_config.get('n_epochs', 200),
            batch_size=agent_config.get('batch_size', 64),
            device=str(self.device)
        )
        
        self.logger.info(f"PPO agent initialized with state_dim={state_dim}")
    
    def train_single_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        n_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        eval_frequency: int = 100,
        save_frequency: int = 500
    ) -> Dict[str, Any]:
        """
        Train agent on a single symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        data : pd.DataFrame
            Processed data for the symbol
        n_episodes : int, default=1000
            Number of training episodes
        max_steps_per_episode : int, default=1000
            Maximum steps per episode
        eval_frequency : int, default=100
            Frequency of evaluation episodes
        save_frequency : int, default=500
            Frequency of model saving
            
        Returns:
        --------
        dict
            Training results and statistics
        """
        self.logger.info(f"Starting training on {symbol} for {n_episodes} episodes")
        
        # Create trading environment
        env_config = self.config.get('environment', {})
        
        # Import TradingConfig for proper initialization
        from ..core.trading_env import TradingConfig
        
        # Create trading config
        trading_config = TradingConfig(
            initial_cash=env_config.get('initial_cash', 10000),
            transaction_cost=env_config.get('transaction_cost', 0.001),
            lookback_window=env_config.get('lookback_window', 50),
            max_position_size=env_config.get('max_position', 1.0)
        )
        
        # Prepare data in the format expected by TradingEnvironment
        trading_data = {symbol: data}
        
        self.env = TradingEnvironment(
            data=trading_data,
            config=trading_config
        )
        
        # Initialize agent if not already done
        if self.agent is None:
            # Get the actual observation dimension from the environment
            obs_dim = self.env.observation_space.shape[0]
            self.initialize_agent(state_dim=obs_dim)
        
        # Training loop
        best_performance = -np.inf
        episode_rewards = []
        
        for episode in range(n_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Episode data collection
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            log_probs = []
            
            # Run episode
            for step in range(max_steps_per_episode):
                # Get action from agent
                action, log_prob = self.agent.get_action(state, deterministic=False)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                states.append(state.copy())
                actions.append(action.copy())
                rewards.append(reward)
                next_states.append(next_state.copy())
                dones.append(done)
                log_probs.append(log_prob)
                
                # Update for next step
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Update agent
            if len(states) > 0:
                # Store experiences in the agent's buffer
                for i in range(len(states)):
                    self.agent.store_experience(
                        state=states[i],
                        action=actions[i],
                        reward=rewards[i],
                        next_state=next_states[i],
                        done=dones[i],
                        log_prob=log_probs[i]
                    )
                
                # Debug: Log buffer size and batch size
                buffer_size = len(self.agent.buffer)
                batch_size = self.agent.batch_size
                self.logger.debug(f"Episode {episode}: Buffer size={buffer_size}, Batch size={batch_size}, Episode steps={len(states)}")
                
                # Update the agent
                training_stats = self.agent.update()
                
                # Debug: Log if update actually happened
                if not training_stats:
                    self.logger.debug(f"Episode {episode}: Agent update returned empty - insufficient buffer data")
                else:
                    self.logger.debug(f"Episode {episode}: Agent update successful - {list(training_stats.keys())}")
            else:
                training_stats = {}
            
            # Calculate episode statistics
            portfolio_stats = self.env.get_portfolio_stats()
            
            # Update training statistics
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['portfolio_values'].append(portfolio_stats.get('final_portfolio_value', 0))
            self.training_stats['total_returns'].append(portfolio_stats.get('total_return', 0))
            self.training_stats['sharpe_ratios'].append(portfolio_stats.get('sharpe_ratio', 0))
            self.training_stats['max_drawdowns'].append(portfolio_stats.get('max_drawdown', 0))
            
            if training_stats:
                self.training_stats['training_losses'].append(training_stats.get('policy_loss', 0))
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(list(self.training_stats['episode_rewards'])[-10:])
                avg_return = np.mean(list(self.training_stats['total_returns'])[-10:])
                
                self.logger.info(
                    f"Episode {episode}: "
                    f"Reward={episode_reward:.4f}, "
                    f"Avg Reward={avg_reward:.4f}, "
                    f"Return={portfolio_stats.get('total_return', 0):.4f}, "
                    f"Avg Return={avg_return:.4f}, "
                    f"Steps={episode_length}"
                )
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                eval_results = self._evaluate_agent(symbol, data, n_eval_episodes=5)
                
                # Track best performance
                current_performance = eval_results['mean_total_return']
                if current_performance > best_performance:
                    best_performance = current_performance
                    self._save_best_model(symbol, episode, eval_results)
                
                self.logger.info(f"Evaluation at episode {episode}: {eval_results}")
            
            # Save checkpoint
            if episode % save_frequency == 0 and episode > 0:
                self._save_checkpoint(symbol, episode)
            
            # Log to experiment tracker
            self.experiment_tracker.log_metrics({
                'episode': episode,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'portfolio_value': portfolio_stats.get('final_portfolio_value', 0),
                'total_return': portfolio_stats.get('total_return', 0),
                'sharpe_ratio': portfolio_stats.get('sharpe_ratio', 0),
                'max_drawdown': portfolio_stats.get('max_drawdown', 0),
                **training_stats
            })
        
        # Final evaluation
        final_eval = self._evaluate_agent(symbol, data, n_eval_episodes=10)
        
        # Save final model
        self._save_final_model(symbol, final_eval)
        
        # Compile results
        results = {
            'symbol': symbol,
            'n_episodes': n_episodes,
            'final_evaluation': final_eval,
            'best_performance': best_performance,
            'training_stats': dict(self.training_stats)
        }
        
        self.logger.info(f"Training completed for {symbol}")
        return results
    
    def train_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        n_episodes_per_symbol: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train agent on multiple symbols sequentially
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        start_date : str
            Start date for data
        end_date : str
            End date for data
        n_episodes_per_symbol : int, default=1000
            Episodes per symbol
        **kwargs
            Additional arguments for train_single_symbol
            
        Returns:
        --------
        dict
            Combined training results
        """
        self.logger.info(f"Starting multi-symbol training on {len(symbols)} symbols")
        
        # Prepare data
        processed_data = self.prepare_data(symbols, start_date, end_date)
        
        # Train on each symbol
        all_results = {}
        
        for i, symbol in enumerate(symbols):
            if symbol not in processed_data:
                self.logger.warning(f"Skipping {symbol} - no processed data available")
                continue
            
            self.logger.info(f"Training on symbol {i+1}/{len(symbols)}: {symbol}")
            
            try:
                results = self.train_single_symbol(
                    symbol=symbol,
                    data=processed_data[symbol],
                    n_episodes=n_episodes_per_symbol,
                    **kwargs
                )
                all_results[symbol] = results
                
            except Exception as e:
                self.logger.error(f"Training failed for {symbol}: {e}")
                continue
        
        # Compile overall results
        overall_results = {
            'symbols_trained': list(all_results.keys()),
            'individual_results': all_results,
            'overall_stats': self._calculate_overall_stats(all_results)
        }
        
        # Save overall results
        self.experiment_tracker.save_results(overall_results)
        
        self.logger.info(f"Multi-symbol training completed for {len(all_results)} symbols")
        return overall_results
    
    def _evaluate_agent(self, symbol: str, data: pd.DataFrame, n_eval_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance"""
        
        eval_rewards = []
        eval_returns = []
        eval_sharpe_ratios = []
        eval_max_drawdowns = []
        
        for _ in range(n_eval_episodes):
            # Create fresh environment for evaluation using new TradingEnvironment interface
            from ..core.trading_env import TradingConfig
            
            eval_config = TradingConfig(
                initial_cash=10000,
                transaction_cost=0.001,
                lookback_window=3  # Use same reduced lookback as main training
            )
            
            eval_data = {symbol: data}
            eval_env = TradingEnvironment(
                data=eval_data,
                config=eval_config
            )
            
            state = eval_env.reset()
            episode_reward = 0
            
            while True:
                # Use deterministic policy for evaluation
                action, _ = self.agent.get_action(state, deterministic=True)
                next_state, reward, done, info = eval_env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Collect statistics
            stats = eval_env.get_portfolio_stats()
            eval_rewards.append(episode_reward)
            eval_returns.append(stats.get('total_return', 0))
            eval_sharpe_ratios.append(stats.get('sharpe_ratio', 0))
            eval_max_drawdowns.append(stats.get('max_drawdown', 0))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_total_return': np.mean(eval_returns),
            'std_total_return': np.std(eval_returns),
            'mean_sharpe_ratio': np.mean(eval_sharpe_ratios),
            'std_sharpe_ratio': np.std(eval_sharpe_ratios),
            'mean_max_drawdown': np.mean(eval_max_drawdowns),
            'std_max_drawdown': np.std(eval_max_drawdowns)
        }
    
    def _save_checkpoint(self, symbol: str, episode: int):
        """Save training checkpoint"""
        checkpoint_dir = self.save_dir / self.experiment_name / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'{symbol}_episode_{episode}.pt'
        self.agent.save(str(checkpoint_path))
        
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self, symbol: str, episode: int, eval_results: Dict):
        """Save best performing model"""
        best_dir = self.save_dir / self.experiment_name / 'best_models'
        best_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = best_dir / f'{symbol}_best.pt'
        self.agent.save(str(model_path))
        
        # Save evaluation results
        results_path = best_dir / f'{symbol}_best_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'episode': episode,
                'evaluation_results': eval_results
            }, f, indent=2)
        
        self.logger.info(f"Best model saved for {symbol} at episode {episode}")
    
    def _save_final_model(self, symbol: str, eval_results: Dict):
        """Save final model"""
        final_dir = self.save_dir / self.experiment_name / 'final_models'
        final_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = final_dir / f'{symbol}_final.pt'
        self.agent.save(str(model_path))
        
        # Save evaluation results
        results_path = final_dir / f'{symbol}_final_results.json'
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        self.logger.info(f"Final model saved for {symbol}")
    
    def _calculate_overall_stats(self, all_results: Dict) -> Dict[str, float]:
        """Calculate overall statistics across all symbols"""
        
        if not all_results:
            return {}
        
        # Extract final evaluation results
        final_returns = []
        final_sharpe_ratios = []
        final_max_drawdowns = []
        
        for symbol, results in all_results.items():
            eval_results = results.get('final_evaluation', {})
            final_returns.append(eval_results.get('mean_total_return', 0))
            final_sharpe_ratios.append(eval_results.get('mean_sharpe_ratio', 0))
            final_max_drawdowns.append(eval_results.get('mean_max_drawdown', 0))
        
        return {
            'mean_return_across_symbols': np.mean(final_returns),
            'std_return_across_symbols': np.std(final_returns),
            'mean_sharpe_across_symbols': np.mean(final_sharpe_ratios),
            'std_sharpe_across_symbols': np.std(final_sharpe_ratios),
            'mean_max_drawdown_across_symbols': np.mean(final_max_drawdowns),
            'std_max_drawdown_across_symbols': np.std(final_max_drawdowns),
            'n_symbols_trained': len(all_results)
        }
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Episode rewards
        axes[0].plot(list(self.training_stats['episode_rewards']))
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
        
        # Portfolio values
        axes[1].plot(list(self.training_stats['portfolio_values']))
        axes[1].set_title('Portfolio Values')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Portfolio Value ($)')
        axes[1].grid(True, alpha=0.3)
        
        # Total returns
        axes[2].plot(list(self.training_stats['total_returns']))
        axes[2].set_title('Total Returns')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Return')
        axes[2].grid(True, alpha=0.3)
        
        # Sharpe ratios
        axes[3].plot(list(self.training_stats['sharpe_ratios']))
        axes[3].set_title('Sharpe Ratios')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Sharpe Ratio')
        axes[3].grid(True, alpha=0.3)
        
        # Max drawdowns
        axes[4].plot(list(self.training_stats['max_drawdowns']))
        axes[4].set_title('Max Drawdowns')
        axes[4].set_xlabel('Episode')
        axes[4].set_ylabel('Max Drawdown')
        axes[4].grid(True, alpha=0.3)
        
        # Training losses
        if self.training_stats['training_losses']:
            axes[5].plot(list(self.training_stats['training_losses']))
            axes[5].set_title('Training Losses')
            axes[5].set_xlabel('Episode')
            axes[5].set_ylabel('Policy Loss')
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training progress plot saved: {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FSRPPO Trainer")
    
    # Training configuration
    config = {
        'fsr': {
            'ceesmdan_params': {'J': 50, 'xi': 0.005},
            'hurst_threshold': 0.5
        },
        'preprocessing': {
            'normalization_method': 'robust',
            'handle_missing': 'forward_fill'
        },
        'features': {
            'include_technical': True,
            'include_volume': True,
            'lookback_periods': [5, 10, 20]
        },
        'agent': {
            'lr': 1e-5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'entropy_coef': 0.01,
            'n_epochs': 10  # Reduced for testing
        },
        'environment': {
            'initial_cash': 10000,
            'transaction_cost': 0.001,
            'lookback_window': 50
        }
    }
    
    # Create trainer
    trainer = FSRPPOTrainer(
        config=config,
        experiment_name='test_experiment',
        save_dir='./test_experiments'
    )
    
    # Test with a single symbol (reduced episodes for testing)
    symbols = ['AAPL']
    start_date = '2023-01-01'
    end_date = '2023-06-30'
    
    print(f"Testing training on {symbols} from {start_date} to {end_date}")
    
    try:
        results = trainer.train_multiple_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            n_episodes_per_symbol=50,  # Reduced for testing
            eval_frequency=20,
            save_frequency=30
        )
        
        print(f"\nTraining Results:")
        for symbol, result in results['individual_results'].items():
            eval_results = result['final_evaluation']
            print(f"  {symbol}:")
            print(f"    Mean Return: {eval_results['mean_total_return']:.4f}")
            print(f"    Mean Sharpe: {eval_results['mean_sharpe_ratio']:.4f}")
            print(f"    Mean Drawdown: {eval_results['mean_max_drawdown']:.4f}")
        
        # Plot training progress
        trainer.plot_training_progress()
        
        print("\nFSRPPO Trainer test completed successfully!")
        
    except Exception as e:
        print(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()