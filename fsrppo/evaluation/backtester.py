"""
 Backtesting Framework for FSRPPO

Advanced backtesting system with realistic market simulation, transaction costs,
slippage modeling, and  performance analysis.

Features:
- Walk-forward analysis
- Out-of-sample testing
- Multiple asset backtesting
- Risk management integration
- Realistic market microstructure simulation
- Performance attribution analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from pathlib import Path

from ..core import PPOAgent, TradingEnvironment
from ..data import YahooFinanceDataProvider, DataPreprocessor, FeatureEngineer
from ..signal_processing import FinancialSignalRepresentation
from .metrics import PerformanceMetrics


class Backtester:
    """
     backtesting framework for FSRPPO
    
    This class provides sophisticated backtesting capabilities with realistic
    market simulation and  performance analysis.
    
    Parameters:
    -----------
    agent : PPOAgent
        Trained FSRPPO agent
    data_provider : YahooFinanceDataProvider
        Data provider for market data
    preprocessor : DataPreprocessor
        Data preprocessor
    feature_engineer : FeatureEngineer
        Feature engineering pipeline
    transaction_cost : float, default=0.001
        Transaction cost rate (0.1%)
    slippage_model : str, default='linear'
        Slippage model ('none', 'linear', 'sqrt')
    slippage_rate : float, default=0.0005
        Slippage rate (0.05%)
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        data_provider: YahooFinanceDataProvider,
        preprocessor: DataPreprocessor,
        feature_engineer: FeatureEngineer,
        transaction_cost: float = 0.001,
        slippage_model: str = 'linear',
        slippage_rate: float = 0.0005
    ):
        self.agent = agent
        self.data_provider = data_provider
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.transaction_cost = transaction_cost
        self.slippage_model = slippage_model
        self.slippage_rate = slippage_rate
        
        # Performance metrics calculator
        self.metrics_calculator = PerformanceMetrics()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.backtest_results = {}
        
        self.logger.info("Backtester initialized")
        self.logger.info(f"  Transaction cost: {transaction_cost:.4f}")
        self.logger.info(f"  Slippage model: {slippage_model}")
        self.logger.info(f"  Slippage rate: {slippage_rate:.4f}")
    
    def run_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_cash: float = 10000,
        lookback_window: int = 50,
        rebalance_frequency: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Run  backtest on a single symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to backtest
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
        initial_cash : float, default=10000
            Initial cash amount
        lookback_window : int, default=50
            Lookback window for state representation
        rebalance_frequency : str, default='daily'
            Rebalancing frequency
            
        Returns:
        --------
        dict
             backtest results
        """
        self.logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Download and prepare data
        raw_data = self.data_provider.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if raw_data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Feature engineering
        features_data = self.feature_engineer.create_features(raw_data)
        
        # Preprocessing
        processed_data = self.preprocessor.transform(features_data)
        
        # Create trading environment
        from ..core.trading_env import TradingConfig
        
        trading_config = TradingConfig(
            initial_cash=initial_cash,
            transaction_cost=self.transaction_cost,
            lookback_window=lookback_window
        )
        
        # Prepare data in the format expected by TradingEnvironment
        trading_data = {symbol: processed_data}
        
        env = TradingEnvironment(
            data=trading_data,
            config=trading_config
        )
        
        # Run backtest simulation
        backtest_results = self._simulate_trading(env, processed_data, symbol)
        
        # Debug: Log returns and portfolio values
        returns = backtest_results['returns']
        portfolio_values = backtest_results['portfolio_values']
        benchmark_returns = backtest_results['benchmark_returns']
        
        self.logger.info(f"Returns array length: {len(returns)}")
        self.logger.info(f"Portfolio values length: {len(portfolio_values)}")
        self.logger.info(f"Benchmark returns length: {len(benchmark_returns)}")
        
        if len(returns) > 0:
            self.logger.info(f"Returns stats - Min: {np.min(returns):.6f}, Max: {np.max(returns):.6f}, Mean: {np.mean(returns):.6f}")
            self.logger.info(f"Returns NaN count: {np.sum(np.isnan(returns))}")
            self.logger.info(f"Returns inf count: {np.sum(np.isinf(returns))}")
            self.logger.info(f"First 5 returns: {returns[:5]}")
            self.logger.info(f"Last 5 returns: {returns[-5:]}")
        
        if len(portfolio_values) > 0:
            self.logger.info(f"Portfolio values - Initial: {portfolio_values[0]:.2f}, Final: {portfolio_values[-1]:.2f}")
            self.logger.info(f"Portfolio values NaN count: {np.sum(np.isnan(portfolio_values))}")
            self.logger.info(f"Portfolio values inf count: {np.sum(np.isinf(portfolio_values))}")
        
        # Calculate  metrics
        performance_metrics = self.metrics_calculator.calculate_all_metrics(
            returns=returns,
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_returns,
            trades=backtest_results['trades']
        )
        
        # Combine results
        full_results = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': initial_cash,
            'backtest_data': backtest_results,
            'performance_metrics': performance_metrics,
            'metadata': {
                'total_days': len(processed_data),
                'trading_days': len(backtest_results['portfolio_values']),
                'data_quality': self._assess_data_quality(raw_data)
            }
        }
        
        # Store results
        self.backtest_results[symbol] = full_results
        
        self.logger.info(f"Backtest completed for {symbol}")
        self.logger.info(f"  Total return: {performance_metrics['total_return']:.2%}")
        self.logger.info(f"  Sharpe ratio: {performance_metrics['sharpe_ratio']:.3f}")
        self.logger.info(f"  Max drawdown: {performance_metrics['max_drawdown']:.2%}")
        
        return full_results
    
    def run_walk_forward_analysis(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        train_window_months: int = 12,
        test_window_months: int = 3,
        step_months: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        train_window_months : int, default=12
            Training window in months
        test_window_months : int, default=3
            Testing window in months
        step_months : int, default=1
            Step size in months
        **kwargs
            Additional arguments for backtesting
            
        Returns:
        --------
        dict
            Walk-forward analysis results
        """
        self.logger.info(f"Starting walk-forward analysis for {symbol}")
        
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(
            start_dt, end_dt, train_window_months, test_window_months, step_months
        )
        
        # Run backtest for each window
        window_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            self.logger.info(f"Walk-forward window {i+1}/{len(windows)}: "
                           f"Train {train_start.date()} to {train_end.date()}, "
                           f"Test {test_start.date()} to {test_end.date()}")
            
            try:
                # Note: In a full implementation, you would retrain the agent here
                # For now, we use the existing trained agent
                
                # Run backtest on test period
                test_results = self.run_backtest(
                    symbol=symbol,
                    start_date=test_start.strftime('%Y-%m-%d'),
                    end_date=test_end.strftime('%Y-%m-%d'),
                    **kwargs
                )
                
                window_result = {
                    'window_id': i,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'test_results': test_results
                }
                
                window_results.append(window_result)
                
            except Exception as e:
                self.logger.error(f"Walk-forward window {i+1} failed: {e}")
                continue
        
        # Aggregate results
        aggregated_results = self._aggregate_walk_forward_results(window_results)
        
        walk_forward_results = {
            'symbol': symbol,
            'analysis_type': 'walk_forward',
            'parameters': {
                'train_window_months': train_window_months,
                'test_window_months': test_window_months,
                'step_months': step_months
            },
            'window_results': window_results,
            'aggregated_results': aggregated_results,
            'n_windows': len(window_results)
        }
        
        self.logger.info(f"Walk-forward analysis completed: {len(window_results)} windows")
        
        return walk_forward_results
    
    def run_multi_asset_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        allocation_method: str = 'equal_weight',
        rebalance_frequency: str = 'monthly',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run backtest on multiple assets
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
        allocation_method : str, default='equal_weight'
            Portfolio allocation method
        rebalance_frequency : str, default='monthly'
            Rebalancing frequency
        **kwargs
            Additional arguments for backtesting
            
        Returns:
        --------
        dict
            Multi-asset backtest results
        """
        self.logger.info(f"Starting multi-asset backtest for {len(symbols)} symbols")
        
        # Run individual backtests
        individual_results = {}
        
        for symbol in symbols:
            try:
                result = self.run_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                individual_results[symbol] = result
                
            except Exception as e:
                self.logger.error(f"Backtest failed for {symbol}: {e}")
                continue
        
        if not individual_results:
            raise ValueError("No successful backtests completed")
        
        # Create portfolio
        portfolio_results = self._create_portfolio(
            individual_results, allocation_method, rebalance_frequency
        )
        
        # Calculate portfolio metrics
        portfolio_metrics = self.metrics_calculator.calculate_all_metrics(
            returns=portfolio_results['returns'],
            portfolio_values=portfolio_results['portfolio_values'],
            benchmark_returns=portfolio_results['benchmark_returns']
        )
        
        multi_asset_results = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'allocation_method': allocation_method,
            'rebalance_frequency': rebalance_frequency,
            'individual_results': individual_results,
            'portfolio_results': portfolio_results,
            'portfolio_metrics': portfolio_metrics,
            'n_successful_assets': len(individual_results)
        }
        
        self.logger.info(f"Multi-asset backtest completed for {len(individual_results)} assets")
        
        return multi_asset_results
    
    def _simulate_trading(self, env: TradingEnvironment, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Simulate trading using the trained agent"""
        
        # Reset environment
        state = env.reset()
        
        # Storage for results
        portfolio_values = [env.config.initial_cash]
        returns = []
        actions_taken = []
        trades = []
        positions = [0]  # Starting position
        cash_history = [env.config.initial_cash]
        
        # Benchmark (buy and hold)
        initial_price = data['Close'].iloc[env.config.lookback_window]
        benchmark_shares = env.config.initial_cash / initial_price
        benchmark_values = [env.config.initial_cash]
        
        step = 0
        while True:
            # Get action from agent
            action, _ = self.agent.get_action(state, deterministic=True)
            
            # Apply slippage if enabled
            if self.slippage_model != 'none':
                action = self._apply_slippage(action, env, step)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Record results
            portfolio_values.append(info['portfolio_value'])
            cash_history.append(info['cash'])
            positions.append(info['positions'])
            actions_taken.append(action.copy())
            
            # Calculate return
            if len(portfolio_values) > 1:
                ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                returns.append(ret)
            
            # Record trade if executed
            if info['trade_info']['executed_trades']:
                current_price = data['Close'].iloc[env.current_step - 1]
                trade_record = {
                    'step': step,
                    'date': data.index[env.current_step - 1] if hasattr(data.index, 'date') else step,
                    'action_type': 'trade',  # Simplified action type
                    'price': current_price,
                    'quantity': abs(positions[-1] - positions[-2]) if len(positions) > 1 else 0,
                    'transaction_cost': info['total_fees'],
                    'portfolio_value_before': portfolio_values[-2] if len(portfolio_values) > 1 else portfolio_values[-1],
                    'portfolio_value_after': portfolio_values[-1]
                }
                trades.append(trade_record)
            
            # Update benchmark
            if env.current_step < len(data):
                current_price = data['Close'].iloc[env.current_step - 1]
                benchmark_value = benchmark_shares * current_price
                benchmark_values.append(benchmark_value)
            
            # Update state
            state = next_state
            step += 1
            
            if done:
                break
        
        # Calculate benchmark returns
        benchmark_returns = []
        for i in range(1, len(benchmark_values)):
            ret = (benchmark_values[i] - benchmark_values[i-1]) / benchmark_values[i-1]
            benchmark_returns.append(ret)
        
        # Ensure equal length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        return {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'benchmark_returns': benchmark_returns,
            'benchmark_values': benchmark_values,
            'actions_taken': actions_taken,
            'trades': trades,
            'positions': positions,
            'cash_history': cash_history,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': len(trades),
            'total_transaction_costs': sum(trade['transaction_cost'] for trade in trades)
        }
    
    def _apply_slippage(self, action: np.ndarray, env: TradingEnvironment, step: int) -> np.ndarray:
        """Apply slippage to trading action"""
        
        if self.slippage_model == 'none':
            return action
        
        # Decode action to get trade size
        direction, amount = action
        
        # Calculate slippage based on trade size
        if self.slippage_model == 'linear':
            slippage_factor = 1 + self.slippage_rate * amount
        elif self.slippage_model == 'sqrt':
            slippage_factor = 1 + self.slippage_rate * np.sqrt(amount)
        else:
            slippage_factor = 1.0
        
        # Apply slippage (reduces effective trade amount)
        adjusted_amount = amount / slippage_factor
        
        return np.array([direction, adjusted_amount])
    
    def _generate_walk_forward_windows(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        train_window_months: int,
        test_window_months: int,
        step_months: int
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate walk-forward analysis windows"""
        
        windows = []
        current_start = start_date
        
        while True:
            # Calculate window dates
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=train_window_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_window_months)
            
            # Check if we've reached the end
            if test_end > end_date:
                break
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # Move to next window
            current_start = current_start + pd.DateOffset(months=step_months)
        
        return windows
    
    def _aggregate_walk_forward_results(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward analysis results"""
        
        if not window_results:
            return {}
        
        # Extract metrics from each window
        all_returns = []
        all_sharpe_ratios = []
        all_max_drawdowns = []
        all_total_returns = []
        
        for window in window_results:
            metrics = window['test_results']['performance_metrics']
            all_returns.extend(window['test_results']['backtest_data']['returns'])
            all_sharpe_ratios.append(metrics['sharpe_ratio'])
            all_max_drawdowns.append(metrics['max_drawdown'])
            all_total_returns.append(metrics['total_return'])
        
        # Calculate aggregated metrics
        aggregated = {
            'mean_return': np.mean(all_returns),
            'volatility': np.std(all_returns) * np.sqrt(252),
            'sharpe_ratio': np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252),
            'mean_window_sharpe': np.mean(all_sharpe_ratios),
            'mean_window_total_return': np.mean(all_total_returns),
            'mean_window_max_drawdown': np.mean(all_max_drawdowns),
            'consistency_ratio': np.sum(np.array(all_total_returns) > 0) / len(all_total_returns),
            'worst_window_return': np.min(all_total_returns),
            'best_window_return': np.max(all_total_returns)
        }
        
        return aggregated
    
    def _create_portfolio(
        self,
        individual_results: Dict[str, Dict],
        allocation_method: str,
        rebalance_frequency: str
    ) -> Dict[str, Any]:
        """Create portfolio from individual asset results"""
        
        # For simplicity, implement equal weight portfolio
        # In a full implementation, you would support various allocation methods
        
        symbols = list(individual_results.keys())
        n_assets = len(symbols)
        
        if allocation_method == 'equal_weight':
            weights = {symbol: 1.0 / n_assets for symbol in symbols}
        else:
            # Default to equal weight
            weights = {symbol: 1.0 / n_assets for symbol in symbols}
        
        # Get returns for each asset
        asset_returns = {}
        min_length = float('inf')
        
        for symbol in symbols:
            returns = individual_results[symbol]['backtest_data']['returns']
            asset_returns[symbol] = returns
            min_length = min(min_length, len(returns))
        
        # Truncate all return series to same length
        for symbol in symbols:
            asset_returns[symbol] = asset_returns[symbol][:min_length]
        
        # Calculate portfolio returns
        portfolio_returns = []
        for i in range(min_length):
            portfolio_return = sum(
                weights[symbol] * asset_returns[symbol][i] 
                for symbol in symbols
            )
            portfolio_returns.append(portfolio_return)
        
        # Calculate portfolio values
        initial_value = 10000  # Assume same initial value
        portfolio_values = [initial_value]
        
        for ret in portfolio_returns:
            new_value = portfolio_values[-1] * (1 + ret)
            portfolio_values.append(new_value)
        
        # Create benchmark (equal weight buy and hold)
        benchmark_returns = []
        for i in range(min_length):
            benchmark_return = sum(
                weights[symbol] * individual_results[symbol]['backtest_data']['benchmark_returns'][i]
                for symbol in symbols
            )
            benchmark_returns.append(benchmark_return)
        
        return {
            'returns': portfolio_returns,
            'portfolio_values': portfolio_values,
            'benchmark_returns': benchmark_returns,
            'weights': weights,
            'asset_returns': asset_returns
        }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality"""
        
        return {
            'total_records': len(data),
            'missing_values': data.isnull().sum().sum(),
            'date_range_days': (data.index[-1] - data.index[0]).days,
            'avg_volume': data['Volume'].mean() if 'Volume' in data.columns else 0,
            'price_range': {
                'min': data['Close'].min(),
                'max': data['Close'].max(),
                'mean': data['Close'].mean()
            }
        }
    
    def get_backtest_summary(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary of backtest results
        
        Parameters:
        -----------
        symbol : str, optional
            Specific symbol to summarize, or None for all
            
        Returns:
        --------
        pd.DataFrame
            Summary of backtest results
        """
        if not self.backtest_results:
            return pd.DataFrame()
        
        symbols_to_include = [symbol] if symbol else list(self.backtest_results.keys())
        
        summary_data = []
        
        for sym in symbols_to_include:
            if sym in self.backtest_results:
                result = self.backtest_results[sym]
                metrics = result['performance_metrics']
                
                summary_data.append({
                    'Symbol': sym,
                    'Total Return': metrics['total_return'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Volatility': metrics['volatility'],
                    'Win Rate': metrics.get('win_rate', 0),
                    'Total Trades': len(result['backtest_data']['trades']),
                    'Final Value': result['backtest_data']['final_portfolio_value']
                })
        
        return pd.DataFrame(summary_data)

