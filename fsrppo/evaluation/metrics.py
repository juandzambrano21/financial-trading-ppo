"""
Performance Metrics for FSRPPO

 performance metrics and risk analysis for trading strategies.
Includes traditional financial metrics, risk-adjusted returns, and advanced
statistical measures for robust strategy evaluation.

Features:
- Return-based metrics (Sharpe, Sortino, Calmar ratios)
- Risk metrics (VaR, CVaR, Maximum Drawdown)
- Statistical measures (Skewness, Kurtosis, Tail ratios)
- Benchmark comparison metrics
- Rolling performance analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
import warnings


class PerformanceMetrics:
    """
     performance metrics calculator
    
    This class provides a wide range of performance and risk metrics
    specifically designed for trading strategy evaluation.
    
    Parameters:
    -----------
    risk_free_rate : float, default=0.02
        Annual risk-free rate for Sharpe ratio calculation
    confidence_levels : List[float], default=[0.95, 0.99]
        Confidence levels for VaR and CVaR calculations
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        confidence_levels: List[float] = [0.95, 0.99]
    ):
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Performance metrics calculator initialized")
        self.logger.info(f"  Risk-free rate: {risk_free_rate:.2%}")
        self.logger.info(f"  Confidence levels: {confidence_levels}")
    
    def calculate_all_metrics(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        portfolio_values: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
        benchmark_returns: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
        trades: Optional[List[Dict]] = None,
        frequency: int = 252
    ) -> Dict[str, float]:
        """
        Calculate  performance metrics
        
        Parameters:
        -----------
        returns : array-like
            Strategy returns
        portfolio_values : array-like, optional
            Portfolio values over time
        benchmark_returns : array-like, optional
            Benchmark returns for comparison
        trades : List[Dict], optional
            Trade records for trade-based metrics
        frequency : int, default=252
            Number of periods per year (252 for daily)
            
        Returns:
        --------
        dict
            Dictionary of all calculated metrics
        """
        # Convert inputs to numpy arrays
        returns = np.asarray(returns)
        
        if len(returns) == 0:
            self.logger.warning("Empty returns array provided")
            return {}
        
        # Calculate all metric categories
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns, frequency))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns, frequency))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns, frequency))
        
        # Statistical metrics
        metrics.update(self._calculate_statistical_metrics(returns))
        
        # Drawdown metrics
        if portfolio_values is not None:
            portfolio_values = np.asarray(portfolio_values)
            metrics.update(self._calculate_drawdown_metrics(portfolio_values))
        
        # Benchmark comparison metrics
        if benchmark_returns is not None:
            benchmark_returns = np.asarray(benchmark_returns)
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns, frequency))
        
        # Trade-based metrics
        if trades is not None:
            metrics.update(self._calculate_trade_metrics(trades))
        
        # Value at Risk metrics
        metrics.update(self._calculate_var_metrics(returns, frequency))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: np.ndarray, frequency: int) -> Dict[str, float]:
        """Calculate basic return metrics"""
        
        if len(returns) == 0:
            return {}
        
        # Remove any NaN values
        clean_returns = returns[~np.isnan(returns)]
        
        if len(clean_returns) == 0:
            return {}
        
        metrics = {
            'total_return': np.prod(1 + clean_returns) - 1,
            'annualized_return': np.prod(1 + clean_returns) ** (frequency / len(clean_returns)) - 1,
            'mean_return': np.mean(clean_returns),
            'median_return': np.median(clean_returns),
            'geometric_mean_return': np.prod(1 + clean_returns) ** (1 / len(clean_returns)) - 1,
            'cumulative_return': np.prod(1 + clean_returns) - 1
        }
        
        return metrics
    
    def _calculate_risk_metrics(self, returns: np.ndarray, frequency: int) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        if len(returns) == 0:
            return {}
        
        clean_returns = returns[~np.isnan(returns)]
        
        if len(clean_returns) == 0:
            return {}
        
        metrics = {
            'volatility': np.std(clean_returns) * np.sqrt(frequency),
            'downside_volatility': self._calculate_downside_volatility(clean_returns) * np.sqrt(frequency),
            'semi_deviation': np.std(clean_returns[clean_returns < 0]) * np.sqrt(frequency) if np.any(clean_returns < 0) else 0,
            'tracking_error': np.std(clean_returns) * np.sqrt(frequency)  # Will be updated if benchmark provided
        }
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, returns: np.ndarray, frequency: int) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        
        if len(returns) == 0:
            return {}
        
        clean_returns = returns[~np.isnan(returns)]
        
        if len(clean_returns) == 0:
            return {}
        
        # Daily risk-free rate
        daily_rf_rate = self.risk_free_rate / frequency
        
        # Excess returns
        excess_returns = clean_returns - daily_rf_rate
        
        # Sharpe ratio
        if np.std(clean_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(clean_returns) * np.sqrt(frequency)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_vol = self._calculate_downside_volatility(clean_returns)
        if downside_vol > 0:
            sortino_ratio = np.mean(excess_returns) / downside_vol * np.sqrt(frequency)
        else:
            sortino_ratio = 0
        
        # Calmar ratio (requires drawdown calculation)
        max_dd = self._calculate_max_drawdown_from_returns(clean_returns)
        if abs(max_dd) > 0:
            annualized_return = np.prod(1 + clean_returns) ** (frequency / len(clean_returns)) - 1
            calmar_ratio = annualized_return / abs(max_dd)
        else:
            calmar_ratio = 0
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': sharpe_ratio  # Will be updated if benchmark provided
        }
        
        return metrics
    
    def _calculate_statistical_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate statistical metrics"""
        
        if len(returns) == 0:
            return {}
        
        clean_returns = returns[~np.isnan(returns)]
        
        if len(clean_returns) < 3:  # Need at least 3 observations for skewness/kurtosis
            return {}
        
        metrics = {
            'skewness': stats.skew(clean_returns),
            'kurtosis': stats.kurtosis(clean_returns),
            'jarque_bera_stat': stats.jarque_bera(clean_returns)[0],
            'jarque_bera_pvalue': stats.jarque_bera(clean_returns)[1],
            'positive_periods': np.sum(clean_returns > 0) / len(clean_returns),
            'negative_periods': np.sum(clean_returns < 0) / len(clean_returns),
            'win_rate': np.sum(clean_returns > 0) / len(clean_returns),
            'loss_rate': np.sum(clean_returns < 0) / len(clean_returns)
        }
        
        # Tail ratios
        if np.sum(clean_returns < 0) > 0:
            avg_win = np.mean(clean_returns[clean_returns > 0]) if np.any(clean_returns > 0) else 0
            avg_loss = np.mean(clean_returns[clean_returns < 0])
            metrics['profit_loss_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            metrics['profit_loss_ratio'] = 0
        
        return metrics
    
    def _calculate_drawdown_metrics(self, portfolio_values: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown-based metrics"""
        
        if len(portfolio_values) == 0:
            return {}
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
        
        # Drawdown duration
        in_drawdown = drawdowns < 0
        if np.any(in_drawdown):
            # Find drawdown periods
            drawdown_periods = []
            start = None
            
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start is None:
                    start = i
                elif not is_dd and start is not None:
                    drawdown_periods.append(i - start)
                    start = None
            
            # Handle case where drawdown continues to end
            if start is not None:
                drawdown_periods.append(len(in_drawdown) - start)
            
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        else:
            max_drawdown_duration = 0
            avg_drawdown_duration = 0
        
        metrics = {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'recovery_factor': (portfolio_values[-1] / portfolio_values[0] - 1) / abs(max_drawdown) if max_drawdown != 0 else 0
        }
        
        return metrics
    
    def _calculate_benchmark_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        frequency: int
    ) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return {}
        
        # Align lengths
        min_length = min(len(returns), len(benchmark_returns))
        strategy_returns = returns[:min_length]
        bench_returns = benchmark_returns[:min_length]
        
        # Remove NaN values
        mask = ~(np.isnan(strategy_returns) | np.isnan(bench_returns))
        strategy_returns = strategy_returns[mask]
        bench_returns = bench_returns[mask]
        
        if len(strategy_returns) == 0:
            return {}
        
        # Excess returns
        excess_returns = strategy_returns - bench_returns
        
        # Tracking error
        tracking_error = np.std(excess_returns) * np.sqrt(frequency)
        
        # Information ratio
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(frequency) if np.std(excess_returns) > 0 else 0
        
        # Beta calculation
        if np.var(bench_returns) > 0:
            beta = np.cov(strategy_returns, bench_returns)[0, 1] / np.var(bench_returns)
        else:
            beta = 0
        
        # Alpha calculation (CAPM)
        daily_rf_rate = self.risk_free_rate / frequency
        strategy_excess = np.mean(strategy_returns) - daily_rf_rate
        benchmark_excess = np.mean(bench_returns) - daily_rf_rate
        alpha = (strategy_excess - beta * benchmark_excess) * frequency
        
        # Up/Down capture ratios
        up_periods = bench_returns > 0
        down_periods = bench_returns < 0
        
        if np.any(up_periods) and np.std(bench_returns[up_periods]) > 0:
            up_capture = np.mean(strategy_returns[up_periods]) / np.mean(bench_returns[up_periods])
        else:
            up_capture = 0
        
        if np.any(down_periods) and np.std(bench_returns[down_periods]) > 0:
            down_capture = np.mean(strategy_returns[down_periods]) / np.mean(bench_returns[down_periods])
        else:
            down_capture = 0
        
        metrics = {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'excess_return': np.mean(excess_returns) * frequency
        }
        
        return metrics
    
    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trade-based metrics"""
        
        if not trades:
            return {}
        
        # Extract trade information
        trade_returns = []
        trade_costs = []
        
        for trade in trades:
            if 'transaction_cost' in trade:
                trade_costs.append(trade['transaction_cost'])
        
        # Basic trade statistics
        metrics = {
            'total_trades': len(trades),
            'total_transaction_costs': sum(trade_costs) if trade_costs else 0,
            'avg_transaction_cost': np.mean(trade_costs) if trade_costs else 0,
            'transaction_cost_ratio': sum(trade_costs) / trades[0].get('portfolio_value_before', 1) if trade_costs and trades else 0
        }
        
        return metrics
    
    def _calculate_var_metrics(self, returns: np.ndarray, frequency: int) -> Dict[str, float]:
        """Calculate Value at Risk metrics"""
        
        if len(returns) == 0:
            return {}
        
        clean_returns = returns[~np.isnan(returns)]
        
        if len(clean_returns) == 0:
            return {}
        
        metrics = {}
        
        for confidence_level in self.confidence_levels:
            # Historical VaR
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(clean_returns, var_percentile)
            
            # Conditional VaR (Expected Shortfall)
            cvar = np.mean(clean_returns[clean_returns <= var])
            
            # Annualized versions
            var_annual = var * np.sqrt(frequency)
            cvar_annual = cvar * np.sqrt(frequency)
            
            confidence_str = f"{int(confidence_level * 100)}"
            
            metrics.update({
                f'var_{confidence_str}': var,
                f'cvar_{confidence_str}': cvar,
                f'var_{confidence_str}_annual': var_annual,
                f'cvar_{confidence_str}_annual': cvar_annual
            })
        
        return metrics
    
    def _calculate_downside_volatility(self, returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate downside volatility"""
        
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0
        
        return np.sqrt(np.mean((downside_returns - target_return) ** 2))
    
    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        
        if len(returns) == 0:
            return 0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        return np.min(drawdowns)
    
    def calculate_rolling_metrics(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        window: int = 252,
        metrics: List[str] = ['sharpe_ratio', 'volatility', 'max_drawdown']
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Parameters:
        -----------
        returns : array-like
            Strategy returns
        window : int, default=252
            Rolling window size
        metrics : List[str], default=['sharpe_ratio', 'volatility', 'max_drawdown']
            Metrics to calculate
            
        Returns:
        --------
        pd.DataFrame
            Rolling metrics over time
        """
        returns_series = pd.Series(returns)
        
        rolling_results = {}
        
        for metric in metrics:
            if metric == 'sharpe_ratio':
                daily_rf = self.risk_free_rate / 252
                excess_returns = returns_series - daily_rf
                rolling_results[metric] = (
                    excess_returns.rolling(window).mean() / 
                    returns_series.rolling(window).std() * np.sqrt(252)
                )
            
            elif metric == 'volatility':
                rolling_results[metric] = returns_series.rolling(window).std() * np.sqrt(252)
            
            elif metric == 'max_drawdown':
                rolling_results[metric] = returns_series.rolling(window).apply(
                    lambda x: self._calculate_max_drawdown_from_returns(x.values)
                )
            
            elif metric == 'total_return':
                rolling_results[metric] = returns_series.rolling(window).apply(
                    lambda x: np.prod(1 + x) - 1
                )
        
        return pd.DataFrame(rolling_results)


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Performance Metrics Calculator")
    
    # Generate synthetic returns data
    np.random.seed(42)
    n_days = 1000
    
    # Strategy returns (with some positive drift and volatility)
    strategy_returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual vol, positive drift
    
    # Benchmark returns (market-like)
    benchmark_returns = np.random.normal(0.0005, 0.015, n_days)  # ~15% annual vol
    
    # Portfolio values
    portfolio_values = [10000]
    for ret in strategy_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # Create some trade records
    trades = []
    for i in range(0, n_days, 50):  # Trade every 50 days
        trades.append({
            'step': i,
            'transaction_cost': np.random.uniform(1, 10),
            'portfolio_value_before': portfolio_values[i] if i < len(portfolio_values) else 10000
        })
    
    print(f"Generated {n_days} days of synthetic data")
    print(f"Strategy mean return: {np.mean(strategy_returns):.4f}")
    print(f"Strategy volatility: {np.std(strategy_returns) * np.sqrt(252):.4f}")
    
    # Create metrics calculator
    metrics_calc = PerformanceMetrics(
        risk_free_rate=0.02,
        confidence_levels=[0.95, 0.99]
    )
    
    # Calculate  metrics
    print("\nCalculating  metrics...")
    
    all_metrics = metrics_calc.calculate_all_metrics(
        returns=strategy_returns,
        portfolio_values=portfolio_values,
        benchmark_returns=benchmark_returns,
        trades=trades
    )
    
    # Display key metrics
    print(f"\nKey Performance Metrics:")
    print(f"  Total Return: {all_metrics['total_return']:.2%}")
    print(f"  Annualized Return: {all_metrics['annualized_return']:.2%}")
    print(f"  Volatility: {all_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {all_metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {all_metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {all_metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {all_metrics['win_rate']:.2%}")
    
    if 'alpha' in all_metrics:
        print(f"  Alpha: {all_metrics['alpha']:.2%}")
        print(f"  Beta: {all_metrics['beta']:.3f}")
        print(f"  Information Ratio: {all_metrics['information_ratio']:.3f}")
    
    print(f"  VaR (95%): {all_metrics['var_95']:.2%}")
    print(f"  CVaR (95%): {all_metrics['cvar_95']:.2%}")
    
    print(f"  Total Trades: {all_metrics['total_trades']}")
    print(f"  Total Transaction Costs: ${all_metrics['total_transaction_costs']:.2f}")
    
    # Test rolling metrics
    print("\nCalculating rolling metrics...")
    rolling_metrics = metrics_calc.calculate_rolling_metrics(
        returns=strategy_returns,
        window=252,  # 1-year rolling window
        metrics=['sharpe_ratio', 'volatility', 'max_drawdown']
    )
    
    print(f"Rolling metrics shape: {rolling_metrics.shape}")
    print(f"Final rolling Sharpe ratio: {rolling_metrics['sharpe_ratio'].iloc[-1]:.3f}")
    
    print("\nPerformance Metrics test completed successfully!")