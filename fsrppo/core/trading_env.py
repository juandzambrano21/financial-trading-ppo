"""
Real Trading Environment

Robust trading environment that works with real market data,
implements proper portfolio management, transaction costs, and risk controls.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for trading environment"""
    initial_cash: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.3  # Max 30% in single position
    lookback_window: int = 50
    min_periods: int = 20  # Minimum periods before trading
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


class TradingEnvironment(gym.Env):
    """
    Robust trading environment for FSRPPO
    
    This environment handles:
    - Real market data integration
    - Portfolio management with multiple assets
    - Transaction costs and slippage
    - Risk management and position limits
    - Proper reward calculation
    """
    
    def __init__(self, 
                 data: Dict[str, pd.DataFrame],
                 config: Optional[TradingConfig] = None,
                 features: Optional[pd.DataFrame] = None):
        """
        Initialize trading environment
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary mapping symbols to their OHLCV data
        config : Optional[TradingConfig]
            Trading configuration
        features : Optional[pd.DataFrame]
            Engineered features for the environment
        """
        super().__init__()
        
        self.config = config or TradingConfig()
        self.data = data
        self.features = features
        self.symbols = list(data.keys())
        self.n_assets = len(self.symbols)
        
        # Validate data
        self._validate_data()
        
        # Align all data to common dates
        self._align_data()
        
        # Initialize environment state
        self._initialize_environment()
        
        # Define action and observation spaces
        self._define_spaces()
        
        logger.info(f"TradingEnvironment initialized with {self.n_assets} assets, "
                   f"{len(self.dates)} trading days")
    
    def _validate_data(self):
        """Validate input data"""
        if not self.data:
            raise ValueError("No data provided")
        
        for symbol, df in self.data.items():
            if df.empty:
                raise ValueError(f"Empty data for symbol {symbol}")
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for {symbol}: {missing_cols}")
    
    def _align_data(self):
        """Align all data to common trading dates"""
        # Get common dates across all symbols
        all_dates = None
        for symbol, df in self.data.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        if not all_dates:
            raise ValueError("No common dates found across symbols")
        
        # Sort dates
        self.dates = sorted(list(all_dates))
        
        # Align data to common dates
        aligned_data = {}
        for symbol, df in self.data.items():
            aligned_data[symbol] = df.loc[self.dates].copy()
        
        self.data = aligned_data
        
        # Align features if provided
        if self.features is not None:
            common_feature_dates = set(self.features.index).intersection(set(self.dates))
            if common_feature_dates:
                feature_dates = sorted(list(common_feature_dates))
                self.features = self.features.loc[feature_dates].copy()
                # Update dates to match features
                self.dates = feature_dates
            else:
                logger.warning("No common dates between features and price data")
                self.features = None
        
        logger.info(f"Data aligned to {len(self.dates)} common trading days")
    
    def _initialize_environment(self):
        """Initialize environment state variables"""
        self.current_step = 0
        self.max_steps = len(self.dates) - self.config.lookback_window - 1
        
        # Portfolio state
        self.cash = self.config.initial_cash
        self.positions = np.zeros(self.n_assets)  # Number of shares
        self.portfolio_value = self.config.initial_cash
        self.total_value = self.config.initial_cash
        
        # Trading history
        self.trade_history = []
        self.portfolio_history = []
        self.reward_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.config.initial_cash
        
        # Risk management
        self.daily_returns = []
        self.volatility = 0.0
        
        logger.debug("Environment state initialized")
    
    def _define_spaces(self):
        """Define action and observation spaces"""
        # Action space: [position_weight_1, position_weight_2, ..., position_weight_n]
        # Each weight is between -1 (short) and 1 (long)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space: price features + portfolio state + market features
        obs_dim = self._get_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logger.debug(f"Action space: {self.action_space.shape}, "
                    f"Observation space: {self.observation_space.shape}")
    
    def _get_observation_dim(self) -> int:
        """Calculate observation dimension"""
        # Base features: OHLCV for each asset * lookback_window
        base_dim = self.n_assets * 5 * self.config.lookback_window
        
        # Portfolio state: cash, positions, portfolio_value, total_value
        portfolio_dim = 1 + self.n_assets + 2
        
        # Market features: volatility, returns, etc.
        market_dim = self.n_assets * 3  # returns, volatility, momentum
        
        # Additional features if provided
        feature_dim = 0
        if self.features is not None:
            feature_dim = len(self.features.columns)
        
        total_dim = base_dim + portfolio_dim + market_dim + feature_dim
        return total_dim
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self._initialize_environment()
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one trading step
        
        Parameters:
        -----------
        action : np.ndarray
            Trading action (position weights)
            
        Returns:
        --------
        tuple
            (observation, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            raise ValueError("Environment is done, call reset()")
        
        # Clip and normalize actions
        action = np.clip(action, -1.0, 1.0)
        
        # Execute trades
        trade_info = self._execute_trades(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update state
        self.current_step += 1
        
        # Check if done
        done = (self.current_step >= self.max_steps or 
                self.total_value <= self.config.initial_cash * 0.1)  # Stop loss at 90% loss
        
        # Prepare info
        info = {
            'portfolio_value': self.portfolio_value,
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'trade_info': trade_info,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_fees': self.total_fees,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'current_date': self.dates[self.current_step + self.config.lookback_window]
        }
        
        # Store history
        self.portfolio_history.append(self.total_value)
        self.reward_history.append(reward)
        
        return self._get_observation(), reward, done, info
    
    def _execute_trades(self, target_weights: np.ndarray) -> Dict:
        """
        Execute trades to achieve target portfolio weights
        
        Parameters:
        -----------
        target_weights : np.ndarray
            Target position weights for each asset
            
        Returns:
        --------
        Dict
            Trade execution information
        """
        current_date_idx = self.current_step + self.config.lookback_window
        current_date = self.dates[current_date_idx]
        
        # Get current prices (use Close prices)
        current_prices = np.array([
            self.data[symbol].loc[current_date, 'Close'] 
            for symbol in self.symbols
        ])
        
        # Calculate current portfolio value
        position_values = self.positions * current_prices
        self.portfolio_value = np.sum(position_values)
        self.total_value = self.cash + self.portfolio_value
        
        # Apply position limits
        target_weights = self._apply_position_limits(target_weights)
        
        # Calculate target position values
        target_values = target_weights * self.total_value
        
        # Calculate required trades
        current_values = position_values
        trade_values = target_values - current_values
        
        # Execute trades
        total_fees = 0.0
        executed_trades = []
        
        for i, (symbol, trade_value) in enumerate(zip(self.symbols, trade_values)):
            if abs(trade_value) < 1.0:  # Skip tiny trades
                continue
            
            price = current_prices[i]
            
            # Apply slippage
            if trade_value > 0:  # Buying
                execution_price = price * (1 + self.config.slippage)
            else:  # Selling
                execution_price = price * (1 - self.config.slippage)
            
            # Calculate shares to trade
            shares_to_trade = trade_value / execution_price
            
            # Check if we have enough cash for buying
            if trade_value > 0 and trade_value > self.cash:
                # Adjust trade size to available cash
                shares_to_trade = self.cash / execution_price
                trade_value = shares_to_trade * execution_price
            
            # Calculate transaction cost
            fee = abs(trade_value) * self.config.transaction_cost
            
            # Execute trade if profitable after fees
            if abs(trade_value) > fee:
                self.positions[i] += shares_to_trade
                self.cash -= trade_value + fee
                total_fees += fee
                self.total_fees += fee
                self.total_trades += 1
                
                # Track winning trades (simplified)
                if (trade_value > 0 and target_weights[i] > 0) or \
                   (trade_value < 0 and target_weights[i] < 0):
                    self.winning_trades += 1
                
                executed_trades.append({
                    'symbol': symbol,
                    'shares': shares_to_trade,
                    'price': execution_price,
                    'value': trade_value,
                    'fee': fee
                })
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
        return {
            'executed_trades': executed_trades,
            'total_fees': total_fees,
            'portfolio_value': self.portfolio_value,
            'total_value': self.total_value
        }
    
    def _apply_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Apply position size limits"""
        # Limit individual position sizes
        weights = np.clip(weights, -self.config.max_position_size, self.config.max_position_size)
        
        # Ensure total absolute weight doesn't exceed 1.0
        total_abs_weight = np.sum(np.abs(weights))
        if total_abs_weight > 1.0:
            weights = weights / total_abs_weight
        
        return weights
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward for current step
        
        Uses a combination of:
        - Portfolio return
        - Risk-adjusted return (Sharpe-like)
        - Transaction cost penalty
        """
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Portfolio return
        current_value = self.total_value
        previous_value = self.portfolio_history[-1] if self.portfolio_history else self.config.initial_cash
        
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Risk adjustment
        if len(self.daily_returns) > 10:
            #TODO: CHANGE 10 TO AN ARBITRARY DEFINED PARAMETER
            volatility = np.std(self.daily_returns[-10:])  # Rolling volatility
            if volatility > 0:
                risk_adjusted_return = portfolio_return / volatility
            else:
                risk_adjusted_return = portfolio_return
        else:
            risk_adjusted_return = portfolio_return
        
        # Transaction cost penalty
        recent_fees = 0.0
        if len(self.portfolio_history) > 0:
            recent_fees = self.total_fees - (self.portfolio_history[-1] if hasattr(self, '_last_fees') else 0)
        
        fee_penalty = recent_fees / self.total_value
        
        # Combined reward
        reward = risk_adjusted_return - fee_penalty
        
        # Scale reward
        reward *= 100  # Scale for better learning
        
        return float(reward)
    
    def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        # Calculate daily return
        if len(self.portfolio_history) > 0:
            daily_return = (self.total_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
            self.daily_returns.append(daily_return)
        
        # Update peak value and drawdown
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        
        current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update volatility
        if len(self.daily_returns) > 1:
            self.volatility = np.std(self.daily_returns)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
        --------
        np.ndarray
            Current state observation
        """
        if self.current_step + self.config.lookback_window >= len(self.dates):
            # Return last valid observation
            return self._last_observation if hasattr(self, '_last_observation') else np.zeros(self.observation_space.shape[0])
        
        current_idx = self.current_step + self.config.lookback_window
        start_idx = self.current_step
        
        # Price features (OHLCV for lookback window)
        price_features = []
        for symbol in self.symbols:
            symbol_data = self.data[symbol].iloc[start_idx:current_idx]
            
            # Normalize prices by current close price
            close_price = symbol_data['Close'].iloc[-1]
            normalized_data = symbol_data[['Open', 'High', 'Low', 'Close']].values / close_price
            
            # Normalize volume by mean volume
            volume = symbol_data['Volume'].values
            mean_volume = np.mean(volume) if np.mean(volume) > 0 else 1
            normalized_volume = volume / mean_volume
            
            # Combine OHLCV
            symbol_features = np.column_stack([normalized_data, normalized_volume.reshape(-1, 1)])
            price_features.append(symbol_features.flatten())
        
        price_features = np.concatenate(price_features)
        
        # Portfolio state
        portfolio_weights = self.positions * np.array([
            self.data[symbol].iloc[current_idx-1]['Close'] for symbol in self.symbols
        ]) / max(self.total_value, 1)

        

        portfolio_state = np.array([
            self.cash / self.config.initial_cash,  # Normalized cash
            *portfolio_weights,  # Position weights
            self.portfolio_value / self.config.initial_cash,  # Normalized portfolio value
            self.total_value / self.config.initial_cash  # Normalized total value
        ])

        logger.log(f"portfolio_state: {portfolio_state}")
        
        # Market features
        market_features = []
        for symbol in self.symbols:
            symbol_data = self.data[symbol].iloc[start_idx:current_idx]
            
            # Returns
            returns = symbol_data['Close'].pct_change().fillna(0).values[-10:]  # Last 10 returns
            avg_return = np.mean(returns)
            
            # Volatility
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Momentum
            momentum = (symbol_data['Close'].iloc[-1] / symbol_data['Close'].iloc[0] - 1) if len(symbol_data) > 1 else 0
            
            market_features.extend([avg_return, volatility, momentum])
        
        market_features = np.array(market_features)
        
        # Additional features
        additional_features = np.array([])
        if self.features is not None and current_idx-1 < len(self.features):
            additional_features = self.features.iloc[current_idx-1].values
            # Handle NaN values
            additional_features = np.nan_to_num(additional_features, nan=0.0)
        
        # Combine all features
        observation = np.concatenate([
            price_features,
            portfolio_state,
            market_features,
            additional_features
        ])
        
        # Ensure observation has correct shape
        expected_dim = self.observation_space.shape[0]
        if len(observation) != expected_dim:
            # Pad or truncate to match expected dimension
            if len(observation) < expected_dim:
                observation = np.pad(observation, (0, expected_dim - len(observation)))
            else:
                observation = observation[:expected_dim]
        
        # Handle any remaining NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self._last_observation = observation
        return observation.astype(np.float32)
    
    def get_portfolio_stats(self) -> Dict:
        """Get  portfolio statistics"""
        if not self.portfolio_history:
            return {}
        
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (self.total_value - self.config.initial_cash) / self.config.initial_cash
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0  # Annualized
        sharpe_ratio = (np.mean(returns) * 252 - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak_values = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak_values - portfolio_values) / peak_values
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(portfolio_values)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees,
            'final_value': self.total_value,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            stats = self.get_portfolio_stats()
            print(f"Step: {self.current_step}")
            print(f"Total Value: ${self.total_value:,.2f}")
            print(f"Total Return: {stats.get('total_return', 0):.2%}")
            print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {stats.get('win_rate', 0):.2%}")
            print("-" * 40)