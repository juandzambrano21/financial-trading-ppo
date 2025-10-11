"""
Feature Engineering for FSRPPO

Advanced feature engineering pipeline specifically designed for financial trading
with FSR integration. Creates  feature sets for the FSRPPO algorithm.

Features:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price-based features (returns, volatility, momentum)
- Volume-based features
- FSR-enhanced features
- Market microstructure features
- Time-based features
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from ..signal_processing import FinancialSignalRepresentation, hurst_exponent


class FeatureEngineer:
    """
    Advanced feature engineering for financial time series
    
    This class creates  feature sets specifically designed
    for the FSRPPO trading algorithm with FSR integration.
    
    Parameters:
    -----------
    fsr_processor : FinancialSignalRepresentation, optional
        FSR processor for enhanced features
    include_technical : bool, default=True
        Whether to include technical indicators
    include_volume : bool, default=True
        Whether to include volume-based features
    include_time : bool, default=True
        Whether to include time-based features
    lookback_periods : List[int], default=[5, 10, 20, 50]
        Lookback periods for rolling features
    """
    
    def __init__(
        self,
        fsr_processor: Optional[FinancialSignalRepresentation] = None,
        include_technical: bool = True,
        include_volume: bool = True,
        include_time: bool = True,
        lookback_periods: List[int] = [5, 10, 20, 50]
    ):
        self.fsr_processor = fsr_processor
        self.include_technical = include_technical
        self.include_volume = include_volume
        self.include_time = include_time
        self.lookback_periods = lookback_periods
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Feature engineer initialized:")
        self.logger.info(f"  Technical indicators: {include_technical}")
        self.logger.info(f"  Volume features: {include_volume}")
        self.logger.info(f"  Time features: {include_time}")
        self.logger.info(f"  FSR enabled: {fsr_processor is not None}")
        self.logger.info(f"  Lookback periods: {lookback_periods}")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create  feature set from OHLCV data
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV price data
            
        Returns:
        --------
        pd.DataFrame
            Enhanced data with engineered features
        """
        self.logger.info(f"Creating features for data with shape {data.shape}")
        
        # Start with original data
        features_df = data.copy()
        
        # Basic price features
        features_df = self._add_price_features(features_df)
        
        # Technical indicators
        if self.include_technical:
            features_df = self._add_technical_indicators(features_df)
        
        # Volume features
        if self.include_volume and 'Volume' in data.columns:
            features_df = self._add_volume_features(features_df)
        
        # Time-based features
        if self.include_time:
            features_df = self._add_time_features(features_df)
        
        # FSR-enhanced features
        if self.fsr_processor is not None:
            features_df = self._add_fsr_features(features_df)
        
        # Market microstructure features
        features_df = self._add_microstructure_features(features_df)
        
        # Rolling statistical features
        features_df = self._add_rolling_features(features_df)
        
        # Remove any infinite or extremely large values
        features_df = self._clean_features(features_df)
        
        n_features = len(features_df.columns) - len(data.columns)
        self.logger.info(f"Created {n_features} new features")
        
        return features_df
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        df = data.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price ratios
        df['hl_ratio'] = df['High'] / df['Low']
        df['oc_ratio'] = df['Open'] / df['Close']
        
        # Price position within daily range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Gap features
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_up'] = (df['gap'] > 0).astype(int)
        df['gap_down'] = (df['gap'] < 0).astype(int)
        
        # Intraday features
        df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
        df['overnight_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        return df
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = data.copy()
        
        # Simple Moving Averages
        for period in self.lookback_periods:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_to_ema_{period}'] = df['Close'] / df[f'ema_{period}']
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_30'] = self._calculate_rsi(df['Close'], 30)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Stochastic Oscillator
        df['stoch_k'] = self._calculate_stochastic_k(df, 14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_ratio'] = df['atr_14'] / df['Close']
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df, 14)
        
        # Commodity Channel Index (CCI)
        df['cci_20'] = self._calculate_cci(df, 20)
        
        return df
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df = data.copy()
        
        # Volume moving averages
        for period in self.lookback_periods:
            df[f'volume_sma_{period}'] = df['Volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_sma_{period}']
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['Volume'] * df['returns']).cumsum()
        
        # On-Balance Volume (OBV)
        df['obv'] = (df['Volume'] * np.sign(df['returns'])).cumsum()
        
        # Volume Rate of Change
        df['volume_roc_10'] = df['Volume'].pct_change(10)
        
        # Price-Volume features
        df['price_volume'] = df['Close'] * df['Volume']
        df['vwap'] = (df['price_volume'].rolling(window=20).sum() / 
                     df['Volume'].rolling(window=20).sum())
        df['price_to_vwap'] = df['Close'] / df['vwap']
        
        # Volume volatility
        df['volume_volatility_20'] = df['Volume'].rolling(window=20).std()
        
        return df
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = data.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.index = pd.to_datetime(df['Date'])
            else:
                self.logger.warning("No datetime index or Date column found for time features")
                return df
        
        # Day of week (Monday=0, Sunday=6)
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Month
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Day of month
        df['day_of_month'] = df.index.day
        df['is_month_end'] = (df.index.day >= 28).astype(int)
        df['is_month_start'] = (df.index.day <= 3).astype(int)
        
        # Year
        df['year'] = df.index.year
        
        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_fsr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add FSR-enhanced features"""
        df = data.copy()
        
        try:
            # Apply FSR to Close price
            if len(df['Close']) >= 50:
                fsr_close = self.fsr_processor.extract_representation(df['Close'].values)
                df['close_fsr'] = fsr_close
                
                # FSR-based features
                df['fsr_returns'] = df['close_fsr'].pct_change()
                df['fsr_volatility_20'] = df['fsr_returns'].rolling(window=20).std()
                df['price_to_fsr'] = df['Close'] / df['close_fsr']
                
                # FSR momentum
                df['fsr_momentum_5'] = df['close_fsr'] / df['close_fsr'].shift(5)
                df['fsr_momentum_20'] = df['close_fsr'] / df['close_fsr'].shift(20)
                
                # Hurst exponent on rolling windows
                df['hurst_50'] = self._calculate_rolling_hurst(df['Close'], 50)
                df['hurst_100'] = self._calculate_rolling_hurst(df['Close'], 100)
                
                self.logger.debug("Added FSR features successfully")
            else:
                self.logger.warning("Insufficient data for FSR features")
                
        except Exception as e:
            self.logger.warning(f"Failed to add FSR features: {e}")
        
        return df
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        df = data.copy()
        
        # Bid-ask spread proxy (High-Low spread)
        df['hl_spread'] = df['High'] - df['Low']
        df['hl_spread_ratio'] = df['hl_spread'] / df['Close']
        
        # Price impact measures
        if 'Volume' in df.columns:
            df['price_impact'] = np.abs(df['returns']) / (df['Volume'] + 1)
            df['amihud_illiquidity'] = np.abs(df['returns']) / (df['Volume'] * df['Close'] + 1)
        
        # Tick direction (simplified)
        df['tick_direction'] = np.sign(df['Close'] - df['Close'].shift(1))
        df['tick_runs'] = self._calculate_runs(df['tick_direction'])
        
        # Realized volatility
        df['realized_vol_5'] = df['returns'].rolling(window=5).std() * np.sqrt(252)
        df['realized_vol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _add_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features"""
        df = data.copy()
        
        # Rolling statistics for returns
        for period in [5, 10, 20]:
            df[f'returns_mean_{period}'] = df['returns'].rolling(window=period).mean()
            df[f'returns_std_{period}'] = df['returns'].rolling(window=period).std()
            df[f'returns_skew_{period}'] = df['returns'].rolling(window=period).skew()
            df[f'returns_kurt_{period}'] = df['returns'].rolling(window=period).kurt()
        
        # Rolling min/max
        for period in [10, 20, 50]:
            df[f'high_max_{period}'] = df['High'].rolling(window=period).max()
            df[f'low_min_{period}'] = df['Low'].rolling(window=period).min()
            df[f'close_rank_{period}'] = df['Close'].rolling(window=period).rank(pct=True)
        
        # Rolling correlations (if multiple price series available)
        # This would be useful for pairs trading or market-relative features
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        return k_percent
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close_prev = np.abs(data['High'] - data['Close'].shift(1))
        low_close_prev = np.abs(data['Low'] - data['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['High'].rolling(window=period).max()
        low_min = data['Low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
        return williams_r
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_rolling_hurst(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Hurst exponent"""
        hurst_values = []
        
        for i in range(len(prices)):
            if i < window - 1:
                hurst_values.append(np.nan)
            else:
                try:
                    price_window = prices.iloc[i-window+1:i+1].values
                    h = hurst_exponent(price_window)
                    hurst_values.append(h)
                except:
                    hurst_values.append(np.nan)
        
        return pd.Series(hurst_values, index=prices.index)
    
    def _calculate_runs(self, series: pd.Series) -> pd.Series:
        """Calculate runs of consecutive same values"""
        runs = []
        current_run = 1
        
        for i in range(1, len(series)):
            if series.iloc[i] == series.iloc[i-1]:
                current_run += 1
            else:
                current_run = 1
            runs.append(current_run)
        
        return pd.Series([1] + runs, index=series.index)
    
    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling infinite and extreme values"""
        df = data.copy()
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values (beyond 5 standard deviations)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].std() > 0:  # Avoid division by zero
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                
                # Count extreme values
                extreme_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if extreme_count > 0:
                    self.logger.debug(f"Capping {extreme_count} extreme values in {col}")
                
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df
    
    def get_feature_importance_proxy(self, data: pd.DataFrame, target_col: str = 'returns') -> pd.Series:
        """
        Calculate feature importance proxy using correlation
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with features
        target_col : str, default='returns'
            Target column for importance calculation
            
        Returns:
        --------
        pd.Series
            Feature importance scores (absolute correlation)
        """
        if target_col not in data.columns:
            self.logger.warning(f"Target column {target_col} not found")
            return pd.Series()
        
        # Calculate correlations with target
        correlations = data.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove the target itself
        correlations = correlations.drop(target_col, errors='ignore')
        
        return correlations


# Example usage and testing
if __name__ == "__main__":
    import logging
    from ..signal_processing import FinancialSignalRepresentation
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Feature Engineer for FSRPPO")
    
    # Generate synthetic financial data
    np.random.seed(42)
    n_days = 500
    
    # Create realistic price series
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame with OHLCV data
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.array(prices[:-1]) * (1 + np.random.normal(0, 0.001, n_days)),
        'High': np.array(prices[:-1]) * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        'Low': np.array(prices[:-1]) * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        'Close': prices[:-1],
        'Volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    data.set_index('Date', inplace=True)
    
    print(f"Created synthetic data with shape: {data.shape}")
    
    # Create FSR processor
    fsr_processor = FinancialSignalRepresentation()
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(
        fsr_processor=fsr_processor,
        include_technical=True,
        include_volume=True,
        include_time=True,
        lookback_periods=[5, 10, 20, 50]
    )
    
    # Create features
    print(f"\nCreating features...")
    features_df = feature_engineer.create_features(data)
    
    print(f"Original columns: {len(data.columns)}")
    print(f"Total columns after feature engineering: {len(features_df.columns)}")
    print(f"New features created: {len(features_df.columns) - len(data.columns)}")
    
    # Show some feature names
    new_features = [col for col in features_df.columns if col not in data.columns]
    print(f"\nSample new features:")
    for i, feature in enumerate(new_features[:20]):
        print(f"  {i+1}. {feature}")
    if len(new_features) > 20:
        print(f"  ... and {len(new_features) - 20} more")
    
    # Calculate feature importance proxy
    print(f"\nCalculating feature importance...")
    importance = feature_engineer.get_feature_importance_proxy(features_df, 'returns')
    
    print(f"Top 10 most important features:")
    for i, (feature, score) in enumerate(importance.head(10).items()):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Check for missing values
    missing_counts = features_df.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    
    if len(features_with_missing) > 0:
        print(f"\nFeatures with missing values:")
        for feature, count in features_with_missing.items():
            print(f"  {feature}: {count} ({count/len(features_df)*100:.1f}%)")
    else:
        print(f"\nNo missing values in engineered features")
    
    print("\nFeature engineering test completed successfully!")