"""
Yahoo Finance Data Provider for FSRPPO

Real-time and historical financial data integration using Yahoo Finance API.
This module provides robust data acquisition with error handling, caching,
and data quality validation for the FSRPPO trading system.

Features:
- Historical price data download
- Real-time price feeds
- Multiple asset support
- Data caching and persistence
- Quality validation and cleaning
- Rate limiting and error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import os
import pickle
from pathlib import Path


class YahooFinanceDataProvider:
    """
    Yahoo Finance Data Provider with caching and error handling
    
    This class provides robust access to Yahoo Finance data with features
    specifically designed for algorithmic trading applications.
    
    Parameters:
    -----------
    cache_dir : str, default='./data_cache'
        Directory for caching downloaded data
    rate_limit_delay : float, default=0.1
        Delay between API calls to respect rate limits
    max_retries : int, default=3
        Maximum number of retry attempts for failed requests
    """
    
    def __init__(
        self,
        cache_dir: str = './data_cache',
        rate_limit_delay: float = 0.1,
        max_retries: int = 3
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Cache for ticker info
        self._ticker_cache = {}
        
        self.logger.info(f"Yahoo Finance provider initialized with cache: {cache_dir}")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download historical price data for a symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        interval : str, default='1d'
            Data interval ('1d', '1h', '5m', etc.)
        use_cache : bool, default=True
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Historical price data with OHLCV columns
        """
        # Convert dates to strings for caching
        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = start_date
            
        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = end_date
        
        # Check cache first
        cache_key = f"{symbol}_{start_str}_{end_str}_{interval}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cached data is recent enough (within 1 day for daily data)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if interval == '1d' and cache_age < timedelta(hours=6):  # Refresh daily data every 6 hours
                    self.logger.debug(f"Using cached data for {symbol}")
                    return cached_data
                elif interval != '1d' and cache_age < timedelta(minutes=30):  # Refresh intraday data every 30 min
                    self.logger.debug(f"Using cached data for {symbol}")
                    return cached_data
                    
            except Exception as e:
                self.logger.warning(f"Failed to load cached data: {e}")
        
        # Download fresh data
        self.logger.info(f"Downloading {symbol} data from {start_str} to {end_str}")
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Download data
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )
                
                if data.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Validate data quality
                data = self._validate_and_clean_data(data, symbol)
                
                # Cache the data
                if use_cache:
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(data, f)
                        self.logger.debug(f"Cached data for {symbol}")
                    except Exception as e:
                        self.logger.warning(f"Failed to cache data: {e}")
                
                return data
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to download data for {symbol} after {self.max_retries} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = '1d',
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        interval : str, default='1d'
            Data interval
        use_cache : bool, default=True
            Whether to use cached data
            
        Returns:
        --------
        dict
            Dictionary mapping symbols to their price data
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_historical_data(
                    symbol, start_date, end_date, interval, use_cache
                )
                results[symbol] = data
                self.logger.info(f"Successfully downloaded {symbol}: {len(data)} records")
                
            except Exception as e:
                self.logger.error(f"Failed to download {symbol}: {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed downloads
        
        return results
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest price for a symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        float or None
            Current price or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price_fields = ['regularMarketPrice', 'currentPrice', 'previousClose']
            
            for field in price_fields:
                if field in info and info[field] is not None:
                    return float(info[field])
            
            # Fallback: get latest from recent history
            recent_data = ticker.history(period='1d', interval='1m')
            if not recent_data.empty:
                return float(recent_data['Close'].iloc[-1])
                
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
        
        return None
    
    def get_ticker_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a ticker
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        dict
            Ticker information
        """
        if symbol in self._ticker_cache:
            return self._ticker_cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            ticker_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'country': info.get('country', 'Unknown')
            }
            
            # Cache the info
            self._ticker_cache[symbol] = ticker_info
            
            return ticker_info
            
        except Exception as e:
            self.logger.error(f"Failed to get ticker info for {symbol}: {e}")
            return {'symbol': symbol, 'name': symbol}
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean downloaded data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw price data
        symbol : str
            Stock symbol for logging
            
        Returns:
        --------
        pd.DataFrame
            Cleaned price data
        """
        original_length = len(data)
        
        # Remove rows with missing OHLC data
        required_columns = ['Open', 'High', 'Low', 'Close']
        data = data.dropna(subset=required_columns)
        
        # Remove rows with zero or negative prices
        for col in required_columns:
            data = data[data[col] > 0]
        
        # Remove rows where High < Low (data errors)
        data = data[data['High'] >= data['Low']]
        
        # Remove rows where Close is outside High-Low range
        data = data[(data['Close'] >= data['Low']) & (data['Close'] <= data['High'])]
        
        # Remove extreme outliers (price changes > 50% in one day)
        if len(data) > 1:
            price_changes = data['Close'].pct_change().abs()
            data = data[price_changes <= 0.5]
        
        # Ensure volume is non-negative
        if 'Volume' in data.columns:
            data = data[data['Volume'] >= 0]
        
        cleaned_length = len(data)
        
        if cleaned_length < original_length:
            self.logger.info(f"Cleaned {symbol} data: {original_length} -> {cleaned_length} records")
        
        if cleaned_length == 0:
            raise ValueError(f"No valid data remaining after cleaning for {symbol}")
        
        return data
    
    def get_sp500_symbols(self) -> List[str]:
        """
        Get list of S&P 500 symbols
        
        Returns:
        --------
        List[str]
            List of S&P 500 stock symbols
        """
        try:
            # Download S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (remove dots, etc.)
            cleaned_symbols = []
            for symbol in symbols:
                # Replace dots with dashes for Yahoo Finance compatibility
                cleaned_symbol = symbol.replace('.', '-')
                cleaned_symbols.append(cleaned_symbol)
            
            self.logger.info(f"Retrieved {len(cleaned_symbols)} S&P 500 symbols")
            return cleaned_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to get S&P 500 symbols: {e}")
            # Fallback to a few major stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data
        
        Parameters:
        -----------
        symbol : str, optional
            Specific symbol to clear, or None to clear all
        """
        if symbol is None:
            # Clear all cache files
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            self.logger.info("Cleared all cached data")
        else:
            # Clear specific symbol
            for cache_file in self.cache_dir.glob(f'{symbol}_*.pkl'):
                cache_file.unlink()
            self.logger.info(f"Cleared cached data for {symbol}")
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for price data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        dict
            Summary statistics
        """
        if data.empty:
            return {}
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        summary = {
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'n_days': len(data),
            'start_price': float(data['Close'].iloc[0]),
            'end_price': float(data['Close'].iloc[-1]),
            'min_price': float(data['Close'].min()),
            'max_price': float(data['Close'].max()),
            'total_return': float((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1),
            'volatility': float(returns.std() * np.sqrt(252)),  # Annualized
            'avg_volume': float(data['Volume'].mean()) if 'Volume' in data.columns else 0,
            'max_drawdown': self._calculate_max_drawdown(data['Close'])
        }
        
        return summary
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Yahoo Finance Data Provider")
    
    # Create data provider
    provider = YahooFinanceDataProvider(cache_dir='./test_cache')
    
    # Test single symbol download
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    print(f"Downloading {symbol} data...")
    data = provider.get_historical_data(symbol, start_date, end_date)
    
    print(f"Downloaded {len(data)} records")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Test data summary
    summary = provider.get_data_summary(data)
    print(f"\nData Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test current price
    current_price = provider.get_current_price(symbol)
    if current_price:
        print(f"\nCurrent {symbol} price: ${current_price:.2f}")
    
    # Test ticker info
    info = provider.get_ticker_info(symbol)
    print(f"\nTicker Info:")
    print(f"  Name: {info['name']}")
    print(f"  Sector: {info['sector']}")
    print(f"  Exchange: {info['exchange']}")
    
    # Test multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    print(f"\nDownloading multiple symbols: {symbols}")
    
    multi_data = provider.get_multiple_symbols(
        symbols, '2023-06-01', '2023-12-31'
    )
    
    for sym, df in multi_data.items():
        if not df.empty:
            print(f"  {sym}: {len(df)} records")
        else:
            print(f"  {sym}: Failed to download")
    
    print("\nYahoo Finance provider test completed successfully!")