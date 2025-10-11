"""
Real Yahoo Finance Data Provider

Robust data provider that fetches real market data from Yahoo Finance
with proper error handling, caching, and data validation.
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pickle
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class YahooFinanceDataProvider:
    """
    Robust Yahoo Finance data provider with caching and error handling
    """
    
    def __init__(self, cache_dir: str = "data_cache", cache_duration_hours: int = 24):
        """
        Initialize Yahoo Finance data provider
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cached data
        cache_duration_hours : int
            How long to cache data before refreshing
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(hours=cache_duration_hours)
        
        logger.info(f"YahooFinanceDataProvider initialized with cache_dir: {cache_dir}")
    
    def _get_cache_path(self, symbol: str, start_date: str, end_date: str, interval: str) -> Path:
        """Get cache file path for given parameters"""
        filename = f"{symbol}_{start_date}_{end_date}_{interval}.pkl"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_duration
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path):
        """Save data to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Data cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Data loaded from cache: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
            return None
    
    def fetch_data(self, 
                   symbol: str, 
                   start_date: str, 
                   end_date: str,
                   interval: str = '1d',
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol from Yahoo Finance
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL', 'GOOGL')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        interval : str
            Data interval ('1d', '1h', '5m', etc.)
        use_cache : bool
            Whether to use cached data
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date, interval)
        
        # Try to load from cache first
        if use_cache and self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
        
        # Fetch fresh data from Yahoo Finance
        logger.info(f"Fetching fresh data for {symbol} from {start_date} to {end_date}")
        
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=True,
                        prepost=False
                    )
                    
                    if not data.empty:
                        break
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean and validate data
            data = self._clean_data(data, symbol)
            
            # Cache the data
            if use_cache:
                self._save_to_cache(data, cache_path)
            
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate the fetched data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw data from Yahoo Finance
        symbol : str
            Stock symbol for logging
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        original_len = len(data)
        
        # Remove rows with missing OHLC data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Remove rows where High < Low (data errors)
        invalid_rows = data['High'] < data['Low']
        if invalid_rows.any():
            logger.warning(f"Removing {invalid_rows.sum()} rows with High < Low for {symbol}")
            data = data[~invalid_rows]
        
        # Remove rows with zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        invalid_prices = (data[price_cols] <= 0).any(axis=1)
        if invalid_prices.any():
            logger.warning(f"Removing {invalid_prices.sum()} rows with invalid prices for {symbol}")
            data = data[~invalid_prices]
        
        # Fill missing volume with 0
        data['Volume'] = data['Volume'].fillna(0)
        
        # Sort by date
        data = data.sort_index()
        
        cleaned_len = len(data)
        if cleaned_len < original_len:
            logger.info(f"Cleaned data for {symbol}: {original_len} -> {cleaned_len} rows")
        
        return data
    
    def fetch_multiple_symbols(self, 
                              symbols: List[str], 
                              start_date: str, 
                              end_date: str,
                              interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        interval : str
            Data interval
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping symbols to their data
        """
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                data = self.fetch_data(symbol, start_date, end_date, interval)
                results[symbol] = data
                logger.info(f"Successfully fetched data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        float
            Latest closing price
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price_fields = ['regularMarketPrice', 'currentPrice', 'previousClose']
            for field in price_fields:
                if field in info and info[field] is not None:
                    return float(info[field])
            
            # Fallback to recent history
            recent_data = ticker.history(period='1d')
            if not recent_data.empty:
                return float(recent_data['Close'].iloc[-1])
            
            raise ValueError(f"Could not get latest price for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get basic information about a symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        Dict
            Symbol information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            symbol_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A')
            }
            
            return symbol_info
            
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to validate
            
        Returns:
        --------
        bool
            True if symbol is valid
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get recent data
            recent_data = ticker.history(period='5d')
            
            return not recent_data.empty
            
        except Exception as e:
            logger.debug(f"Symbol validation failed for {symbol}: {e}")
            return False
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data
        
        Parameters:
        -----------
        symbol : Optional[str]
            If provided, clear cache only for this symbol
        """
        if symbol:
            # Clear cache for specific symbol
            cache_files = list(self.cache_dir.glob(f"{symbol}_*.pkl"))
        else:
            # Clear all cache
            cache_files = list(self.cache_dir.glob("*.pkl"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.debug(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {len(cache_files)} cache files")


# Convenience functions for common use cases
def get_stock_data(symbol: str, 
                   period: str = '1y',
                   interval: str = '1d') -> pd.DataFrame:
    """
    Quick function to get stock data for a symbol
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    period : str
        Period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    interval : str
        Data interval
        
    Returns:
    --------
    pd.DataFrame
        Stock data
    """
    provider = YahooFinanceDataProvider()
    
    # Convert period to start/end dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    if period == '1d':
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    elif period == '5d':
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    elif period == '1mo':
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    elif period == '3mo':
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    elif period == '6mo':
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    elif period == '1y':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    elif period == '2y':
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    elif period == '5y':
        start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    else:
        start_date = '2010-01-01'  # Default to long period
    
    return provider.fetch_data(symbol, start_date, end_date, interval)


def get_portfolio_data(symbols: List[str], 
                      period: str = '1y',
                      interval: str = '1d') -> Dict[str, pd.DataFrame]:
    """
    Quick function to get data for multiple symbols
    
    Parameters:
    -----------
    symbols : List[str]
        List of stock symbols
    period : str
        Period
    interval : str
        Data interval
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping symbols to their data
    """
    provider = YahooFinanceDataProvider()
    
    # Convert period to start/end dates (same logic as above)
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    if period == '1y':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    elif period == '2y':
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    return provider.fetch_multiple_symbols(symbols, start_date, end_date, interval)