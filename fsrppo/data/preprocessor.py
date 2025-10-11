"""
Data Preprocessor for FSRPPO

Advanced data preprocessing pipeline specifically designed for financial time series
and the FSRPPO trading system. Includes FSR integration, normalization, and
quality assurance for robust trading performance.

Features:
- Financial Signal Representation (FSR) integration
- Multiple normalization techniques
- Missing data handling
- Outlier detection and treatment
- Data quality validation
- Train/validation/test splitting for time series
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings

from ..signal_processing import FinancialSignalRepresentation


class DataPreprocessor:
    """
    Advanced data preprocessor for financial time series
    
    This class provides  preprocessing capabilities specifically
    designed for financial trading applications with FSR integration.
    
    Parameters:
    -----------
    fsr_processor : FinancialSignalRepresentation, optional
        FSR processor for signal cleaning
    normalization_method : str, default='robust'
        Normalization method ('standard', 'minmax', 'robust', 'none')
    handle_missing : str, default='forward_fill'
        Missing data handling ('forward_fill', 'backward_fill', 'interpolate', 'drop')
    outlier_method : str, default='iqr'
        Outlier detection method ('iqr', 'zscore', 'none')
    outlier_threshold : float, default=3.0
        Threshold for outlier detection
    """
    
    def __init__(
        self,
        fsr_processor: Optional[FinancialSignalRepresentation] = None,
        normalization_method: str = 'robust',
        handle_missing: str = 'forward_fill',
        outlier_method: str = 'iqr',
        outlier_threshold: float = 3.0
    ):
        self.fsr_processor = fsr_processor
        self.normalization_method = normalization_method
        self.handle_missing = handle_missing
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Initialize scalers
        self.scalers = {}
        self._fitted = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Data preprocessor initialized:")
        self.logger.info(f"  Normalization: {normalization_method}")
        self.logger.info(f"  Missing data: {handle_missing}")
        self.logger.info(f"  Outlier method: {outlier_method}")
        self.logger.info(f"  FSR enabled: {fsr_processor is not None}")
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data to fit on
            
        Returns:
        --------
        DataPreprocessor
            Self for method chaining
        """
        self.logger.info("Fitting preprocessor on training data")
        
        # Validate input
        if data.empty:
            raise ValueError("Cannot fit on empty data")
        
        # Handle missing values first
        clean_data = self._handle_missing_values(data.copy())
        
        # Detect and handle outliers
        clean_data = self._handle_outliers(clean_data, fit=True)
        
        # Fit normalizers
        if self.normalization_method != 'none':
            self._fit_normalizers(clean_data)
        
        self._fitted = True
        self.logger.info("Preprocessor fitting completed")
        
        return self
    
    def transform(self, data: pd.DataFrame, apply_fsr: bool = True) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to transform
        apply_fsr : bool, default=True
            Whether to apply FSR processing
            
        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        self.logger.debug(f"Transforming data with shape {data.shape}")
        
        # Copy data to avoid modifying original
        transformed_data = data.copy()
        
        # Handle missing values
        transformed_data = self._handle_missing_values(transformed_data)
        
        # Handle outliers (using fitted parameters)
        transformed_data = self._handle_outliers(transformed_data, fit=False)
        
        # Apply normalization
        if self.normalization_method != 'none':
            transformed_data = self._apply_normalization(transformed_data)
        
        # Apply FSR processing if requested and available
        if apply_fsr and self.fsr_processor is not None:
            transformed_data = self._apply_fsr_processing(transformed_data)
        
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame, apply_fsr: bool = True) -> pd.DataFrame:
        """
        Fit preprocessor and transform data in one step
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to fit and transform
        apply_fsr : bool, default=True
            Whether to apply FSR processing
            
        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        return self.fit(data).transform(data, apply_fsr=apply_fsr)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to specified method"""
        if data.isnull().sum().sum() == 0:
            return data
        
        self.logger.debug(f"Handling {data.isnull().sum().sum()} missing values")
        
        if self.handle_missing == 'forward_fill':
            data = data.fillna(method='ffill')
            # Handle any remaining NaNs at the beginning
            data = data.fillna(method='bfill')
            
        elif self.handle_missing == 'backward_fill':
            data = data.fillna(method='bfill')
            # Handle any remaining NaNs at the end
            data = data.fillna(method='ffill')
            
        elif self.handle_missing == 'interpolate':
            # Use linear interpolation for time series
            data = data.interpolate(method='linear')
            # Handle any remaining NaNs at edges
            data = data.fillna(method='ffill').fillna(method='bfill')
            
        elif self.handle_missing == 'drop':
            data = data.dropna()
            
        else:
            raise ValueError(f"Unknown missing data method: {self.handle_missing}")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Detect and handle outliers"""
        if self.outlier_method == 'none':
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.outlier_bounds = {}
        
        for col in numeric_columns:
            if fit:
                # Calculate outlier bounds
                if self.outlier_method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.outlier_threshold * IQR
                    upper_bound = Q3 + self.outlier_threshold * IQR
                    
                elif self.outlier_method == 'zscore':
                    mean = data[col].mean()
                    std = data[col].std()
                    lower_bound = mean - self.outlier_threshold * std
                    upper_bound = mean + self.outlier_threshold * std
                    
                else:
                    raise ValueError(f"Unknown outlier method: {self.outlier_method}")
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
            
            # Apply outlier bounds
            if col in self.outlier_bounds:
                lower_bound, upper_bound = self.outlier_bounds[col]
                
                # Count outliers
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0:
                    self.logger.debug(f"Clipping {outliers} outliers in {col}")
                
                # Clip outliers to bounds
                data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        return data
    
    def _fit_normalizers(self, data: pd.DataFrame) -> None:
        """Fit normalization scalers"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if self.normalization_method == 'standard':
                scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.normalization_method == 'robust':
                scaler = RobustScaler()
            else:
                continue
            
            # Fit scaler on column data
            scaler.fit(data[[col]])
            self.scalers[col] = scaler
    
    def _apply_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted normalization"""
        normalized_data = data.copy()
        
        for col, scaler in self.scalers.items():
            if col in normalized_data.columns:
                normalized_data[col] = scaler.transform(normalized_data[[col]]).flatten()
        
        return normalized_data
    
    def _apply_fsr_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply FSR processing to price columns"""
        fsr_data = data.copy()
        
        # Identify price columns (typically Close, Open, High, Low)
        price_columns = ['Close', 'Open', 'High', 'Low', 'Adj Close']
        available_price_cols = [col for col in price_columns if col in data.columns]
        
        if not available_price_cols:
            self.logger.warning("No price columns found for FSR processing")
            return fsr_data
        
        # Apply FSR to each price column
        for col in available_price_cols:
            try:
                if len(data[col]) >= 50:  # FSR requires minimum length
                    clean_signal = self.fsr_processor.extract_representation(data[col].values)
                    fsr_data[f'{col}_FSR'] = clean_signal
                    self.logger.debug(f"Applied FSR to {col}")
                else:
                    self.logger.warning(f"Insufficient data for FSR on {col}")
                    
            except Exception as e:
                self.logger.warning(f"FSR processing failed for {col}: {e}")
        
        return fsr_data
    
    def create_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int = 50,
        target_column: str = 'Close',
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed data
        sequence_length : int, default=50
            Length of input sequences
        target_column : str, default='Close'
            Column to use as target
        prediction_horizon : int, default=1
            Number of steps ahead to predict
            
        Returns:
        --------
        tuple
            (X_sequences, y_targets)
        """
        if len(data) < sequence_length + prediction_horizon:
            raise ValueError("Data too short for sequence creation")
        
        # Prepare features (all columns except target)
        feature_columns = [col for col in data.columns if col != target_column]
        X_data = data[feature_columns].values
        y_data = data[target_column].values
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X_seq = X_data[i:i + sequence_length]
            X_sequences.append(X_seq)
            
            # Target (future value)
            y_target = y_data[i + sequence_length + prediction_horizon - 1]
            y_targets.append(y_target)
        
        return np.array(X_sequences), np.array(y_targets)
    
    def split_time_series(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data maintaining temporal order
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data
        train_ratio : float, default=0.7
            Fraction for training set
        val_ratio : float, default=0.15
            Fraction for validation set
        test_ratio : float, default=0.15
            Fraction for test set
            
        Returns:
        --------
        tuple
            (train_data, val_data, test_data)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        self.logger.info(f"Data split: Train={len(train_data)}, "
                        f"Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate  data quality report
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to analyze
            
        Returns:
        --------
        dict
            Data quality metrics
        """
        report = {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'data_types': data.dtypes.to_dict(),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Statistics for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            report['numeric_stats'] = {
                'mean': numeric_data.mean().to_dict(),
                'std': numeric_data.std().to_dict(),
                'min': numeric_data.min().to_dict(),
                'max': numeric_data.max().to_dict(),
                'skewness': numeric_data.skew().to_dict(),
                'kurtosis': numeric_data.kurtosis().to_dict()
            }
            
            # Detect potential outliers
            outlier_counts = {}
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((numeric_data[col] < lower_bound) | 
                           (numeric_data[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
            
            report['outlier_counts'] = outlier_counts
        
        return report


# Example usage and testing
if __name__ == "__main__":
    import logging
    from ..signal_processing import FinancialSignalRepresentation
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Data Preprocessor for FSRPPO")
    
    # Generate synthetic financial data
    np.random.seed(42)
    n_days = 1000
    
    # Create realistic price series
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame with OHLCV data
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.array(prices[:-1]) * (1 + np.random.normal(0, 0.001, n_days)),
        'High': np.array(prices[:-1]) * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        'Low': np.array(prices[:-1]) * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        'Close': prices[:-1],
        'Volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # Add some missing values and outliers for testing
    data.loc[100:105, 'Close'] = np.nan
    data.loc[500, 'High'] = data.loc[500, 'Close'] * 2  # Outlier
    
    print(f"Created synthetic data with shape: {data.shape}")
    
    # Create FSR processor
    fsr_processor = FinancialSignalRepresentation()
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        fsr_processor=fsr_processor,
        normalization_method='robust',
        handle_missing='forward_fill',
        outlier_method='iqr'
    )
    
    # Generate data quality report
    quality_report = preprocessor.get_data_quality_report(data)
    print(f"\nData Quality Report:")
    print(f"  Shape: {quality_report['shape']}")
    print(f"  Missing values: {sum(quality_report['missing_values'].values())}")
    print(f"  Outliers detected: {sum(quality_report['outlier_counts'].values())}")
    
    # Split data
    train_data, val_data, test_data = preprocessor.split_time_series(data)
    
    # Fit and transform
    print(f"\nFitting preprocessor on training data...")
    train_processed = preprocessor.fit_transform(train_data)
    
    print(f"Training data shape: {train_processed.shape}")
    print(f"Columns after processing: {list(train_processed.columns)}")
    
    # Transform validation data
    val_processed = preprocessor.transform(val_data)
    print(f"Validation data shape: {val_processed.shape}")
    
    # Create sequences for modeling
    print(f"\nCreating sequences...")
    X_sequences, y_targets = preprocessor.create_sequences(
        train_processed, 
        sequence_length=50,
        target_column='Close'
    )
    
    print(f"Sequence shapes: X={X_sequences.shape}, y={y_targets.shape}")
    
    print("\nData preprocessor test completed successfully!")