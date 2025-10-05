"""
Modified Rescaled Range Analysis (MRS) for Hurst Exponent Calculation

Implementation of Algorithm 4 from Wang & Wang (2024) FSRPPO paper.
This is the modified version by Sánchez Granero et al. (2008) that addresses
shortcomings of the original R/S analysis.

Key features:
- Corrects bias in original R/S analysis
- Uses modified correction factor ψ(R/S)_v
- Specifically designed for financial time series analysis
- Provides robust long-term memory detection

References:
- Wang & Wang (2024): FSRPPO paper Algorithm 4
- Sánchez Granero et al. (2008): Modified rescaled range analysis
- Hurst (1951): Original rescaled range analysis
"""

import numpy as np
import logging
from typing import Union, Optional
from scipy.special import gamma
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def hurst_exponent(Z: np.ndarray, min_window: int = 10, max_window: Optional[int] = None) -> float:
    """
    Calculate Hurst exponent using Modified Rescaled Range Analysis (Algorithm 4)
    
    Parameters:
    -----------
    Z : np.ndarray
        Time series to be analyzed {z_n}_{n=1}^N
    min_window : int, default=10
        Minimum window size for analysis
    max_window : int, optional
        Maximum window size (default: N//2)
        
    Returns:
    --------
    float
        Hurst exponent H ∈ [0, 1]
        - H ≤ 0.5: Short-term memory (anti-persistent)
        - H > 0.5: Long-term memory (persistent)
    """
    # Input validation
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 1:
        raise ValueError("Input must be 1-dimensional array")
    
    N = len(Z)
    if N < 20:
        raise ValueError("Time series too short for reliable Hurst analysis (minimum 20 points)")
    
    if max_window is None:
        max_window = N // 2
    
    if min_window < 2:
        min_window = 2
    if max_window > N // 2:
        max_window = N // 2
    
    logger = logging.getLogger(__name__)
    logger.debug(f"Computing Hurst exponent for series of length {N}")
    
    # Storage for regression data
    ln_v_values = []
    ln_H_v_values = []
    
    # Main algorithm loop: for v = 2 to floor(N/2)
    for v in range(max(min_window, 2), max_window + 1):
        # Calculate number of complete subsequences
        I = N // v
        
        if I < 1:
            continue
        
        # Divide time series into I adjacent subsequences of length v
        # Discard redundant data {z_n}_{n=I×v+1}^N
        subsequences = []
        for i in range(I):
            start_idx = i * v
            end_idx = start_idx + v
            Z_v_i = Z[start_idx:end_idx]
            subsequences.append(Z_v_i)
        
        # Calculate statistics for each subsequence
        R_v_i_list = []
        S_v_i_list = []
        
        for i, Z_v_i in enumerate(subsequences):
            # Calculate mean
            M_v_i = np.mean(Z_v_i)
            
            # Calculate standard deviation
            S_v_i = np.std(Z_v_i, ddof=0)  # Population standard deviation
            
            # Calculate range R_v_i
            cumsum_deviations = np.cumsum(Z_v_i - M_v_i)
            R_v_i = np.max(cumsum_deviations) - np.min(cumsum_deviations)
            
            R_v_i_list.append(R_v_i)
            S_v_i_list.append(S_v_i)
        
        # Calculate (R/S)_v = (1/I) × Σ(R_v_i / S_v_i)
        RS_v_values = []
        for R_v_i, S_v_i in zip(R_v_i_list, S_v_i_list):
            if S_v_i > 0:
                RS_v_values.append(R_v_i / S_v_i)
            else:
                # Handle zero standard deviation case
                RS_v_values.append(0.0)
        
        RS_v = np.mean(RS_v_values)
        
        # Calculate correction factor ψ(R/S)_v
        psi_RS_v = _calculate_correction_factor(v)
        
        # Calculate ln H_v = ln(R/S)_v - ln ψ(R/S)_v + (ln v)/2
        if RS_v > 0 and psi_RS_v > 0:
            ln_H_v = np.log(RS_v) - np.log(psi_RS_v) + 0.5 * np.log(v)
            ln_v = np.log(v)
            
            ln_v_values.append(ln_v)
            ln_H_v_values.append(ln_H_v)
    
    if len(ln_v_values) < 3:
        logger.warning("Insufficient data points for reliable Hurst estimation")
        return 0.5  # Return neutral value
    
    # Construct regression equation: ln H_v = ln C + H × ln v
    # Use least squares to find H
    ln_v_array = np.array(ln_v_values)
    ln_H_v_array = np.array(ln_H_v_values)
    
    # Linear regression: y = a + b*x where y = ln_H_v, x = ln_v, b = H
    A = np.vstack([np.ones(len(ln_v_array)), ln_v_array]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, ln_H_v_array, rcond=None)
    
    ln_C, H = coeffs
    
    # Ensure H is within valid range [0, 1]
    H = np.clip(H, 0.0, 1.0)
    
    logger.debug(f"Hurst exponent calculated: H = {H:.4f}")
    
    return float(H)


def _calculate_correction_factor(v: int) -> float:
    """
    Calculate correction factor ψ(R/S)_v according to Algorithm 4
    
    Parameters:
    -----------
    v : int
        Window size
        
    Returns:
    --------
    float
        Correction factor ψ(R/S)_v
    """
    if v <= 1:
        return 1.0
    
    try:
        if v <= 340:
            # Use exact formula with gamma function
            gamma_term = gamma((v - 1) / 2) / (np.sqrt(np.pi) * gamma(v / 2))
            
            # Calculate sum: Σ_{k=1}^{v-1} √((v-k)/k)
            sum_term = 0.0
            for k in range(1, v):
                sum_term += np.sqrt((v - k) / k)
            
            psi_RS_v = ((v - 0.5) / v) * gamma_term * sum_term
            
        else:
            # Use approximation for large v
            approx_gamma = 1.0 / np.sqrt(v * np.pi / 2)
            
            # Calculate sum: Σ_{k=1}^{v-1} √((v-k)/k)
            sum_term = 0.0
            for k in range(1, v):
                sum_term += np.sqrt((v - k) / k)
            
            psi_RS_v = ((v - 0.5) / v) * approx_gamma * sum_term
        
        # Ensure positive result
        if psi_RS_v <= 0:
            psi_RS_v = 1.0
            
    except (OverflowError, ZeroDivisionError, ValueError):
        # Fallback for numerical issues
        psi_RS_v = 1.0
    
    return psi_RS_v


class HurstAnalyzer:
    """
    Advanced Hurst exponent analyzer with additional features
    
    Parameters:
    -----------
    min_window : int, default=10
        Minimum window size for analysis
    max_window : int, optional
        Maximum window size (default: N//2)
    """
    
    def __init__(self, min_window: int = 10, max_window: Optional[int] = None):
        self.min_window = min_window
        self.max_window = max_window
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, time_series: np.ndarray) -> dict:
        """
         Hurst analysis
        
        Parameters:
        -----------
        time_series : np.ndarray
            Time series to analyze
            
        Returns:
        --------
        dict
            Analysis results including Hurst exponent and interpretation
        """
        # Calculate Hurst exponent
        H = hurst_exponent(time_series, self.min_window, self.max_window)
        
        # Interpret results
        if H < 0.5:
            memory_type = "anti-persistent"
            interpretation = "Mean-reverting behavior, short-term memory"
        elif H == 0.5:
            memory_type = "random"
            interpretation = "Random walk behavior, no memory"
        else:
            memory_type = "persistent"
            interpretation = "Trending behavior, long-term memory"
        
        # Calculate confidence metrics
        confidence = self._calculate_confidence(time_series, H)
        
        return {
            'hurst_exponent': H,
            'memory_type': memory_type,
            'interpretation': interpretation,
            'confidence': confidence,
            'series_length': len(time_series),
            'is_long_term_memory': H > 0.5
        }
    
    def _calculate_confidence(self, time_series: np.ndarray, H: float) -> str:
        """
        Calculate confidence level based on series length and H value
        
        Parameters:
        -----------
        time_series : np.ndarray
            Time series
        H : float
            Hurst exponent
            
        Returns:
        --------
        str
            Confidence level
        """
        N = len(time_series)
        
        # Distance from 0.5 (random walk)
        distance_from_random = abs(H - 0.5)
        
        if N < 50:
            return "low"
        elif N < 200:
            if distance_from_random > 0.1:
                return "medium"
            else:
                return "low"
        else:
            if distance_from_random > 0.15:
                return "high"
            elif distance_from_random > 0.05:
                return "medium"
            else:
                return "low"


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate test signals with known properties
    np.random.seed(42)
    N = 1000
    
    # 1. Random walk (H ≈ 0.5)
    random_walk = np.cumsum(np.random.randn(N))
    
    # 2. Anti-persistent signal (H < 0.5)
    anti_persistent = np.zeros(N)
    for i in range(1, N):
        anti_persistent[i] = anti_persistent[i-1] - 0.3 * anti_persistent[i-1] + np.random.randn()
    
    # 3. Persistent signal (H > 0.5)
    persistent = np.zeros(N)
    for i in range(1, N):
        persistent[i] = persistent[i-1] + 0.3 * persistent[i-1] + np.random.randn()
    
    # 4. Financial-like signal
    t = np.linspace(0, 10, N)
    trend = 0.1 * t
    seasonal = np.sin(2 * np.pi * t)
    noise = 0.5 * np.random.randn(N)
    financial_signal = trend + seasonal + noise
    
    # Test signals
    test_signals = {
        'Random Walk': random_walk,
        'Anti-persistent': anti_persistent,
        'Persistent': persistent,
        'Financial-like': financial_signal
    }
    
    # Analyze each signal
    analyzer = HurstAnalyzer(min_window=10)
    
    print("Hurst Exponent Analysis Results:")
    print("=" * 50)
    
    for name, signal in test_signals.items():
        result = analyzer.analyze(signal)
        
        print(f"\n{name}:")
        print(f"  Hurst Exponent: {result['hurst_exponent']:.4f}")
        print(f"  Memory Type: {result['memory_type']}")
        print(f"  Interpretation: {result['interpretation']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Long-term Memory: {result['is_long_term_memory']}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (name, signal) in enumerate(test_signals.items()):
        result = analyzer.analyze(signal)
        H = result['hurst_exponent']
        
        axes[i].plot(signal, linewidth=1)
        axes[i].set_title(f'{name}\nH = {H:.4f} ({result["memory_type"]})')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nHurst analysis completed successfully!")