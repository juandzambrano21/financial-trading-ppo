"""
Financial Signal Representation (FSR) based on Decomposition-Reconstruction

Implementation according to Wang & Wang (2024) FSRPPO paper Section 2.1.4.
This module combines CEESMDAN and Modified Rescaled Range Analysis (MRS) to create
a robust financial signal representation that filters noise and preserves trends.

Key Steps (from paper):
1. Decompose original financial signal into IMFs using CEESMDAN
2. Identify memory characteristics of each IMF using MRS (Hurst exponent)
3. Remove IMFs with short-term memory (H ≤ 0.5) - these are noise-like
4. Merge IMFs with long-term memory (H > 0.5) - these contain trends

Memory Characteristics Rules:
- 0 ≤ H ≤ 0.5: Short-term memory, high frequency, noise-like (remove)
- 0.5 < H ≤ 1: Long-term memory, low frequency, trend-like (keep)

References:
- Wang & Wang (2024): FSRPPO paper Section 2.1.4
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from .ceesmdan import CEESMDAN
from .hurst import hurst_exponent, HurstAnalyzer


class FinancialSignalRepresentation:
    """
    Financial Signal Representation using CEESMDAN + MRS
    
    This class implements the FSR technique described in the FSRPPO paper
    for extracting clean financial signals by filtering out noise components.
    
    Parameters:
    -----------
    ceesmdan_params : dict, optional
        Parameters for CEESMDAN decomposition
    hurst_params : dict, optional
        Parameters for Hurst analysis
    hurst_threshold : float, default=0.5
        Threshold for separating short/long-term memory components
    """
    
    def __init__(
        self,
        ceesmdan_params: Optional[Dict] = None,
        hurst_params: Optional[Dict] = None,
        hurst_threshold: float = 0.5
    ):
        # Default CEESMDAN parameters (from paper)
        default_ceesmdan = {
            'J': 100,  # Number of ensemble realizations
            'xi': 0.005,  # Noise coefficient
            'varpi_prime': 6,  # Extrema threshold
            'C': 2,  # Cubic spline lines
            'delta': 0.001,  # Convergence threshold
            'D': 100,  # Max iterations
            'varpi': 6,  # ESMD extrema threshold
            'seed': 42  # For reproducibility
        }
        
        # Default Hurst parameters
        default_hurst = {
            'min_window': 10,
            'max_window': None
        }
        
        # Update with user parameters
        self.ceesmdan_params = {**default_ceesmdan, **(ceesmdan_params or {})}
        self.hurst_params = {**default_hurst, **(hurst_params or {})}
        self.hurst_threshold = hurst_threshold
        
        # Initialize components
        self.ceesmdan = CEESMDAN(
            J=self.ceesmdan_params.get('J', 100),
            xi=self.ceesmdan_params.get('xi', 0.005),
            varpi_prime=self.ceesmdan_params.get('varpi_prime', 6),
            C=self.ceesmdan_params.get('C', 2),
            delta=self.ceesmdan_params.get('delta', 0.001),
            D=self.ceesmdan_params.get('D', 100),
            varpi=self.ceesmdan_params.get('varpi', 6),
            seed=self.ceesmdan_params.get('seed', 42)
        )
        self.hurst_analyzer = HurstAnalyzer(**self.hurst_params)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.last_decomposition = None
        self.last_hurst_analysis = None
        self.last_representation = None
    
    def extract_representation(self, financial_signal: np.ndarray) -> np.ndarray:
        """
        Extract financial signal representation using FSR technique
        
        Parameters:
        -----------
        financial_signal : np.ndarray
            Original financial time series (e.g., price data)
            
        Returns:
        --------
        np.ndarray
            Clean financial signal representation with noise removed
        """
        # Input validation
        financial_signal = np.asarray(financial_signal, dtype=np.float64)
        if financial_signal.ndim != 1:
            raise ValueError("Financial signal must be 1-dimensional")
        if len(financial_signal) < 50:
            raise ValueError("Financial signal too short for reliable FSR analysis")
        
        self.logger.info("Starting Financial Signal Representation extraction")
        
        # Step 1: Decompose using CEESMDAN
        self.logger.debug("Step 1: CEESMDAN decomposition")
        imfs, residue = self.ceesmdan.decompose(financial_signal)
        
        # Step 2: Analyze memory characteristics using MRS
        self.logger.debug("Step 2: Memory characteristics analysis")
        hurst_results = []
        
        for i, imf in enumerate(imfs):
            try:
                # Calculate Hurst exponent for each IMF
                H = hurst_exponent(imf, **self.hurst_params)
                analysis = self.hurst_analyzer.analyze(imf)
                
                hurst_results.append({
                    'imf_index': i,
                    'hurst_exponent': H,
                    'memory_type': analysis['memory_type'],
                    'is_long_term': H > self.hurst_threshold,
                    'confidence': analysis['confidence']
                })
                
                self.logger.debug(f"IMF {i+1}: H = {H:.4f} ({analysis['memory_type']})")
                
            except Exception as e:
                self.logger.warning(f"Hurst analysis failed for IMF {i+1}: {e}")
                # Default to short-term memory (will be filtered out)
                hurst_results.append({
                    'imf_index': i,
                    'hurst_exponent': 0.4,
                    'memory_type': 'anti-persistent',
                    'is_long_term': False,
                    'confidence': 'low'
                })
        
        # Analyze residue
        try:
            residue_H = hurst_exponent(residue, **self.hurst_params)
            residue_analysis = self.hurst_analyzer.analyze(residue)
            residue_is_long_term = residue_H > self.hurst_threshold
            
            self.logger.debug(f"Residue: H = {residue_H:.4f} ({residue_analysis['memory_type']})")
            
        except Exception as e:
            self.logger.warning(f"Hurst analysis failed for residue: {e}")
            residue_H = 0.6  # Assume trend-like
            residue_is_long_term = True
        
        # Step 3: Filter and merge components
        self.logger.debug("Step 3: Filtering and merging components")
        
        # Separate short-term and long-term memory components
        short_term_imfs = []
        long_term_imfs = []
        
        for i, (imf, result) in enumerate(zip(imfs, hurst_results)):
            if result['is_long_term']:
                long_term_imfs.append(imf)
                self.logger.debug(f"Keeping IMF {i+1} (H = {result['hurst_exponent']:.4f})")
            else:
                short_term_imfs.append(imf)
                self.logger.debug(f"Removing IMF {i+1} (H = {result['hurst_exponent']:.4f})")
        
        # Merge long-term memory components
        if long_term_imfs:
            merged_long_term = np.sum(long_term_imfs, axis=0)
        else:
            self.logger.warning("No long-term memory IMFs found, using zero signal")
            merged_long_term = np.zeros_like(financial_signal)
        
        # Add residue if it has long-term memory
        if residue_is_long_term:
            final_representation = merged_long_term + residue
            self.logger.debug("Including residue in final representation")
        else:
            final_representation = merged_long_term
            self.logger.debug("Excluding residue from final representation")
        
        # Store results for analysis
        self.last_decomposition = {
            'imfs': imfs,
            'residue': residue,
            'original_signal': financial_signal
        }
        
        self.last_hurst_analysis = {
            'imf_results': hurst_results,
            'residue_hurst': residue_H,
            'residue_is_long_term': residue_is_long_term,
            'n_long_term_imfs': len(long_term_imfs),
            'n_short_term_imfs': len(short_term_imfs)
        }
        
        self.last_representation = final_representation
        
        # Calculate noise reduction metrics
        noise_signal = np.sum(short_term_imfs, axis=0)
        if not residue_is_long_term:
            noise_signal += residue
        
        noise_power = np.var(noise_signal)
        signal_power = np.var(final_representation)
        snr_improvement = 10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))
        
        self.logger.info(f"FSR extraction completed:")
        self.logger.info(f"  - Total IMFs: {len(imfs)}")
        self.logger.info(f"  - Long-term IMFs kept: {len(long_term_imfs)}")
        self.logger.info(f"  - Short-term IMFs removed: {len(short_term_imfs)}")
        self.logger.info(f"  - SNR improvement: {snr_improvement:.2f} dB")
        
        return final_representation
    
    def get_analysis_results(self) -> Dict:
        """
        Get detailed analysis results from last FSR extraction
        
        Returns:
        --------
        dict
             analysis results
        """
        if self.last_decomposition is None:
            raise ValueError("No FSR analysis available. Run extract_representation() first.")
        
        return {
            'decomposition': self.last_decomposition,
            'hurst_analysis': self.last_hurst_analysis,
            'representation': self.last_representation,
            'parameters': {
                'ceesmdan': self.ceesmdan_params,
                'hurst': self.hurst_params,
                'hurst_threshold': self.hurst_threshold
            }
        }
    
    def validate_representation(self) -> Dict:
        """
        Validate the quality of signal representation
        
        Returns:
        --------
        dict
            Validation metrics
        """
        if self.last_decomposition is None or self.last_representation is None:
            raise ValueError("No FSR results available for validation")
        
        original = self.last_decomposition['original_signal']
        representation = self.last_representation
        
        # Reconstruction quality
        imfs = self.last_decomposition['imfs']
        residue = self.last_decomposition['residue']
        full_reconstruction = np.sum(imfs, axis=0) + residue
        
        recon_error = np.linalg.norm(original - full_reconstruction)
        recon_relative_error = recon_error / (np.linalg.norm(original) + 1e-12)
        
        # Signal preservation
        correlation = np.corrcoef(original, representation)[0, 1]
        
        # Noise reduction
        noise_components = []
        for i, result in enumerate(self.last_hurst_analysis['imf_results']):
            if not result['is_long_term']:
                noise_components.append(imfs[i])
        
        if noise_components:
            noise_signal = np.sum(noise_components, axis=0)
            noise_power_ratio = np.var(noise_signal) / (np.var(original) + 1e-12)
        else:
            noise_power_ratio = 0.0
        
        return {
            'reconstruction_error': recon_error,
            'reconstruction_relative_error': recon_relative_error,
            'signal_correlation': correlation,
            'noise_power_ratio': noise_power_ratio,
            'noise_reduction_db': -10 * np.log10(noise_power_ratio + 1e-12),
            'n_components_kept': self.last_hurst_analysis['n_long_term_imfs'],
            'n_components_removed': self.last_hurst_analysis['n_short_term_imfs']
        }
    
    def plot_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot  FSR analysis results
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size for plots
        """
        if self.last_decomposition is None:
            raise ValueError("No FSR results available for plotting")
        
        import matplotlib.pyplot as plt
        
        original = self.last_decomposition['original_signal']
        imfs = self.last_decomposition['imfs']
        residue = self.last_decomposition['residue']
        representation = self.last_representation
        hurst_results = self.last_hurst_analysis['imf_results']
        
        # Create subplots
        n_plots = len(imfs) + 4  # Original, IMFs, Residue, Representation, Hurst
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        
        # Original signal
        axes[0].plot(original, 'b-', linewidth=1)
        axes[0].set_title('Original Financial Signal')
        axes[0].grid(True, alpha=0.3)
        
        # IMFs with Hurst information
        for i, (imf, result) in enumerate(zip(imfs, hurst_results)):
            color = 'green' if result['is_long_term'] else 'red'
            label = f"H={result['hurst_exponent']:.3f} ({'Keep' if result['is_long_term'] else 'Remove'})"
            
            axes[i + 1].plot(imf, color=color, linewidth=1)
            axes[i + 1].set_title(f'IMF {i+1} - {label}')
            axes[i + 1].grid(True, alpha=0.3)
        
        # Residue
        residue_color = 'green' if self.last_hurst_analysis['residue_is_long_term'] else 'red'
        residue_label = f"H={self.last_hurst_analysis['residue_hurst']:.3f}"
        
        axes[len(imfs) + 1].plot(residue, color=residue_color, linewidth=1)
        axes[len(imfs) + 1].set_title(f'Residue - {residue_label}')
        axes[len(imfs) + 1].grid(True, alpha=0.3)
        
        # Final representation
        axes[len(imfs) + 2].plot(representation, 'purple', linewidth=1)
        axes[len(imfs) + 2].set_title('Final FSR Representation (Noise Removed)')
        axes[len(imfs) + 2].grid(True, alpha=0.3)
        
        # Comparison
        axes[len(imfs) + 3].plot(original, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[len(imfs) + 3].plot(representation, 'purple', alpha=0.9, label='FSR', linewidth=1)
        axes[len(imfs) + 3].set_title('Original vs FSR Representation')
        axes[len(imfs) + 3].legend()
        axes[len(imfs) + 3].grid(True, alpha=0.3)
        axes[len(imfs) + 3].set_xlabel('Time')
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic financial signal
    np.random.seed(42)
    N = 1000
    t = np.linspace(0, 20, N)
    
    # Create realistic financial signal components
    # Long-term trend
    trend = 0.05 * t + 0.01 * t**1.5
    
    # Medium-term cycles (business cycles)
    business_cycle = 2 * np.sin(2 * np.pi * 0.1 * t) + 1.5 * np.cos(2 * np.pi * 0.05 * t)
    
    # Short-term fluctuations (should be filtered out)
    short_term = 0.8 * np.sin(2 * np.pi * 2 * t) + 0.6 * np.sin(2 * np.pi * 5 * t)
    
    # High-frequency noise (should be filtered out)
    noise = 0.5 * np.random.randn(N)
    
    # Combined financial signal
    financial_signal = trend + business_cycle + short_term + noise
    
    print("Testing Financial Signal Representation on synthetic data")
    
    # Create FSR processor
    fsr = FinancialSignalRepresentation(
        ceesmdan_params={'J': 50, 'xi': 0.005},  # Reduced J for faster testing
        hurst_params={'min_window': 20},
        hurst_threshold=0.5
    )
    
    # Extract representation
    clean_signal = fsr.extract_representation(financial_signal)
    
    # Get analysis results
    results = fsr.get_analysis_results()
    validation = fsr.validate_representation()
    
    print(f"\nFSR Analysis Results:")
    print(f"Signal correlation: {validation['signal_correlation']:.4f}")
    print(f"Noise reduction: {validation['noise_reduction_db']:.2f} dB")
    print(f"Components kept: {validation['n_components_kept']}")
    print(f"Components removed: {validation['n_components_removed']}")
    
    # Plot results
    fsr.plot_analysis(figsize=(15, 12))
    
    print("\nFSR analysis completed successfully!")