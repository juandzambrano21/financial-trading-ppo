#!/usr/bin/python
# coding: UTF-8
"""
CEESMDAN: Complete Ensemble Extreme-point Symmetric Mode Decomposition with Adaptive Noise

Implementation according to Wang & Wang (2024) FSRPPO paper.
This combines ESMD with ensemble and adaptive noise techniques to overcome mode mixing.

Algorithm 3 from the paper:
- Uses ESMD as base decomposition method
- Adds adaptive white noise to create ensemble
- Averages results to eliminate noise effects
- Specifically designed for financial signal processing

References:
- Wang & Wang (2024): FSRPPO paper
- Wang & Li (2013): ESMD method
"""

import numpy as np
import logging
from typing import List, Optional, Tuple
from .esmd import ESMD


class CEESMDAN:
    """
    Complete Ensemble Extreme-point Symmetric Mode Decomposition with Adaptive Noise
    
    Implementation of Algorithm 3 from Wang & Wang (2024) FSRPPO paper.
    
    Parameters:
    -----------
    J : int, default=100
        Number of ensemble realizations (noise trials)
    xi : float, default=0.005
        Noise coefficient for adaptive noise scaling
    varpi_prime : int, default=6
        Threshold for number of local extreme points (stopping criterion)
    C : int, default=2
        Number of cubic spline interpolation lines for ESMD
    delta : float, default=0.001
        Threshold for ESMD convergence
    D : int, default=100
        Maximum iterations for ESMD sifting
    varpi : int, default=6
        Threshold for number of local extreme points in ESMD
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        J: int = 100,
        xi: float = 0.005,
        varpi_prime: int = 6,
        C: int = 2,
        delta: float = 0.001,
        D: int = 100,
        varpi: int = 6,
        seed: Optional[int] = None
    ):
        self.J = J  # Number of ensemble realizations
        self.xi = xi  # Noise coefficient
        self.varpi_prime = varpi_prime  # Stopping criterion threshold
        
        # ESMD parameters
        self.esmd_params = {
            'C': C,
            'delta': delta,
            'D': D,
            'varpi': varpi
        }
        
        # Random state for reproducibility
        self.random_state = np.random.RandomState(seed)
        
        # Initialize ESMD decomposer
        self.esmd = ESMD(**self.esmd_params)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.imfs = None
        self.residue = None
    
    def decompose(self, X: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Perform CEESMDAN decomposition according to Algorithm 3
        
        Parameters:
        -----------
        X : np.ndarray
            Original time series data
            
        Returns:
        --------
        tuple
            (list_of_imfs, final_residue)
        """
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 1:
            raise ValueError("Input must be 1-dimensional array")
        if len(X) < 10:
            raise ValueError("Input signal too short for decomposition")
        
        N = len(X)
        
        # Generate white noise sequences W_j for j=1,2,...,J
        self.logger.debug(f"Generating {self.J} white noise sequences")
        W = self.random_state.randn(self.J, N)
        
        # Initialize
        i = 1
        L_bar_prev = X.copy()  # L̄_{i-1}
        imfs = []
        
        self.logger.info("Starting CEESMDAN decomposition")
        
        # Main decomposition loop
        while self._count_extrema(L_bar_prev) > self.varpi_prime:
            self.logger.debug(f"Extracting IMF {i}")
            
            # Calculate adaptive noise coefficient ξ_{i-1}
            xi_i_minus_1 = self.xi * np.std(L_bar_prev)
            
            # Initialize accumulator for ensemble averaging
            imf_sum = np.zeros(N)
            
            # Ensemble loop: process each noise realization
            for j in range(self.J):
                # Add adaptive white noise: X + ξ_{i-1} * G_{i-1}(W_j)
                if i == 1:
                    # For first IMF, G_0(W_j) = W_j (no previous IMF)
                    noisy_signal = X + xi_i_minus_1 * W[j]
                else:
                    # For subsequent IMFs, use (i-1)th IMF of noise
                    # This requires decomposing noise W_j and taking its (i-1)th component
                    noise_imfs, _ = self.esmd.decompose(W[j])
                    if len(noise_imfs) >= i - 1:
                        G_i_minus_1_Wj = noise_imfs[i - 2]  # (i-1)th IMF (0-indexed)
                    else:
                        G_i_minus_1_Wj = W[j]  # Fallback to original noise
                    
                    noisy_signal = X + xi_i_minus_1 * G_i_minus_1_Wj
                
                # Decompose noisy signal using ESMD
                try:
                    signal_imfs, _ = self.esmd.decompose(noisy_signal)
                    
                    # Extract first IMF: G_1(X + ξ_{i-1} * G_{i-1}(W_j))
                    if len(signal_imfs) > 0:
                        imf_sum += signal_imfs[0]
                    else:
                        # If no IMF extracted, use the signal itself
                        imf_sum += noisy_signal
                        
                except Exception as e:
                    self.logger.warning(f"ESMD failed for trial {j}: {e}")
                    # Fallback: use original signal
                    imf_sum += noisy_signal
            
            # Calculate average IMF: IMF̄_i = (1/J) * Σ G_1(X + ξ_{i-1} * G_{i-1}(W_j))
            imf_i = imf_sum / self.J
            imfs.append(imf_i)
            
            # Calculate residue: L̄_i = X - IMF̄_i
            L_bar_i = X - imf_i
            
            # Update for next iteration
            X = L_bar_i.copy()
            L_bar_prev = L_bar_i.copy()
            i += 1
            
            # Safety check to prevent infinite loops
            if i > 20:  # Maximum reasonable number of IMFs
                self.logger.warning("Maximum IMF limit reached")
                break
        
        # Final residue
        final_residue = L_bar_prev
        
        self.logger.info(f"CEESMDAN completed: {len(imfs)} IMFs extracted")
        
        # Store results
        self.imfs = imfs
        self.residue = final_residue
        
        return imfs, final_residue
    
    def _count_extrema(self, signal: np.ndarray) -> int:
        """
        Count number of local extrema in signal
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        int
            Number of local extrema (maxima + minima)
        """
        if len(signal) < 3:
            return 0
        
        # Find local maxima and minima
        maxima = 0
        minima = 0
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima += 1
            elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                minima += 1
        
        return maxima + minima
    
    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct original signal from IMFs and residue
        
        Returns:
        --------
        np.ndarray
            Reconstructed signal
        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No decomposition results available. Run decompose() first.")
        
        reconstructed = np.sum(self.imfs, axis=0) + self.residue
        return reconstructed
    
    def get_imfs_and_residue(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get IMFs and residue from last decomposition
        
        Returns:
        --------
        tuple
            (list_of_imfs, residue)
        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No decomposition results available. Run decompose() first.")
        
        return self.imfs, self.residue
    
    def validate_decomposition(self, original: np.ndarray) -> dict:
        """
        Validate decomposition quality
        
        Parameters:
        -----------
        original : np.ndarray
            Original signal
            
        Returns:
        --------
        dict
            Validation metrics
        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No decomposition results available")
        
        # Reconstruction
        reconstructed = self.reconstruct()
        
        # Reconstruction error
        recon_error = np.linalg.norm(original - reconstructed)
        relative_error = recon_error / (np.linalg.norm(original) + 1e-12)
        
        # Energy conservation
        original_energy = np.sum(original ** 2)
        reconstructed_energy = np.sum(reconstructed ** 2)
        energy_ratio = reconstructed_energy / (original_energy + 1e-12)
        
        return {
            'reconstruction_error': recon_error,
            'relative_error': relative_error,
            'energy_conservation_ratio': energy_ratio,
            'n_imfs': len(self.imfs),
            'n_extrema_residue': self._count_extrema(self.residue)
        }


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic financial-like signal
    np.random.seed(42)
    N = 1000
    t = np.linspace(0, 10, N)
    
    # Multi-component signal mimicking financial data
    trend = 0.1 * t  # Linear trend
    seasonal = 2 * np.sin(2 * np.pi * 0.5 * t)  # Seasonal component
    high_freq = 0.5 * np.sin(2 * np.pi * 5 * t)  # High frequency
    noise = 0.2 * np.random.randn(N)  # Noise
    
    signal = trend + seasonal + high_freq + noise
    
    print("Testing CEESMDAN on synthetic financial signal")
    
    # CEESMDAN decomposition
    ceesmdan = CEESMDAN(
        J=50,  # Reduced for faster testing
        xi=0.005,
        varpi_prime=6,
        C=2,
        delta=0.001,
        D=100,
        varpi=6,
        seed=42
    )
    
    imfs, residue = ceesmdan.decompose(signal)
    
    # Validation
    validation = ceesmdan.validate_decomposition(signal)
    
    print(f"\nDecomposition Results:")
    print(f"Number of IMFs: {len(imfs)}")
    print(f"Reconstruction error: {validation['relative_error']:.2e}")
    print(f"Energy conservation: {validation['energy_conservation_ratio']:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(len(imfs) + 2, 1, figsize=(12, 2 * (len(imfs) + 2)))
    
    # Original signal
    axes[0].plot(t, signal, 'b-', linewidth=1)
    axes[0].set_title('Original Signal')
    axes[0].grid(True, alpha=0.3)
    
    # IMFs
    for i, imf in enumerate(imfs):
        axes[i + 1].plot(t, imf, 'g-', linewidth=1)
        axes[i + 1].set_title(f'IMF {i + 1}')
        axes[i + 1].grid(True, alpha=0.3)
    
    # Residue
    axes[-1].plot(t, residue, 'r-', linewidth=1)
    axes[-1].set_title('Residue')
    axes[-1].set_xlabel('Time')
    axes[-1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nCEESMDAN analysis completed successfully!")