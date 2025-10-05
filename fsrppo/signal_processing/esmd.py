"""
ESMD: Extreme-Point Symmetric Mode Decomposition

Implementation of Algorithm 2 from Wang & Wang (2024) FSRPPO paper.
This is the base decomposition method used in CEESMDAN.

Key features according to the paper:
- Uses cubic spline interpolation of midpoints between extrema
- Iterative sifting process with convergence criteria
- Designed to overcome endpoint effects and mode mixing
- Specifically optimized for financial signal processing

Reference:
- Wang & Wang (2024): FSRPPO paper Algorithm 2
- Wang & Li (2013): Original ESMD method
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ESMD:
    """
    Extreme-Point Symmetric Mode Decomposition
    
    Implementation of Algorithm 2 from Wang & Wang (2024) FSRPPO paper.
    
    Parameters:
    -----------
    C : int, default=2
        Number of cubic spline interpolation lines
    delta : float, default=0.001
        Threshold for convergence (max |CSI_mean|)
    D : int, default=100
        Maximum iterations for sifting process
    varpi : int, default=6
        Threshold for number of local extreme points (stopping criterion)
    """
    
    def __init__(
        self,
        C: int = 2,
        delta: float = 0.001,
        D: int = 100,
        varpi: int = 6
    ):
        # Validation
        if C < 1:
            raise ValueError("C must be >= 1")
        if delta <= 0:
            raise ValueError("delta must be > 0")
        if D < 1:
            raise ValueError("D must be >= 1")
        if varpi < 1:
            raise ValueError("varpi must be >= 1")
        
        self.C = C  # Number of cubic spline interpolation lines
        self.delta = delta  # Convergence threshold
        self.D = D  # Maximum iterations
        self.varpi = varpi  # Extrema threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def decompose(self, X: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Perform ESMD decomposition according to Algorithm 2
        
        Parameters:
        -----------
        X : np.ndarray
            Original data {x_n}_{n=1}^N
            
        Returns:
        --------
        tuple
            (K_intrinsic_mode_functions, final_residue_L_K)
        """
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 1:
            raise ValueError("Input must be 1-dimensional array")
        if len(X) < 10:
            raise ValueError("Input signal too short for decomposition")
        
        # Algorithm 2 implementation
        i = 1
        L_prev = X.copy()  # L_0 = X
        imfs = []
        
        self.logger.info("Starting ESMD decomposition")
        
        # Main decomposition loop
        while self._count_extrema(L_prev) > self.varpi:
            self.logger.debug(f"Extracting IMF {i}")
            
            # Initialize sifting process
            d = 1
            Y = L_prev.copy()  # Y = X (current residue)
            
            # Sifting loop
            while d <= self.D:
                # Find all local extremum points
                extrema_indices = self._find_extrema_indices(Y)
                
                if len(extrema_indices) < 2:
                    self.logger.debug(f"Insufficient extrema at iteration {d}")
                    break
                
                # Calculate midpoints between adjacent extrema
                midpoints_indices, midpoints_values = self._calculate_midpoints(Y, extrema_indices)
                
                if len(midpoints_indices) < 2:
                    self.logger.debug(f"Insufficient midpoints at iteration {d}")
                    break
                
                # Supplement boundary midpoints by interpolation
                extended_indices, extended_values = self._supplement_boundaries(
                    Y, midpoints_indices, midpoints_values
                )
                
                # Construct C cubic spline interpolation lines
                csi_lines = []
                for j in range(self.C):
                    # Select points for j-th spline: {f_0, f_l, f_m} where l ≡ j (mod C)
                    selected_indices = [0]  # Always include f_0
                    for l in range(1, len(extended_indices) - 1):
                        if l % self.C == j:
                            selected_indices.append(l)
                    selected_indices.append(len(extended_indices) - 1)  # Always include f_m
                    
                    if len(selected_indices) >= 2:
                        # Create cubic spline
                        try:
                            cs = CubicSpline(
                                extended_indices[selected_indices],
                                extended_values[selected_indices],
                                bc_type='natural'
                            )
                            # Evaluate spline at all data points
                            csi_j = cs(np.arange(len(Y)))
                            csi_lines.append(csi_j)
                        except Exception as e:
                            self.logger.warning(f"Spline construction failed: {e}")
                            # Fallback to linear interpolation
                            csi_j = np.interp(
                                np.arange(len(Y)),
                                extended_indices[selected_indices],
                                extended_values[selected_indices]
                            )
                            csi_lines.append(csi_j)
                    else:
                        # Not enough points, use zero curve
                        csi_lines.append(np.zeros(len(Y)))
                
                # Calculate mean curve of all cubic spline interpolation lines
                if csi_lines:
                    csi_mean = np.mean(csi_lines, axis=0)
                else:
                    csi_mean = np.zeros(len(Y))
                
                # Check convergence criterion
                max_csi_mean = np.max(np.abs(csi_mean))
                if max_csi_mean <= self.delta:
                    self.logger.debug(f"Converged at iteration {d}, max|CSI_mean|={max_csi_mean:.6f}")
                    break
                
                # Update Y
                Y = Y - csi_mean
                d += 1
            
            # Extract IMF
            if csi_lines:
                imf_i = np.mean(csi_lines, axis=0)
            else:
                imf_i = np.zeros(len(Y))
            
            imfs.append(imf_i)
            
            # Calculate residue: L_i = X - IMF_i
            L_i = L_prev - imf_i
            
            # Update for next iteration
            L_prev = L_i.copy()
            i += 1
            
            # Safety check
            if i > 20:  # Maximum reasonable number of IMFs
                self.logger.warning("Maximum IMF limit reached")
                break
        
        # Final residue
        final_residue = L_prev
        
        self.logger.info(f"ESMD completed: {len(imfs)} IMFs extracted")
        
        return imfs, final_residue
    
    def _find_extrema_indices(self, signal: np.ndarray) -> np.ndarray:
        """
        Find all local extremum points (maxima and minima)
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        np.ndarray
            Indices of local extrema, sorted
        """
        if len(signal) < 3:
            return np.array([])
        
        extrema = []
        
        # Find local maxima and minima
        for i in range(1, len(signal) - 1):
            if ((signal[i] > signal[i-1] and signal[i] > signal[i+1]) or
                (signal[i] < signal[i-1] and signal[i] < signal[i+1])):
                extrema.append(i)
        
        return np.array(extrema)
    
    def _calculate_midpoints(self, signal: np.ndarray, extrema_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate midpoints between adjacent extrema
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        extrema_indices : np.ndarray
            Indices of extrema points
            
        Returns:
        --------
        tuple
            (midpoint_indices, midpoint_values)
        """
        if len(extrema_indices) < 2:
            return np.array([]), np.array([])
        
        midpoint_indices = []
        midpoint_values = []
        
        for k in range(len(extrema_indices) - 1):
            i1, i2 = extrema_indices[k], extrema_indices[k + 1]
            mid_idx = (i1 + i2) / 2
            mid_val = (signal[i1] + signal[i2]) / 2
            
            midpoint_indices.append(mid_idx)
            midpoint_values.append(mid_val)
        
        return np.array(midpoint_indices), np.array(midpoint_values)
    
    def _supplement_boundaries(self, signal: np.ndarray, midpoint_indices: np.ndarray, 
                             midpoint_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Supplement boundary midpoints by interpolation
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        midpoint_indices : np.ndarray
            Midpoint indices
        midpoint_values : np.ndarray
            Midpoint values
            
        Returns:
        --------
        tuple
            (extended_indices, extended_values) including f_0 and f_m
        """
        if len(midpoint_indices) == 0:
            # No midpoints, create boundary points
            f_0_idx = 0
            f_0_val = signal[0]
            f_m_idx = len(signal) - 1
            f_m_val = signal[-1]
            
            return np.array([f_0_idx, f_m_idx]), np.array([f_0_val, f_m_val])
        
        # Calculate boundary points by extrapolation
        if len(midpoint_indices) >= 2:
            # Linear extrapolation for left boundary
            slope_left = (midpoint_values[1] - midpoint_values[0]) / (midpoint_indices[1] - midpoint_indices[0])
            f_0_val = midpoint_values[0] - slope_left * midpoint_indices[0]
            
            # Linear extrapolation for right boundary
            slope_right = (midpoint_values[-1] - midpoint_values[-2]) / (midpoint_indices[-1] - midpoint_indices[-2])
            f_m_val = midpoint_values[-1] + slope_right * (len(signal) - 1 - midpoint_indices[-1])
        else:
            # Only one midpoint, use signal boundary values
            f_0_val = signal[0]
            f_m_val = signal[-1]
        
        # Combine boundary points with midpoints
        extended_indices = np.concatenate([[0], midpoint_indices, [len(signal) - 1]])
        extended_values = np.concatenate([[f_0_val], midpoint_values, [f_m_val]])
        
        return extended_indices, extended_values
    
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
            Number of local extrema
        """
        extrema_indices = self._find_extrema_indices(signal)
        return len(extrema_indices)


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic signal
    np.random.seed(42)
    N = 500
    t = np.linspace(0, 10, N)
    
    # Multi-component signal
    signal = (2 * np.sin(2 * np.pi * 1 * t) + 
              1 * np.sin(2 * np.pi * 3 * t) + 
              0.5 * np.sin(2 * np.pi * 8 * t) + 
              0.1 * np.random.randn(N))
    
    print("Testing ESMD on synthetic signal")
    
    # ESMD decomposition
    esmd = ESMD(C=2, delta=0.001, D=100, varpi=6)
    imfs, residue = esmd.decompose(signal)
    
    # Reconstruction
    reconstructed = np.sum(imfs, axis=0) + residue
    recon_error = np.linalg.norm(signal - reconstructed) / np.linalg.norm(signal)
    
    print(f"\nDecomposition Results:")
    print(f"Number of IMFs: {len(imfs)}")
    print(f"Reconstruction error: {recon_error:.2e}")
    
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
    
    print("\nESMD analysis completed successfully!")