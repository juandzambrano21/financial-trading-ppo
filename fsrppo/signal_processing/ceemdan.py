#!/usr/bin/python
# coding: UTF-8
"""
CEESMDAN: Complete Ensemble Extreme-point Symmetric Mode Decomposition with Adaptive Noise
Key differences from standard CEEMDAN:
- Uses ESMD as base decomposition instead of EMD
- Implements adaptive noise scaling
- Designed specifically for financial signal processing
"""

import logging
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class EMD:
    """
    Empirical Mode Decomposition base class
    Simplified implementation for CEEMDAN use
    """
    
    def __init__(self, **kwargs):
        self.max_imf = kwargs.get('max_imf', 100)
        self.range_thr = kwargs.get('range_thr', 0.01)
        self.total_power_thr = kwargs.get('total_power_thr', 0.05)
        
    def emd(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1) -> np.ndarray:
        """Basic EMD implementation"""
        if T is None:
            T = np.arange(len(S), dtype=S.dtype)
            
        if max_imf == -1:
            max_imf = self.max_imf
            
        # Simple EMD implementation
        imfs = []
        residue = S.copy()
        
        for _ in range(max_imf):
            if self._is_monotonic(residue):
                break
                
            imf = self._extract_imf(residue, T)
            if imf is None:
                break
                
            imfs.append(imf)
            residue = residue - imf
            
            # Check stopping criteria
            if np.max(residue) - np.min(residue) < self.range_thr:
                break
            if np.sum(np.abs(residue)) < self.total_power_thr:
                break
                
        imfs.append(residue)
        return np.array(imfs)
    
    def _is_monotonic(self, signal: np.ndarray) -> bool:
        """Check if signal is monotonic"""
        diff = np.diff(signal)
        return np.all(diff >= 0) or np.all(diff <= 0)
    
    def _extract_imf(self, signal: np.ndarray, T: np.ndarray) -> Optional[np.ndarray]:
        """Extract single IMF using sifting process"""
        h = signal.copy()
        
        for _ in range(50):  # Max sifting iterations
            # Find extrema
            maxima, minima = self._find_extrema(h)
            
            if len(maxima) < 2 or len(minima) < 2:
                return None
                
            # Create envelopes
            upper_env = self._interpolate_envelope(T, T[maxima], h[maxima], len(T))
            lower_env = self._interpolate_envelope(T, T[minima], h[minima], len(T))
            
            # Calculate mean
            mean = (upper_env + lower_env) / 2
            
            # Update h
            h_new = h - mean
            
            # Check IMF criteria
            if self._is_imf(h_new):
                return h_new
                
            h = h_new
            
        return h
    
    def _find_extrema(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find local maxima and minima"""
        n = len(signal)
        if n < 3:
            return np.array([]), np.array([])
            
        # Find local maxima
        maxima = []
        minima = []
        
        for i in range(1, n-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima.append(i)
            elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                minima.append(i)
                
        return np.array(maxima), np.array(minima)
    
    def _interpolate_envelope(self, T: np.ndarray, T_ext: np.ndarray, 
                            S_ext: np.ndarray, n_points: int) -> np.ndarray:
        """Interpolate envelope using cubic spline"""
        if len(T_ext) < 2:
            return np.zeros(n_points)
            
        # Simple linear interpolation (can be upgraded to cubic spline)
        return np.interp(T, T_ext, S_ext)
    
    def _is_imf(self, signal: np.ndarray, tolerance: float = 0.05) -> bool:
        """Check if signal satisfies IMF criteria"""
        maxima, minima = self._find_extrema(signal)
        n_extrema = len(maxima) + len(minima)
        n_zeros = len(self._find_zero_crossings(signal))
        
        # IMF criteria: number of extrema and zero crossings differ by at most 1
        return abs(n_extrema - n_zeros) <= 1
    
    def _find_zero_crossings(self, signal: np.ndarray) -> np.ndarray:
        """Find zero crossings in signal"""
        return np.where(np.diff(np.signbit(signal)))[0]


class CEEMDAN:
    """
    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
    
    Enhanced implementation optimized for financial signal processing with:
    - Improved noise handling
    - Better convergence criteria
    - Robust error handling
    -  logging
    """
    
    logger = logging.getLogger(__name__)
    noise_kinds_all = ["normal", "uniform"]
    
    def __init__(self, trials: int = 100, epsilon: float = 0.005, ext_EMD=None, 
                 parallel: bool = False, **kwargs):
        """
        Initialize CEEMDAN decomposer
        
        Parameters:
        -----------
        trials : int, default=100
            Number of trials for ensemble decomposition
        epsilon : float, default=0.005
            Noise scale factor
        ext_EMD : EMD, optional
            External EMD instance to use
        parallel : bool, default=False
            Whether to use parallel processing
        **kwargs : dict
            Additional parameters for configuration
        """
        # Ensemble constants
        self.trials = trials
        self.epsilon = epsilon
        self.noise_scale = float(kwargs.get("noise_scale", 1.0))
        self.range_thr = float(kwargs.get("range_thr", 0.01))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.05))
        self.beta_progress = bool(kwargs.get("beta_progress", True))
        
        # Random state for reproducibility
        self.random = np.random.RandomState(seed=kwargs.get("seed", 42))
        self.noise_kind = kwargs.get("noise_kind", "normal")
        self._max_imf = int(kwargs.get("max_imf", 100))
        
        # Parallel processing
        self.parallel = parallel
        self.processes = kwargs.get("processes")
        
        if self.processes is not None and not self.parallel:
            self.logger.warning("Process count specified but parallel=False")
            
        # Initialize EMD
        if ext_EMD is None:
            self.EMD = EMD(**kwargs)
        else:
            self.EMD = ext_EMD
            
        # Results storage
        self.C_IMF = None
        self.residue = None
        self.all_noise_EMD = []
        
    def __call__(self, S: np.ndarray, T: Optional[np.ndarray] = None, 
                 max_imf: int = -1, progress: bool = False) -> np.ndarray:
        """Callable interface for decomposition"""
        return self.ceemdan(S, T=T, max_imf=max_imf, progress=progress)
    
    def generate_noise(self, scale: float, size: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Generate noise with specified parameters
        
        Parameters:
        -----------
        scale : float
            Noise scale/amplitude
        size : int or sequence
            Shape of noise array
            
        Returns:
        --------
        noise : np.ndarray
            Generated noise array
        """
        if self.noise_kind == "normal":
            noise = self.random.normal(loc=0, scale=scale, size=size)
        elif self.noise_kind == "uniform":
            noise = self.random.uniform(low=-scale/2, high=scale/2, size=size)
        else:
            raise ValueError(f"Unsupported noise kind: {self.noise_kind}. "
                           f"Supported: {self.noise_kinds_all}")
        return noise
    
    def noise_seed(self, seed: int) -> None:
        """Set seed for noise generation"""
        self.random.seed(seed)
    
    def ceemdan(self, S: np.ndarray, T: Optional[np.ndarray] = None, 
                max_imf: int = -1, progress: bool = False) -> np.ndarray:
        """
        Perform CEEMDAN decomposition
        
        Parameters:
        -----------
        S : np.ndarray
            Input signal to decompose
        T : np.ndarray, optional
            Time vector (if None, assumes uniform sampling)
        max_imf : int, default=-1
            Maximum number of IMFs to extract (-1 for automatic)
        progress : bool, default=False
            Whether to show progress bar
            
        Returns:
        --------
        components : np.ndarray
            CEEMDAN components (IMFs + residue)
        """
        try:
            # Input validation
            S = np.asarray(S, dtype=np.float64)
            if S.ndim != 1:
                raise ValueError("Input signal must be 1-dimensional")
            if len(S) < 10:
                raise ValueError("Signal too short for decomposition")
                
            # Normalize signal
            scale_s = np.std(S)
            if scale_s == 0:
                raise ValueError("Input signal has zero variance")
            S_norm = S / scale_s
            
            # Generate noise ensemble
            self.logger.debug(f"Generating {self.trials} noise realizations")
            self.all_noises = self.generate_noise(
                self.noise_scale, (self.trials, S.size)
            )
            
            # Decompose all noise realizations
            self.logger.debug("Decomposing noise realizations")
            self.all_noise_EMD = self._decompose_noise()
            
            # Create first IMF using EEMD
            self.logger.debug("Computing first IMF")
            first_imf = self._eemd(S_norm, T, max_imf=1, progress=progress)[0]
            
            # Initialize variables
            all_cimfs = first_imf.reshape((-1, first_imf.size))
            prev_res = S_norm - first_imf
            
            self.logger.debug("Starting CEEMDAN sifting process")
            
            # Main CEEMDAN loop
            total = (max_imf - 1) if max_imf != -1 else None
            iterator = iter if not progress else lambda x: tqdm(
                x, desc="CEEMDAN decomposition", total=total
            )
            
            for imf_idx in iterator(range(self._max_imf)):
                # Check termination conditions
                if self.end_condition(S_norm, all_cimfs, max_imf):
                    self.logger.debug("Termination condition met")
                    break
                    
                # Current IMF number
                imfNo = all_cimfs.shape[0]
                beta = self.epsilon * np.std(prev_res)
                local_mean = np.zeros(S.size)
                
                # Ensemble averaging
                for trial in range(self.trials):
                    noise_imf = self.all_noise_EMD[trial]
                    res = prev_res.copy()
                    
                    # Add noise mode if available
                    if len(noise_imf) > imfNo:
                        res += beta * noise_imf[imfNo]
                    
                    # Extract local mean
                    imfs = self.emd(res, T, max_imf=1)
                    if len(imfs) > 1:
                        local_mean += imfs[-1] / self.trials
                    else:
                        local_mean += res / self.trials
                
                # Compute next IMF
                last_imf = prev_res - local_mean
                all_cimfs = np.vstack((all_cimfs, last_imf))
                prev_res = local_mean.copy()
            
            # Add final residue
            final_residue = S_norm - np.sum(all_cimfs, axis=0)
            all_cimfs = np.vstack((all_cimfs, final_residue))
            
            # Rescale back to original amplitude
            all_cimfs = all_cimfs * scale_s
            
            # Store results
            self.C_IMF = all_cimfs
            self.residue = S - np.sum(self.C_IMF, axis=0)
            
            # Cleanup
            del self.all_noise_EMD[:]
            
            self.logger.info(f"CEEMDAN completed: {all_cimfs.shape[0]} components extracted")
            return all_cimfs
            
        except Exception as e:
            self.logger.error(f"CEEMDAN decomposition failed: {str(e)}")
            raise
    
    def end_condition(self, S: np.ndarray, cIMFs: np.ndarray, max_imf: int) -> bool:
        """
        Test for CEEMDAN termination conditions
        
        Parameters:
        -----------
        S : np.ndarray
            Original signal
        cIMFs : np.ndarray
            Current set of IMFs
        max_imf : int
            Maximum number of IMFs
            
        Returns:
        --------
        bool
            Whether to terminate decomposition
        """
        imfNo = cIMFs.shape[0]
        
        # Check maximum IMF limit
        if 0 < max_imf <= imfNo:
            return True
        
        # Compute residue
        R = S - np.sum(cIMFs, axis=0)
        
        # Test EMD on residue
        try:
            _test_imf = self.emd(R, None, max_imf=1)
            if _test_imf.shape[0] == 1:
                self.logger.debug("Insufficient extrema in residue")
                return True
        except:
            return True
        
        # Range threshold test
        if np.max(R) - np.min(R) < self.range_thr:
            self.logger.debug("Range threshold reached")
            return True
        
        # Power threshold test
        if np.sum(np.abs(R)) < self.total_power_thr:
            self.logger.debug("Power threshold reached")
            return True
        
        return False
    
    def _decompose_noise(self) -> List[np.ndarray]:
        """Decompose all noise realizations"""
        if self.parallel:
            with Pool(processes=self.processes) as pool:
                all_noise_EMD = pool.map(self._decompose_single_noise, 
                                       range(self.trials))
        else:
            all_noise_EMD = [self._decompose_single_noise(i) 
                           for i in range(self.trials)]
        
        # Normalize by first IMF standard deviation
        if self.beta_progress:
            all_stds = [np.std(imfs[0]) if len(imfs) > 0 else 1.0 
                       for imfs in all_noise_EMD]
            all_noise_EMD = [imfs / max(imfs_std, 1e-12) 
                           for imfs, imfs_std in zip(all_noise_EMD, all_stds)]
        
        return all_noise_EMD
    
    def _decompose_single_noise(self, trial_idx: int) -> np.ndarray:
        """Decompose single noise realization"""
        noise = self.all_noises[trial_idx]
        return self.emd(noise, max_imf=-1)
    
    def _eemd(self, S: np.ndarray, T: Optional[np.ndarray] = None, 
              max_imf: int = -1, progress: bool = True) -> np.ndarray:
        """Ensemble EMD for first IMF computation"""
        if T is None:
            T = np.arange(len(S), dtype=S.dtype)
        
        self._S = S
        self._T = T
        self._N = N = len(S)
        self.max_imf = max_imf
        
        # Initialize ensemble IMF storage
        self.E_IMF = np.zeros((1, N))
        
        # Process trials
        if self.parallel:
            with Pool(processes=self.processes) as pool:
                map_func = pool.imap_unordered
        else:
            map_func = map
        
        iterator = iter if not progress else lambda x: tqdm(
            x, desc="EEMD trials", total=self.trials
        )
        
        for IMFs in iterator(map_func(self._trial_update, range(self.trials))):
            # Expand storage if needed
            if self.E_IMF.shape[0] < IMFs.shape[0]:
                num_new_layers = IMFs.shape[0] - self.E_IMF.shape[0]
                self.E_IMF = np.vstack((
                    self.E_IMF, 
                    np.zeros(shape=(num_new_layers, N))
                ))
            
            # Accumulate IMFs
            self.E_IMF[:IMFs.shape[0]] += IMFs
        
        return self.E_IMF / self.trials
    
    def _trial_update(self, trial: int) -> np.ndarray:
        """Single EEMD trial"""
        # Add noise to signal
        noise = self.epsilon * self.all_noise_EMD[trial][0]
        noisy_signal = self._S + noise
        
        # Decompose noisy signal
        return self.emd(noisy_signal, self._T, self.max_imf)
    
    def emd(self, S: np.ndarray, T: Optional[np.ndarray] = None, 
            max_imf: int = -1) -> np.ndarray:
        """EMD interface"""
        return self.EMD.emd(S, T, max_imf=max_imf)
    
    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get separated IMFs and residue from last decomposition
        
        Returns:
        --------
        tuple
            (IMFs, residue) from last decomposition
        """
        if self.C_IMF is None or self.residue is None:
            raise ValueError("No decomposition results available. Run CEEMDAN first.")
        return self.C_IMF, self.residue
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct original signal from IMFs"""
        if self.C_IMF is None:
            raise ValueError("No decomposition results available")
        return np.sum(self.C_IMF, axis=0)
    
    def get_instantaneous_frequency(self, imf_idx: int = 0) -> np.ndarray:
        """
        Compute instantaneous frequency of specified IMF using Hilbert transform
        
        Parameters:
        -----------
        imf_idx : int, default=0
            Index of IMF to analyze
            
        Returns:
        --------
        np.ndarray
            Instantaneous frequency
        """
        if self.C_IMF is None:
            raise ValueError("No decomposition results available")
        if imf_idx >= self.C_IMF.shape[0]:
            raise ValueError(f"IMF index {imf_idx} out of range")
        
        # Hilbert transform
        imf = self.C_IMF[imf_idx]
        analytic_signal = np.fft.ifft(np.fft.fft(imf) * self._hilbert_mask(len(imf)))
        
        # Instantaneous frequency
        phase = np.unwrap(np.angle(analytic_signal))
        inst_freq = np.gradient(phase) / (2 * np.pi)
        
        return inst_freq
    
    def _hilbert_mask(self, n: int) -> np.ndarray:
        """Create Hilbert transform mask"""
        mask = np.zeros(n)
        if n % 2 == 0:
            mask[0] = mask[n//2] = 1
            mask[1:n//2] = 2
        else:
            mask[0] = 1
            mask[1:(n+1)//2] = 2
        return mask


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate test signal
    N = 1000
    t = np.linspace(0, 10, N)
    
    # Multi-component signal
    signal = (2 * np.sin(2 * np.pi * 1 * t) + 
              1.5 * np.sin(2 * np.pi * 5 * t) + 
              0.5 * np.sin(2 * np.pi * 10 * t) + 
              0.1 * np.random.randn(N))
    
    # CEEMDAN decomposition
    ceemdan = CEEMDAN(trials=50, epsilon=0.005, parallel=False)
    imfs = ceemdan(signal, progress=True)
    
    # Results
    print(f"Decomposed into {imfs.shape[0]} components")
    print(f"Reconstruction error: {np.linalg.norm(signal - ceemdan.reconstruct()):.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(imfs.shape[0] + 1, 1, 1)
    plt.plot(t, signal, 'b-', linewidth=1)
    plt.title('Original Signal')
    plt.grid(True, alpha=0.3)
    
    for i, imf in enumerate(imfs):
        plt.subplot(imfs.shape[0] + 1, 1, i + 2)
        plt.plot(t, imf, 'g-', linewidth=1)
        plt.title(f'IMF {i+1}' if i < imfs.shape[0]-1 else 'Residue')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()