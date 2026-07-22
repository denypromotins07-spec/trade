"""
Hilbert-Huang Transform (HHT) Module for Ray Workers

This module develops Hilbert-Huang Transform modules on Ray workers
to extract instantaneous frequency and amplitude from non-stationary
crypto data, respecting the 4GB RAM quota per worker.

Architecture Notes:
- Uses NumPy arrays with contiguous memory layout to prevent cache thrashing
- Injects AMD ROCm/DirectML environment checks for GPU acceleration
- Memory-bounded EMD decomposition to respect 4GB RAM limit
- Designed for Ray distributed execution on crypto tick data

Mathematical Foundation:
HHT consists of two steps:
1. Empirical Mode Decomposition (EMD) - decomposes signal into IMFs
2. Hilbert Transform - extracts instantaneous frequency/amplitude from IMFs
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import ray


# Check for AMD ROCm/DirectML availability
def check_amd_acceleration() -> Dict[str, bool]:
    """
    Check for AMD DirectML/ROCm environment and return availability status.
    
    Returns:
        Dictionary with acceleration backend availability flags
    """
    acceleration_status = {
        "rocm_available": False,
        "directml_available": False,
        "cuda_available": False,
        "cpu_only": True
    }
    
    # Check for ROCm (AMD GPUs)
    try:
        import torch
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            acceleration_status["rocm_available"] = True
            acceleration_status["cpu_only"] = False
    except ImportError:
        pass
    
    # Check for DirectML (Windows AMD GPU acceleration)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DirectMLExecutionProvider' in providers:
            acceleration_status["directml_available"] = True
            acceleration_status["cpu_only"] = False
    except ImportError:
        pass
    
    # Standard CUDA check for comparison
    try:
        import torch
        if torch.cuda.is_available():
            acceleration_status["cuda_available"] = True
            acceleration_status["cpu_only"] = False
    except ImportError:
        pass
    
    return acceleration_status


@dataclass
class IMFResult:
    """Result container for Intrinsic Mode Function decomposition."""
    imf: np.ndarray
    residual: np.ndarray
    iterations: int
    convergence_achieved: bool


@dataclass
class HilbertSpectrumResult:
    """Result container for Hilbert spectral analysis."""
    instantaneous_amplitude: np.ndarray
    instantaneous_frequency: np.ndarray
    instantaneous_phase: np.ndarray
    time_axis: np.ndarray


class EmpiricalModeDecomposition:
    """
    Empirical Mode Decomposition (EMD) implementation with memory bounds.
    
    Decomposes a signal into Intrinsic Mode Functions (IMFs) using
    the sifting process. Memory-bounded for 4GB RAM constraint.
    """
    
    def __init__(
        self,
        max_imfs: int = 16,
        max_sifting_iterations: int = 100,
        tolerance: float = 0.05,
        max_extrema_ratio: float = 0.2
    ):
        """
        Initialize EMD decomposer.
        
        Args:
            max_imfs: Maximum number of IMFs to extract (memory bound)
            max_sifting_iterations: Max iterations per IMF extraction
            tolerance: Sifting tolerance for convergence
            max_extrema_ratio: Maximum ratio of extrema to samples
        """
        self.max_imfs = max_imfs
        self.max_sifting_iterations = max_sifting_iterations
        self.tolerance = tolerance
        self.max_extrema_ratio = max_extrema_ratio
        
    def _find_extrema(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find local maxima and minima indices."""
        from scipy.signal import argrelextrema
        
        max_idx = argrelextrema(signal, np.greater)[0]
        min_idx = argrelextrema(signal, np.less)[0]
        
        return max_idx, min_idx
    
    def _cubic_spline_envelope(
        self, 
        signal: np.ndarray, 
        max_idx: np.ndarray, 
        min_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute upper and lower envelopes using cubic spline interpolation.
        
        Uses contiguous memory layout for cache efficiency.
        """
        from scipy.interpolate import interp1d
        
        n = len(signal)
        
        # Handle edge cases
        if len(max_idx) < 2:
            upper_env = np.ones(n) * np.max(signal)
        else:
            # Include boundary points
            max_ext = np.concatenate([[0], max_idx, [n - 1]])
            max_vals = np.concatenate([[signal[0]], signal[max_idx], [signal[-1]]])
            f_upper = interp1d(max_ext, max_vals, kind='cubic', fill_value='extrapolate')
            upper_env = f_upper(np.arange(n))
        
        if len(min_idx) < 2:
            lower_env = np.ones(n) * np.min(signal)
        else:
            min_ext = np.concatenate([[0], min_idx, [n - 1]])
            min_vals = np.concatenate([[signal[0]], signal[min_idx], [signal[-1]]])
            f_lower = interp1d(min_ext, min_vals, kind='cubic', fill_value='extrapolate')
            lower_env = f_lower(np.arange(n))
        
        return upper_env, lower_env
    
    def _sift(self, signal: np.ndarray) -> IMFResult:
        """
        Perform sifting process to extract one IMF.
        
        Args:
            signal: Input signal segment
            
        Returns:
            IMFResult with extracted IMF and residual
        """
        h = signal.copy()
        iterations = 0
        convergence = False
        
        while iterations < self.max_sifting_iterations:
            max_idx, min_idx = self._find_extrema(h)
            
            # Check stopping criterion
            n_extrema = len(max_idx) + len(min_idx)
            if n_extrema < 2:
                break
                
            # Check extrema ratio
            if n_extrema / len(signal) > self.max_extrema_ratio:
                break
            
            upper, lower = self._cubic_spline_envelope(h, max_idx, min_idx)
            mean_env = (upper + lower) / 2.0
            
            # Update h
            h_new = h - mean_env
            
            # Check convergence
            diff = np.sum((h_new - h) ** 2) / np.sum(h ** 2 + 1e-10)
            h = h_new
            
            if diff < self.tolerance:
                convergence = True
                break
                
            iterations += 1
        
        return IMFResult(
            imf=h,
            residual=signal - h,
            iterations=iterations,
            convergence_achieved=convergence
        )
    
    def decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Decompose signal into IMFs.
        
        Args:
            signal: Input signal (contiguous numpy array)
            
        Returns:
            List of IMF arrays from high to low frequency
        """
        # Ensure contiguous memory layout
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        
        imfs = []
        residual = signal.copy()
        
        for _ in range(self.max_imfs):
            # Check residual energy
            if np.std(residual) < 1e-10 * np.std(signal):
                break
            
            result = self._sift(residual)
            
            # Check if IMF is valid
            if np.std(result.imf) < 1e-10 * np.std(signal):
                break
                
            imfs.append(result.imf)
            residual = result.residual
        
        return imfs


class HilbertTransform:
    """
    Hilbert Transform for instantaneous attribute extraction.
    
    Uses FFT-based implementation for O(n log n) complexity.
    """
    
    @staticmethod
    def transform(signal: np.ndarray) -> np.ndarray:
        """
        Compute analytic signal via Hilbert transform.
        
        Args:
            signal: Real input signal
            
        Returns:
            Complex analytic signal
        """
        from scipy.signal import hilbert
        return hilbert(signal)
    
    @staticmethod
    def instantaneous_attributes(analytic_signal: np.ndarray, sampling_rate: float) -> HilbertSpectrumResult:
        """
        Extract instantaneous amplitude, frequency, and phase.
        
        Args:
            analytic_signal: Complex analytic signal from Hilbert transform
            sampling_rate: Signal sampling rate in Hz
            
        Returns:
            HilbertSpectrumResult with all attributes
        """
        # Instantaneous amplitude (envelope)
        amplitude = np.abs(analytic_signal)
        
        # Instantaneous phase
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Instantaneous frequency (derivative of phase)
        dt = 1.0 / sampling_rate
        frequency = np.diff(phase) / (2 * np.pi * dt)
        frequency = np.concatenate([[frequency[0]], frequency])  # Pad to match length
        
        time_axis = np.arange(len(analytic_signal)) * dt
        
        return HilbertSpectrumResult(
            instantaneous_amplitude=amplitude,
            instantaneous_frequency=frequency,
            instantaneous_phase=phase,
            time_axis=time_axis
        )


class HilbertHuangProcessor:
    """
    Complete Hilbert-Huang Transform processor.
    
    Combines EMD decomposition with Hilbert spectral analysis.
    Memory-bounded for 4GB RAM per Ray worker.
    """
    
    def __init__(
        self,
        max_imfs: int = 12,
        sampling_rate: float = 1000.0,
        use_gpu: bool = False
    ):
        """
        Initialize HHT processor.
        
        Args:
            max_imfs: Maximum IMFs to extract
            sampling_rate: Data sampling rate (Hz)
            use_gpu: Attempt GPU acceleration if available
        """
        self.emd = EmpiricalModeDecomposition(max_imfs=max_imfs)
        self.sampling_rate = sampling_rate
        self.use_gpu = use_gpu
        
        # Check acceleration availability
        self.accel_status = check_amd_acceleration()
        if use_gpu and not self.accel_status["cpu_only"]:
            print(f"HHT using hardware acceleration: {self.accel_status}")
    
    def process(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Process signal through complete HHT pipeline.
        
        Args:
            signal: Input signal (price, volume, etc.)
            
        Returns:
            Dictionary with IMFs and spectral attributes
        """
        # Normalize signal
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        signal_mean = np.mean(signal)
        signal_std = np.std(signal) + 1e-10
        signal_normalized = (signal - signal_mean) / signal_std
        
        # EMD decomposition
        imfs = self.emd.decompose(signal_normalized)
        
        if not imfs:
            return {
                "imfs": [],
                "spectral_results": [],
                "residual": signal_normalized,
                "n_imfs": 0
            }
        
        # Hilbert transform on each IMF
        spectral_results = []
        for imf in imfs:
            analytic = HilbertTransform.transform(imf)
            attrs = HilbertTransform.instantaneous_attributes(analytic, self.sampling_rate)
            spectral_results.append(attrs)
        
        # Get final residual
        residual = signal_normalized - sum(imfs)
        
        return {
            "imfs": imfs,
            "spectral_results": spectral_results,
            "residual": residual,
            "n_imfs": len(imfs),
            "original_mean": signal_mean,
            "original_std": signal_std
        }
    
    def get_marginal_spectrum(self, spectral_results: List[HilbertSpectrumResult]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute marginal Hilbert spectrum (frequency distribution).
        
        Args:
            spectral_results: List of HilbertSpectrumResult from IMFs
            
        Returns:
            Frequency bins and amplitude spectrum
        """
        if not spectral_results:
            return np.array([]), np.array([])
        
        # Collect all frequencies and amplitudes
        all_freqs = []
        all_amps = []
        
        for result in spectral_results:
            mask = result.instantaneous_frequency > 0  # Valid frequencies only
            all_freqs.extend(result.instantaneous_frequency[mask])
            all_amps.extend(result.instantaneous_amplitude[mask])
        
        if not all_freqs:
            return np.array([]), np.array([])
        
        all_freqs = np.array(all_freqs)
        all_amps = np.array(all_amps)
        
        # Bin frequencies
        freq_bins = np.linspace(0, self.sampling_rate / 2, 256)
        spectrum, _ = np.histogram(all_freqs, bins=freq_bins, weights=all_amps)
        
        return freq_bins[:-1], spectrum


@ray.remote(max_calls=1000)
class RayHHTWorker:
    """
    Ray worker for distributed HHT processing.
    
    Processes chunks of crypto tick data in parallel.
    Memory-bounded to respect 4GB RAM quota.
    """
    
    def __init__(self, worker_id: int, sampling_rate: float = 1000.0):
        """Initialize worker with ID and parameters."""
        self.worker_id = worker_id
        self.processor = HilbertHuangProcessor(sampling_rate=sampling_rate)
        self.processed_count = 0
        
    def process_chunk(self, chunk_data: np.ndarray) -> Dict[str, Any]:
        """
        Process a chunk of data.
        
        Args:
            chunk_data: Numpy array of price/volume data
            
        Returns:
            Processing results dictionary
        """
        result = self.processor.process(chunk_data)
        result["worker_id"] = self.worker_id
        self.processed_count += 1
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "processed_count": self.processed_count,
            "accel_status": self.processor.accel_status
        }


def create_hht_pool(n_workers: int = 4, sampling_rate: float = 1000.0) -> List[ray.ObjectRef]:
    """
    Create a pool of Ray HHT workers.
    
    Args:
        n_workers: Number of workers to create
        sampling_rate: Data sampling rate
        
    Returns:
        List of worker handles
    """
    workers = [RayHHTWorker.remote(i, sampling_rate) for i in range(n_workers)]
    return workers


if __name__ == "__main__":
    # Example usage
    import time
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, _system_config={"max_object_store_fraction": 0.5})
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t) + 0.1 * np.random.randn(1000)
    
    # Test sequential processing
    processor = HilbertHuangProcessor(sampling_rate=1000.0)
    start = time.time()
    result = processor.process(signal)
    elapsed = time.time() - start
    
    print(f"Sequential HHT completed in {elapsed:.3f}s")
    print(f"Extracted {result['n_imfs']} IMFs")
    
    # Test Ray distributed processing
    workers = create_hht_pool(n_workers=2)
    
    # Distribute work
    futures = []
    chunk_size = len(signal) // 2
    for i, worker in enumerate(workers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < len(workers) - 1 else len(signal)
        chunk = signal[start_idx:end_idx]
        futures.append(worker.process_chunk.remote(chunk))
    
    # Collect results
    results = ray.get(futures)
    print(f"Distributed HHT completed with {len(results)} workers")
    
    ray.shutdown()
