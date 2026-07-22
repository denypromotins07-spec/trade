"""
Shannon and Approximate Entropy Metrics for Market Regime Detection

This module implements entropy-based metrics using Numba JIT compilation
to measure market regime complexity and noise levels. Dynamically filters
out low-signal environments for the RL agent.

Key Features:
- Shannon Entropy for distribution complexity
- Approximate Entropy (ApEn) for time series regularity
- Sample Entropy (SampEn) for robustness
- Permutation Entropy for ordinal patterns
- AMD ROCm/DirectML acceleration checks
- Strict 4GB RAM quota enforcement

AMD Ryzen AI 5 Optimizations:
- Numba JIT compilation for SIMD vectorization
- Parallel entropy computation
- Cache-efficient sliding windows
"""

import numpy as np
from numba import jit, prange, float64, int64
from typing import Tuple, List, Dict, Optional
import os
import warnings


def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD ROCm/DirectML availability."""
    acceleration = {
        'rocm_available': False,
        'directml_available': False,
        'numba_simd_available': True,
        'llvm_optimized': True
    }
    
    try:
        import torch
        if hasattr(torch.version, 'hip') or (torch.cuda.is_available() and 'ROCm' in str(torch.version.cuda)):
            acceleration['rocm_available'] = True
    except ImportError:
        pass
    
    try:
        import torch_directml
        acceleration['directml_available'] = True
    except ImportError:
        pass
    
    # Numba provides SIMD optimizations on AMD CPUs
    try:
        from numba import __version__ as numba_version
        acceleration['numba_simd_available'] = True
    except ImportError:
        acceleration['numba_simd_available'] = False
    
    return acceleration


@jit(nopython=True, cache=True, parallel=False)
def _shannon_entropy_numba(probabilities: np.ndarray) -> float64:
    """
    Compute Shannon entropy using Numba JIT.
    
    H(X) = -sum(p * log2(p))
    
    Args:
        probabilities: Array of probabilities (must sum to 1)
        
    Returns:
        Shannon entropy in bits
    """
    entropy = 0.0
    n = len(probabilities)
    for i in range(n):
        p = probabilities[i]
        if p > 1e-12:
            entropy -= p * np.log2(p)
    return entropy


@jit(nopython=True, cache=True, parallel=False)
def _compute_histogram_numba(data: np.ndarray, n_bins: int64, 
                              data_min: float64, data_max: float64) -> np.ndarray:
    """
    Compute histogram with Numba JIT optimization.
    
    Args:
        data: Input data array
        n_bins: Number of bins
        data_min: Minimum value for binning
        data_max: Maximum value for binning
        
    Returns:
        Histogram counts as float64 array
    """
    hist = np.zeros(n_bins, dtype=np.float64)
    n = len(data)
    bin_width = (data_max - data_min) / n_bins
    
    for i in range(n):
        val = data[i]
        # Clamp value to range
        if val < data_min:
            val = data_min
        elif val >= data_max:
            val = data_max - 1e-12
        
        bin_idx = int((val - data_min) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        hist[bin_idx] += 1.0
    
    return hist


@jit(nopython=True, cache=True, parallel=True)
def _shannon_entropy_parallel(data: np.ndarray, n_bins: int64) -> float64:
    """
    Compute Shannon entropy with parallel histogram computation.
    
    Args:
        data: Input data array
        n_bins: Number of bins for discretization
        
    Returns:
        Shannon entropy in bits
    """
    n = len(data)
    if n == 0:
        return 0.0
    
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max - data_min < 1e-12:
        return 0.0  # Constant signal has zero entropy
    
    hist = _compute_histogram_numba(data, n_bins, data_min, data_max)
    
    # Normalize to probabilities
    total = np.sum(hist)
    if total == 0:
        return 0.0
    
    probs = hist / total
    
    # Remove zero probabilities
    non_zero_count = 0
    for i in range(n_bins):
        if probs[i] > 0:
            non_zero_count += 1
    
    if non_zero_count == 0:
        return 0.0
    
    entropy = 0.0
    for i in range(n_bins):
        p = probs[i]
        if p > 1e-12:
            entropy -= p * np.log2(p)
    
    return entropy


def shannon_entropy(data: np.ndarray, n_bins: int = 64) -> float:
    """
    Compute Shannon entropy of a signal.
    
    Measures the average information content or uncertainty in the signal.
    Higher entropy indicates more complex/unpredictable distributions.
    
    Args:
        data: Input time series or distribution
        n_bins: Number of bins for discretization
        
    Returns:
        Shannon entropy in bits
    """
    data = np.asarray(data, dtype=np.float64)
    return float(_shannon_entropy_parallel(data, n_bins))


@jit(nopython=True, cache=True)
def _match_count_numba(template: np.ndarray, candidate: np.ndarray, 
                       r: float64) -> int64:
    """Count matches within tolerance r."""
    m = len(template)
    max_dist = 0.0
    for i in range(m):
        dist = np.abs(template[i] - candidate[i])
        if dist > max_dist:
            max_dist = dist
    return 1 if max_dist <= r else 0


@jit(nopython=True, cache=True, parallel=False)
def _approximate_entropy_numba(data: np.ndarray, m: int64, r: float64) -> float64:
    """
    Compute Approximate Entropy (ApEn) using Numba JIT.
    
    ApEn measures the regularity/complexity of a time series.
    Lower ApEn indicates more regularity/predictability.
    
    Args:
        data: Input time series
        m: Pattern length (embedding dimension)
        r: Tolerance threshold (typically 0.1-0.25 * std(data))
        
    Returns:
        Approximate entropy value
    """
    n = len(data)
    if n <= m:
        return 0.0
    
    # Normalize data
    data_std = np.std(data)
    if data_std < 1e-12:
        return 0.0
    
    normalized_data = (data - np.mean(data)) / data_std
    
    # Count matches for pattern length m
    phi_m = 0.0
    count_m = 0
    
    for i in range(n - m):
        template = normalized_data[i:i+m]
        matches = 0
        for j in range(n - m):
            if i != j:
                max_dist = 0.0
                for k in range(m):
                    dist = np.abs(template[k] - normalized_data[j+k])
                    if dist > max_dist:
                        max_dist = dist
                if max_dist <= r:
                    matches += 1
        if (n - m - 1) > 0:
            phi_m += np.log(matches / (n - m - 1))
        count_m += 1
    
    phi_m /= count_m if count_m > 0 else 1
    
    # Count matches for pattern length m+1
    phi_m1 = 0.0
    count_m1 = 0
    
    for i in range(n - m - 1):
        template = normalized_data[i:i+m+1]
        matches = 0
        for j in range(n - m - 1):
            if i != j:
                max_dist = 0.0
                for k in range(m + 1):
                    dist = np.abs(template[k] - normalized_data[j+k])
                    if dist > max_dist:
                        max_dist = dist
                if max_dist <= r:
                    matches += 1
        if (n - m - 2) > 0:
            phi_m1 += np.log(matches / (n - m - 2))
        count_m1 += 1
    
    phi_m1 /= count_m1 if count_m1 > 0 else 1
    
    apen = phi_m - phi_m1
    return apen


def approximate_entropy(data: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Compute Approximate Entropy (ApEn) for market regime detection.
    
    ApEn quantifies the unpredictability of fluctuations in a time series.
    - Low ApEn: Regular, predictable market (trending or ranging)
    - High ApEn: Chaotic, unpredictable market (high volatility, noise)
    
    Args:
        data: Price returns or other time series
        m: Pattern length (default 2)
        r_factor: Tolerance as fraction of standard deviation (default 0.2)
        
    Returns:
        Approximate entropy value
    """
    data = np.asarray(data, dtype=np.float64)
    r = r_factor * np.std(data)
    return float(_approximate_entropy_numba(data, m, r))


@jit(nopython=True, cache=True, parallel=False)
def _sample_entropy_numba(data: np.ndarray, m: int64, r: float64) -> float64:
    """
    Compute Sample Entropy (SampEn) using Numba JIT.
    
    SampEn is similar to ApEn but avoids self-matching bias.
    More robust for shorter time series.
    
    Args:
        data: Input time series
        m: Pattern length
        r: Tolerance threshold
        
    Returns:
        Sample entropy value
    """
    n = len(data)
    if n <= m:
        return 0.0
    
    # Normalize
    data_std = np.std(data)
    if data_std < 1e-12:
        return 0.0
    
    normalized_data = (data - np.mean(data)) / data_std
    
    # Count matches for length m (excluding self-matches)
    def count_matches(length):
        count = 0
        total = 0
        for i in range(n - length):
            for j in range(i + 1, n - length):
                max_dist = 0.0
                for k in range(length):
                    dist = np.abs(normalized_data[i+k] - normalized_data[j+k])
                    if dist > max_dist:
                        max_dist = dist
                if max_dist <= r:
                    count += 1
                total += 1
        return count, total
    
    count_m, total_m = count_matches(m)
    count_m1, total_m1 = count_matches(m + 1)
    
    if count_m == 0 or count_m1 == 0:
        return 0.0
    
    # SampEn = -log(A/B) where A and B are match probabilities
    sampen = -np.log((count_m1 / total_m1) / (count_m / total_m))
    return sampen


def sample_entropy(data: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Compute Sample Entropy (SampEn) for robust complexity measurement.
    
    Args:
        data: Input time series
        m: Pattern length
        r_factor: Tolerance as fraction of std
        
    Returns:
        Sample entropy value
    """
    data = np.asarray(data, dtype=np.float64)
    r = r_factor * np.std(data)
    return float(_sample_entropy_numba(data, m, r))


@jit(nopython=True, cache=True, parallel=True)
def _permutation_entropy_numba(data: np.ndarray, order: int64, delay: int64) -> float64:
    """
    Compute Permutation Entropy using Numba JIT.
    
    Permutation entropy measures the complexity based on ordinal patterns.
    Robust to noise and monotonic transformations.
    
    Args:
        data: Input time series
        order: Order of permutation (typically 3-7)
        delay: Time delay between elements
        
    Returns:
        Permutation entropy in bits
    """
    n = len(data)
    if n <= order * delay:
        return 0.0
    
    # Number of possible permutations
    n_perms = 1
    for i in range(1, order + 1):
        n_perms *= i
    
    # Count permutation patterns
    perm_counts = np.zeros(n_perms, dtype=np.float64)
    
    for i in range(n - (order - 1) * delay):
        # Extract pattern
        pattern = np.zeros(order, dtype=np.int64)
        for j in range(order):
            pattern[j] = i + j * delay
        
        # Compute rank order (permutation index)
        perm_idx = 0
        factorial = 1
        for j in range(order):
            count_smaller = 0
            for k in range(j + 1, order):
                if data[pattern[j]] > data[pattern[k]]:
                    count_smaller += 1
            perm_idx += count_smaller * factorial
            factorial *= (j + 1)
        
        if perm_idx < n_perms:
            perm_counts[perm_idx] += 1
    
    # Normalize to probabilities
    total = np.sum(perm_counts)
    if total == 0:
        return 0.0
    
    probs = perm_counts / total
    
    # Compute entropy
    entropy = 0.0
    for i in range(n_perms):
        p = probs[i]
        if p > 1e-12:
            entropy -= p * np.log2(p)
    
    # Normalize by maximum entropy
    max_entropy = np.log2(n_perms)
    if max_entropy > 0:
        entropy /= max_entropy
    
    return entropy


def permutation_entropy(data: np.ndarray, order: int = 5, delay: int = 1) -> float:
    """
    Compute Permutation Entropy for ordinal pattern complexity.
    
    Args:
        data: Input time series
        order: Order of permutation (3-7 recommended)
        delay: Time delay between elements
        
    Returns:
        Normalized permutation entropy [0, 1]
    """
    data = np.asarray(data, dtype=np.float64)
    return float(_permutation_entropy_numba(data, order, delay))


class EntropyRegimeDetector:
    """
    Multi-metric entropy-based market regime detector.
    
    Combines multiple entropy measures to classify market states:
    - Low Signal (high noise, random walk)
    - Trending (low entropy, directional)
    - Ranging (medium entropy, mean-reverting)
    - Chaotic (very high entropy, unpredictable)
    """
    
    def __init__(self, window_size: int = 100, memory_limit_mb: int = 3800):
        """
        Initialize regime detector.
        
        Args:
            window_size: Rolling window size for analysis
            memory_limit_mb: Memory limit in MB
        """
        self.window_size = window_size
        self.memory_limit_mb = memory_limit_mb
        self.acceleration = check_amd_acceleration()
    
    def _check_memory(self):
        """Validate memory usage."""
        import psutil
        process = psutil.Process(os.getpid())
        current_mem_mb = process.memory_info().rss / (1024 * 1024)
        if current_mem_mb > self.memory_limit_mb:
            raise MemoryError(f"Memory {current_mem_mb:.0f}MB exceeds limit {self.memory_limit_mb}MB")
    
    def compute_all_entropies(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute all entropy metrics for a signal.
        
        Args:
            data: Input time series
            
        Returns:
            Dictionary of entropy metrics
        """
        data = np.asarray(data, dtype=np.float64)
        
        results = {
            'shannon_entropy': shannon_entropy(data, n_bins=64),
            'approximate_entropy': approximate_entropy(data, m=2, r_factor=0.2),
            'sample_entropy': sample_entropy(data, m=2, r_factor=0.2),
            'permutation_entropy': permutation_entropy(data, order=5, delay=1),
            'acceleration': self.acceleration
        }
        
        self._check_memory()
        return results
    
    def classify_regime(self, data: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Classify market regime based on entropy metrics.
        
        Args:
            data: Price returns or other signal
            
        Returns:
            Tuple of (regime_name, metrics_dict)
        """
        metrics = self.compute_all_entropies(data)
        
        pe = metrics['permutation_entropy']
        se = metrics['sample_entropy']
        ae = metrics['approximate_entropy']
        
        # Normalize metrics for classification
        avg_complexity = (pe + min(se, 2.0) / 2.0) / 2.0
        
        if pe > 0.9 and ae > 1.5:
            regime = 'chaotic'
            confidence = min(pe, 1.0)
        elif pe < 0.4 and ae < 0.5:
            regime = 'trending'
            confidence = 1.0 - pe
        elif 0.4 <= pe <= 0.7 and ae < 1.0:
            regime = 'ranging'
            confidence = 1.0 - abs(pe - 0.55) * 2
        else:
            regime = 'low_signal'
            confidence = 0.5
        
        metrics['classified_regime'] = regime
        metrics['regime_confidence'] = confidence
        
        return regime, metrics
    
    def rolling_entropy_analysis(self, data: np.ndarray, 
                                  step: int = 10) -> List[Dict]:
        """
        Compute rolling entropy analysis over time.
        
        Args:
            data: Full time series
            step: Step size between windows
            
        Returns:
            List of analysis results for each window
        """
        n = len(data)
        results = []
        
        for start in range(0, n - self.window_size, step):
            window = data[start:start + self.window_size]
            regime, metrics = self.classify_regime(window)
            
            results.append({
                'timestamp_index': start,
                'regime': regime,
                **metrics
            })
            
            self._check_memory()
        
        return results
    
    def filter_low_signal_periods(self, data: np.ndarray, 
                                   threshold: float = 0.7) -> np.ndarray:
        """
        Filter out low-signal periods unsuitable for RL training.
        
        Args:
            data: Full time series
            threshold: Permutation entropy threshold for filtering
            
        Returns:
            Boolean mask indicating valid (high-signal) periods
        """
        n = len(data)
        mask = np.ones(n, dtype=bool)
        
        for start in range(0, n - self.window_size, self.window_size // 2):
            window = data[start:start + self.window_size]
            pe = permutation_entropy(window)
            
            if pe < threshold:
                # Mark this window as low signal
                end = min(start + self.window_size, n)
                mask[start:end] = False
        
        return mask


if __name__ == '__main__':
    print("Checking AMD acceleration...")
    accel = check_amd_acceleration()
    print(f"Acceleration: {accel}")
    
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate different market regimes
    trending = np.cumsum(np.random.randn(500) + 0.1)
    ranging = np.sin(np.linspace(0, 20, 500)) + np.random.randn(500) * 0.1
    chaotic = np.cumsum(np.random.randn(500) * 2)
    
    detector = EntropyRegimeDetector(window_size=100)
    
    print("\nTrending regime:")
    regime, metrics = detector.classify_regime(np.diff(trending))
    print(f"  Classified: {regime} (confidence: {metrics['regime_confidence']:.2f})")
    
    print("\nRanging regime:")
    regime, metrics = detector.classify_regime(ranging)
    print(f"  Classified: {regime} (confidence: {metrics['regime_confidence']:.2f})")
    
    print("\nChaotic regime:")
    regime, metrics = detector.classify_regime(np.diff(chaotic))
    print(f"  Classified: {regime} (confidence: {metrics['regime_confidence']:.2f})")
