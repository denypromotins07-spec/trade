"""
Chapter 2: Information Theory & Feature Selection
File 5: python/features/entropy.py

Shannon and Approximate Entropy metrics computed via Numba JIT
to measure market regime complexity and noise levels.
Dynamically filters out low-signal environments for the RL agent.

Optimized for AMD Ryzen AI 5 with ROCm/DirectML checks.
Uses Numba SIMD auto-vectorization for throughput.
"""

import numpy as np
from numba import jit, prange, float64, int64
from typing import Tuple, Optional, List, Dict
import ray

# Memory limit (4GB quota)
MAX_MEMORY_MB = 4096


def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD ROCm/DirectML availability."""
    accel_info = {
        'rocm_available': False,
        'directml_available': False,
        'cuda_available': False,
        'numba_available': False,
        'recommended_backend': 'numpy'
    }
    
    try:
        import torch
        if torch.version.hip is not None:
            accel_info['rocm_available'] = True
            accel_info['recommended_backend'] = 'pytorch_rocm'
        elif hasattr(torch.backends, 'directml'):
            accel_info['directml_available'] = True
            accel_info['recommended_backend'] = 'pytorch_directml'
        elif torch.cuda.is_available():
            accel_info['cuda_available'] = True
            accel_info['recommended_backend'] = 'pytorch_cuda'
    except ImportError:
        pass
    
    # Check Numba availability
    try:
        from numba import cuda
        accel_info['numba_available'] = True
    except ImportError:
        pass
    
    return accel_info


@jit(nopython=True, cache=True, fastmath=True)
def _shannon_entropy_numba(probabilities: np.ndarray) -> float64:
    """
    Calculate Shannon entropy using Numba JIT.
    
    H(X) = -sum(p * log(p))
    
    Parameters
    ----------
    probabilities : np.ndarray
        Probability distribution (must sum to 1)
        
    Returns
    -------
    float64
        Shannon entropy in nats
    """
    entropy = 0.0
    n = len(probabilities)
    for i in range(n):
        p = probabilities[i]
        if p > 1e-10:
            entropy -= p * np.log(p)
    return entropy


@jit(nopython=True, cache=True, fastmath=True)
def _discretize_series(data: np.ndarray, bins: int64) -> np.ndarray:
    """
    Discretize continuous time series into bins.
    SIMD-optimized via Numba parallelization.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    bins : int64
        Number of bins
        
    Returns
    -------
    np.ndarray
        Discretized series (integer labels)
    """
    n = len(data)
    result = np.empty(n, dtype=np.int64)
    
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    
    if range_val < 1e-10:
        return np.zeros(n, dtype=np.int64)
    
    bin_width = range_val / bins
    
    for i in range(n):
        bin_idx = int64((data[i] - min_val) / bin_width)
        bin_idx = min(bin_idx, bins - 1)
        result[i] = bin_idx
    
    return result


@jit(nopython=True, cache=True, fastmath=True)
def _count_patterns(series: np.ndarray, pattern_len: int64) -> np.ndarray:
    """
    Count occurrence of patterns for approximate entropy.
    
    Parameters
    ----------
    series : np.ndarray
        Discretized time series
    pattern_len : int64
        Length of patterns to count
        
    Returns
    -------
    np.ndarray
        Pattern counts
    """
    n = len(series)
    if n < pattern_len:
        return np.array([0.0])
    
    # Use dictionary-like approach with fixed size array
    max_patterns = min(10000, n - pattern_len + 1)
    counts = np.zeros(max_patterns, dtype=np.float64)
    pattern_hashes = np.zeros(max_patterns, dtype=np.int64)
    unique_count = 0
    
    for i in range(n - pattern_len + 1):
        # Create hash of pattern
        pattern_hash = 0
        for j in range(pattern_len):
            pattern_hash = pattern_hash * 31 + series[i + j]
        
        # Find or create entry
        found = False
        for k in range(unique_count):
            if pattern_hashes[k] == pattern_hash:
                counts[k] += 1
                found = True
                break
        
        if not found and unique_count < max_patterns:
            pattern_hashes[unique_count] = pattern_hash
            counts[unique_count] = 1
            unique_count += 1
    
    return counts[:unique_count]


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def shannon_entropy(data: np.ndarray, bins: int64 = 50) -> float64:
    """
    Calculate Shannon entropy of a time series.
    
    Parallelized via Numba for SIMD throughput.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    bins : int64
        Number of bins for discretization
        
    Returns
    -------
    float64
        Shannon entropy in nats
    """
    n = len(data)
    if n < 2:
        return 0.0
    
    # Discretize
    discrete = _discretize_series(data, bins)
    
    # Count frequencies
    freqs = np.zeros(bins, dtype=np.float64)
    for i in range(n):
        freqs[discrete[i]] += 1
    
    # Convert to probabilities
    for i in range(bins):
        freqs[i] /= n
    
    return _shannon_entropy_numba(freqs)


@jit(nopython=True, cache=True, fastmath=True)
def approximate_entropy(
    data: np.ndarray,
    m: int64 = 2,
    r_factor: float64 = 0.2
) -> float64:
    """
    Calculate Approximate Entropy (ApEn) of a time series.
    
    ApEn measures regularity/complexity:
    - Low ApEn: Regular, predictable
    - High ApEn: Complex, noisy
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    m : int64
        Pattern length (embedding dimension)
    r_factor : float64
        Tolerance factor (multiplied by std)
        
    Returns
    -------
    float64
        Approximate entropy value
    """
    n = len(data)
    if n < m + 1:
        return 0.0
    
    # Calculate tolerance
    std = np.std(data)
    r = r_factor * std if std > 1e-10 else 0.2
    
    def count_matches(embed_dim: int64) -> float64:
        """Count matching patterns within tolerance."""
        total_matches = 0.0
        n_embed = n - embed_dim + 1
        
        for i in range(n_embed):
            matches = 0
            for j in range(n_embed):
                if i == j:
                    continue
                
                # Check if patterns match within tolerance
                max_diff = 0.0
                for k in range(embed_dim):
                    diff = abs(data[i + k] - data[j + k])
                    if diff > max_diff:
                        max_diff = diff
                
                if max_diff <= r:
                    matches += 1
            
            total_matches += matches / (n_embed - 1)
        
        return total_matches / n_embed
    
    phi_m = np.log(count_matches(m))
    phi_m1 = np.log(count_matches(m + 1))
    
    apen = phi_m - phi_m1
    
    return max(0.0, apen)


@jit(nopython=True, cache=True, fastmath=True)
def sample_entropy(data: np.ndarray, m: int64 = 2, r_factor: float64 = 0.2) -> float64:
    """
    Calculate Sample Entropy (SampEn) - improved version of ApEn.
    
    SampEn is less biased than ApEn for short series.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    m : int64
        Pattern length
    r_factor : float64
        Tolerance factor
        
    Returns
    -------
    float64
        Sample entropy value
    """
    n = len(data)
    if n < m + 1:
        return 0.0
    
    std = np.std(data)
    r = r_factor * std if std > 1e-10 else 0.2
    
    def count_similar_pairs(embed_dim: int64) -> Tuple[int64, int64]:
        """Count similar pattern pairs."""
        n_embed = n - embed_dim
        num_pairs = 0
        num_matches = 0
        
        for i in range(n_embed):
            for j in range(i + 1, n_embed):
                num_pairs += 1
                
                # Check match
                max_diff = 0.0
                for k in range(embed_dim):
                    diff = abs(data[i + k] - data[j + k])
                    if diff > max_diff:
                        max_diff = diff
                
                if max_diff <= r:
                    num_matches += 1
        
        return num_pairs, num_matches
    
    pairs_m, matches_m = count_similar_pairs(m)
    pairs_m1, matches_m1 = count_similar_pairs(m + 1)
    
    if matches_m1 == 0 or pairs_m == 0 or pairs_m1 == 0:
        return 2.0  # Maximum entropy indicator
    
    # SampEn = -log(A/B) where A and B are match probabilities
    prob_m = matches_m / pairs_m if pairs_m > 0 else 0
    prob_m1 = matches_m1 / pairs_m1 if pairs_m1 > 0 else 0
    
    if prob_m1 < 1e-10:
        return 2.0
    
    sampen = -np.log(prob_m1 / prob_m)
    
    return max(0.0, sampen)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def permutation_entropy(data: np.ndarray, dim: int64 = 3, delay: int64 = 1) -> float64:
    """
    Calculate Permutation Entropy - measures ordinal pattern complexity.
    
    Robust to noise and computationally efficient.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    dim : int64
        Embedding dimension (typically 3-7)
    delay : int64
        Time delay between samples
        
    Returns
    -------
    float64
        Permutation entropy (normalized 0-1)
    """
    n = len(data)
    if n < dim * delay:
        return 0.0
    
    # Number of possible permutations
    n_perms = 1
    for i in range(1, dim + 1):
        n_perms *= i
    
    # Count permutation patterns
    perm_counts = np.zeros(n_perms, dtype=np.float64)
    n_vectors = n - (dim - 1) * delay
    
    for i in range(n_vectors):
        # Extract pattern indices
        pattern = np.zeros(dim, dtype=np.int64)
        for d in range(dim):
            pattern[d] = i + d * delay
        
        # Determine ordinal pattern (rank ordering)
        # Simple bubble sort for ranking
        rank = np.zeros(dim, dtype=np.int64)
        for d in range(dim):
            rank[d] = d
        
        for a in range(dim - 1):
            for b in range(a + 1, dim):
                if data[pattern[rank[a]]] > data[pattern[rank[b]]]:
                    tmp = rank[a]
                    rank[a] = rank[b]
                    rank[b] = tmp
        
        # Convert rank to permutation index
        perm_idx = 0
        factorial = 1
        for d in range(dim):
            count = 0
            for e in range(d):
                if rank[e] < rank[d]:
                    count += 1
            perm_idx += count * factorial
            factorial *= (d + 1)
        
        if perm_idx < n_perms:
            perm_counts[perm_idx] += 1
    
    # Normalize to probabilities
    total = np.sum(perm_counts)
    if total < 1e-10:
        return 0.0
    
    probs = perm_counts / total
    
    # Calculate entropy
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * np.log(p)
    
    # Normalize by maximum entropy
    max_entropy = np.log(n_perms)
    if max_entropy > 1e-10:
        entropy /= max_entropy
    
    return min(1.0, max(0.0, entropy))


@jit(nopython=True, cache=True, fastmath=True)
def spectral_entropy(data: np.ndarray, normalize: bool = True) -> float64:
    """
    Calculate Spectral Entropy from FFT power spectrum.
    
    Measures frequency domain complexity.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    normalize : bool
        If True, normalize to [0, 1]
        
    Returns
    -------
    float64
        Spectral entropy
    """
    n = len(data)
    if n < 4:
        return 0.0
    
    # Remove mean
    data_centered = data - np.mean(data)
    
    # Simple DFT magnitude (Numba-compatible)
    n_freq = n // 2
    power = np.zeros(n_freq, dtype=np.float64)
    
    for k in range(n_freq):
        real_sum = 0.0
        imag_sum = 0.0
        for t in range(n):
            angle = 2.0 * np.pi * k * t / n
            real_sum += data_centered[t] * np.cos(angle)
            imag_sum -= data_centered[t] * np.sin(angle)
        power[k] = (real_sum * real_sum + imag_sum * imag_sum) / n
    
    # Normalize to probability distribution
    total_power = np.sum(power)
    if total_power < 1e-10:
        return 0.0
    
    probs = power / total_power
    
    # Calculate entropy
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * np.log(p)
    
    if normalize:
        max_entropy = np.log(n_freq)
        if max_entropy > 1e-10:
            entropy /= max_entropy
    
    return min(1.0, max(0.0, entropy))


class MarketRegimeClassifier:
    """
    Classify market regimes using entropy metrics.
    
    Regimes:
    - LOW_VOLATILITY: Low entropy, predictable
    - HIGH_VOLATILITY: High entropy, chaotic
    - TRENDING: Medium entropy with directional bias
    - MEAN_REVERTING: Low-medium entropy, oscillating
    """
    
    REGIME_NAMES = ['low_volatility', 'high_volatility', 'trending', 'mean_reverting']
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.accel_info = check_amd_acceleration()
    
    def classify(self, returns: np.ndarray) -> Dict[str, any]:
        """
        Classify market regime based on entropy features.
        
        Parameters
        ----------
        returns : np.ndarray
            Price returns time series
            
        Returns
        -------
        dict
            Classification results with regime name and confidence
        """
        if len(returns) < self.window_size:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        # Use recent window
        window = returns[-self.window_size:]
        
        # Calculate entropy features
        shannon = shannon_entropy(window, bins=20)
        apen = approximate_entropy(window, m=2, r_factor=0.2)
        sampen = sample_entropy(window, m=2, r_factor=0.2)
        perm_en = permutation_entropy(window, dim=3, delay=1)
        spec_en = spectral_entropy(window)
        
        # Composite entropy score
        composite = (shannon + perm_en + spec_en) / 3.0
        
        # Volatility
        vol = np.std(window)
        
        # Classify regime
        if composite < 0.3 and vol < 0.01:
            regime = 'low_volatility'
            confidence = 1.0 - composite
        elif composite > 0.7 or vol > 0.03:
            regime = 'high_volatility'
            confidence = composite
        elif vol > 0.015 and np.mean(returns[-50:]) > 0.001:
            regime = 'trending'
            confidence = 0.6
        else:
            regime = 'mean_reverting'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'features': {
                'shannon_entropy': shannon,
                'approximate_entropy': apen,
                'sample_entropy': sampen,
                'permutation_entropy': perm_en,
                'spectral_entropy': spec_en,
                'volatility': vol
            },
            'is_low_signal': composite < 0.2  # Filter for RL agent
        }


@ray.remote(max_calls=10)
class DistributedEntropyCalculator:
    """Ray actor for distributed entropy calculations."""
    
    def __init__(self, memory_limit_mb: int = MAX_MEMORY_MB):
        self.memory_limit_mb = memory_limit_mb
        self.accel_info = check_amd_acceleration()
    
    def calculate_all_entropies(
        self,
        data: np.ndarray,
        symbols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate all entropy metrics for multiple symbols."""
        results = {}
        
        for i, symbol in enumerate(symbols):
            series = data[:, i] if len(data.shape) > 1 else data
            
            results[symbol] = {
                'shannon': float(shannon_entropy(series, bins=30)),
                'approximate': float(approximate_entropy(series)),
                'sample': float(sample_entropy(series)),
                'permutation': float(permutation_entropy(series)),
                'spectral': float(spectral_entropy(series))
            }
        
        return results
    
    def get_stats(self) -> Dict:
        return {
            'acceleration': self.accel_info,
            'memory_limit_mb': self.memory_limit_mb
        }


if __name__ == "__main__":
    print("AMD Acceleration:", check_amd_acceleration())
    
    # Test with sample data
    np.random.seed(42)
    
    # Generate different types of series
    random_series = np.random.randn(1000)
    trend_series = np.cumsum(np.random.randn(1000) * 0.01)
    mean_rev = np.sin(np.linspace(0, 20*np.pi, 1000)) + np.random.randn(1000) * 0.1
    
    print(f"\nRandom Series:")
    print(f"  Shannon: {shannon_entropy(random_series):.4f}")
    print(f"  Approx En: {approximate_entropy(random_series):.4f}")
    print(f"  Sample En: {sample_entropy(random_series):.4f}")
    print(f"  Perm En: {permutation_entropy(random_series):.4f}")
    
    print(f"\nTrend Series:")
    classifier = MarketRegimeClassifier()
    returns = np.diff(trend_series) / trend_series[:-1]
    result = classifier.classify(returns)
    print(f"  Regime: {result['regime']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Low Signal: {result['is_low_signal']}")
