"""
Model Drift Monitoring Module for Nautilus/Ray Trading Bot

Implements Population Stability Index (PSI) and Kolmogorov-Smirnov tests
to continuously monitor feature distributions and trigger automated model
retraining if drift exceeds 5%.

Features:
- Real-time PSI calculation for categorical and binned continuous features
- Two-sample Kolmogorov-Smirnov test for distribution comparison
- Automated retraining triggers with configurable thresholds
- Ray-distributed monitoring across workers
- AMD ROCm/DirectML environment checks

Compatible with /START and /KILL PowerShell orchestration.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import deque

# Check for AMD ROCm/DirectML availability
def check_rocm_availability() -> bool:
    """Check if AMD ROCm is available for GPU acceleration."""
    try:
        import torch
        if hasattr(torch, 'dml'):
            return True
        rocm_path = os.environ.get('ROCM_PATH', '')
        hip_path = os.environ.get('HIP_PATH', '')
        return bool(rocm_path or hip_path)
    except ImportError:
        return False


def check_directml_availability() -> bool:
    """Check if DirectML is available for Windows GPU acceleration."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return 'DmlExecutionProvider' in providers
        except ImportError:
            return False
    except ImportError:
        return False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROCM_AVAILABLE = check_rocm_availability()
DIRECTML_AVAILABLE = check_directml_availability()
logger.info(f"AMD ROCm available: {ROCM_AVAILABLE}")
logger.info(f"DirectML available: {DIRECTML_AVAILABLE}")


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    # PSI thresholds
    psi_warning_threshold: float = 0.1   # 10% drift - warning
    psi_critical_threshold: float = 0.25 # 25% drift - critical
    psi_retrain_threshold: float = 0.5   # 50% drift - immediate retrain
    
    # KS test parameters
    ks_significance_level: float = 0.05  # p-value threshold
    
    # Window sizes
    reference_window_size: int = 10000   # Samples in reference distribution
    current_window_size: int = 1000      # Samples in current window
    
    # Number of bins for continuous features
    n_bins: int = 20
    
    # Minimum samples before checking
    min_samples_for_check: int = 500
    
    # Features to monitor
    monitored_features: Optional[List[str]] = None


@dataclass
class DriftResult:
    """Result from drift detection."""
    feature_name: str
    psi_value: float
    ks_statistic: float
    ks_pvalue: float
    drift_severity: str  # "none", "warning", "critical", "retrain"
    should_retrain: bool
    timestamp_ns: int
    details: Dict[str, Any]


class PopulationStabilityIndex:
    """
    Calculate Population Stability Index (PSI) for feature drift detection.
    
    PSI measures how much a population has shifted over time:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change
    - PSI >= 0.25: Significant change
    """
    
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.reference_distribution: Optional[np.ndarray] = None
    
    def fit_reference(self, data: np.ndarray) -> 'PopulationStabilityIndex':
        """Establish the reference distribution."""
        if data.ndim > 1:
            data = data.flatten()
        
        # Create bins based on reference data
        self.bin_edges = np.histogram_bin_edges(data, bins=self.n_bins)
        
        # Calculate reference distribution (percentages)
        hist, _ = np.histogram(data, bins=self.bin_edges)
        self.reference_distribution = (hist + 1) / (len(data) + self.n_bins)  # Laplace smoothing
        
        return self
    
    def calculate_psi(self, current_data: np.ndarray) -> float:
        """
        Calculate PSI between reference and current distributions.
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        if self.reference_distribution is None or self.bin_edges is None:
            raise ValueError("Reference distribution not established. Call fit_reference first.")
        
        if current_data.ndim > 1:
            current_data = current_data.flatten()
        
        # Calculate current distribution
        hist, _ = np.histogram(current_data, bins=self.bin_edges)
        current_dist = (hist + 1) / (len(current_data) + self.n_bins)  # Laplace smoothing
        
        # Calculate PSI
        psi = 0.0
        for actual, expected in zip(current_dist, self.reference_distribution):
            if actual > 0 and expected > 0:
                psi += (actual - expected) * np.log(actual / expected)
        
        return psi
    
    def get_contribution_by_bin(self, current_data: np.ndarray) -> np.ndarray:
        """Get PSI contribution from each bin (for debugging)."""
        if self.reference_distribution is None or self.bin_edges is None:
            raise ValueError("Reference distribution not established.")
        
        if current_data.ndim > 1:
            current_data = current_data.flatten()
        
        hist, _ = np.histogram(current_data, bins=self.bin_edges)
        current_dist = (hist + 1) / (len(current_data) + self.n_bins)
        
        contributions = []
        for actual, expected in zip(current_dist, self.reference_distribution):
            if actual > 0 and expected > 0:
                contrib = (actual - expected) * np.log(actual / expected)
            else:
                contrib = 0.0
            contributions.append(contrib)
        
        return np.array(contributions)


class KolmogorovSmirnovTest:
    """
    Two-sample Kolmogorov-Smirnov test for distribution comparison.
    
    Tests whether two samples come from the same distribution.
    Returns KS statistic (max distance between CDFs) and p-value.
    """
    
    @staticmethod
    def ks_2samp(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """
        Perform two-sample KS test.
        
        Returns:
            (statistic, p-value)
        """
        n1 = len(sample1)
        n2 = len(sample2)
        
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        
        # Sort both samples
        s1 = np.sort(sample1)
        s2 = np.sort(sample2)
        
        # Combine and sort all data points
        all_data = np.concatenate([s1, s2])
        all_data = np.sort(np.unique(all_data))
        
        # Calculate empirical CDFs at each point
        cdf1 = np.searchsorted(s1, all_data, side='right') / n1
        cdf2 = np.searchsorted(s2, all_data, side='right') / n2
        
        # KS statistic is maximum absolute difference
        d = np.max(np.abs(cdf1 - cdf2))
        
        # Approximate p-value using asymptotic distribution
        # For large samples, use the Kolmogorov distribution approximation
        en = np.sqrt(n1 * n2 / (n1 + n2))
        
        # Simplified p-value calculation (sufficient for drift detection)
        # More accurate methods exist but this is fast and adequate
        if d == 0:
            p_value = 1.0
        else:
            # Approximation: p ≈ 2 * exp(-2 * (en * d)^2)
            p_value = 2 * np.exp(-2 * (en * d) ** 2)
            p_value = min(1.0, max(0.0, p_value))
        
        return d, p_value


class DriftDetector:
    """
    Main drift detection orchestrator.
    
    Monitors multiple features for distribution drift using both
    PSI and KS tests, triggering retraining when thresholds are exceeded.
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        
        # Per-feature detectors
        self.psi_calculators: Dict[str, PopulationStabilityIndex] = {}
        
        # Rolling windows for current data
        self.reference_windows: Dict[str, deque] = {}
        self.current_windows: Dict[str, deque] = {}
        
        # Feature names
        self.feature_names: List[str] = []
        
        # Initialization flag
        self.initialized = False
        
        # Drift history
        self.drift_history: deque = deque(maxlen=1000)
        
        logger.info("DriftDetector initialized")
    
    def initialize_features(self, feature_names: List[str], 
                           reference_data: Dict[str, np.ndarray]) -> None:
        """Initialize monitoring for specified features with reference data."""
        self.feature_names = feature_names or list(reference_data.keys())
        
        for name in self.feature_names:
            # Initialize PSI calculator
            self.psi_calculators[name] = PopulationStabilityIndex(self.config.n_bins)
            
            # Set reference distribution
            if name in reference_data:
                self.psi_calculators[name].fit_reference(reference_data[name])
            
            # Initialize windows
            self.reference_windows[name] = deque(maxlen=self.config.reference_window_size)
            self.current_windows[name] = deque(maxlen=self.config.current_window_size)
            
            # Fill reference window
            if name in reference_data:
                ref_data = reference_data[name]
                if ref_data.ndim > 1:
                    ref_data = ref_data.flatten()
                for val in ref_data[:self.config.reference_window_size]:
                    self.reference_windows[name].append(val)
        
        self.initialized = True
        logger.info(f"Initialized drift detection for {len(self.feature_names)} features")
    
    def update(self, features: Dict[str, float]) -> None:
        """Update current windows with new feature values."""
        if not self.initialized:
            return
        
        for name, value in features.items():
            if name in self.current_windows:
                self.current_windows[name].append(value)
    
    def check_drift(self, timestamp_ns: int) -> Dict[str, DriftResult]:
        """
        Check all features for drift.
        
        Returns dictionary of results per feature.
        """
        if not self.initialized:
            return {}
        
        results = {}
        
        for name in self.feature_names:
            current_data = np.array(list(self.current_windows[name]))
            
            if len(current_data) < self.config.min_samples_for_check:
                continue
            
            # Calculate PSI
            psi_value = self.psi_calculators[name].calculate_psi(current_data)
            
            # Calculate KS test
            ref_data = np.array(list(self.reference_windows[name]))
            ks_stat, ks_pval = KolmogorovSmirnovTest.ks_2samp(ref_data, current_data)
            
            # Determine severity
            if psi_value >= self.config.psi_retrain_threshold:
                severity = "retrain"
                should_retrain = True
            elif psi_value >= self.config.psi_critical_threshold:
                severity = "critical"
                should_retrain = True
            elif psi_value >= self.config.psi_warning_threshold:
                severity = "warning"
                should_retrain = False
            else:
                severity = "none"
                should_retrain = False
            
            # Also consider KS test p-value
            if ks_pval < self.config.ks_significance_level and not should_retrain:
                severity = "warning"
                should_retrain = False
            
            result = DriftResult(
                feature_name=name,
                psi_value=float(psi_value),
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pval),
                drift_severity=severity,
                should_retrain=should_retrain,
                timestamp_ns=timestamp_ns,
                details={
                    "current_samples": len(current_data),
                    "reference_samples": len(ref_data),
                    "rocm_available": ROCM_AVAILABLE,
                    "directml_available": DIRECTML_AVAILABLE,
                }
            )
            
            results[name] = result
            self.drift_history.append(result)
        
        return results
    
    def get_overall_drift_status(self) -> Dict[str, Any]:
        """Get summary of overall drift status."""
        if not self.drift_history:
            return {"status": "unknown", "features_monitored": len(self.feature_names)}
        
        recent_results = list(self.drift_history)[-len(self.feature_names):]
        
        retrain_count = sum(1 for r in recent_results if r.should_retrain)
        critical_count = sum(1 for r in recent_results if r.drift_severity == "critical")
        warning_count = sum(1 for r in recent_results if r.drift_severity == "warning")
        
        if retrain_count > 0:
            status = "retrain_required"
        elif critical_count > 0:
            status = "critical"
        elif warning_count > 0:
            status = "warning"
        else:
            status = "stable"
        
        avg_psi = np.mean([r.psi_value for r in recent_results]) if recent_results else 0.0
        
        return {
            "status": status,
            "features_monitored": len(self.feature_names),
            "avg_psi": float(avg_psi),
            "retrain_count": retrain_count,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "recommendation": self._get_recommendation(status, avg_psi),
        }
    
    def _get_recommendation(self, status: str, avg_psi: float) -> str:
        """Generate human-readable recommendation."""
        if status == "retrain_required":
            return "IMMEDIATE ACTION: Retrain models due to significant drift"
        elif status == "critical":
            return "URGENT: Consider retraining; drift approaching critical levels"
        elif status == "warning":
            return f"MONITOR: Mild drift detected (avg PSI: {avg_psi:.3f})"
        else:
            return "OK: Feature distributions stable"
    
    def reset_reference(self, feature_name: Optional[str] = None) -> None:
        """Reset reference distribution to current data."""
        if feature_name:
            names_to_reset = [feature_name]
        else:
            names_to_reset = self.feature_names
        
        for name in names_to_reset:
            if name in self.current_windows and len(self.current_windows[name]) > 0:
                current_data = np.array(list(self.current_windows[name]))
                self.psi_calculators[name].fit_reference(current_data)
                
                # Move current to reference
                self.reference_windows[name].clear()
                for val in current_data:
                    self.reference_windows[name].append(val)
        
        logger.info(f"Reset reference distribution for: {names_to_reset}")


# Ray actor for distributed drift monitoring
try:
    import ray
    
    @ray.remote(max_restarts=-1)
    class RayDriftMonitor:
        """Ray-distributed drift monitor worker."""
        
        def __init__(self, worker_id: int, config: Optional[Dict] = None):
            self.worker_id = worker_id
            self.config = DriftConfig(**config) if config else DriftConfig()
            self.detector = DriftDetector(self.config)
            
            logger.info(f"DriftMonitor Worker {worker_id} initialized")
        
        def initialize(self, feature_names: List[str], 
                      reference_data: Dict[str, np.ndarray]) -> bool:
            """Initialize monitoring."""
            self.detector.initialize_features(feature_names, reference_data)
            return True
        
        def update_and_check(self, features: Dict[str, float], 
                            timestamp_ns: int) -> Dict[str, Dict]:
            """Update with new data and check for drift."""
            self.detector.update(features)
            results = self.detector.check_drift(timestamp_ns)
            
            # Convert to serializable format
            return {
                name: {
                    "psi": r.psi_value,
                    "ks_stat": r.ks_statistic,
                    "ks_pval": r.ks_pvalue,
                    "severity": r.drift_severity,
                    "retrain": r.should_retrain,
                }
                for name, r in results.items()
            }
        
        def get_status(self) -> Dict:
            """Get current drift status."""
            return {
                "worker_id": self.worker_id,
                **self.detector.get_overall_drift_status(),
                "rocm_available": ROCM_AVAILABLE,
                "directml_available": DIRECTML_AVAILABLE,
            }
        
        def trigger_retrain(self) -> bool:
            """Check if retrain should be triggered."""
            status = self.detector.get_overall_drift_status()
            return status.get("status") in ["retrain_required", "critical"]

except ImportError:
    logger.warning("Ray not available, using local execution")
    RayDriftMonitor = None  # type: ignore


if __name__ == "__main__":
    # Test the drift detector
    config = DriftConfig(
        reference_window_size=1000,
        current_window_size=200,
        min_samples_for_check=100,
    )
    detector = DriftDetector(config)
    
    # Generate reference data (normal distribution)
    np.random.seed(42)
    reference_data = {
        "feature_1": np.random.randn(1000),
        "feature_2": np.random.randn(1000) * 0.5 + 2,
    }
    
    detector.initialize_features(["feature_1", "feature_2"], reference_data)
    
    # Simulate normal updates
    print("=== Normal Updates ===")
    for i in range(200):
        features = {
            "feature_1": float(np.random.randn()),
            "feature_2": float(np.random.randn() * 0.5 + 2),
        }
        detector.update(features)
    
    results = detector.check_drift(timestamp_ns=1234567890)
    for name, result in results.items():
        print(f"{name}: PSI={result.psi_value:.4f}, KS p-value={result.ks_pvalue:.4f}, Severity={result.drift_severity}")
    
    print("\nOverall Status:", detector.get_overall_drift_status())
    
    # Simulate drift (shifted distribution)
    print("\n=== After Distribution Shift ===")
    for i in range(200):
        features = {
            "feature_1": float(np.random.randn() + 2),  # Shifted mean
            "feature_2": float(np.random.randn() * 0.5 + 2),
        }
        detector.update(features)
    
    results = detector.check_drift(timestamp_ns=1234567890)
    for name, result in results.items():
        print(f"{name}: PSI={result.psi_value:.4f}, KS p-value={result.ks_pvalue:.4f}, Severity={result.drift_severity}")
    
    print("\nOverall Status:", detector.get_overall_drift_status())
