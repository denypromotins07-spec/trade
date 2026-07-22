"""
Real-Time Sentiment Z-Score Aggregator

Builds a real-time sentiment Z-score aggregator that fuses multi-source text data
into a single continuous signal for the RL agent's observation space. Optimized
for microsecond updates and strict 4GB RAM quota enforcement.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import local modules
from .fasttext import FastTextClassifier, SentimentResult, SentimentLabel
from .lexicon import CryptoLexiconScorer, SentimentScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Sources of sentiment data."""
    TWITTER = "twitter"
    NEWS = "news"
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    DISCORD = "discord"
    WIRE = "wire"  # Bloomberg Terminal


@dataclass
class TimeWeightedSample:
    """A sentiment sample with time decay weighting."""
    value: float
    timestamp: float
    source: DataSource
    confidence: float
    half_life_sec: float = 300.0  # 5 minute half-life
    
    @property
    def weight(self) -> float:
        """Compute exponential time decay weight."""
        age = time.time() - self.timestamp
        return np.exp(-np.log(2) * age / self.half_life_sec)


@dataclass
class AggregatedSignal:
    """Aggregated sentiment signal for RL observation."""
    z_score: float
    raw_score: float
    smoothed_score: float
    momentum: float
    volatility: float
    sample_count: int
    sources_active: List[str]
    timestamp: float
    processing_time_us: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


class SentimentZScoreAggregator:
    """
    Real-time sentiment Z-score aggregator.
    
    Features:
    - Multi-source fusion (Twitter, News, Telegram, etc.)
    - Exponential time decay weighting
    - Rolling Z-score computation
    - EWMA smoothing for noise reduction
    - Momentum and volatility indicators
    - Memory-efficient circular buffers
    - 4GB RAM quota enforcement
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        ewma_span: int = 50,
        z_score_threshold: float = 2.0,
        max_ram_mb: int = 100,  # Per-aggregator budget
    ):
        self.window_size = window_size
        self.ewma_span = ewma_span
        self.z_score_threshold = z_score_threshold
        
        # Circular buffer for memory efficiency
        self.samples: Deque[TimeWeightedSample] = deque(maxlen=window_size)
        
        # Source-specific buffers
        self.source_buffers: Dict[DataSource, Deque[float]] = {
            source: deque(maxlen=window_size // len(DataSource))
            for source in DataSource
        }
        
        # EWMA state
        self.ewma_value: Optional[float] = None
        self.ewma_variance: float = 0.0
        self.alpha = 2.0 / (ewma_span + 1)  # EWMA smoothing factor
        
        # Statistics cache
        self._mean_cache: Optional[float] = None
        self._std_cache: Optional[float] = None
        self._last_update: float = 0.0
        self._update_interval_sec: float = 0.1  # Update stats every 100ms
        
        # Momentum tracking
        self.momentum_buffer: Deque[float] = deque(maxlen=20)
        
        # Volatility estimation
        self.volatility_buffer: Deque[float] = deque(maxlen=50)
        
        # Memory tracking
        self.max_ram_bytes = max_ram_mb * 1024 * 1024
        
        logger.info(f"SentimentZScoreAggregator initialized (window={window_size}, ewma_span={ewma_span})")
    
    def add_sample(
        self,
        value: float,
        source: DataSource,
        confidence: float = 1.0,
        half_life_sec: float = 300.0,
    ) -> None:
        """
        Add a new sentiment sample to the aggregator.
        
        Args:
            value: Raw sentiment score (-1 to 1)
            source: Data source identifier
            confidence: Confidence in the sample (0 to 1)
            half_life_sec: Half-life for time decay
        """
        sample = TimeWeightedSample(
            value=value,
            timestamp=time.time(),
            source=source,
            confidence=confidence,
            half_life_sec=half_life_sec,
        )
        
        self.samples.append(sample)
        self.source_buffers[source].append(value)
        
        # Update EWMA
        if self.ewma_value is None:
            self.ewma_value = value
        else:
            self.ewma_value = self.alpha * value + (1 - self.alpha) * self.ewma_value
        
        # Track momentum
        if len(self.samples) > 1:
            prev_value = self.samples[-2].value if len(self.samples) > 1 else value
            self.momentum_buffer.append(value - prev_value)
        
        # Track volatility
        if self.ewma_value is not None:
            self.volatility_buffer.append(abs(value - self.ewma_value))
        
        # Invalidate caches
        self._mean_cache = None
        self._std_cache = None
        self._last_update = time.time()
    
    def add_fasttext_result(self, result: SentimentResult, source: DataSource) -> None:
        """Add a FastText classification result."""
        # Convert label to numeric score
        label_map = {
            SentimentLabel.POSITIVE: 1.0,
            SentimentLabel.NEGATIVE: -1.0,
            SentimentLabel.NEUTRAL: 0.0,
            SentimentLabel.UNKNOWN: 0.0,
        }
        value = label_map.get(result.label, 0.0) * result.confidence
        self.add_sample(value, source, result.confidence)
    
    def add_lexicon_result(self, result: SentimentScore, source: DataSource) -> None:
        """Add a lexicon-based sentiment score."""
        self.add_sample(result.compound, source, 
                       confidence=(result.positive + result.negative) / 2)
    
    def _compute_weighted_stats(self) -> Tuple[float, float]:
        """Compute weighted mean and std using time-decay weights."""
        if not self.samples:
            return 0.0, 1.0
        
        # Check cache
        if (self._mean_cache is not None and 
            time.time() - self._last_update < self._update_interval_sec):
            return self._mean_cache, self._std_cache or 1.0
        
        values = []
        weights = []
        
        for sample in self.samples:
            w = sample.weight * sample.confidence
            values.append(sample.value)
            weights.append(w)
        
        values = np.array(values)
        weights = np.array(weights)
        
        total_weight = weights.sum()
        if total_weight < 1e-10:
            return 0.0, 1.0
        
        # Weighted mean
        weighted_mean = np.sum(values * weights) / total_weight
        
        # Weighted variance
        weighted_var = np.sum(weights * (values - weighted_mean) ** 2) / total_weight
        weighted_std = np.sqrt(weighted_var) if weighted_var > 0 else 1.0
        
        # Avoid division by zero
        if weighted_std < 1e-6:
            weighted_std = 1.0
        
        self._mean_cache = weighted_mean
        self._std_cache = weighted_std
        
        return weighted_mean, weighted_std
    
    def compute_z_score(self, value: Optional[float] = None) -> float:
        """
        Compute Z-score for current or given value.
        
        Z-score represents how many standard deviations from the mean.
        Values > 2 or < -2 indicate extreme sentiment.
        """
        mean, std = self._compute_weighted_stats()
        
        if value is None:
            # Use most recent sample
            if not self.samples:
                return 0.0
            value = self.samples[-1].value
        
        return (value - mean) / std
    
    def get_signal(self) -> AggregatedSignal:
        """
        Compute the full aggregated signal for RL observation.
        
        Returns comprehensive metrics including Z-score, momentum,
        volatility, and confidence intervals.
        """
        start_time = time.perf_counter()
        
        if not self.samples:
            elapsed = (time.perf_counter() - start_time) * 1_000_000
            return AggregatedSignal(
                z_score=0.0,
                raw_score=0.0,
                smoothed_score=0.0,
                momentum=0.0,
                volatility=0.0,
                sample_count=0,
                sources_active=[],
                timestamp=time.time(),
                processing_time_us=elapsed,
            )
        
        # Compute statistics
        mean, std = self._compute_weighted_stats()
        z_score = self.compute_z_score()
        
        # Current raw score (most recent)
        raw_score = self.samples[-1].value
        
        # Smoothed score (EWMA)
        smoothed_score = self.ewma_value if self.ewma_value is not None else raw_score
        
        # Momentum (rate of change)
        momentum = np.mean(list(self.momentum_buffer)) if self.momentum_buffer else 0.0
        
        # Volatility (standard deviation of residuals)
        volatility = np.std(list(self.volatility_buffer)) if self.volatility_buffer else 0.0
        
        # Active sources
        sources_active = [
            src.value for src, buf in self.source_buffers.items()
            if len(buf) > 0
        ]
        
        # Confidence interval (95%)
        ci_margin = 1.96 * std / np.sqrt(len(self.samples))
        confidence_interval = (mean - ci_margin, mean + ci_margin)
        
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        
        return AggregatedSignal(
            z_score=z_score,
            raw_score=raw_score,
            smoothed_score=smoothed_score,
            momentum=momentum,
            volatility=volatility,
            sample_count=len(self.samples),
            sources_active=sources_active,
            timestamp=time.time(),
            processing_time_us=elapsed_us,
            confidence_interval=confidence_interval,
        )
    
    def get_rl_observation(self) -> np.ndarray:
        """
        Generate normalized observation vector for RL agent.
        
        Returns a fixed-size numpy array suitable for neural network input.
        """
        signal = self.get_signal()
        
        # Normalize components to [-1, 1] range where possible
        obs = np.array([
            np.clip(signal.z_score, -3, 3) / 3.0,           # Normalized Z-score
            np.clip(signal.raw_score, -1, 1),               # Raw sentiment
            np.clip(signal.smoothed_score, -1, 1),          # Smoothed
            np.tanh(signal.momentum * 10),                  # Momentum (scaled)
            np.tanh(signal.volatility * 5),                 # Volatility (scaled)
            min(1.0, len(self.samples) / self.window_size), # Buffer fill ratio
            len(signal.sources_active) / len(DataSource),   # Source coverage
            np.clip((signal.z_score - signal.momentum), -2, 2) / 2.0,  # Divergence
        ], dtype=np.float32)
        
        return obs
    
    def detect_extreme_sentiment(self) -> Optional[str]:
        """
        Detect extreme sentiment conditions for alerting.
        
        Returns:
            String describing the condition, or None if normal
        """
        signal = self.get_signal()
        
        if signal.z_score > self.z_score_threshold:
            return "EXTREME_POSITIVE"
        elif signal.z_score < -self.z_score_threshold:
            return "EXTREME_NEGATIVE"
        elif signal.volatility > 0.5 and len(self.samples) > 100:
            return "HIGH_VOLATILITY"
        elif abs(signal.momentum) > 0.3:
            return "RAPID_SHIFT"
        
        return None
    
    def reset(self) -> None:
        """Reset all state (for regime changes)."""
        self.samples.clear()
        for buf in self.source_buffers.values():
            buf.clear()
        self.ewma_value = None
        self.ewma_variance = 0.0
        self.momentum_buffer.clear()
        self.volatility_buffer.clear()
        self._mean_cache = None
        self._std_cache = None
        logger.info("SentimentZScoreAggregator reset")
    
    def check_memory_usage(self) -> float:
        """Check current memory usage and enforce limits."""
        import sys
        
        # Estimate memory usage
        sample_size = sys.getsizeof(TimeWeightedSample(0, 0, DataSource.TWITTER, 1.0))
        estimated_bytes = len(self.samples) * sample_size * 2  # Account for overhead
        
        if estimated_bytes > self.max_ram_bytes:
            # Prune oldest samples
            prune_count = len(self.samples) // 4
            for _ in range(prune_count):
                if self.samples:
                    self.samples.popleft()
            logger.warning(f"Memory limit reached, pruned {prune_count} samples")
        
        return estimated_bytes / (1024 * 1024)  # Return MB


class MultiAssetSentimentTracker:
    """
    Track sentiment signals across multiple crypto assets.
    
    Maintains separate aggregators per asset while computing
    cross-asset correlation and divergence signals.
    """
    
    def __init__(self, max_assets: int = 50):
        self.max_assets = max_assets
        self.aggregators: Dict[str, SentimentZScoreAggregator] = {}
        self.asset_correlations: Dict[Tuple[str, str], float] = {}
    
    def get_aggregator(self, symbol: str) -> SentimentZScoreAggregator:
        """Get or create aggregator for a symbol."""
        if symbol not in self.aggregators:
            if len(self.aggregators) >= self.max_assets:
                # Remove least recently used
                lru_symbol = min(
                    self.aggregators.keys(),
                    key=lambda s: self.aggregators[s]._last_update
                )
                del self.aggregators[lru_symbol]
            
            self.aggregators[symbol] = SentimentZScoreAggregator()
        
        return self.aggregators[symbol]
    
    def add_sentiment(
        self,
        symbol: str,
        value: float,
        source: DataSource,
        confidence: float = 1.0,
    ) -> None:
        """Add sentiment sample for a specific asset."""
        agg = self.get_aggregator(symbol)
        agg.add_sample(value, source, confidence)
    
    def get_market_regime(self) -> str:
        """
        Determine overall market regime based on cross-asset sentiment.
        
        Returns:
            Regime string: 'BULLISH', 'BEARISH', 'NEUTRAL', 'MIXED'
        """
        if not self.aggregators:
            return "UNKNOWN"
        
        z_scores = [
            agg.get_signal().z_score
            for agg in self.aggregators.values()
            if len(agg.samples) > 10  # Require minimum samples
        ]
        
        if not z_scores:
            return "UNKNOWN"
        
        avg_z = np.mean(z_scores)
        dispersion = np.std(z_scores)
        
        if dispersion > 1.5:
            return "MIXED"  # High divergence between assets
        elif avg_z > 1.0:
            return "BULLISH"
        elif avg_z < -1.0:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_cross_asset_signals(self) -> Dict[str, float]:
        """
        Compute relative sentiment signals across assets.
        
        Returns dict of symbol -> relative strength vs market
        """
        if not self.aggregators:
            return {}
        
        # Get market average
        all_z = [agg.get_signal().z_score for agg in self.aggregators.values()]
        market_avg = np.mean(all_z) if all_z else 0.0
        
        # Compute relative strength
        relative = {}
        for symbol, agg in self.aggregators.items():
            signal = agg.get_signal()
            relative[symbol] = signal.z_score - market_avg
        
        return relative


if __name__ == "__main__":
    # Example usage
    aggregator = SentimentZScoreAggregator(window_size=500)
    
    # Simulate incoming sentiment data
    import random
    
    print("Simulating sentiment stream...")
    for i in range(200):
        # Simulate different sources
        for source in [DataSource.TWITTER, DataSource.NEWS, DataSource.TELEGRAM]:
            base_sentiment = 0.3 * np.sin(i / 20)  # Cyclical pattern
            noise = random.gauss(0, 0.2)
            value = np.clip(base_sentiment + noise, -1, 1)
            
            aggregator.add_sample(
                value=value,
                source=source,
                confidence=random.uniform(0.7, 1.0),
            )
        
        # Periodically get signal
        if i % 20 == 0:
            signal = aggregator.get_signal()
            print(f"\nt={i}: Z={signal.z_score:+.2f}, Raw={signal.raw_score:+.2f}, "
                  f"Momentum={signal.momentum:+.3f}, Vol={signal.volatility:.3f}")
            
            extreme = aggregator.detect_extreme_sentiment()
            if extreme:
                print(f"  *** ALERT: {extreme} ***")
    
    # Get RL observation
    obs = aggregator.get_rl_observation()
    print(f"\nRL Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
