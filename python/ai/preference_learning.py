"""
Pairwise Preference Learning for Strategy Trajectory Ranking

This module codes pairwise preference learning on Ray to rank strategy trajectories
based on strict quantitative drawdown metrics, respecting the 4GB Python RAM quota.

Optimized for:
- Streaming mini-batch processing
- 4GB Python RAM quota enforcement
- AMD ROCm/DirectML acceleration checks
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc


@dataclass
class TrajectorySegment:
    """Represents a segment of trading trajectory"""
    trajectory_id: str
    start_idx: int
    end_idx: int
    returns: np.ndarray
    drawdowns: np.ndarray
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair:
    """A pairwise comparison between two trajectory segments"""
    preferred_id: str
    non_preferred_id: str
    preference_strength: float  # 0.5 to 1.0
    metric_basis: str  # Which metric drove the preference


class DrawdownAnalyzer:
    """Analyzes drawdown metrics for trajectory evaluation"""
    
    @staticmethod
    def compute_drawdowns(returns: np.ndarray) -> np.ndarray:
        """Compute running drawdown series from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns
    
    @staticmethod
    def compute_max_drawdown(returns: np.ndarray) -> float:
        """Compute maximum drawdown from returns"""
        drawdowns = DrawdownAnalyzer.compute_drawdowns(returns)
        return abs(np.min(drawdowns))
    
    @staticmethod
    def compute_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute Calmar ratio (annualized return / max drawdown)"""
        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        annualized = (1 + total_return) ** (1 / max(years, 1/periods_per_year)) - 1
        
        max_dd = DrawdownAnalyzer.compute_max_drawdown(returns)
        
        if max_dd < 1e-10:
            return float('inf') if annualized > 0 else 0.0
        
        return annualized / max_dd
    
    @staticmethod
    def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                            periods_per_year: int = 252) -> float:
        """Compute Sharpe ratio"""
        excess_returns = returns - risk_free_rate / periods_per_year
        
        if np.std(excess_returns) < 1e-10:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)


class PreferenceModel(nn.Module):
    """Neural network for learning preferences between trajectories"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning probability that A is preferred over B.
        """
        encoded_a = self.encoder(features_a)
        encoded_b = self.encoder(features_b)
        
        score_a = self.scorer(encoded_a)
        score_b = self.scorer(encoded_b)
        
        # Bradley-Terry model: P(A > B) = exp(score_a) / (exp(score_a) + exp(score_b))
        diff = score_a - score_b
        return torch.sigmoid(diff)
    
    def score_single(self, features: torch.Tensor) -> torch.Tensor:
        """Get raw score for a single trajectory"""
        encoded = self.encoder(features)
        return self.scorer(encoded)


class PreferenceLearner:
    """
    Learns preferences from pairwise comparisons with memory management.
    """
    
    def __init__(
        self,
        feature_dim: int,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        max_buffer_size: int = 10000
    ):
        self.feature_dim = feature_dim
        self.max_buffer_size = max_buffer_size
        
        # Device selection with AMD checks
        self.device = self._select_device(device)
        
        # Initialize model
        self.model = PreferenceModel(feature_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Experience buffer (bounded)
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        
        # Memory tracking
        self.memory_used_mb = 0
        self.max_memory_mb = 4096
    
    def _select_device(self, requested_device: Optional[str]) -> str:
        """Select best available device with AMD ROCm/DirectML checks."""
        if requested_device:
            return requested_device
        
        if torch.cuda.is_available():
            return 'cuda'
        
        try:
            import torch_directml
            return 'dml'
        except ImportError:
            pass
        
        if torch.version.hip is not None:
            return 'cuda'
        
        return 'cpu'
    
    def _check_memory_quota(self) -> bool:
        """Check if we're within 4GB Python RAM quota."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.memory_used_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        return self.memory_used_mb < self.max_memory_mb * 0.9
    
    def extract_features(self, trajectory: TrajectorySegment) -> np.ndarray:
        """Extract feature vector from trajectory segment"""
        features = np.array([
            trajectory.sharpe_ratio,
            trajectory.max_drawdown,
            trajectory.calmar_ratio,
            trajectory.volatility,
            np.mean(trajectory.returns),
            np.std(trajectory.returns),
            np.skew(trajectory.returns) if len(trajectory.returns) > 2 else 0.0,
            np.kurtosis(trajectory.returns) if len(trajectory.returns) > 3 else 0.0,
            np.sum(trajectory.drawdowns < -0.01),  # Count of >1% drawdowns
            np.max(trajectory.drawdowns < -0.05),  # Any >5% drawdown
        ])
        
        # Normalize features
        features[0] = np.clip(features[0], -5, 5) / 5.0
        features[1] = np.clip(features[1], 0, 1)
        features[2] = np.clip(features[2], -10, 10) / 10.0
        features[3] = np.clip(features[3], 0, 0.5) / 0.5
        features[4] = np.clip(features[4], -1, 1)
        features[5] = np.clip(features[5], 0, 0.5) / 0.5
        
        return features.astype(np.float32)
    
    def create_preference_pair(
        self,
        traj_a: TrajectorySegment,
        traj_b: TrajectorySegment
    ) -> Optional[PreferencePair]:
        """Create a preference pair based on drawdown metrics"""
        # Primary ranking by Calmar ratio
        if traj_a.calmar_ratio > traj_b.calmar_ratio * 1.1:
            return PreferencePair(
                preferred_id=traj_a.trajectory_id,
                non_preferred_id=traj_b.trajectory_id,
                preference_strength=min(0.5 + (traj_a.calmar_ratio - traj_b.calmar_ratio) * 0.5, 0.95),
                metric_basis='calmar_ratio'
            )
        elif traj_b.calmar_ratio > traj_a.calmar_ratio * 1.1:
            return PreferencePair(
                preferred_id=traj_b.trajectory_id,
                non_preferred_id=traj_a.trajectory_id,
                preference_strength=min(0.5 + (traj_b.calmar_ratio - traj_a.calmar_ratio) * 0.5, 0.95),
                metric_basis='calmar_ratio'
            )
        
        # Secondary ranking by max drawdown
        if traj_a.max_drawdown < traj_b.max_drawdown * 0.9:
            return PreferencePair(
                preferred_id=traj_a.trajectory_id,
                non_preferred_id=traj_b.trajectory_id,
                preference_strength=0.6,
                metric_basis='max_drawdown'
            )
        elif traj_b.max_drawdown < traj_a.max_drawdown * 0.9:
            return PreferencePair(
                preferred_id=traj_b.trajectory_id,
                non_preferred_id=traj_a.trajectory_id,
                preference_strength=0.6,
                metric_basis='max_drawdown'
            )
        
        return None
    
    def add_experience(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        preference_strength: float
    ):
        """Add experience to buffer with memory check"""
        if not self._check_memory_quota():
            # Clear oldest experiences
            self.buffer = self.buffer[-self.max_buffer_size // 2:]
        
        self.buffer.append((features_a, features_b, preference_strength))
        
        # Enforce max buffer size
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]
    
    def train_step(self, batch_size: int = 64) -> float:
        """Perform one training step"""
        if len(self.buffer) < batch_size:
            return 0.0
        
        # Sample mini-batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        features_a_list = []
        features_b_list = []
        strengths = []
        
        for idx in indices:
            fa, fb, s = self.buffer[idx]
            features_a_list.append(fa)
            features_b_list.append(fb)
            strengths.append(s)
        
        # Convert to tensors
        features_a = torch.FloatTensor(np.array(features_a_list)).to(self.device)
        features_b = torch.FloatTensor(np.array(features_b_list)).to(self.device)
        targets = torch.FloatTensor(strengths).unsqueeze(1).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(features_a, features_b)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def get_trajectory_scores(self, trajectories: List[TrajectorySegment]) -> Dict[str, float]:
        """Score all trajectories and return ranking"""
        self.model.eval()
        
        scores = {}
        with torch.no_grad():
            for traj in trajectories:
                features = self.extract_features(traj)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                score = self.model.score_single(features_tensor).item()
                scores[traj.trajectory_id] = score
        
        return scores
    
    def cleanup(self):
        """Cleanup to maintain memory quota"""
        self.buffer = self.buffer[-self.max_buffer_size // 2:]
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


@ray.remote(max_calls=500)
class DistributedPreferenceWorker:
    """Ray-distributed worker for preference learning"""
    
    def __init__(self, feature_dim: int = 10):
        self.learner = PreferenceLearner(feature_dim)
        self.trajectories_processed = 0
    
    def process_trajectories(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> List[PreferencePair]:
        """Process batch of trajectories into preference pairs"""
        if not self.learner._check_memory_quota():
            raise MemoryError("Exceeded 4GB Python RAM quota")
        
        # Convert dicts to TrajectorySegment objects
        segments = []
        for t in trajectories:
            seg = TrajectorySegment(
                trajectory_id=t['id'],
                start_idx=t.get('start_idx', 0),
                end_idx=t.get('end_idx', 0),
                returns=np.array(t['returns']),
                drawdowns=DrawdownAnalyzer.compute_drawdowns(np.array(t['returns'])),
                sharpe_ratio=DrawdownAnalyzer.compute_sharpe_ratio(np.array(t['returns'])),
                max_drawdown=DrawdownAnalyzer.compute_max_drawdown(np.array(t['returns'])),
                calmar_ratio=DrawdownAnalyzer.compute_calmar_ratio(np.array(t['returns'])),
                volatility=float(np.std(t['returns']))
            )
            segments.append(seg)
        
        self.trajectories_processed += len(segments)
        
        # Create preference pairs
        pairs = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                pair = self.learner.create_preference_pair(segments[i], segments[j])
                if pair:
                    pairs.append(pair)
                    
                    # Add to experience buffer
                    fa = self.learner.extract_features(segments[i])
                    fb = self.learner.extract_features(segments[j])
                    
                    if pair.preferred_id == segments[i].trajectory_id:
                        self.learner.add_experience(fa, fb, pair.preference_strength)
                    else:
                        self.learner.add_experience(fb, fa, pair.preference_strength)
        
        return pairs
    
    def train(self, batch_size: int = 64) -> float:
        """Run training step"""
        return self.learner.train_step(batch_size)
    
    def get_rankings(self, trajectory_dicts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get current rankings for trajectories"""
        segments = []
        for t in trajectory_dicts:
            seg = TrajectorySegment(
                trajectory_id=t['id'],
                start_idx=t.get('start_idx', 0),
                end_idx=t.get('end_idx', 0),
                returns=np.array(t['returns']),
                drawdowns=DrawdownAnalyzer.compute_drawdowns(np.array(t['returns'])),
                sharpe_ratio=DrawdownAnalyzer.compute_sharpe_ratio(np.array(t['returns'])),
                max_drawdown=DrawdownAnalyzer.compute_max_drawdown(np.array(t['returns'])),
                calmar_ratio=DrawdownAnalyzer.compute_calmar_ratio(np.array(t['returns'])),
                volatility=float(np.std(t['returns']))
            )
            segments.append(seg)
        
        return self.learner.get_trajectory_scores(segments)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            'trajectories_processed': self.trajectories_processed,
            'memory_mb': self.learner.memory_used_mb,
            'buffer_size': len(self.learner.buffer),
            'device': self.learner.device
        }


if __name__ == '__main__':
    import time
    
    # Initialize Ray with memory limits
    ray.init(
        ignore_reinit_error=True,
        _system_config={"object_store_memory": 1024*1024*1024}
    )
    
    # Create workers
    workers = [DistributedPreferenceWorker.remote() for _ in range(4)]
    
    # Generate test trajectories
    test_trajectories = []
    for i in range(20):
        returns = np.random.randn(100) * 0.01 + 0.0005  # Small positive drift
        test_trajectories.append({
            'id': f'traj_{i}',
            'returns': returns.tolist()
        })
    
    # Distribute work
    start = time.time()
    futures = [w.process_trajectories.remote(test_trajectories) for w in workers]
    results = ray.get(futures)
    elapsed = time.time() - start
    
    print(f"Processed {sum(len(r) for r in results)} preference pairs in {elapsed*1000:.2f}ms")
    
    # Training step
    train_futures = [w.train.remote() for w in workers]
    losses = ray.get(train_futures)
    print(f"Training losses: {losses}")
    
    ray.shutdown()
