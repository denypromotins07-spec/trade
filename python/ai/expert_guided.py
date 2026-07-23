"""
Expert-Guided Reinforcement Learning with SMC Rules

This module implements Expert-Guided RL where hard-coded Smart Money Concepts (SMC) rules 
act as the expert to shape the reward function, strictly avoiding LLM latency while 
enforcing market geometry logic.

Optimized for:
- Microsecond decision latency
- 4GB Python RAM quota enforcement
- AMD ROCm/DirectML acceleration checks
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import ray
import torch
import torch.nn as nn
import os


@dataclass
class SMCSignal:
    """Smart Money Concepts signal structure"""
    signal_type: str  # 'order_block', 'fair_value_gap', 'liquidity_sweep', 'breaker'
    direction: int  # 1 for bullish, -1 for bearish
    strength: float  # 0.0 to 1.0
    price_level: float
    timestamp_ns: int
    confidence: float = 0.0
    expiry_ns: int = 0


@dataclass
class ExpertGuidance:
    """Expert guidance container for RL reward shaping"""
    action_mask: Optional[np.ndarray] = None
    reward_shaping: float = 0.0
    priority_override: Optional[int] = None
    risk_adjustment: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartMoneyConceptsExpert:
    """
    Hard-coded SMC rule engine that provides expert guidance to RL agent.
    Avoids LLM latency by using deterministic rule-based logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # SMC parameters
        self.order_block_threshold = self.config.get('order_block_threshold', 0.7)
        self.fvg_size_threshold = self.config.get('fvg_size_threshold', 0.001)  # 0.1%
        self.liquidity_sweep_multiplier = self.config.get('liquidity_sweep_multiplier', 1.5)
        
        # Recent signals cache (bounded for memory)
        self.max_cached_signals = 100
        self.recent_signals: List[SMCSignal] = []
        
    def detect_order_blocks(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> List[SMCSignal]:
        """
        Detect order blocks based on significant candle patterns.
        Order blocks are areas where institutional orders were placed.
        """
        signals = []
        n = len(closes)
        if n < 5:
            return signals
        
        # Bullish order block: strong up candle after downtrend
        for i in range(2, n - 1):
            # Check for downtrend
            prev_downtrend = closes[i-2] > closes[i-1] > closes[i]
            
            # Strong bullish candle
            candle_range = highs[i] - lows[i]
            body = closes[i] - opens[i] if (opens := getattr(self, '_opens', closes)) is not None else closes[i] - lows[i]
            body_ratio = abs(body) / max(candle_range, 1e-10)
            
            if prev_downtrend and body_ratio > 0.7 and body > 0:
                strength = min(body_ratio * (volumes[i] / max(np.mean(volumes[:i+1]), 1e-10)), 1.0)
                
                if strength > self.order_block_threshold:
                    signals.append(SMCSignal(
                        signal_type='order_block',
                        direction=1,
                        strength=strength,
                        price_level=lows[i],
                        timestamp_ns=i,
                        confidence=strength,
                        expiry_ns=i + 50
                    ))
        
        # Bearish order block: strong down candle after uptrend
        for i in range(2, n - 1):
            prev_uptrend = closes[i-2] < closes[i-1] < closes[i]
            
            candle_range = highs[i] - lows[i]
            body = closes[i] - opens[i] if (opens := getattr(self, '_opens', closes)) is not None else highs[i] - closes[i]
            body_ratio = abs(body) / max(candle_range, 1e-10)
            
            if prev_uptrend and body_ratio > 0.7 and body < 0:
                strength = min(body_ratio * (volumes[i] / max(np.mean(volumes[:i+1]), 1e-10)), 1.0)
                
                if strength > self.order_block_threshold:
                    signals.append(SMCSignal(
                        signal_type='order_block',
                        direction=-1,
                        strength=strength,
                        price_level=highs[i],
                        timestamp_ns=i,
                        confidence=strength,
                        expiry_ns=i + 50
                    ))
        
        return signals[-self.max_cached_signals:]
    
    def detect_fair_value_gaps(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> List[SMCSignal]:
        """
        Detect Fair Value Gaps (FVG) - imbalances between buying and selling pressure.
        """
        signals = []
        n = len(closes)
        if n < 3:
            return signals
        
        for i in range(1, n - 1):
            # Bullish FVG: current low > previous high with gap
            if lows[i] > highs[i-1]:
                gap_size = (lows[i] - highs[i-1]) / max(closes[i-1], 1e-10)
                
                if gap_size > self.fvg_size_threshold:
                    strength = min(gap_size / self.fvg_size_threshold, 1.0)
                    signals.append(SMCSignal(
                        signal_type='fair_value_gap',
                        direction=1,
                        strength=strength,
                        price_level=(lows[i] + highs[i-1]) / 2,
                        timestamp_ns=i,
                        confidence=strength,
                        expiry_ns=i + 30
                    ))
            
            # Bearish FVG: current high < previous low with gap
            if highs[i] < lows[i-1]:
                gap_size = (lows[i-1] - highs[i]) / max(closes[i-1], 1e-10)
                
                if gap_size > self.fvg_size_threshold:
                    strength = min(gap_size / self.fvg_size_threshold, 1.0)
                    signals.append(SMCSignal(
                        signal_type='fair_value_gap',
                        direction=-1,
                        strength=strength,
                        price_level=(highs[i] + lows[i-1]) / 2,
                        timestamp_ns=i,
                        confidence=strength,
                        expiry_ns=i + 30
                    ))
        
        return signals[-self.max_cached_signals:]
    
    def detect_liquidity_sweeps(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        lookback: int = 20
    ) -> List[SMCSignal]:
        """
        Detect liquidity sweeps - wicks that take out recent highs/lows but close back inside range.
        """
        signals = []
        n = len(closes)
        if n < lookback:
            return signals
        
        for i in range(lookback, n):
            # Recent range
            recent_high = np.max(highs[i-lookback:i])
            recent_low = np.min(lows[i-lookback:i])
            
            # Bullish sweep: took out low but closed above
            if lows[i] < recent_low and closes[i] > recent_low:
                sweep_depth = (recent_low - lows[i]) / max(recent_low, 1e-10)
                recovery = (closes[i] - recent_low) / max(recent_low, 1e-10)
                
                strength = min(sweep_depth * self.liquidity_sweep_multiplier, 1.0)
                signals.append(SMCSignal(
                    signal_type='liquidity_sweep',
                    direction=1,
                    strength=strength,
                    price_level=lows[i],
                    timestamp_ns=i,
                    confidence=strength,
                    expiry_ns=i + 20
                ))
            
            # Bearish sweep: took out high but closed below
            if highs[i] > recent_high and closes[i] < recent_high:
                sweep_depth = (highs[i] - recent_high) / max(recent_high, 1e-10)
                recovery = (recent_high - closes[i]) / max(recent_high, 1e-10)
                
                strength = min(sweep_depth * self.liquidity_sweep_multiplier, 1.0)
                signals.append(SMCSignal(
                    signal_type='liquidity_sweep',
                    direction=-1,
                    strength=strength,
                    price_level=highs[i],
                    timestamp_ns=i,
                    confidence=strength,
                    expiry_ns=i + 20
                ))
        
        return signals[-self.max_cached_signals:]
    
    def generate_expert_guidance(
        self,
        current_price: float,
        position_direction: int,
        action_space_size: int
    ) -> ExpertGuidance:
        """
        Generate expert guidance based on active SMC signals.
        """
        # Filter active signals
        active_signals = [
            s for s in self.recent_signals
            if s.expiry_ns > 0  # Simplified expiry check
        ]
        
        if not active_signals:
            return ExpertGuidance()
        
        # Aggregate signals by direction
        bullish_strength = sum(s.strength * s.confidence for s in active_signals if s.direction == 1)
        bearish_strength = sum(s.strength * s.confidence for s in active_signals if s.direction == -1)
        
        net_signal = bullish_strength - bearish_strength
        
        # Create action mask based on expert opinion
        action_mask = np.ones(action_space_size, dtype=np.float32)
        
        if net_signal > 0.3:
            # Expert suggests long - mask out sell actions
            action_mask[action_space_size // 3:] = 0.3  # Reduce probability of neutral/sell
        elif net_signal < -0.3:
            # Expert suggests short - mask out buy actions
            action_mask[:action_space_size // 3] = 0.3  # Reduce probability of buy/neutral
        
        # Reward shaping based on alignment with expert
        if position_direction == 1 and net_signal > 0:
            reward_shaping = 0.1 * net_signal
        elif position_direction == -1 and net_signal < 0:
            reward_shaping = 0.1 * abs(net_signal)
        else:
            reward_shaping = -0.05 * abs(net_signal)  # Penalty for going against expert
        
        return ExpertGuidance(
            action_mask=action_mask,
            reward_shaping=reward_shaping,
            risk_adjustment=1.0 + abs(net_signal) * 0.2
        )


class ExpertGuidedRLAgent:
    """
    RL Agent with expert guidance integration.
    Uses SMC rules to shape rewards and mask actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expert_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Check for AMD ROCm/DirectML availability
        self.device = self._select_device(device)
        
        # Initialize expert system
        self.expert = SmartMoneyConceptsExpert(expert_config)
        
        # Simple policy network (would be larger in production)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)
        
        # Memory tracking for 4GB quota
        self.memory_used_mb = 0
        self.max_memory_mb = 4096  # 4GB limit
        
    def _select_device(self, requested_device: Optional[str]) -> str:
        """Select best available device with AMD ROCm/DirectML checks."""
        if requested_device:
            return requested_device
        
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            return 'cuda'
        
        # Check for DirectML (Windows AMD acceleration)
        try:
            import torch_directml
            return 'dml'
        except ImportError:
            pass
        
        # Check for ROCm (AMD Linux)
        if torch.version.hip is not None:
            return 'cuda'  # PyTorch uses cuda device type for ROCm
        
        return 'cpu'
    
    def _check_memory_quota(self) -> bool:
        """Check if we're within 4GB Python RAM quota."""
        import gc
        
        # Estimate current memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.memory_used_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        return self.memory_used_mb < self.max_memory_mb * 0.95  # 95% threshold
    
    def select_action(
        self,
        state: np.ndarray,
        market_data: Dict[str, np.ndarray]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action with expert guidance.
        Returns action and metadata including expert signals.
        """
        # Run expert analysis
        highs = market_data.get('highs', np.array([]))
        lows = market_data.get('lows', np.array([]))
        closes = market_data.get('close', market_data.get('closes', np.array([])))
        volumes = market_data.get('volumes', np.array([]))
        
        # Detect SMC signals
        ob_signals = self.expert.detect_order_blocks(highs, lows, closes, volumes)
        fvg_signals = self.expert.detect_fair_value_gaps(highs, lows, closes)
        sweep_signals = self.expert.detect_liquidity_sweeps(highs, lows, closes)
        
        # Update recent signals cache
        self.expert.recent_signals = ob_signals + fvg_signals + sweep_signals
        self.expert.recent_signals = self.expert.recent_signals[-self.expert.max_cached_signals:]
        
        # Get expert guidance
        current_price = closes[-1] if len(closes) > 0 else 0.0
        guidance = self.expert.generate_expert_guidance(
            current_price=current_price,
            position_direction=0,  # Would track actual position
            action_space_size=self.action_dim
        )
        
        # Get policy output
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            
            # Apply expert action mask
            if guidance.action_mask is not None:
                mask_tensor = torch.FloatTensor(guidance.action_mask).to(self.device)
                logits = logits * mask_tensor
            
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        return action, {
            'expert_signals': len(self.expert.recent_signals),
            'guidance_strength': guidance.reward_shaping,
            'risk_adjustment': guidance.risk_adjustment
        }
    
    def compute_reward(
        self,
        raw_reward: float,
        trade_result: Dict[str, Any]
    ) -> float:
        """
        Compute shaped reward incorporating expert guidance.
        """
        position_dir = trade_result.get('direction', 0)
        
        # Get expert guidance for reward shaping
        guidance = self.expert.generate_expert_guidance(
            current_price=trade_result.get('price', 0),
            position_direction=position_dir,
            action_space_size=self.action_dim
        )
        
        # Apply reward shaping
        shaped_reward = raw_reward + guidance.reward_shaping
        
        # Apply risk adjustment
        shaped_reward *= guidance.risk_adjustment
        
        return shaped_reward
    
    def cleanup(self):
        """Cleanup to maintain memory quota."""
        import gc
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        gc.collect()


# Ray actor for distributed expert-guided RL
@ray.remote(max_calls=1000)
class DistributedExpertAgent:
    """Ray-distributed expert-guided RL agent with memory monitoring."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expert_config: Optional[Dict[str, Any]] = None
    ):
        self.agent = ExpertGuidedRLAgent(state_dim, action_dim, expert_config)
        self.episode_count = 0
        
    def run_step(self, state: np.ndarray, market_data: Dict[str, np.ndarray]) -> Tuple[int, Dict]:
        """Run single step with memory check."""
        if not self.agent._check_memory_quota():
            raise MemoryError("Exceeded 4GB Python RAM quota")
        
        action, metadata = self.agent.select_action(state, market_data)
        self.episode_count += 1
        
        return action, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'episodes': self.episode_count,
            'memory_mb': self.agent.memory_used_mb,
            'device': self.agent.device
        }


if __name__ == '__main__':
    # Example usage
    import time
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, _system_config={"object_store_memory": 1024*1024*1024})
    
    # Create distributed agents
    agents = [
        DistributedExpertAgent.remote(state_dim=64, action_dim=9)
        for _ in range(4)
    ]
    
    # Test run
    test_state = np.random.randn(64).astype(np.float32)
    test_market = {
        'highs': np.random.randn(100).cumsum() + 50000,
        'lows': np.random.randn(100).cumsum() + 49000,
        'closes': np.random.randn(100).cumsum() + 49500,
        'volumes': np.random.rand(100) * 1000
    }
    
    start = time.time()
    results = ray.get([agent.run_step.remote(test_state, test_market) for agent in agents])
    elapsed = time.time() - start
    
    print(f"Distributed expert-guided RL step completed in {elapsed*1000:.2f}ms")
    print(f"Results: {results}")
    
    ray.shutdown()
