"""
Automated Domain Randomization for Robust RL Training

This module implements domain randomization that dynamically injects
synthetic latency, slippage, and spread widening into the training
environment to ensure robust live execution.

Key features:
- Synthetic latency injection with realistic distributions
- Slippage modeling based on order size and volatility
- Spread widening simulation during stress periods
- Black-swan event injection for robustness testing
- Agent quarantine system for failed adaptation detection
- AMD DirectML/ROCm environment integration

Usage:
    randomizer = DomainRandomizer()
    obs = randomizer.randomize_observation(raw_obs)
    reward = randomizer.randomize_reward(raw_reward, action)
    if randomizer.should_quarantine(agent_id):
        handle_quarantined_agent(agent_id)
"""

import os
import time
import random
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

import torch
import numpy as np

# Import hardware detection from attention module
try:
    from .attention import DIRECTML_AVAILABLE, ROCM_AVAILABLE, RECOMMENDED_DEVICE
except ImportError:
    DIRECTML_AVAILABLE = False
    ROCM_AVAILABLE = False
    RECOMMENDED_DEVICE = "cpu"


class EventType(Enum):
    """Types of randomization events."""
    NORMAL = "normal"
    HIGH_LATENCY = "high_latency"
    HIGH_SLIPPAGE = "high_slippage"
    SPREAD_WIDENING = "spread_widening"
    BLACK_SWAN = "black_swan"
    CONNECTION_ISSUES = "connection_issues"


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization."""
    # Latency parameters (milliseconds)
    base_latency_ms: float = 5.0
    max_latency_ms: float = 100.0
    latency_std_ms: float = 2.0
    
    # Slippage parameters (basis points)
    base_slippage_bps: float = 1.0
    max_slippage_bps: float = 50.0
    slippage_volatility_coef: float = 0.1
    
    # Spread parameters (basis points)
    base_spread_bps: float = 10.0
    max_spread_bps: float = 200.0
    spread_widening_prob: float = 0.05
    
    # Black swan parameters
    black_swan_prob: float = 0.001  # 0.1% chance per step
    black_swan_severity: float = 5.0  # Multiplier for all adverse effects
    
    # Quarantine parameters
    performance_threshold: float = -0.5  # Below this Sharpe ratio triggers quarantine
    quarantine_window_steps: int = 1000
    min_steps_before_evaluation: int = 100


@dataclass
class RandomizationState:
    """Current state of randomization."""
    current_event: EventType = EventType.NORMAL
    latency_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    event_duration_remaining: int = 0
    step_count: int = 0


class LatencyInjector:
    """
    Realistic network latency simulation.
    
    Models various latency components:
    - Network propagation delay
    - Exchange processing time
    - Order queue wait time
    """
    
    def __init__(self, config: RandomizationConfig):
        self.config = config
        self.base_latency = config.base_latency_ms
        
        # Historical latency tracking
        self.latency_history: deque = deque(maxlen=1000)
    
    def inject_latency(self, base_value: float) -> float:
        """
        Apply latency injection to a value.
        
        Returns delayed value in milliseconds.
        """
        # Base latency with Gaussian noise
        latency = self.base_latency + np.random.normal(0, self.config.latency_std_ms)
        
        # Add occasional spikes
        if random.random() < 0.01:  # 1% chance of spike
            latency += np.random.exponential(self.config.max_latency_ms / 10)
        
        latency = max(0, latency)
        self.latency_history.append(latency)
        
        return base_value + latency
    
    def get_simulated_latency(self) -> float:
        """Get current simulated latency for this step."""
        base = self.base_latency
        
        # Add some autocorrelation (real networks have memory)
        if len(self.latency_history) > 0:
            prev_latency = self.latency_history[-1]
            base = 0.7 * base + 0.3 * prev_latency
        
        noise = np.random.normal(0, self.config.latency_std_ms)
        return max(0, base + noise)
    
    def reset(self):
        """Reset latency tracker."""
        self.latency_history.clear()


class SlippageModel:
    """
    Realistic slippage simulation based on market microstructure.
    
    Slippage depends on:
    - Order size relative to average volume
    - Current volatility
    - Market depth (simulated)
    """
    
    def __init__(self, config: RandomizationConfig):
        self.config = config
        self.base_slippage = config.base_slippage_bps
        
        # Simulated market depth
        self.market_depth_factor = 1.0
        
        # Volatility estimate
        self.current_volatility = 0.02  # 2% daily
    
    def calculate_slippage(
        self,
        order_size: float,
        avg_volume: float,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate slippage in basis points.
        
        Args:
            order_size: Size of the order
            avg_volume: Average market volume
            volatility: Current volatility (uses internal estimate if None)
        
        Returns:
            Slippage in basis points
        """
        vol = volatility if volatility is not None else self.current_volatility
        
        # Base slippage scaled by order size ratio
        size_ratio = order_size / max(avg_volume, 1.0)
        base_slip = self.base_slippage * (1 + size_ratio * 10)
        
        # Volatility adjustment
        vol_adjustment = 1 + (vol / 0.02 - 1) * self.config.slippage_volatility_coef
        
        # Random component
        random_component = np.random.exponential(self.base_slippage / 2)
        
        total_slippage = base_slip * vol_adjustment + random_component
        
        return min(total_slippage, self.config.max_slippage_bps)
    
    def update_volatility(self, volatility: float):
        """Update current volatility estimate."""
        # Exponential moving average
        self.current_volatility = 0.9 * self.current_volatility + 0.1 * volatility


class SpreadSimulator:
    """
    Bid-ask spread simulation with dynamic widening.
    
    Models spread behavior during:
    - Normal market conditions
    - High volatility periods
    - Liquidity crises
    """
    
    def __init__(self, config: RandomizationConfig):
        self.config = config
        self.base_spread = config.base_spread_bps
        
        # Spread regime tracking
        self.current_regime = "normal"
        self.regime_duration = 0
    
    def get_spread(self, volatility: float = 0.02) -> float:
        """
        Get current simulated spread.
        
        Args:
            volatility: Current market volatility
        
        Returns:
            Spread in basis points
        """
        # Base spread adjusted for volatility
        vol_factor = volatility / 0.02
        base = self.base_spread * vol_factor
        
        # Regime-based adjustments
        if self.current_regime == "stressed":
            base *= 2.0
        elif self.current_regime == "crisis":
            base *= 5.0
        
        # Small random fluctuation
        noise = np.random.normal(0, base * 0.1)
        
        return max(self.base_spread, min(base + noise, self.config.max_spread_bps))
    
    def trigger_regime_change(self, regime: str, duration: int):
        """Trigger a spread regime change."""
        self.current_regime = regime
        self.regime_duration = duration
    
    def step(self):
        """Advance spread simulation by one step."""
        if self.regime_duration > 0:
            self.regime_duration -= 1
            if self.regime_duration == 0:
                self.current_regime = "normal"


class BlackSwanGenerator:
    """
    Black swan event generator for stress testing.
    
    Generates rare, severe market events:
    - Flash crashes
    - Liquidity evaporation
    - Extreme volatility spikes
    - Exchange outages
    """
    
    def __init__(self, config: RandomizationConfig):
        self.config = config
        self.event_history: List[Dict[str, Any]] = []
        self.last_event_step = -10000
    
    def check_for_event(self, step: int) -> Optional[EventType]:
        """
        Check if a black swan event should occur.
        
        Returns:
            EventType if event triggered, None otherwise
        """
        if step - self.last_event_step < 1000:  # Minimum 1000 steps between events
            return None
        
        if random.random() < self.config.black_swan_prob:
            event_type = random.choice([
                EventType.FLASH_CRASH,
                EventType.LIQUIDITY_CRISIS,
                EventType.VOLATILITY_SPIKE,
                EventType.EXCHANGE_OUTAGE,
            ])
            
            self.event_history.append({
                'type': event_type,
                'step': step,
                'severity': random.uniform(1.0, self.config.black_swan_severity),
            })
            self.last_event_step = step
            
            return event_type
        
        return None
    
    def get_event_effects(self, event_type: EventType) -> Dict[str, float]:
        """
        Get the effects of a black swan event.
        
        Returns multipliers for various market parameters.
        """
        severity = self.event_history[-1]['severity'] if self.event_history else 1.0
        
        effects = {
            'latency_mult': 1.0,
            'slippage_mult': 1.0,
            'spread_mult': 1.0,
            'volatility_mult': 1.0,
        }
        
        if event_type == EventType.FLASH_CRASH:
            effects['volatility_mult'] = 10.0 * severity
            effects['slippage_mult'] = 5.0 * severity
        elif event_type == EventType.LIQUIDITY_CRISIS:
            effects['spread_mult'] = 10.0 * severity
            effects['slippage_mult'] = 8.0 * severity
        elif event_type == EventType.VOLATILITY_SPIKE:
            effects['volatility_mult'] = 5.0 * severity
            effects['spread_mult'] = 3.0 * severity
        elif event_type == EventType.EXCHANGE_OUTAGE:
            effects['latency_mult'] = 100.0 * severity
        
        return effects


class PerformanceTracker:
    """
    Track agent performance for quarantine decisions.
    
    Monitors:
    - Cumulative returns
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    """
    
    def __init__(self, config: RandomizationConfig):
        self.config = config
        self.returns: deque = deque(maxlen=config.quarantine_window_steps)
        self.cumulative_return = 0.0
        self.peak_return = 0.0
        self.max_drawdown = 0.0
    
    def record_step(self, reward: float):
        """Record a step's reward."""
        self.returns.append(reward)
        self.cumulative_return += reward
        
        # Update peak and drawdown
        if self.cumulative_return > self.peak_return:
            self.peak_return = self.cumulative_return
        
        current_drawdown = self.peak_return - self.cumulative_return
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio over the window."""
        if len(self.returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(self.returns)
        std_return = statistics.stdev(self.returns) if len(self.returns) > 1 else 1.0
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 252 trading days, scaled)
        return (mean_return / std_return) * np.sqrt(len(self.returns))
    
    def should_quarantine(self) -> bool:
        """Check if agent should be quarantined."""
        if len(self.returns) < self.config.min_steps_before_evaluation:
            return False
        
        sharpe = self.calculate_sharpe()
        return sharpe < self.config.performance_threshold
    
    def reset(self):
        """Reset performance tracker."""
        self.returns.clear()
        self.cumulative_return = 0.0
        self.peak_return = 0.0
        self.max_drawdown = 0.0


class DomainRandomizer:
    """
    Main domain randomization controller.
    
    Coordinates all randomization components and manages
    agent quarantine decisions.
    """
    
    def __init__(self, config: Optional[RandomizationConfig] = None):
        self.config = config or RandomizationConfig()
        self.state = RandomizationState()
        
        # Initialize components
        self.latency_injector = LatencyInjector(self.config)
        self.slippage_model = SlippageModel(self.config)
        self.spread_simulator = SpreadSimulator(self.config)
        self.black_swan_generator = BlackSwanGenerator(self.config)
        
        # Agent performance tracking
        self.agent_trackers: Dict[str, PerformanceTracker] = {}
        self.quarantined_agents: set = set()
        
        # Device configuration
        self.device = RECOMMENDED_DEVICE
    
    def randomize_observation(
        self,
        observation: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Apply randomization to observation.
        
        Primarily adds observation noise and latency effects.
        """
        if not add_noise:
            return observation
        
        # Ensure tensor is on correct device
        if isinstance(observation, torch.Tensor):
            if observation.device.type != self.device:
                observation = observation.to(self.device)
        
        # Add small Gaussian noise
        noise_scale = 0.001 * self.state.latency_multiplier
        if isinstance(observation, torch.Tensor):
            noise = torch.randn_like(observation) * noise_scale
            return observation + noise
        else:
            noise = np.random.randn(*observation.shape) * noise_scale
            return observation + noise
    
    def randomize_reward(
        self,
        reward: float,
        action: Optional[Any] = None,
        order_size: float = 1.0,
    ) -> float:
        """
        Apply slippage and transaction costs to reward.
        
        This makes the training environment more realistic.
        """
        # Calculate slippage cost
        slippage_bps = self.slippage_model.calculate_slippage(order_size, avg_volume=100.0)
        slippage_cost = reward * (slippage_bps / 10000.0) * self.state.slippage_multiplier
        
        # Apply spread cost (implicit transaction cost)
        spread_cost = abs(reward) * (self.spread_simulator.get_spread() / 10000.0)
        
        # Total adjusted reward
        adjusted_reward = reward - slippage_cost - spread_cost
        
        return adjusted_reward
    
    def apply_latency(
        self,
        timestamp: float,
        operation: str = "order",
    ) -> float:
        """
        Apply latency to a timestamp.
        
        Returns the delayed timestamp.
        """
        base_latency = self.latency_injector.get_simulated_latency()
        
        # Apply event multiplier
        effective_latency = base_latency * self.state.latency_multiplier
        
        # Add operation-specific latency
        if operation == "cancel":
            effective_latency *= 0.5  # Cancels are faster
        elif operation == "query":
            effective_latency *= 0.3  # Queries are fastest
        
        return timestamp + effective_latency / 1000.0  # Convert ms to seconds
    
    def step(self, step_count: int, volatility: float = 0.02):
        """
        Advance randomization state by one step.
        
        Should be called at each environment step.
        """
        self.state.step_count = step_count
        
        # Update slippage model volatility
        self.slippage_model.update_volatility(volatility)
        
        # Advance spread simulator
        self.spread_simulator.step()
        
        # Check for black swan events
        event = self.black_swan_generator.check_for_event(step_count)
        if event is not None:
            self._trigger_event(event)
        
        # Decay event multipliers if in an event
        if self.state.event_duration_remaining > 0:
            self.state.event_duration_remaining -= 1
            if self.state.event_duration_remaining <= 0:
                self._reset_event_state()
    
    def _trigger_event(self, event_type: EventType):
        """Trigger a randomization event."""
        self.state.current_event = event_type
        
        if event_type == EventType.BLACK_SWAN:
            effects = self.black_swan_generator.get_event_effects(event_type)
            self.state.latency_multiplier = effects['latency_mult']
            self.state.slippage_multiplier = effects['slippage_mult']
            self.state.spread_multiplier = effects['spread_mult']
            self.state.event_duration_remaining = 100  # Event lasts 100 steps
        else:
            # Standard event
            duration = random.randint(10, 50)
            self.state.event_duration_remaining = duration
            
            if event_type == EventType.HIGH_LATENCY:
                self.state.latency_multiplier = random.uniform(2.0, 5.0)
            elif event_type == EventType.HIGH_SLIPPAGE:
                self.state.slippage_multiplier = random.uniform(2.0, 4.0)
            elif event_type == EventType.SPREAD_WIDENING:
                self.state.spread_multiplier = random.uniform(2.0, 5.0)
                self.spread_simulator.trigger_regime_change("stressed", duration)
    
    def _reset_event_state(self):
        """Reset to normal event state."""
        self.state.current_event = EventType.NORMAL
        self.state.latency_multiplier = 1.0
        self.state.slippage_multiplier = 1.0
        self.state.spread_multiplier = 1.0
    
    def record_agent_performance(self, agent_id: str, reward: float):
        """Record performance for an agent."""
        if agent_id not in self.agent_trackers:
            self.agent_trackers[agent_id] = PerformanceTracker(self.config)
        
        self.agent_trackers[agent_id].record_step(reward)
    
    def should_quarantine(self, agent_id: str) -> bool:
        """Check if an agent should be quarantined."""
        if agent_id not in self.agent_trackers:
            return False
        
        if self.agent_trackers[agent_id].should_quarantine():
            self.quarantined_agents.add(agent_id)
            return True
        
        return False
    
    def is_quarantined(self, agent_id: str) -> bool:
        """Check if an agent is currently quarantined."""
        return agent_id in self.quarantined_agents
    
    def release_from_quarantine(self, agent_id: str):
        """Release an agent from quarantine."""
        if agent_id in self.quarantined_agents:
            self.quarantined_agents.remove(agent_id)
            if agent_id in self.agent_trackers:
                self.agent_trackers[agent_id].reset()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current randomization state for logging."""
        return {
            'event': self.state.current_event.value,
            'latency_mult': self.state.latency_multiplier,
            'slippage_mult': self.state.slippage_multiplier,
            'spread_mult': self.state.spread_multiplier,
            'quarantined_agents': list(self.quarantined_agents),
        }
    
    def reset(self):
        """Reset all randomization state."""
        self.state = RandomizationState()
        self.latency_injector.reset()
        for tracker in self.agent_trackers.values():
            tracker.reset()
        self.quarantined_agents.clear()


def create_randomizer(
    config: Optional[RandomizationConfig] = None,
    enable_black_swan: bool = True,
) -> DomainRandomizer:
    """
    Factory function to create domain randomizer.
    
    Configures appropriate settings for HFT training.
    """
    if config is None:
        config = RandomizationConfig()
    
    if not enable_black_swan:
        config.black_swan_prob = 0.0
    
    return DomainRandomizer(config)


if __name__ == "__main__":
    # Test domain randomization
    print("Testing Domain Randomization Module...")
    print(f"DirectML Available: {DIRECTML_AVAILABLE}")
    print(f"ROCm Available: {ROCM_AVAILABLE}")
    
    # Create randomizer
    randomizer = create_randomizer(enable_black_swan=True)
    
    # Simulate some steps
    print("\nSimulating 100 steps...")
    for step in range(100):
        randomizer.step(step, volatility=0.02 + random.random() * 0.01)
        
        # Record some agent performance
        reward = random.gauss(0.001, 0.01)
        randomizer.record_agent_performance("agent_1", reward)
        
        if step % 20 == 0:
            state = randomizer.get_current_state()
            print(f"Step {step}: Event={state['event']}, Latency mult={state['latency_mult']:.2f}")
    
    # Test quarantine system
    print("\nTesting quarantine system...")
    bad_agent_rewards = [-0.01] * 150  # Consistently bad performance
    for i, reward in enumerate(bad_agent_rewards):
        randomizer.record_agent_performance("bad_agent", reward)
        randomizer.step(100 + i, volatility=0.02)
    
    if randomizer.should_quarantine("bad_agent"):
        print("Bad agent has been quarantined!")
    
    print(f"\nQuarantined agents: {randomizer.quarantined_agents}")
    print("\nDomain Randomization test complete!")
