"""
Multi-Agent Reinforcement Learning (MARL) Framework using Ray RLlib

This module implements a MARL architecture where distinct agents specialize in:
- Trend-following strategies
- Mean-reversion strategies  
- Market-making strategies

Optimized for AMD Ryzen AI 5 with DirectML/ROCm acceleration checks.
Respects strict 4GB Python RAM quota during Ray distribution.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
import gymnasium as gym
from gymnasium import spaces

# AMD DirectML/ROCm environment check
def check_amd_acceleration() -> Dict[str, Any]:
    """Check for AMD DirectML/ROCm availability and return configuration."""
    config = {
        "directml_available": False,
        "rocm_available": False,
        "gpu_device": None,
        "recommended_backend": "cpu"
    }
    
    try:
        # Check for DirectML (Windows)
        if os.name == 'nt':
            try:
                import torch
                if torch.cuda.is_available():
                    config["directml_available"] = True
                    config["gpu_device"] = "DirectML"
                    config["recommended_backend"] = "cuda"
            except ImportError:
                pass
        
        # Check for ROCm (Linux with AMD GPU)
        try:
            import torch
            if torch.version.hip is not None:
                config["rocm_available"] = True
                config["gpu_device"] = f"ROCm ({torch.cuda.get_device_name(0)})"
                config["recommended_backend"] = "cuda"
        except (ImportError, AttributeError):
            pass
            
    except Exception as e:
        print(f"[WARN] AMD acceleration check failed: {e}")
    
    return config


class CryptoTradingEnv(gym.Env):
    """
    Gymnasium environment for crypto trading simulation.
    Supports multiple agents with different strategy specializations.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_balance: float = 100000.0,
        max_steps: int = 10000,
        agent_type: str = "trend",  # trend, mean_revert, market_maker
    ):
        super().__init__()
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.agent_type = agent_type
        
        # Action space: [position_size, limit_price_offset]
        # position_size: -1.0 (full short) to 1.0 (full long)
        # limit_price_offset: -0.05 to 0.05 (±5% price offset)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -0.05], dtype=np.float32),
            high=np.array([1.0, 0.05], dtype=np.float32),
        )
        
        # Observation space varies by agent type
        obs_dim = 20  # Base observation dimension
        if agent_type == "market_maker":
            obs_dim += 10  # Additional order book features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Reset internal state variables."""
        self.balance = self.initial_balance
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.current_step = 0
        self.total_pnl = 0.0
        self.trades = []
        self.price_history = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self._reset_state()
        
        # Generate initial observation
        obs = self._generate_observation()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [position_size, limit_price_offset]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Simulate price movement (random walk with drift based on regime)
        current_price = self._simulate_price()
        
        # Execute action
        target_position, price_offset = action
        execution_price = current_price * (1 + price_offset)
        
        # Calculate position change
        position_delta = target_position - self.position
        
        # Apply transaction costs (maker/taker fees)
        fee_rate = 0.0004 if abs(price_offset) < 0.001 else 0.0001  # Taker vs Maker
        transaction_cost = abs(position_delta) * execution_price * fee_rate
        
        # Update balance and position
        if position_delta != 0:
            cost = position_delta * execution_price + transaction_cost
            self.balance -= cost
            
            if self.position == 0:
                self.avg_entry_price = execution_price
            else:
                # Update average entry price
                total_value = self.position * self.avg_entry_price + position_delta * execution_price
                self.position += position_delta
                if self.position != 0:
                    self.avg_entry_price = total_value / self.position
            self.position = target_position
            
            self.trades.append({
                "step": self.current_step,
                "price": execution_price,
                "size": position_delta,
                "fee": transaction_cost
            })
        
        # Calculate unrealized PnL
        unrealized_pnl = self.position * (current_price - self.avg_entry_price)
        self.total_pnl = self.balance - self.initial_balance + unrealized_pnl
        
        # Calculate reward (PnL change with risk penalty)
        reward = self._calculate_reward(current_price, action)
        
        # Generate new observation
        obs = self._generate_observation()
        
        # Check termination
        terminated = self.balance <= 0 or self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "pnl": self.total_pnl,
            "balance": self.balance,
            "position": self.position,
            "price": current_price,
            "step": self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_price(self) -> float:
        """Simulate price movement with regime-dependent dynamics."""
        base_vol = 0.001  # Base volatility
        
        if len(self.price_history) == 0:
            price = 50000.0  # Initial BTC price
        else:
            last_price = self.price_history[-1]
            
            # Add regime-dependent drift
            if self.agent_type == "trend":
                drift = 0.0001 * np.sign(last_price - self.avg_entry_price)
            elif self.agent_type == "mean_revert":
                drift = -0.0002 * (last_price - 50000) / 50000
            else:
                drift = 0.0
            
            noise = np.random.normal(0, base_vol)
            price = last_price * (1 + drift + noise)
        
        self.price_history.append(price)
        return price
    
    def _generate_observation(self) -> np.ndarray:
        """Generate observation vector for the agent."""
        obs = []
        
        # Price features
        if len(self.price_history) > 0:
            current_price = self.price_history[-1]
            obs.extend([
                current_price / 50000,  # Normalized price
                (current_price - self.avg_entry_price) / current_price if self.avg_entry_price > 0 else 0,
                self.balance / self.initial_balance,
                self.position,
                self.total_pnl / self.initial_balance,
            ])
        else:
            obs.extend([0.0] * 5)
        
        # Technical indicators (simplified)
        if len(self.price_history) >= 20:
            recent = self.price_history[-20:]
            ma_20 = np.mean(recent)
            std_20 = np.std(recent)
            obs.extend([
                (self.price_history[-1] - ma_20) / ma_20 if ma_20 > 0 else 0,
                std_20 / ma_20 if ma_20 > 0 else 0,
            ])
        else:
            obs.extend([0.0, 0.0])
        
        # Fill remaining dimensions with zeros or additional features
        while len(obs) < self.observation_space.shape[0]:
            obs.append(0.0)
        
        return np.array(obs[:self.observation_space.shape[0]], dtype=np.float32)
    
    def _calculate_reward(self, current_price: float, action: np.ndarray) -> float:
        """Calculate reward based on PnL and risk metrics."""
        # Base reward: PnL change
        pnl_reward = self.total_pnl / self.initial_balance
        
        # Risk penalty: position size and drawdown
        risk_penalty = abs(self.position) * 0.001
        
        # Transaction cost penalty
        fee_penalty = abs(action[0]) * 0.0004
        
        # Sharpe-like component
        if len(self.price_history) > 10:
            returns = np.diff(self.price_history[-10:]) / self.price_history[-10:-1]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                pnl_reward += sharpe * 0.01
        
        return pnl_reward - risk_penalty - fee_penalty


class MultiAgentCryptoEnv(MultiAgentEnv):
    """
    Multi-agent environment wrapping individual trading environments.
    Each agent specializes in a different strategy type.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        agents_per_symbol: int = 3,
    ):
        super().__init__()
        
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.agents_per_symbol = agents_per_symbol
        
        # Create environments for each agent
        self.envs: Dict[str, CryptoTradingEnv] = {}
        self.agent_ids: List[str] = []
        
        agent_types = ["trend", "mean_revert", "market_maker"]
        
        for symbol in self.symbols:
            for i, agent_type in enumerate(agent_types[:agents_per_symbol]):
                agent_id = f"{symbol}_{agent_type}"
                self.envs[agent_id] = CryptoTradingEnv(
                    symbol=symbol,
                    agent_type=agent_type
                )
                self.agent_ids.append(agent_id)
        
        # Build observation/action spaces mapping
        self.observation_space = self.envs[self.agent_ids[0]].observation_space
        self.action_space = self.envs[self.agent_ids[0]].action_space
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset all agent environments."""
        observations = {}
        infos = {}
        
        for agent_id in self.agent_ids:
            obs, info = self.envs[agent_id].reset(seed=seed)
            observations[agent_id] = obs
            infos[agent_id] = info
        
        return observations, infos
    
    def step(
        self, 
        actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """Step all agent environments."""
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        for agent_id, action in actions.items():
            if agent_id in self.envs:
                result = self.envs[agent_id].step(action)
                obs, reward, terminated, truncated, info = result
                
                observations[agent_id] = obs
                rewards[agent_id] = reward
                terminateds[agent_id] = terminated
                truncateds[agent_id] = truncated
                infos[agent_id] = info
        
        return observations, rewards, terminateds, truncateds, infos


def build_marl_config(
    num_agents: int = 9,
    use_gpu: bool = False,
    ram_limit_gb: float = 4.0,
) -> PPOConfig:
    """
    Build Ray RLlib configuration for multi-agent training.
    
    Args:
        num_agents: Total number of agents across all symbols
        use_gpu: Whether to use GPU acceleration
        ram_limit_gb: Memory limit for Ray workers
    
    Returns:
        Configured PPOConfig object
    """
    # Check AMD acceleration
    amd_config = check_amd_acceleration()
    if amd_config["recommended_backend"] == "cuda" and not use_gpu:
        print(f"[INFO] AMD acceleration available: {amd_config['gpu_device']}")
        use_gpu = True
    
    # Calculate memory per worker
    num_workers = min(4, num_agents)
    memory_per_worker = (ram_limit_gb * 1024) / (num_workers + 1)  # MB
    
    config = (
        PPOConfig()
        .environment(env=MultiAgentCryptoEnv)
        .rollouts(num_rollout_workers=num_workers)
        .training(
            model={
                "fcnet_hiddens": [256, 128, 64],
                "fcnet_activation": "relu",
                "use_lstm": False,
            },
            train_batch_size=4096,
            grad_clip=1.0,
            kl_coeff=0.2,
        )
        .resources(
            num_gpus=1 if use_gpu else 0,
            num_cpus_per_worker=2,
            num_gpus_per_worker=0.25 if use_gpu else 0,
            memory=memory_per_worker,
            object_store_memory=memory_per_worker * 0.5,
        )
        .multi_agent(
            policies={
                f"trend_policy": PolicySpec(None),
                f"mean_revert_policy": PolicySpec(None),
                f"market_maker_policy": PolicySpec(None),
            },
            policy_mapping_fn=lambda agent_id, episode: (
                "trend_policy" if "trend" in agent_id else
                "mean_revert_policy" if "mean_revert" in agent_id else
                "market_maker_policy"
            ),
        )
    )
    
    return config


def train_marl_agents(
    num_episodes: int = 100,
    checkpoint_dir: str = "/tmp/marl_checkpoints",
    ram_limit_gb: float = 4.0,
) -> str:
    """
    Train multi-agent reinforcement learning agents.
    
    Args:
        num_episodes: Number of training episodes
        checkpoint_dir: Directory to save checkpoints
        ram_limit_gb: Memory limit for Ray
    
    Returns:
        Path to final checkpoint
    """
    # Initialize Ray with memory limits
    if not ray.is_initialized():
        ray.init(
            object_store_memory=int(ram_limit_gb * 1024**3 * 0.3),
            _system_memory=int(ram_limit_gb * 1024**3),
        )
    
    # Build configuration
    config = build_marl_config(ram_limit_gb=ram_limit_gb)
    
    # Create trainer
    algo = config.build()
    
    # Training loop
    for episode in range(num_episodes):
        result = algo.train()
        
        if episode % 10 == 0:
            print(f"Episode {episode}: reward={result['episode_reward_mean']:.4f}")
    
    # Save final checkpoint
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"Training complete. Checkpoint saved to: {checkpoint_path}")
    
    ray.shutdown()
    return checkpoint_path


if __name__ == "__main__":
    # Example usage
    print("Checking AMD acceleration...")
    amd_info = check_amd_acceleration()
    print(f"AMD Config: {amd_info}")
    
    print("\nInitializing MARL training...")
    checkpoint = train_marl_agents(num_episodes=50, ram_limit_gb=4.0)
    print(f"Training completed: {checkpoint}")
