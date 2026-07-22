"""
Ensemble Machine Learning - Multi-Agent Meta-Strategy with Ray

This module combines multiple weak RL agents into a strong meta-agent using Ray,
applying stacking techniques to dynamically weight predictions based on current
market regime confidence. Optimized for AMD Ryzen AI 5 with ROCm/DirectML support.

Features:
- Ray-based distributed ensemble training
- Dynamic agent weighting by regime confidence
- Stacking meta-learner for prediction fusion
- AMD ROCm/DirectML environment detection
- Memory-efficient worker management (4GB ceiling)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy import Policy
from ray.tune.registry import register_env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification for dynamic weighting"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"


@dataclass
class AgentConfig:
    """Configuration for individual ensemble agent"""
    agent_id: str
    model_type: str  # ppo, dqn, a2c, etc.
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    clip_param: float = 0.2
    hidden_size: int = 256
    num_layers: int = 2
    use_gpu: bool = False
    memory_limit_mb: int = 512  # Per-agent memory limit


@dataclass
class EnsemblePrediction:
    """Weighted ensemble prediction output"""
    action: int
    confidence: float
    action_probs: np.ndarray
    agent_weights: Dict[str, float]
    regime: MarketRegime
    timestamp_ns: int


def detect_amd_rocm() -> bool:
    """
    Detect AMD ROCm availability for GPU acceleration.
    Returns True if ROCm is available and properly configured.
    """
    try:
        # Check for ROCm environment variables
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        if not os.path.exists(rocm_path):
            logger.info(f"ROCm path not found at {rocm_path}")
            return False
        
        # Check for hip libraries
        hip_lib = os.path.join(rocm_path, 'lib', 'libhipblas.so')
        if not os.path.exists(hip_lib):
            logger.info("HIP BLAS library not found")
            return False
        
        # Try importing torch with ROCm
        try:
            import torch
            if torch.version.hip is not None:
                logger.info(f"ROCm detected: {torch.version.hip}")
                return True
        except ImportError:
            pass
        
        logger.info("ROCm detected but PyTorch ROCm not available")
        return False
        
    except Exception as e:
        logger.warning(f"Error detecting ROCm: {e}")
        return False


def detect_directml() -> bool:
    """
    Detect Microsoft DirectML availability for AMD GPU on Windows.
    Returns True if DirectML is available.
    """
    try:
        # Check for DirectML package
        import torch
        try:
            import torch_directml
            logger.info("DirectML detected")
            return True
        except ImportError:
            logger.info("DirectML not available (torch_directml not installed)")
            return False
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Error detecting DirectML: {e}")
        return False


def get_device_config() -> Dict[str, Any]:
    """
    Get optimal device configuration based on available hardware.
    Prioritizes ROCm > DirectML > CPU for AMD Ryzen AI 5.
    """
    config = {
        "device": "cpu",
        "use_gpu": False,
        "gpu_device_id": 0,
    }
    
    if detect_amd_rocm():
        config["device"] = "cuda"  # PyTorch uses 'cuda' for ROCm too
        config["use_gpu"] = True
        logger.info("Using AMD ROCm for GPU acceleration")
    elif detect_directml():
        config["device"] = "directml"
        config["use_gpu"] = True
        logger.info("Using DirectML for GPU acceleration")
    else:
        logger.info("Using CPU (no GPU acceleration available)")
    
    return config


@ray.remote(max_calls=1000)  # Restart workers after 1000 calls to prevent memory leaks
class EnsembleAgent:
    """
    Individual agent in the ensemble.
    Runs as a Ray actor for distributed training and inference.
    """
    
    def __init__(self, config: AgentConfig, env_config: Dict[str, Any]):
        self.config = config
        self.env_config = env_config
        self.agent_id = config.agent_id
        self.model_type = config.model_type
        self.training_steps = 0
        self.last_confidence = 0.0
        self.regime_specialization: Optional[MarketRegime] = None
        
        # Enforce memory limit
        memory_limit = config.memory_limit_mb * 1024 * 1024
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            logger.info(f"Agent {self.agent_id} memory limited to {config.memory_limit_mb}MB")
        except Exception as e:
            logger.warning(f"Could not set memory limit: {e}")
        
        # Initialize RL trainer
        self._init_trainer()
    
    def _init_trainer(self):
        """Initialize the RL trainer based on model type"""
        device_config = get_device_config()
        
        ray_config = {
            "env": self.env_config.get("env_name", "CryptoTradingEnv"),
            "gamma": self.config.gamma,
            "lr": self.config.learning_rate,
            "train_batch_size": 4000,
            "num_gpus": 1 if device_config["use_gpu"] else 0,
            "framework": "torch",
            "entropy_coeff": self.config.entropy_coeff,
            "clip_param": self.config.clip_param,
            "model": {
                "fcnet_hiddens": [self.config.hidden_size] * self.config.num_layers,
                "use_lstm": False,
            },
            "rollout_fragment_length": 1000,
            "batch_mode": "truncate_episodes",
        }
        
        if self.model_type == "ppo":
            self.trainer = PPOTrainer(config=ray_config)
        else:
            # Default to PPO for now
            self.trainer = PPOTrainer(config=ray_config)
        
        logger.info(f"Agent {self.agent_id} initialized with {self.model_type}")
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """
        Generate prediction for given observation.
        Returns action and confidence scores.
        """
        if self.trainer is None:
            return 0, {"confidence": 0.0}
        
        result = self.trainer.compute_single_action(observation)
        action = result[0]
        action_info = result[2] if len(result) > 2 else {}
        
        # Extract confidence from action distribution
        if "action_dist_inputs" in action_info:
            dist_inputs = action_info["action_dist_inputs"]
            confidence = float(np.max(dist_inputs))
        else:
            confidence = 0.5  # Default confidence
        
        self.last_confidence = confidence
        return int(action), {"confidence": confidence}
    
    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """Perform a training step on a batch of experiences"""
        if self.trainer is None:
            return {"loss": 0.0}
        
        # Store batch in replay buffer (simplified)
        self.training_steps += 1
        
        # Run training iteration
        result = self.trainer.train()
        
        loss = result.get("policy_loss", 0.0)
        return {"loss": float(loss), "steps": self.training_steps}
    
    def get_confidence(self) -> float:
        """Get current confidence level"""
        return self.last_confidence
    
    def set_regime_specialization(self, regime: MarketRegime):
        """Set market regime specialization for this agent"""
        self.regime_specialization = regime
        logger.info(f"Agent {self.agent_id} specialized for regime {regime.value}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for serialization"""
        return {
            "agent_id": self.agent_id,
            "model_type": self.model_type,
            "training_steps": self.training_steps,
            "confidence": self.last_confidence,
            "regime": self.regime_specialization.value if self.regime_specialization else None,
        }


@ray.remote
class RegimeClassifier:
    """
    Market regime classifier for dynamic agent weighting.
    Uses simple heuristics combined with ML for regime detection.
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback = lookback_periods
        self.price_history: List[float] = []
        self.volatility_history: List[float] = []
        self.current_regime = MarketRegime.TRANSITION
    
    def update(self, price: float, volume: float) -> MarketRegime:
        """Update regime classification with new price/volume data"""
        self.price_history.append(price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
        
        # Calculate volatility
        if len(self.price_history) >= 20:
            returns = np.diff(np.log(self.price_history[-20:]))
            volatility = float(np.std(returns))
            self.volatility_history.append(volatility)
            if len(self.volatility_history) > self.lookback:
                self.volatility_history.pop(0)
        
        # Classify regime
        self.current_regime = self._classify_regime()
        return self.current_regime
    
    def _classify_regime(self) -> MarketRegime:
        """Classify current market regime based on price action"""
        if len(self.price_history) < 20:
            return MarketRegime.TRANSITION
        
        prices = np.array(self.price_history)
        
        # Trend detection
        ma_short = np.mean(prices[-10:])
        ma_long = np.mean(prices[-20:])
        trend_strength = (ma_short - ma_long) / ma_long
        
        # Volatility assessment
        if self.volatility_history:
            avg_volatility = np.mean(self.volatility_history)
            current_volatility = self.volatility_history[-1] if self.volatility_history else 0
        else:
            avg_volatility = 0
            current_volatility = 0
        
        high_vol = current_volatility > avg_volatility * 1.5 if avg_volatility > 0 else False
        
        # Classify
        if trend_strength > 0.02:
            return MarketRegime.BULL_TRENDING
        elif trend_strength < -0.02:
            return MarketRegime.BEAR_TRENDING
        elif high_vol:
            return MarketRegime.HIGH_VOLATILITY
        elif current_volatility < avg_volatility * 0.5 if avg_volatility > 0 else False:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGE_BOUND
    
    def get_current_regime(self) -> MarketRegime:
        """Get current market regime"""
        return self.current_regime


class EnsembleMetaAgent:
    """
    Main ensemble meta-agent that combines predictions from multiple agents.
    Uses stacking to dynamically weight agent predictions based on regime confidence.
    """
    
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        env_config: Dict[str, Any],
        stacking_window: int = 100,
        memory_ceiling_gb: float = 4.0,
    ):
        self.agent_configs = agent_configs
        self.env_config = env_config
        self.stacking_window = stacking_window
        self.memory_ceiling_bytes = int(memory_ceiling_gb * 1024**3)
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            # Configure Ray with memory limits
            ray.init(
                object_store_memory=int(memory_ceiling_bytes * 0.3),  # 30% for object store
                _system_config={"max_direct_call_object_size": 1024 * 1024},
            )
            logger.info(f"Ray initialized with {memory_ceiling_gb}GB memory ceiling")
        
        # Initialize regime classifier
        self.regime_classifier = RegimeClassifier.remote()
        
        # Initialize ensemble agents
        self.agents: Dict[str, ray.actor.ActorHandle] = {}
        self._init_agents()
        
        # Stacking weights (learned over time)
        self.agent_weights: Dict[str, float] = {
            cfg.agent_id: 1.0 / len(agent_configs) for cfg in agent_configs
        }
        
        # Performance tracking per regime
        self.regime_performance: Dict[MarketRegime, Dict[str, List[float]]] = {
            regime: {cfg.agent_id: [] for cfg in agent_configs}
            for regime in MarketRegime
        }
        
        # Prediction history for stacking
        self.prediction_history: List[Tuple[EnsemblePrediction, float]] = []  # (prediction, reward)
        
        logger.info(f"EnsembleMetaAgent initialized with {len(self.agents)} agents")
    
    def _init_agents(self):
        """Initialize all ensemble agents as Ray actors"""
        for config in self.agent_configs:
            agent_ref = EnsembleAgent.remote(config, self.env_config)
            self.agents[config.agent_id] = agent_ref
            logger.info(f"Initialized agent: {config.agent_id}")
    
    def predict(self, observation: np.ndarray) -> EnsemblePrediction:
        """
        Generate ensemble prediction by combining all agent outputs.
        Uses dynamic weighting based on regime and recent performance.
        """
        # Get current regime
        # In production, you'd pass actual price/volume here
        regime_future = self.regime_classifier.get_current_regime.remote()
        current_regime = ray.get(regime_future)
        
        # Collect predictions from all agents
        prediction_futures = []
        for agent_id, agent_ref in self.agents.items():
            future = agent_ref.predict.remote(observation)
            prediction_futures.append((agent_id, future))
        
        # Gather predictions
        agent_predictions = {}
        for agent_id, future in prediction_futures:
            try:
                action, info = ray.get(future, timeout=1.0)
                agent_predictions[agent_id] = {
                    "action": action,
                    "confidence": info.get("confidence", 0.5),
                }
            except Exception as e:
                logger.warning(f"Agent {agent_id} prediction failed: {e}")
                agent_predictions[agent_id] = {"action": 0, "confidence": 0.0}
        
        # Compute dynamic weights
        weights = self._compute_weights(current_regime)
        
        # Weighted ensemble prediction
        total_weight = sum(weights.values())
        if total_weight > 0:
            weighted_actions = np.zeros(len(agent_predictions))
            weighted_confidence = 0.0
            
            for agent_id, pred in agent_predictions.items():
                w = weights.get(agent_id, 0.0) / total_weight
                weighted_actions[pred["action"]] += w
                weighted_confidence += w * pred["confidence"]
            
            final_action = int(np.argmax(weighted_actions))
            action_probs = weighted_actions / (weighted_actions.sum() + 1e-8)
        else:
            final_action = 0
            action_probs = np.ones(len(agent_predictions)) / len(agent_predictions)
            weighted_confidence = 0.0
        
        return EnsemblePrediction(
            action=final_action,
            confidence=float(weighted_confidence),
            action_probs=action_probs,
            agent_weights=weights,
            regime=current_regime,
            timestamp_ns=time.time_ns(),
        )
    
    def _compute_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Compute dynamic agent weights based on regime and recent performance.
        Implements stacking meta-learner logic.
        """
        weights = {}
        
        for agent_id in self.agents.keys():
            # Base weight
            base_weight = self.agent_weights.get(agent_id, 0.1)
            
            # Regime specialization bonus
            regime_perf = self.regime_performance.get(regime, {})
            agent_history = regime_perf.get(agent_id, [])
            
            if agent_history:
                # Recent performance (last 10 predictions)
                recent_perf = np.mean(agent_history[-10:]) if len(agent_history) >= 10 else np.mean(agent_history)
                regime_bonus = max(0.0, recent_perf)
            else:
                regime_bonus = 0.0
            
            # Confidence adjustment
            # Agents with higher confidence in current regime get more weight
            weights[agent_id] = base_weight * (1.0 + regime_bonus)
        
        return weights
    
    def update_weights(self, prediction: EnsemblePrediction, reward: float):
        """
        Update stacking weights based on prediction outcome.
        Implements online learning for the meta-learner.
        """
        self.prediction_history.append((prediction, reward))
        
        # Keep history bounded
        if len(self.prediction_history) > self.stacking_window:
            self.prediction_history.pop(0)
        
        # Update regime-specific performance
        regime = prediction.regime
        for agent_id, weight in prediction.agent_weights.items():
            if regime in self.regime_performance:
                perf_list = self.regime_performance[regime].get(agent_id, [])
                perf_list.append(reward)
                # Keep bounded
                if len(perf_list) > self.stacking_window:
                    perf_list.pop(0)
                self.regime_performance[regime][agent_id] = perf_list
        
        # Global weight update (simple exponential moving average)
        alpha = 0.1  # Learning rate
        for agent_id in self.agent_weights.keys():
            # Find if this agent's prediction matched the ensemble
            # Simplified: use reward as proxy
            current_weight = self.agent_weights.get(agent_id, 0.1)
            if reward > 0:
                new_weight = current_weight * (1 + alpha)
            else:
                new_weight = current_weight * (1 - alpha)
            
            # Clip weights
            self.agent_weights[agent_id] = np.clip(new_weight, 0.01, 1.0)
    
    async def train_async(self, experience_batches: Dict[str, List[Dict]]):
        """
        Asynchronously train all agents on their respective experience batches.
        Uses Ray for parallel training.
        """
        train_futures = []
        
        for agent_id, batch in experience_batches.items():
            if agent_id in self.agents and batch:
                agent_ref = self.agents[agent_id]
                future = agent_ref.train_step.remote(batch)
                train_futures.append(future)
        
        # Wait for all training to complete
        if train_futures:
            results = await asyncio.gather(*train_futures, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Training error for agent: {result}")
                else:
                    logger.debug(f"Training result: {result}")
    
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all agents"""
        states = {}
        
        for agent_id, agent_ref in self.agents.items():
            try:
                state = ray.get(agent_ref.get_state.remote())
                states[agent_id] = state
            except Exception as e:
                logger.warning(f"Could not get state for agent {agent_id}: {e}")
        
        return states
    
    def shutdown(self):
        """Shutdown all agents and release resources"""
        for agent_id, agent_ref in self.agents.items():
            try:
                ray.kill(agent_ref)
            except Exception:
                pass
        
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("EnsembleMetaAgent shutdown complete")


# Import for time
import time
import asyncio

# Register environment creator (example)
def create_trading_env(env_config: Dict[str, Any]):
    """Factory function for creating trading environments"""
    # This would import and instantiate your actual trading env
    from gymnasium import Env
    class CryptoTradingEnv(Env):
        def __init__(self, config):
            self.config = config
            self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
        
        def reset(self, seed=None):
            return np.zeros(10), {}
        
        def step(self, action):
            return np.zeros(10), 0.0, False, False, {}
    
    return CryptoTradingEnv(env_config)


# Example usage
if __name__ == "__main__":
    # Configure agents
    agent_configs = [
        AgentConfig(agent_id="ppo_conservative", model_type="ppo", learning_rate=1e-4),
        AgentConfig(agent_id="ppo_aggressive", model_type="ppo", learning_rate=5e-4),
        AgentConfig(agent_id="ppo_balanced", model_type="ppo", learning_rate=3e-4),
    ]
    
    env_config = {"env_name": "CryptoTradingEnv"}
    
    # Create ensemble
    ensemble = EnsembleMetaAgent(
        agent_configs=agent_configs,
        env_config=env_config,
        memory_ceiling_gb=4.0,
    )
    
    # Test prediction
    obs = np.random.randn(10)
    prediction = ensemble.predict(obs)
    
    print(f"Ensemble prediction: action={prediction.action}, confidence={prediction.confidence:.3f}")
    print(f"Regime: {prediction.regime.value}")
    print(f"Agent weights: {prediction.agent_weights}")
    
    # Cleanup
    ensemble.shutdown()
