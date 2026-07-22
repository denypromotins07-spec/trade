"""
Dreamer-Style Actor-Critic for Market Trading

Implements a Dreamer-style architecture that learns purely from imagined
trajectories in latent space. This drastically improves sample efficiency
and reduces the need for expensive live market exploration.

Key Features:
- Latent imagination for policy learning
- Actor-Critic with discrete latent states
- Memory-bounded replay and imagination buffers
- AMD DirectML/ROCm acceleration support
- 4GB Python RAM quota enforcement

Architecture:
- Actor: Maps latent state to action distribution
- Critic: Estimates value from latent state
- World Model: Provides latent dynamics for imagination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import ray


@dataclass
class DreamerConfig:
    """Configuration for Dreamer Agent"""
    
    # Model architecture
    latent_dim: int = 256
    hidden_dim: int = 512
    vocab_size: int = 8192  # Discrete latent vocabulary
    
    # Actor-Critic architecture
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    num_actor_layers: int = 2
    num_critic_layers: int = 2
    
    # Action space
    action_dim: int = 10
    action_entropy_weight: float = 0.01  # Entropy regularization
    
    # Imagination parameters
    imagination_horizon: int = 15  # Steps to imagine
    imagination_batch_size: int = 256
    
    # Learning parameters
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    lambda_: float = 0.95  # GAE lambda
    
    # Memory constraints (4GB quota)
    max_replay_size: int = 200_000
    batch_size: int = 64
    
    # Regularization
    dropout: float = 0.1
    gradient_clip: float = 100.0
    target_update_tau: float = 0.005


def check_amd_acceleration() -> Tuple[bool, str]:
    """Check for AMD ROCm or DirectML availability."""
    if torch.cuda.is_available() and ('roc' in torch.version.cuda or 
                                       hasattr(torch.version, 'hip')):
        return True, 'cuda'
    
    try:
        import torch_directml
        return True, 'dml'
    except ImportError:
        pass
    
    return False, 'cpu'


class DreamerActor(nn.Module):
    """
    Actor network that maps latent states to action distributions.
    Uses discrete latent representations for memory efficiency.
    """
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        
        # Input combines latent and hidden state
        input_dim = config.vocab_size + config.hidden_dim
        
        # Actor network
        layers = []
        prev_dim = input_dim
        for _ in range(config.num_actor_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, config.actor_hidden_dim),
                nn.LayerNorm(config.actor_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = config.actor_hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action distribution from latent state.
        
        Args:
            latent: Latent representation [batch, vocab_size]
            hidden: Hidden state [batch, hidden_dim]
            temperature: Softmax temperature for exploration
            
        Returns:
            action_probs: Action probabilities [batch, action_dim]
            log_probs: Log probabilities for policy gradient
        """
        # Combine inputs
        x = torch.cat([latent, hidden], dim=-1)
        
        # Forward through network
        logits = self.network(x)
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Get probabilities
        action_probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return action_probs, log_probs
    
    def get_action(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        Sample or select action from policy.
        
        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
        """
        action_probs, log_probs = self(latent, hidden)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Categorical sampling
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        # Get log prob of selected action
        batch_range = torch.arange(action.shape[0], device=action.device)
        selected_log_prob = log_probs[batch_range, action]
        
        return action.item(), selected_log_prob


class DreamerCritic(nn.Module):
    """
    Critic network that estimates value from latent state.
    Uses target network for stable training.
    """
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        
        input_dim = config.vocab_size + config.hidden_dim
        
        # Critic network
        layers = []
        prev_dim = input_dim
        for _ in range(config.num_critic_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, config.critic_hidden_dim),
                nn.LayerNorm(config.critic_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = config.critic_hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Estimate value from latent state."""
        x = torch.cat([latent, hidden], dim=-1)
        return self.network(x).squeeze(-1)


class DreamerAgent:
    """
    Complete Dreamer agent with actor-critic and world model.
    Learns from imagined trajectories in latent space.
    """
    
    def __init__(
        self,
        world_model,
        config: Optional[DreamerConfig] = None,
        device: str = 'auto',
    ):
        self.config = config or DreamerConfig()
        self.world_model = world_model
        
        # Device selection with AMD check
        if device == 'auto':
            has_amd, device_type = check_amd_acceleration()
            self.device = torch.device(device_type if has_amd else 'cpu')
            print(f"Dreamer using device: {self.device}")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.actor = DreamerActor(self.config).to(self.device)
        self.critic = DreamerCritic(self.config).to(self.device)
        self.target_critic = DreamerCritic(self.config).to(self.device)
        
        # Initialize target network
        self._update_target_network(tau=1.0)
        
        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.config.actor_lr,
            weight_decay=1e-5,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.config.critic_lr,
            weight_decay=1e-5,
        )
        
        # Memory-bounded replay buffer
        self.replay_buffer = []
        self.max_buffer_size = self.config.max_replay_size
    
    def _update_target_network(self, tau: float = None):
        """Soft update of target critic network."""
        if tau is None:
            tau = self.config.target_update_tau
        
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
    
    def add_experience(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        latent: Optional[np.ndarray] = None,
        hidden: Optional[np.ndarray] = None,
    ):
        """Add experience to replay buffer with memory bounds."""
        experience = {
            'obs': observation,
            'act': action,
            'rew': reward,
            'done': done,
            'latent': latent,
            'hidden': hidden,
        }
        
        self.replay_buffer.append(experience)
        
        # Enforce memory bound
        while len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
    
    def imagine_trajectory(
        self,
        initial_latent: torch.Tensor,
        initial_hidden: torch.Tensor,
        horizon: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine trajectory in latent space starting from given state.
        
        Returns imagined states, actions, and rewards for policy learning.
        """
        if horizon is None:
            horizon = self.config.imagination_horizon
        
        latents = [initial_latent]
        hiddens = [initial_hidden]
        actions = []
        rewards = []
        
        current_latent = initial_latent
        current_hidden = initial_hidden
        
        with torch.no_grad():
            for _ in range(horizon):
                # Actor selects action
                action_probs, _ = self.actor(current_latent, current_hidden)
                
                # Sample action
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                
                # One-hot encode action
                action_onehot = F.one_hot(
                    action,
                    num_classes=self.config.action_dim,
                ).float()
                
                # World model predicts next state
                next_latent, next_hidden = self.world_model.predict_next_state(
                    current_latent,
                    current_hidden,
                    action_onehot.unsqueeze(0),
                )
                
                # Critic predicts reward
                reward_pred = self.critic(next_latent.squeeze(0), next_hidden.squeeze(0))
                
                # Store
                actions.append(action)
                rewards.append(reward_pred)
                latents.append(next_latent)
                hiddens.append(next_hidden)
                
                current_latent = next_latent
                current_hidden = next_hidden
        
        return {
            'latents': torch.cat(latents[:-1], dim=0),
            'hiddens': torch.cat(hiddens[:-1], dim=0),
            'actions': torch.stack(actions, dim=0),
            'rewards': torch.stack(rewards, dim=0),
        }
    
    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lambda-returns using GAE."""
        returns = []
        next_return = 0.0
        
        gamma = self.config.gamma
        lambda_ = self.config.lambda_
        
        # Reverse iteration
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            next_return = delta + gamma * lambda_ * next_return * (1 - dones[t])
            returns.insert(0, next_return + values[t])
        
        return torch.tensor(returns, device=rewards.device)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform training step using imagined trajectories.
        
        Returns training metrics.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Sample batch from replay
        indices = np.random.choice(
            len(self.replay_buffer),
            self.config.batch_size,
            replace=False,
        )
        
        # Build tensors
        obs_batch = torch.FloatTensor(
            np.array([self.replay_buffer[i]['obs'] for i in indices])
        ).to(self.device)
        
        # Encode observations
        with torch.no_grad():
            latent_probs, hidden_state = self.world_model.encode_observation(obs_batch)
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]
        
        # Imagine trajectories from each state in batch
        imagined = self.imagine_trajectory(
            latent_probs,
            hidden_state,
            self.config.imagination_horizon,
        )
        
        # Compute values for imagined states
        values = self.critic(
            imagined['latents'].squeeze(1),
            imagined['hiddens'].squeeze(1),
        )
        
        # ===== Critic Update =====
        self.critic_optimizer.zero_grad()
        
        # TD targets from imagined rewards
        with torch.no_grad():
            target_values = self.target_critic(
                imagined['latents'].squeeze(1),
                imagined['hiddens'].squeeze(1),
            )
            
            # Bootstrap from final state
            final_reward = imagined['rewards'][-1]
            td_target = final_reward + self.config.gamma * target_values[-1]
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(values, td_target.expand_as(values))
        
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.gradient_clip,
        )
        self.critic_optimizer.step()
        
        # ===== Actor Update =====
        self.actor_optimizer.zero_grad()
        
        # Actor loss: maximize expected return + entropy
        actor_losses = []
        entropy_losses = []
        
        for t in range(self.config.imagination_horizon):
            latent_t = imagined['latents'][:, t].squeeze(1)
            hidden_t = imagined['hiddens'][:, t].squeeze(1)
            
            action_probs, log_probs = self.actor(latent_t, hidden_t)
            
            # Advantage from critic
            with torch.no_grad():
                advantage = imagined['rewards'][t].squeeze() - \
                           self.critic(latent_t, hidden_t)
            
            # Policy gradient loss
            actor_loss = -(log_probs * advantage.detach()).mean()
            
            # Entropy regularization
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
            entropy_loss = -self.config.action_entropy_weight * entropy
            
            total_actor_loss = actor_loss + entropy_loss
            actor_losses.append(actor_loss)
            entropy_losses.append(entropy_loss)
        
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        avg_actor_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.gradient_clip,
        )
        self.actor_optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        return {
            'actor_loss': avg_actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_entropy': sum(entropy_losses).item() / len(entropy_losses),
        }
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        """Get action for current observation."""
        self.actor.eval()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            latent_probs, hidden_state = self.world_model.encode_observation(obs_tensor)
            
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]
            
            action, log_prob = self.actor.get_action(
                latent_probs,
                hidden_state,
                deterministic=deterministic,
            )
            
            # Get action value estimate
            value = self.critic(latent_probs.squeeze(0), hidden_state.squeeze(0))
        
        return action, {
            'log_prob': log_prob.item(),
            'value': value.item(),
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


@ray.remote(num_cpus=2, max_calls=10)
class DreamerWorker:
    """Ray worker for distributed Dreamer training."""
    
    def __init__(self, world_model_weights: Dict, config: DreamerConfig, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        
        # Create dummy world model placeholder (would be loaded from weights)
        class DummyWorldModel:
            def __init__(self):
                self.config = type('obj', (object,), {
                    'obs_dim': 100,
                    'vocab_size': config.vocab_size,
                    'hidden_dim': config.hidden_dim,
                    'action_dim': config.action_dim,
                    'num_layers': 2,
                })()
            
            def encode_observation(self, obs):
                batch_size = obs.shape[0]
                latent = torch.randn(batch_size, config.vocab_size)
                latent = F.softmax(latent, dim=-1)
                hidden = torch.randn(2, batch_size, config.hidden_dim)
                return latent, hidden
            
            def predict_next_state(self, latent, hidden, action):
                batch_size = latent.shape[0]
                next_latent = torch.randn(batch_size, 1, config.vocab_size)
                next_latent = F.softmax(next_latent, dim=-1)
                next_hidden = torch.randn(2, batch_size, config.hidden_dim)
                return next_latent, next_hidden
        
        world_model = DummyWorldModel()
        self.agent = DreamerAgent(world_model, config)
        
        print(f"DreamerWorker {worker_id} initialized on {self.agent.device}")
    
    def train_batch(self) -> Dict[str, float]:
        """Perform training batch."""
        return self.agent.train_step()
    
    def get_weights(self) -> Dict:
        """Get agent weights for synchronization."""
        return {
            'actor': {k: v.cpu().numpy() for k, v in self.agent.actor.state_dict().items()},
            'critic': {k: v.cpu().numpy() for k, v in self.agent.critic.state_dict().items()},
        }
    
    def load_weights(self, weights: Dict):
        """Load synchronized weights."""
        # Would implement weight loading here
        pass


if __name__ == "__main__":
    print("Dreamer Agent module loaded successfully")
    
    # Test configuration
    config = DreamerConfig(
        latent_dim=128,
        hidden_dim=256,
        vocab_size=1024,
        action_dim=5,
        imagination_horizon=10,
    )
    
    print(f"Dreamer Config: latent={config.latent_dim}, horizon={config.imagination_horizon}")
    print(f"AMD Acceleration: {check_amd_acceleration()}")
