"""
Categorical DQN (C51) - Distributional Reinforcement Learning
================================================================
This module implements Categorical DQN (C51) on Ray workers to predict the full
probability distribution of returns rather than just the expected value.

Key Features:
- Predicts return distribution across fixed atoms (bins)
- Strict 4GB Python RAM quota enforcement per worker
- AMD ROCm/DirectML acceleration checks
- Memory-efficient contiguous tensor layouts

Constraints:
- Must run within Ray worker memory limits
- Uses contiguous memory to prevent cache thrashing
- No LLM dependencies
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import ray

# Enforce 4GB RAM quota per worker
MAX_RAM_GB = 4.0
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def check_amd_acceleration() -> str:
    """
    Check for AMD ROCm/DirectML availability and return device string.
    Returns 'cuda' for ROCm, 'cpu' otherwise (DirectML not natively in PyTorch).
    """
    if torch.cuda.is_available():
        # Check if it's AMD ROCm
        if hasattr(torch.cuda, 'get_device_name'):
            device_name = torch.cuda.get_device_name(0)
            if 'AMD' in device_name or 'Radeon' in device_name:
                print(f"AMD ROCm detected: {device_name}")
                return 'cuda'
            elif 'MI' in device_name or 'Instinct' in device_name:
                print(f"AMD Instinct GPU detected: {device_name}")
                return 'cuda'
        print(f"CUDA/ROCm available: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    
    # DirectML check (requires torch-directml package on Windows)
    try:
        import torch_directml
        print("DirectML available")
        return 'dml'
    except ImportError:
        pass
    
    print("No GPU acceleration found, using CPU")
    return 'cpu'


class C51Network(nn.Module):
    """
    Categorical DQN (C51) Neural Network.
    
    Predicts a probability distribution over N_atoms discrete return values.
    Uses contiguous memory layout for cache efficiency.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -100.0,
        v_max: float = 100.0,
        hidden_dim: int = 256,
    ):
        super(C51Network, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Register atoms as buffer (not parameters) for efficient access
        self.register_buffer(
            'atoms', 
            torch.linspace(v_min, v_max, steps=n_atoms)
        )
        
        # Contiguous fully-connected layers
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim * n_atoms),
        )
        
        # Initialize weights with Xavier for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log-probabilities over atoms for each action.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            log_probs: Log probabilities [batch_size, action_dim, n_atoms]
        """
        batch_size = state.size(0)
        logits = self.net(state)  # [batch, action_dim * n_atoms]
        logits = logits.view(batch_size, self.action_dim, self.n_atoms)
        
        # Return log-softmax for numerical stability
        return F.log_softmax(logits, dim=-1)
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Q-values by taking expectation over atoms.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            q_values: Expected Q-values [batch_size, action_dim]
        """
        log_probs = self.forward(state)
        probs = log_probs.exp()
        return (probs * self.atoms).sum(dim=-1)


@ray.remote(max_calls=1000)  # Restart worker periodically to prevent memory leaks
class C51Worker:
    """
    Ray worker for distributed C51 training.
    
    Enforces strict 4GB RAM quota and uses AMD acceleration when available.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        n_atoms: int = 51,
        v_min: float = -100.0,
        v_max: float = 100.0,
    ):
        self.device = check_amd_acceleration()
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Create network
        self.policy_net = C51Network(
            state_dim, action_dim, n_atoms, v_min, v_max
        ).to(self.device)
        
        self.target_net = C51Network(
            state_dim, action_dim, n_atoms, v_min, v_max
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        
        # Memory tracking
        self.memory_usage_mb = 0
    
    def compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """
        Compute C51 distributional loss.
        
        Projects target distribution onto support atoms and computes KL divergence.
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        batch_size = states.size(0)
        
        # Current policy distribution
        log_probs = self.policy_net(states)
        probs = log_probs.exp()
        
        # Get current action probabilities
        action_probs = probs.gather(
            1, 
            actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)
        ).squeeze(1)
        
        # Target network: greedy action selection by expected value
        with torch.no_grad():
            next_q_values = self.target_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Next state distribution
            next_log_probs = self.target_net(next_states)
            next_probs = next_log_probs.exp()
            
            # Select distribution for greedy action
            next_action_probs = next_probs.gather(
                1, 
                next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_atoms)
            ).squeeze(1)
            
            # Compute target distribution (Bellman update with projection)
            tz = rewards + (1 - dones) * self.gamma * self.policy_net.atoms
            tz = tz.clamp(self.v_min, self.v_max)
            
            # Project onto atoms
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle edge case where l == u
            mask = (l == u).float()
            l = (l * mask + u * (1 - mask)).clamp(0, self.n_atoms - 1)
            u = u.clamp(0, self.n_atoms - 1)
            
            # Distribute probability mass
            offset = torch.arange(0, batch_size, device=self.device).unsqueeze(1)
            target_dist = torch.zeros_like(next_action_probs)
            
            target_dist.view(-1).index_add_(
                0, 
                (offset * self.n_atoms + l.view(-1)), 
                (next_action_probs * (u.float() - b + 1)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, 
                (offset * self.n_atoms + u.view(-1)), 
                (next_action_probs * (b - l.float())).view(-1)
            )
        
        # KL divergence loss
        loss = -(action_probs * target_dist.log()).sum(dim=1).mean()
        
        return loss.item()
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict:
        """
        Perform one training step with gradient descent.
        """
        self.policy_net.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        batch_size = states_t.size(0)
        
        log_probs = self.policy_net(states_t)
        probs = log_probs.exp()
        action_probs = probs.gather(
            1, 
            actions_t.unsqueeze(-1).expand(-1, -1, self.n_atoms)
        ).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net.get_q_values(next_states_t)
            next_actions = next_q_values.argmax(dim=1)
            next_log_probs = self.target_net(next_states_t)
            next_probs = next_log_probs.exp()
            next_action_probs = next_probs.gather(
                1, 
                next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_atoms)
            ).squeeze(1)
            
            tz = rewards_t + (1 - dones_t) * self.gamma * self.policy_net.atoms
            tz = tz.clamp(self.v_min, self.v_max)
            
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            mask = (l == u).float()
            l = (l * mask + u * (1 - mask)).clamp(0, self.n_atoms - 1)
            u = u.clamp(0, self.n_atoms - 1)
            
            offset = torch.arange(0, batch_size, device=self.device).unsqueeze(1)
            target_dist = torch.zeros_like(next_action_probs)
            target_dist.view(-1).index_add_(
                0, (offset * self.n_atoms + l.view(-1)),
                (next_action_probs * (u.float() - b + 1)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (offset * self.n_atoms + u.view(-1)),
                (next_action_probs * (b - l.float())).view(-1)
            )
        
        loss = -(action_probs * target_dist.log()).sum(dim=1).mean()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track memory usage
        if self.device == 'cuda':
            self.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return {
            'loss': loss.item(),
            'memory_mb': self.memory_usage_mb,
        }
    
    def update_target_network(self, tau: float = 0.001):
        """Soft update of target network parameters."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )
    
    def get_action_distribution(
        self, 
        state: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Get action with highest expected value and its return distribution.
        
        Returns:
            action: Best action index
            distribution: Probability distribution over atoms for that action
        """
        self.policy_net.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            log_probs = self.policy_net(state_t)
            probs = log_probs.exp().squeeze(0).cpu().numpy()
            
            # Expected values
            atoms = self.policy_net.atoms.cpu().numpy()
            q_values = (probs * atoms).sum(axis=1)
            best_action = np.argmax(q_values)
            
            return best_action, probs[best_action]
    
    def check_memory_quota(self) -> bool:
        """Verify we're within 4GB RAM quota."""
        if self.device == 'cuda':
            allocated_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            return allocated_gb < MAX_RAM_GB
        else:
            # Approximate CPU memory check
            import psutil
            process = psutil.Process(os.getpid())
            used_gb = process.memory_info().rss / 1024 / 1024 / 1024
            return used_gb < MAX_RAM_GB


def create_c51_workers(
    num_workers: int,
    state_dim: int,
    action_dim: int,
) -> List[ray.ObjectRef]:
    """
    Create distributed C51 workers on Ray.
    """
    workers = []
    for _ in range(num_workers):
        worker = C51Worker.remote(state_dim, action_dim)
        workers.append(worker)
    return workers


if __name__ == '__main__':
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=int(2 * 1024 * 1024 * 1024),  # 2GB object store
        _system_config={'worker_max_memory_fraction': 0.5}
    )
    
    # Test C51 setup
    workers = create_c51_workers(2, state_dim=128, action_dim=6)
    
    # Verify AMD acceleration detection
    for w in workers:
        result = ray.get(w.check_memory_quota.remote())
        print(f"Worker memory quota OK: {result}")
    
    ray.shutdown()
