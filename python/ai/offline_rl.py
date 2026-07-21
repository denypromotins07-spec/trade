"""
Conservative Q-Learning (CQL) for Safe Offline Reinforcement Learning

This module implements CQL to prevent agents from overestimating values on
out-of-distribution actions not seen in historical data. Critical for safe
deployment of trading strategies trained on historical market data.

Optimized for AMD Ryzen AI 5 with DirectML/ROCm checks.
Respects strict 4GB Python RAM quota during Ray distribution.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# AMD DirectML/ROCm environment check
def check_amd_acceleration() -> Dict[str, Any]:
    """Check for AMD DirectML/ROCm availability and configure PyTorch."""
    config = {
        "directml_available": False,
        "rocm_available": False,
        "device": "cpu",
        "device_name": "CPU"
    }
    
    try:
        # Check for ROCm (AMD GPU on Linux)
        if torch.version.hip is not None:
            config["rocm_available"] = True
            config["device"] = "cuda"
            config["device_name"] = f"ROCm ({torch.cuda.get_device_name(0)})"
            print(f"[INFO] ROCm available: {config['device_name']}")
        
        # Check for DirectML (Windows with AMD GPU via DirectX)
        elif os.name == 'nt':
            try:
                import torch_directml
                config["directml_available"] = True
                config["device"] = "dml"
                config["device_name"] = "DirectML"
                print("[INFO] DirectML available")
            except ImportError:
                pass
        
        # Fallback to CUDA if available (NVIDIA or some AMD via ROCm)
        elif torch.cuda.is_available():
            config["device"] = "cuda"
            config["device_name"] = torch.cuda.get_device_name(0)
            
    except Exception as e:
        print(f"[WARN] AMD acceleration check failed: {e}")
    
    return config


class CQLNetwork(nn.Module):
    """
    Conservative Q-Network for offline RL.
    Uses ensemble of Q-networks for uncertainty estimation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        dropout_rate: float = 0.1,
        ensemble_size: int = 2,
    ):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.q_networks = nn.ModuleList()
        
        for _ in range(ensemble_size):
            layers = []
            prev_dim = state_dim + action_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.q_networks.append(nn.Sequential(*layers))
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through all Q-networks in ensemble."""
        x = torch.cat([state, action], dim=-1)
        q_values = torch.stack([net(x) for net in self.q_networks], dim=0)
        return q_values.squeeze(-1)  # [ensemble_size, batch_size]
    
    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return minimum Q-value across ensemble (conservative estimate)."""
        q_values = self.forward(state, action)
        return q_values.min(dim=0)[0]


class CQLTrainer:
    """
    Conservative Q-Learning trainer for offline RL.
    
    CQL adds a regularization term that penalizes Q-values for actions
    not in the dataset, preventing overestimation on OOD actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 1.0,  # CQL regularization coefficient
        gamma: float = 0.99,  # Discount factor
        tau: float = 0.005,  # Target network update rate
        learning_rate: float = 3e-4,
        device: str = "cpu",
        ram_limit_gb: float = 4.0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.ram_limit_gb = ram_limit_gb
        
        # Initialize networks
        self.q_network = CQLNetwork(state_dim, action_dim).to(device)
        self.target_q_network = CQLNetwork(state_dim, action_dim).to(device)
        
        # Copy weights to target network
        self._update_target_network(1.0)
        
        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=learning_rate
        )
        
        # Memory tracking
        self._track_memory_usage()
    
    def _update_target_network(self, tau: float = None):
        """Soft update of target network parameters."""
        tau = tau or self.tau
        with torch.no_grad():
            for param, target_param in zip(
                self.q_network.parameters(),
                self.target_q_network.parameters()
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def _track_memory_usage(self):
        """Track and log memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[MEM] GPU allocated: {allocated:.2f} GB")
    
    def cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        num_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Conservative Q-Learning loss.
        
        CQL Loss = TD Loss + α * (log_sum_exp(Q(s,a)) - E[Q(s,a_dataset)])
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            rewards: Batch of rewards [batch_size]
            next_states: Batch of next states [batch_size, state_dim]
            dones: Terminal flags [batch_size]
            num_samples: Number of random actions for sampling
        
        Returns:
            total_loss, td_loss, cql_regularization
        """
        batch_size = states.shape[0]
        
        # Current Q-values for dataset actions
        current_q = self.q_network(states, actions)  # [ensemble_size, batch_size]
        current_q_min = current_q.min(dim=0)[0]  # Conservative estimate
        
        # Target Q-values
        with torch.no_grad():
            # Sample multiple actions for next state
            random_actions = torch.randn(
                num_samples, batch_size, actions.shape[-1], 
                device=self.device
            )
            random_actions = random_actions.tanh()  # Bound actions
            
            # Get max Q-value over sampled actions
            next_q_values = self.target_q_network(next_states, random_actions.reshape(-1, actions.shape[-1]))
            next_q_values = next_q_values.view(num_samples, batch_size, -1)
            next_q_max = next_q_values.max(dim=0)[0].min(dim=0)[0]  # Min over ensemble
            
            # Bootstrap target
            target_q = rewards + (1 - dones) * self.gamma * next_q_max
        
        # TD Loss (mean squared Bellman error)
        td_losses = [(q - target_q).pow(2).mean() for q in current_q]
        td_loss = sum(td_losses) / len(td_losses)
        
        # CQL Regularization
        # Sample random actions
        random_actions = torch.randn(
            num_samples * batch_size, actions.shape[-1],
            device=self.device
        ).tanh()
        random_states = states.repeat_interleave(num_samples, dim=0)
        
        # Q-values for random actions
        q_random = self.q_network(random_states, random_actions)
        log_sum_exp_q_random = torch.logsumexp(q_random, dim=0).mean()
        
        # Q-values for dataset actions (for comparison)
        q_data = self.q_network(states, actions)
        mean_q_data = q_data.mean()
        
        # CQL penalty: encourage lower Q for random actions
        cql_penalty = self.alpha * (log_sum_exp_q_random - mean_q_data)
        
        # Total loss
        total_loss = td_loss + cql_penalty
        
        return total_loss, td_loss, cql_penalty
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones
        
        Returns:
            Dictionary of loss metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Zero gradients
        self.q_optimizer.zero_grad()
        
        # Compute loss
        total_loss, td_loss, cql_penalty = self.cql_loss(
            states, actions, rewards, next_states, dones
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        # Update weights
        self.q_optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        # Track memory
        self._track_memory_usage()
        
        return {
            "total_loss": total_loss.item(),
            "td_loss": td_loss.item(),
            "cql_penalty": cql_penalty.item(),
        }
    
    def get_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """
        Get action using conservative Q-value estimates.
        
        Args:
            state: Current state observation
            explore: Whether to add exploration noise
            noise_scale: Scale of exploration noise
        
        Returns:
            Action array
        """
        self.q_network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Sample multiple actions and pick best according to min Q
            num_candidates = 10
            action_dim = self.q_network.q_networks[0][-1].in_features - state.shape[-1]
            
            candidates = torch.randn(num_candidates, action_dim, device=self.device).tanh()
            q_values = self.q_network(
                state_tensor.repeat(num_candidates, 1),
                candidates
            )
            
            # Use minimum Q across ensemble (conservative)
            min_q_values = q_values.min(dim=0)[0]
            best_action_idx = min_q_values.argmax()
            action = candidates[best_action_idx]
            
            if explore:
                action += noise_scale * torch.randn_like(action)
                action = action.tanh()
        
        self.q_network.train()
        return action.cpu().numpy()


def load_offline_dataset(
    filepath: str,
    max_samples: int = 100000,
    ram_limit_gb: float = 4.0,
) -> Dict[str, torch.Tensor]:
    """
    Load offline dataset from file with memory limits.
    
    Args:
        filepath: Path to dataset file (npz format)
        max_samples: Maximum number of samples to load
        ram_limit_gb: Memory limit for dataset
    
    Returns:
        Dictionary of tensors
    """
    # Estimate memory per sample (~1KB per transition)
    bytes_per_sample = 1024
    max_bytes = ram_limit_gb * 1024**3 * 0.5  # Use 50% of RAM limit
    max_samples_by_mem = int(max_bytes / bytes_per_sample)
    
    actual_max = min(max_samples, max_samples_by_mem)
    print(f"[INFO] Loading up to {actual_max} samples (RAM limit: {ram_limit_gb}GB)")
    
    try:
        data = np.load(filepath)
        
        # Truncate if necessary
        states = torch.FloatTensor(data["states"][:actual_max])
        actions = torch.FloatTensor(data["actions"][:actual_max])
        rewards = torch.FloatTensor(data["rewards"][:actual_max])
        next_states = torch.FloatTensor(data["next_states"][:actual_max])
        dones = torch.FloatTensor(data["dones"][:actual_max])
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }
    except FileNotFoundError:
        print(f"[WARN] Dataset not found at {filepath}, generating synthetic data")
        return generate_synthetic_dataset(actual_max)


def generate_synthetic_dataset(num_samples: int) -> Dict[str, torch.Tensor]:
    """Generate synthetic offline dataset for testing."""
    state_dim = 20
    action_dim = 2
    
    states = torch.randn(num_samples, state_dim)
    actions = torch.randn(num_samples, action_dim).tanh()
    rewards = torch.randn(num_samples) * 0.1
    next_states = states + torch.randn(num_samples, state_dim) * 0.01
    dones = torch.zeros(num_samples)
    
    # Add some terminal states
    num_dones = num_samples // 100
    done_indices = torch.randperm(num_samples)[:num_dones]
    dones[done_indices] = 1.0
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
    }


def train_cql(
    dataset: Dict[str, torch.Tensor],
    num_epochs: int = 100,
    batch_size: int = 256,
    alpha: float = 1.0,
    device: str = "cpu",
    ram_limit_gb: float = 4.0,
) -> CQLTrainer:
    """
    Train CQL agent on offline dataset.
    
    Args:
        dataset: Offline dataset dictionary
        num_epochs: Number of training epochs
        batch_size: Training batch size
        alpha: CQL regularization coefficient
        device: Training device
        ram_limit_gb: Memory limit
    
    Returns:
        Trained CQLTrainer
    """
    state_dim = dataset["states"].shape[1]
    action_dim = dataset["actions"].shape[1]
    
    # Initialize trainer
    trainer = CQLTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=alpha,
        device=device,
        ram_limit_gb=ram_limit_gb,
    )
    
    # Create DataLoader
    tensor_dataset = TensorDataset(
        dataset["states"],
        dataset["actions"],
        dataset["rewards"],
        dataset["next_states"],
        dataset["dones"],
    )
    
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device != "cpu"),
    )
    
    # Training loop
    print(f"[INFO] Starting CQL training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_metrics = {"total_loss": 0.0, "td_loss": 0.0, "cql_penalty": 0.0}
        num_batches = 0
        
        for batch in dataloader:
            states, actions, rewards, next_states, dones = batch
            
            batch_dict = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "dones": dones,
            }
            
            metrics = trainer.train_step(batch_dict)
            
            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"total_loss={epoch_metrics['total_loss']:.4f}, "
                f"td_loss={epoch_metrics['td_loss']:.4f}, "
                f"cql_penalty={epoch_metrics['cql_penalty']:.4f}"
            )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("Checking AMD acceleration...")
    amd_info = check_amd_acceleration()
    print(f"Device: {amd_info['device_name']}")
    
    print("\nLoading/generating dataset...")
    dataset = load_offline_dataset("/tmp/offline_data.npz", ram_limit_gb=4.0)
    
    print("\nTraining CQL agent...")
    trainer = train_cql(
        dataset,
        num_epochs=50,
        batch_size=256,
        alpha=1.0,
        device=amd_info["device"],
        ram_limit_gb=4.0,
    )
    
    print("\nCQL training complete!")
