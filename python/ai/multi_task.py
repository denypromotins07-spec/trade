"""
Multi-Task Reinforcement Learning Architecture on Ray

This module implements a Multi-Task RL architecture that shares lower-level
feature representations across trend, mean-reversion, and market-making heads
to improve sample efficiency. Runs on Ray with strict 4GB RAM quota.

Optimized for:
- Shared feature backbone for multiple trading objectives
- 4GB Python RAM quota per worker
- AMD ROCm/DirectML acceleration detection
- Memory-bounded replay buffers
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

# AMD ROCm/DirectML environment detection
def detect_amd_acceleration() -> Dict[str, bool]:
    """Detect available AMD acceleration hardware."""
    result = {
        "rocm_available": False,
        "directml_available": False,
        "hip_available": False,
        "cuda_available": torch.cuda.is_available() if torch.cuda.is_available() else False,
    }
    
    try:
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            result["rocm_available"] = True
            result["hip_available"] = True
    except (ImportError, AttributeError):
        pass
    
    try:
        import torch_directml
        result["directml_available"] = True
    except ImportError:
        pass
    
    return result


def get_device() -> torch.device:
    """Get optimal compute device based on available hardware."""
    amd_status = detect_amd_acceleration()
    
    if amd_status["rocm_available"]:
        return torch.device("cuda")  # ROCm uses CUDA interface
    elif amd_status["directml_available"]:
        return torch.device("privateuseone")  # DirectML device
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@dataclass
class TaskOutput:
    """Output from a specific task head."""
    action_logits: Optional[torch.Tensor]
    value_estimate: torch.Tensor
    auxiliary_outputs: Dict[str, torch.Tensor]


class SharedBackbone(nn.Module):
    """
    Shared feature extraction backbone for multi-task learning.
    
    Extracts common features from LOB state that are useful across
    all trading objectives (trend, mean-reversion, market-making).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Shared transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Global context aggregation
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared backbone.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Shared features of shape (batch, hidden_dim)
        """
        # Project input
        x = self.input_proj(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Aggregate sequence dimension (mean pooling)
        context = x.mean(dim=1)
        
        # Final projection
        context = self.context_proj(context)
        
        return context


class TrendHead(nn.Module):
    """Task head for trend-following strategies."""
    
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, features: torch.Tensor) -> TaskOutput:
        x = self.net(features)
        logits = self.action_head(x)
        value = self.value_head(x)
        
        return TaskOutput(
            action_logits=logits,
            value_estimate=value,
            auxiliary_outputs={"trend_strength": torch.sigmoid(value)},
        )


class MeanReversionHead(nn.Module):
    """Task head for mean-reversion strategies."""
    
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)
        # Additional output for reversion speed prediction
        self.speed_head = nn.Linear(64, 1)
        
    def forward(self, features: torch.Tensor) -> TaskOutput:
        x = self.net(features)
        logits = self.action_head(x)
        value = self.value_head(x)
        speed = self.speed_head(x)
        
        return TaskOutput(
            action_logits=logits,
            value_estimate=value,
            auxiliary_outputs={"reversion_speed": torch.sigmoid(speed)},
        )


class MarketMakingHead(nn.Module):
    """Task head for market-making strategies."""
    
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.bid_ask_head = nn.Linear(64, num_actions * 2)  # Separate bid/ask
        self.spread_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, features: torch.Tensor) -> TaskOutput:
        x = self.net(features)
        bid_ask = self.bid_ask_head(x)
        spread = self.spread_head(x)
        value = self.value_head(x)
        
        return TaskOutput(
            action_logits=bid_ask,
            value_estimate=value,
            auxiliary_outputs={"optimal_spread": torch.sigmoid(spread)},
        )


class MultiTaskRLNetwork(nn.Module):
    """
    Complete Multi-Task RL network with shared backbone and task-specific heads.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int = 3,  # Buy, Hold, Sell
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.shared_backbone = SharedBackbone(input_dim, hidden_dim)
        
        self.trend_head = TrendHead(hidden_dim, num_actions)
        self.mean_rev_head = MeanReversionHead(hidden_dim, num_actions)
        self.mm_head = MarketMakingHead(hidden_dim, num_actions)
        
        # Task weighting (learnable)
        self.task_weights = nn.Parameter(torch.ones(3))
        
    def forward(
        self,
        x: torch.Tensor,
        task: str = "all",
    ) -> Dict[str, TaskOutput]:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            task: Specific task or "all" for all tasks
        
        Returns:
            Dictionary of task outputs
        """
        features = self.shared_backbone(x)
        
        outputs = {}
        
        if task in ("all", "trend"):
            outputs["trend"] = self.trend_head(features)
        
        if task in ("all", "mean_reversion"):
            outputs["mean_reversion"] = self.mean_rev_head(features)
        
        if task in ("all", "market_making"):
            outputs["market_making"] = self.mm_head(features)
        
        return outputs
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get normalized task weights."""
        weights = F.softmax(self.task_weights, dim=0)
        return {
            "trend": weights[0].item(),
            "mean_reversion": weights[1].item(),
            "market_making": weights[2].item(),
        }


@ray.remote(num_cpus=2, memory=4 * 1024 * 1024 * 1024)
class MultiTaskWorker:
    """
    Ray worker for distributed multi-task RL training.
    
    Enforces 4GB RAM quota and handles gradient computation for
    specific task heads.
    """
    
    def __init__(
        self,
        worker_id: int,
        input_dim: int,
        primary_task: str,
        ram_quota_mb: int = 3500,
    ):
        self.worker_id = worker_id
        self.primary_task = primary_task
        self.ram_quota_mb = ram_quota_mb
        
        self.device = get_device()
        self.amd_status = detect_amd_acceleration()
        
        # Initialize network
        self.network = MultiTaskRLNetwork(input_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )
        
        # Memory-bounded buffer for gradients
        self.gradient_buffer: List[Dict[str, torch.Tensor]] = []
        self.max_gradient_batch = 32
        
        # Training statistics
        self.total_updates = 0
        self.task_losses: Dict[str, List[float]] = {
            "trend": [],
            "mean_reversion": [],
            "market_making": [],
        }
        
    def compute_gradients(
        self,
        batch_states: np.ndarray,
        batch_actions: np.ndarray,
        batch_rewards: np.ndarray,
        task: str,
    ) -> Dict[str, Any]:
        """Compute gradients for a specific task."""
        states = torch.FloatTensor(batch_states).to(self.device)
        actions = torch.LongTensor(batch_actions).to(self.device)
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        
        # Forward pass
        outputs = self.network(states, task=task)
        
        if task not in outputs:
            return {"error": f"Unknown task: {task}"}
        
        task_output = outputs[task]
        
        # Policy loss (REINFORCE-style)
        log_probs = F.log_softmax(task_output.action_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(selected_log_probs * rewards).mean()
        
        # Value loss
        value_loss = F.mse_loss(task_output.value_estimate.squeeze(), rewards)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Store gradients
        grad_dict = {}
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.cpu().clone()
        
        self.gradient_buffer.append(grad_dict)
        
        # Trim buffer if too large
        if len(self.gradient_buffer) > self.max_gradient_batch:
            self.gradient_buffer = self.gradient_buffer[-self.max_gradient_batch:]
        
        self.total_updates += 1
        
        # Track losses
        self.task_losses[task].append(loss.item())
        if len(self.task_losses[task]) > 1000:
            self.task_losses[task] = self.task_losses[task][-1000:]
        
        return {
            "worker_id": self.worker_id,
            "task": task,
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "memory_mb": self._get_memory_usage_mb(),
        }
    
    def apply_aggregated_gradients(
        self,
        aggregated_grads: Dict[str, np.ndarray],
    ) -> Dict[str, bool]:
        """Apply pre-aggregated gradients from parameter server."""
        for name, grad_np in aggregated_grads.items():
            grad_tensor = torch.FloatTensor(grad_np).to(self.device)
            
            # Find the parameter
            for param_name, param in self.network.named_parameters():
                if param_name == name and param.grad is not None:
                    param.data -= 0.001 * grad_tensor  # Simple update
        
        return {"applied": True}
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get current network weights."""
        weights = {}
        for name, param in self.network.state_dict().items():
            weights[name] = param.cpu().numpy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Set network weights from parameter server."""
        state_dict = {}
        for name, weight_np in weights.items():
            state_dict[name] = torch.FloatTensor(weight_np)
        
        self.network.load_state_dict(state_dict)
        return {"loaded": True}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            "worker_id": self.worker_id,
            "primary_task": self.primary_task,
            "total_updates": self.total_updates,
            "amd_status": self.amd_status,
            "device": str(self.device),
            "memory_mb": self._get_memory_usage_mb(),
            "within_quota": self._get_memory_usage_mb() < self.ram_quota_mb,
        }
        
        # Add recent loss averages
        for task, losses in self.task_losses.items():
            if losses:
                stats[f"{task}_avg_loss"] = np.mean(losses[-100:])
        
        return stats
    
    def _get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        import gc
        mem = 0
        
        # Model parameters
        for param in self.network.parameters():
            mem += param.numel() * param.element_size()
        
        # Gradient buffer
        for grad_dict in self.gradient_buffer:
            for grad in grad_dict.values():
                mem += grad.numel() * grad.element_size()
        
        # Python objects
        for losses in self.task_losses.values():
            mem += len(losses) * 8
        
        return mem / (1024 * 1024)
    
    def reset(self) -> Dict[str, bool]:
        """Reset worker state."""
        self.gradient_buffer.clear()
        self.total_updates = 0
        for task in self.task_losses:
            self.task_losses[task] = []
        return {"reset": True}


def create_multi_task_pool(
    num_workers: int,
    input_dim: int,
    tasks: List[str],
) -> List[ray.actor.ActorHandle]:
    """Create a pool of multi-task workers."""
    workers = []
    for i in range(num_workers):
        task = tasks[i % len(tasks)]
        worker = MultiTaskWorker.remote(i, input_dim, task)
        workers.append(worker)
    return workers


if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=2 * 1024 * 1024 * 1024,
        _system_config={"max_worker_size": 4 * 1024 * 1024 * 1024},
    )
    
    # Test multi-task setup
    tasks = ["trend", "mean_reversion", "market_making"]
    workers = create_multi_task_pool(3, input_dim=128, tasks=tasks)
    
    print(f"Created {len(workers)} multi-task workers")
    
    # Check AMD status
    status = ray.get(workers[0].get_statistics.remote())
    print(f"AMD Status: {status['amd_status']}")
    print(f"Device: {status['device']}")
    
    ray.shutdown()
