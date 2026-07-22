"""
Lightweight Self-Attention for L2 Order Book on Ray

This module implements a non-LLM self-attention mechanism optimized for
L2 order book data. Uses PyTorch with DirectML/ROCm acceleration and
strictly bounds sequence lengths to respect memory limits.

Optimized for:
- Lightweight attention over LOB levels
- 4GB Python RAM quota per worker
- AMD ROCm/DirectML acceleration
- Bounded sequence and feature dimensions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_amd_acceleration() -> Dict[str, bool]:
    """Detect available AMD acceleration hardware."""
    result = {
        "rocm_available": False,
        "directml_available": False,
        "hip_available": False,
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
        return torch.device("cuda")
    elif amd_status["directml_available"]:
        return torch.device("privateuseone")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@dataclass
class AttentionOutput:
    """Output from attention layer."""
    attended_features: torch.Tensor
    attention_weights: torch.Tensor
    value_estimate: Optional[torch.Tensor]


class BoundedSelfAttention(nn.Module):
    """
    Memory-bounded self-attention for L2 order book data.
    
    Key optimizations:
    - Fixed maximum sequence length (LOB levels)
    - Low-rank attention projection
    - Sparse attention pattern option
    """
    
    def __init__(
        self,
        input_dim: int,
        num_levels: int = 20,  # Max LOB levels to attend over
        num_heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.1,
        max_seq_len: int = 50,  # Strict memory bound
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_levels = min(num_levels, 20)  # Cap at 20 levels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # Input projection with memory-efficient dimension
        self.input_proj = nn.Linear(input_dim, num_heads * head_dim)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, input_dim)
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        """
        Forward pass through bounded self-attention.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
               seq_len is bounded by max_seq_len
            mask: Optional attention mask
        
        Returns:
            AttentionOutput with attended features
        """
        batch_size = x.shape[0]
        seq_len = min(x.shape[1], self.max_seq_len)
        
        # Truncate if exceeds max
        x = x[:, :seq_len, :]
        
        # Residual connection
        residual = x
        
        # Normalize
        x = self.layer_norm1(x)
        
        # Project input
        x = self.input_proj(x)
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for multi-head
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        # Residual connection
        output = output + residual
        
        return AttentionOutput(
            attended_features=output,
            attention_weights=attn_weights.mean(dim=1),  # Average across heads
            value_estimate=None,
        )


class LOBAttentionNetwork(nn.Module):
    """
    Complete network for L2 order book attention processing.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_levels: int = 20,
        hidden_dim: int = 128,
        num_attention_layers: int = 2,
    ):
        super().__init__()
        
        # Feature embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Stacked attention layers
        self.attention_layers = nn.ModuleList([
            BoundedSelfAttention(
                input_dim=hidden_dim,
                num_levels=num_levels,
                num_heads=4,
                head_dim=32,
            )
            for _ in range(num_attention_layers)
        ])
        
        # Global pooling and prediction
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Buy, Hold, Sell logits
        )
        
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through LOB attention network.
        
        Args:
            x: Input tensor (batch, levels, features)
        
        Returns:
            Tuple of (action_logits, value, attention_weights)
        """
        # Embed input
        x = self.embedding(x)
        
        # Pass through attention layers
        attn_weights_list = []
        for layer in self.attention_layers:
            output = layer(x)
            x = output.attended_features
            attn_weights_list.append(output.attention_weights)
        
        # Global pooling
        x_pooled = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Predictions
        action_logits = self.prediction_head(x_pooled)
        value = self.value_head(x_pooled)
        
        # Average attention weights across layers
        avg_attn = torch.stack(attn_weights_list).mean(dim=0)
        
        return action_logits, value, avg_attn


@ray.remote(num_cpus=1, memory=4 * 1024 * 1024 * 1024)
class AttentionWorker:
    """
    Ray worker for distributed attention-based LOB processing.
    
    Enforces 4GB RAM quota and bounded sequence lengths.
    """
    
    def __init__(
        self,
        worker_id: int,
        input_dim: int,
        ram_quota_mb: int = 3500,
    ):
        self.worker_id = worker_id
        self.ram_quota_mb = ram_quota_mb
        
        self.device = get_device()
        self.amd_status = detect_amd_acceleration()
        
        # Initialize network with strict bounds
        self.network = LOBAttentionNetwork(
            input_dim=input_dim,
            num_levels=20,  # Max 20 LOB levels
            hidden_dim=128,
            num_attention_layers=2,
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )
        
        # Bounded replay buffer
        self.max_buffer_size = 1000
        self.buffer_states: List[np.ndarray] = []
        self.buffer_actions: List[int] = []
        self.buffer_rewards: List[float] = []
        
        self.total_updates = 0
        
    def process_lob_snapshot(
        self,
        lob_data: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process a single LOB snapshot through attention network.
        
        Args:
            lob_data: Array of shape (levels, features)
        
        Returns:
            Dictionary with action probabilities and attention map
        """
        # Validate input dimensions
        if len(lob_data.shape) != 2:
            return {"error": "Invalid input shape"}
        
        levels = min(lob_data.shape[0], 20)  # Enforce bound
        lob_tensor = torch.FloatTensor(lob_data[:levels]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value, attn_weights = self.network(lob_tensor)
            
            probs = F.softmax(logits.squeeze(), dim=-1)
            
            return {
                "worker_id": self.worker_id,
                "action_probs": probs.cpu().numpy().tolist(),
                "value_estimate": value.squeeze().cpu().item(),
                "attention_map": attn_weights.cpu().numpy().tolist(),
                "levels_processed": levels,
                "device": str(self.device),
            }
    
    def add_to_buffer(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
    ) -> Dict[str, int]:
        """Add experiences to replay buffer with memory bounds."""
        for s, a, r in zip(states, actions, rewards):
            if len(self.buffer_states) >= self.max_buffer_size:
                # Remove oldest
                self.buffer_states.pop(0)
                self.buffer_actions.pop(0)
                self.buffer_rewards.pop(0)
            
            self.buffer_states.append(s)
            self.buffer_actions.append(a)
            self.buffer_rewards.append(r)
        
        return {"buffer_size": len(self.buffer_states)}
    
    def train_batch(
        self,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Train on a batch from replay buffer."""
        if len(self.buffer_states) < batch_size:
            return {"error": "Insufficient buffer size"}
        
        # Sample random batch
        indices = np.random.choice(len(self.buffer_states), batch_size, replace=False)
        
        states = np.array([self.buffer_states[i] for i in indices])
        actions = np.array([self.buffer_actions[i] for i in indices])
        rewards = np.array([self.buffer_rewards[i] for i in indices])
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        
        # Forward pass
        logits, value, _ = self.network(states_t)
        
        # Policy loss
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(selected_log_probs * rewards_t).mean()
        
        # Value loss
        value_loss = F.mse_loss(value.squeeze(), rewards_t)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        
        return {
            "worker_id": self.worker_id,
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "memory_mb": self._get_memory_usage_mb(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "total_updates": self.total_updates,
            "buffer_size": len(self.buffer_states),
            "max_buffer_size": self.max_buffer_size,
            "amd_status": self.amd_status,
            "device": str(self.device),
            "memory_mb": self._get_memory_usage_mb(),
            "within_quota": self._get_memory_usage_mb() < self.ram_quota_mb,
        }
    
    def _get_memory_usage_mb(self) -> float:
        """Estimate memory usage."""
        mem = 0
        
        # Model parameters
        for param in self.network.parameters():
            mem += param.numel() * param.element_size()
        
        # Buffer
        for state in self.buffer_states:
            mem += state.nbytes
        
        mem += len(self.buffer_actions) * 4
        mem += len(self.buffer_rewards) * 8
        
        return mem / (1024 * 1024)
    
    def reset(self) -> Dict[str, bool]:
        """Reset worker state."""
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.total_updates = 0
        return {"reset": True}


def create_attention_pool(
    num_workers: int,
    input_dim: int,
) -> List[ray.actor.ActorHandle]:
    """Create a pool of attention workers."""
    workers = []
    for i in range(num_workers):
        worker = AttentionWorker.remote(i, input_dim)
        workers.append(worker)
    return workers


if __name__ == "__main__":
    ray.init(
        object_store_memory=2 * 1024 * 1024 * 1024,
        _system_config={"max_worker_size": 4 * 1024 * 1024 * 1024},
    )
    
    # Test attention workers
    workers = create_attention_pool(2, input_dim=10)
    
    print(f"Created {len(workers)} attention workers")
    
    status = ray.get(workers[0].get_statistics.remote())
    print(f"AMD Status: {status['amd_status']}")
    print(f"Device: {status['device']}")
    
    ray.shutdown()
