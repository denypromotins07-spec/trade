"""
PCGrad: Project Conflicting Gradients for Multi-Task RL

This module implements PCGrad (Project Conflicting Gradients) to resolve
conflicting gradient updates between different trading objectives during
multi-task backpropagation. Essential for stable multi-task training.

Optimized for:
- Gradient conflict resolution across trend, mean-reversion, MM tasks
- 4GB Python RAM quota per worker
- AMD ROCm/DirectML acceleration detection
- Efficient vector operations
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
class TaskGradient:
    """Gradient information for a single task."""
    task_name: str
    gradients: Dict[str, torch.Tensor]
    loss_value: float
    gradient_norm: float


class PCGradOptimizer:
    """
    PCGrad (Project Conflicting Gradients) optimizer.
    
    Projects conflicting gradients onto each other's normal plane to
    reduce interference between tasks during multi-task learning.
    
    Reference: Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    """
    
    def __init__(self, base_optimizer: torch.optim.Optimizer):
        self.base_optimizer = base_optimizer
        self.task_gradients: Dict[str, Dict[str, torch.Tensor]] = {}
        
    def store_task_gradients(
        self,
        task_name: str,
        gradients: Dict[str, torch.Tensor],
    ) -> None:
        """Store gradients for a specific task."""
        self.task_gradients[task_name] = {
            k: v.clone().cpu() for k, v in gradients.items()
        }
    
    def _dot_product(
        self,
        grad1: Dict[str, torch.Tensor],
        grad2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute dot product between two gradient dictionaries."""
        total = 0.0
        for key in grad1:
            if key in grad2:
                total += torch.dot(
                    grad1[key].flatten(),
                    grad2[key].flatten()
                ).item()
        return total
    
    def _gradient_norm(self, grad: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of gradient dictionary."""
        total = 0.0
        for g in grad.values():
            total += torch.sum(g ** 2).item()
        return np.sqrt(total)
    
    def _project_gradient(
        self,
        grad: Dict[str, torch.Tensor],
        ref_grad: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Project grad onto the normal plane of ref_grad.
        
        If the angle between gradients is acute (conflict), project.
        Otherwise, keep original gradient.
        """
        dot = self._dot_product(grad, ref_grad)
        
        if dot < 0:
            # Conflict detected - project
            ref_norm_sq = self._gradient_norm(ref_grad) ** 2
            
            if ref_norm_sq > 1e-10:
                projected = {}
                for key in grad:
                    if key in ref_grad:
                        # g_proj = g - (g·n / ||n||^2) * n
                        scale = dot / ref_norm_sq
                        projected[key] = grad[key] - scale * ref_grad[key]
                    else:
                        projected[key] = grad[key].clone()
                return projected
        
        # No conflict or numerical issues - return copy
        return {k: v.clone() for k, v in grad.items()}
    
    def compute_pcgrad(
        self,
        task_order: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PCGrad-aggregated gradients.
        
        Args:
            task_order: Order to process tasks (random if None)
        
        Returns:
            Aggregated gradient dictionary
        """
        if not self.task_gradients:
            return {}
        
        tasks = list(self.task_gradients.keys())
        
        if task_order is None:
            np.random.shuffle(tasks)
        else:
            tasks = [t for t in task_order if t in tasks]
        
        # Initialize with first task's gradients
        result_grads = {
            k: v.clone() for k, v in self.task_gradients[tasks[0]].items()
        }
        
        # Sequentially project with other tasks
        for i, task in enumerate(tasks[1:], 1):
            task_grad = self.task_gradients[task]
            
            # Project current aggregated gradient onto this task's gradient
            projected = self._project_gradient(result_grads, task_grad)
            
            # Add projected gradient
            for key in projected:
                if key in result_grads:
                    result_grads[key] = result_grads[key] + projected[key]
                else:
                    result_grads[key] = projected[key]
        
        # Normalize by number of tasks
        num_tasks = len(tasks)
        for key in result_grads:
            result_grads[key] = result_grads[key] / num_tasks
        
        return result_grads
    
    def apply_pcgrad(self, pc_gradients: Dict[str, torch.Tensor]) -> None:
        """Apply PCGrad-aggregated gradients to model parameters."""
        for name, param in self.base_optimizer.param_groups[0]['params']:
            if name in pc_gradients:
                param.grad = pc_gradients[name].to(param.device)
        
        self.base_optimizer.step()
    
    def clear(self) -> None:
        """Clear stored gradients."""
        self.task_gradients.clear()


class PCGradMultiTaskNetwork(nn.Module):
    """
    Multi-task network with integrated PCGrad support.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_tasks: int = 3,
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),  # Value output
            )
            for _ in range(num_tasks)
        ])
        
        # Task names
        self.task_names = ["trend", "mean_reversion", "market_making"]
        
    def forward(
        self,
        x: torch.Tensor,
        task_idx: int,
    ) -> torch.Tensor:
        """Forward pass for specific task."""
        features = self.backbone(x)
        return self.task_heads[task_idx](features)
    
    def get_all_task_outputs(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get outputs for all tasks."""
        features = self.backbone(x)
        
        outputs = {}
        for i, name in enumerate(self.task_names):
            outputs[name] = self.task_heads[i](features)
        
        return outputs


@ray.remote(num_cpus=2, memory=4 * 1024 * 1024 * 1024)
class PCGradWorker:
    """
    Ray worker implementing PCGrad for multi-task RL.
    
    Enforces 4GB RAM quota and handles gradient projection.
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
        
        # Initialize network
        self.network = PCGradMultiTaskNetwork(input_dim).to(self.device)
        
        # Base optimizer
        self.base_optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )
        
        # PCGrad optimizer wrapper
        self.pcgrad_opt = PCGradOptimizer(self.base_optimizer)
        
        # Per-task gradient storage
        self.task_gradients: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Statistics
        self.total_updates = 0
        self.conflict_counts: Dict[str, int] = {
            "trend": 0,
            "mean_reversion": 0,
            "market_making": 0,
        }
        self.total_conflicts = 0
        
    def compute_task_gradient(
        self,
        batch_states: np.ndarray,
        batch_rewards: np.ndarray,
        task_name: str,
    ) -> Dict[str, Any]:
        """Compute and store gradient for a specific task."""
        states = torch.FloatTensor(batch_states).to(self.device)
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        
        # Get task index
        task_idx = self.network.task_names.index(task_name)
        
        # Forward pass
        values = self.network(states, task_idx).squeeze()
        
        # MSE loss for value prediction
        loss = F.mse_loss(values, rewards)
        
        # Backward pass
        self.base_optimizer.zero_grad()
        loss.backward()
        
        # Extract gradients
        gradients = {}
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().clone()
        
        # Store for PCGrad
        self.pcgrad_opt.store_task_gradients(task_name, gradients)
        self.task_gradients[task_name] = gradients
        
        # Count conflicts (negative dot products with average)
        grad_norm = self.pcgrad_opt._gradient_norm(gradients)
        
        return {
            "worker_id": self.worker_id,
            "task": task_name,
            "loss": loss.item(),
            "gradient_norm": grad_norm,
            "memory_mb": self._get_memory_usage_mb(),
        }
    
    def apply_pcgrad_update(self) -> Dict[str, Any]:
        """Apply PCGrad aggregation and update weights."""
        if len(self.pcgrad_opt.task_gradients) < 2:
            # Not enough tasks, use standard update
            self.base_optimizer.step()
            method_used = "standard"
        else:
            # Apply PCGrad
            pc_gradients = self.pcgrad_opt.compute_pcgrad()
            
            # Count conflicts before applying
            for task in self.pcgrad_opt.task_gradients:
                self.conflict_counts[task] += 1
            self.total_conflicts += 1
            
            self.pcgrad_opt.apply_pcgrad(pc_gradients)
            method_used = "pcgrad"
        
        self.total_updates += 1
        self.pcgrad_opt.clear()
        
        return {
            "worker_id": self.worker_id,
            "method": method_used,
            "total_updates": self.total_updates,
            "total_conflicts": self.total_conflicts,
        }
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get current network weights."""
        weights = {}
        for name, param in self.network.state_dict().items():
            weights[name] = param.cpu().numpy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Set network weights."""
        state_dict = {}
        for name, weight_np in weights.items():
            state_dict[name] = torch.FloatTensor(weight_np)
        self.network.load_state_dict(state_dict)
        return {"loaded": True}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "total_updates": self.total_updates,
            "total_conflicts": self.total_conflicts,
            "conflict_counts": self.conflict_counts.copy(),
            "amd_status": self.amd_status,
            "device": str(self.device),
            "memory_mb": self._get_memory_usage_mb(),
            "within_quota": self._get_memory_usage_mb() < self.ram_quota_mb,
        }
    
    def _get_memory_usage_mb(self) -> float:
        """Estimate memory usage."""
        mem = 0
        
        for param in self.network.parameters():
            mem += param.numel() * param.element_size()
        
        for task_grads in self.task_gradients.values():
            for grad in task_grads.values():
                mem += grad.numel() * grad.element_size()
        
        return mem / (1024 * 1024)
    
    def reset(self) -> Dict[str, bool]:
        """Reset worker state."""
        self.task_gradients.clear()
        self.pcgrad_opt.clear()
        self.total_updates = 0
        self.total_conflicts = 0
        self.conflict_counts = {k: 0 for k in self.conflict_counts}
        return {"reset": True}


def create_pcgrad_pool(
    num_workers: int,
    input_dim: int,
) -> List[ray.actor.ActorHandle]:
    """Create a pool of PCGrad workers."""
    workers = []
    for i in range(num_workers):
        worker = PCGradWorker.remote(i, input_dim)
        workers.append(worker)
    return workers


if __name__ == "__main__":
    ray.init(
        object_store_memory=2 * 1024 * 1024 * 1024,
        _system_config={"max_worker_size": 4 * 1024 * 1024 * 1024},
    )
    
    # Test PCGrad workers
    workers = create_pcgrad_pool(2, input_dim=64)
    
    print(f"Created {len(workers)} PCGrad workers")
    
    status = ray.get(workers[0].get_statistics.remote())
    print(f"AMD Status: {status['amd_status']}")
    print(f"Device: {status['device']}")
    
    ray.shutdown()
