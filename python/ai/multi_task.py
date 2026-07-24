"""
Stage 62: AI & Pipeline Audit - File 4/20
Module: python/ai/multi_task.py
Focus: PCGrad Gradient Surgery, Tensor Detachment, Conflicting Gradient Norms
Constraints: 4GB RAM Quota, AMD ROCm Compatibility

AUDIT FIXES APPLIED:
- Fixed PCGrad gradient surgery tensor detachment
- Added gradient norm conflict detection
- Prevented memory leaks via explicit gradient cleanup
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PCGradOptimizer:
    """
    Projected Conflicting Gradients (PCGrad) optimizer.
    FIX: Properly detaches tensors and handles gradient conflicts.
    """
    
    def __init__(self, base_optimizer: torch.optim.Optimizer):
        self.base_optimizer = base_optimizer
        self.task_gradients: Dict[str, List[torch.Tensor]] = {}
        
    def store_gradient(self, task_name: str, params: List[nn.Parameter]) -> None:
        """Store gradients for a specific task."""
        grads = []
        for p in params:
            if p.grad is not None:
                # FIX: Clone and detach to prevent graph retention
                grads.append(p.grad.clone().detach())
            else:
                grads.append(torch.zeros_like(p))
        self.task_gradients[task_name] = grads
    
    def project_conflicts(self, task_order: List[str]) -> None:
        """Project conflicting gradients using PCGrad algorithm."""
        if len(task_order) < 2:
            return
        
        # Get all task gradients
        all_grads = [self.task_gradients[t] for t in task_order]
        num_params = len(all_grads[0])
        
        for i, task_i in enumerate(task_order[:-1]):
            for j, task_j in enumerate(task_order[i+1:], i+1):
                grad_i = all_grads[i]
                grad_j = all_grads[j]
                
                # Compute dot products for each parameter
                for p_idx in range(num_params):
                    g_i = grad_i[p_idx]
                    g_j = grad_j[p_idx]
                    
                    dot_product = torch.sum(g_i * g_j)
                    
                    if dot_product < 0:
                        # Project gradient i onto gradient j's normal plane
                        g_j_norm_sq = torch.sum(g_j ** 2) + 1e-8
                        projection = (dot_product / g_j_norm_sq) * g_j
                        all_grads[i][p_idx] = g_i - projection
                        
                        logger.debug(f"Projected conflict between {task_i} and {task_j}")
        
        # Apply projected gradients back to parameters
        for param_idx, param in enumerate(self.base_optimizer.param_groups[0]['params']):
            if param.grad is not None:
                param.grad.zero_()
                for task_idx, task in enumerate(task_order):
                    if param_idx < len(all_grads[task_idx]):
                        param.grad += all_grads[task_idx][param_idx] / len(task_order)
    
    def step(self) -> None:
        """Perform optimization step."""
        self.base_optimizer.step()
    
    def zero_grad(self) -> None:
        """Clear gradients and task storage."""
        self.base_optimizer.zero_grad()
        self.task_gradients.clear()


class MultiTaskLossBalancer:
    """
    Balances multiple task losses with gradient norm monitoring.
    FIX: Detects and handles conflicting gradient norms.
    """
    
    def __init__(self, task_names: List[str], decay: float = 0.9):
        self.task_names = task_names
        self.decay = decay
        self.running_grad_norms: Dict[str, float] = {t: 1.0 for t in task_names}
        
    def compute_weights(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute task weights based on gradient norms."""
        weights = {}
        
        for task in self.task_names:
            if task in gradients:
                grad_norm = torch.norm(gradients[task]).item() + 1e-8
                
                # Update running average
                self.running_grad_norms[task] = (
                    self.decay * self.running_grad_norms[task] + 
                    (1 - self.decay) * grad_norm
                )
                
                # Inverse gradient norm weighting
                weights[task] = 1.0 / self.running_grad_norms[task]
            else:
                weights[task] = 1.0
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def detect_conflict(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """Detect gradient conflict via cosine similarity."""
        cos_sim = torch.nn.functional.cosine_similarity(
            grad1.flatten(), grad2.flatten(), dim=0
        )
        return cos_sim.item()


if __name__ == "__main__":
    print("Multi-task learning module loaded")
    print("Use PCGradOptimizer for gradient surgery")
