"""
Stage 62: AI & Pipeline Audit - File 10/20
Module: python/soul/knowledge_distill.py
Focus: ONNX Export, Dynamic Axis Shape Mismatch Prevention
Constraints: 4GB RAM Quota

AUDIT FIXES APPLIED:
- Fixed ONNX export dynamic axis shape mismatches
- Added validation for distilled MLP shapes
- Prevented tensor shape corruption during export
"""

from __future__ import annotations
import torch
import torch.nn as nn
import onnx
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DistilledMLP(nn.Module):
    """
    Distilled MLP for knowledge transfer.
    FIX: Ensures consistent tensor shapes for ONNX export.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[-1]}")
        return self.network(x)


class KnowledgeDistiller:
    """
    Knowledge distillation with ONNX export safety.
    FIX: Validates dynamic axes for ONNX export.
    """
    
    def __init__(self, teacher: nn.Module, student: DistilledMLP, temperature: float = 2.0):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        
    def distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss with KL divergence."""
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_div = nn.KLDivLoss(reduction='batchmean')(
            student_log_probs, 
            teacher_probs
        )
        return kl_div * (self.temperature ** 2)
    
    def export_to_onnx(
        self, 
        output_path: str, 
        batch_size: int = 1,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> bool:
        """
        Export student model to ONNX with shape validation.
        FIX: Ensures dynamic axes are correctly specified.
        """
        self.student.eval()
        
        # Create dummy input with explicit batch dimension
        dummy_input = torch.randn(batch_size, self.student.input_dim)
        
        # Default dynamic axes for variable batch sizes
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        try:
            # Export with validation
            torch.onnx.export(
                self.student,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True
            )
            
            # Validate exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"Successfully exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def validate_shapes(self, sample_input: torch.Tensor) -> Tuple[bool, str]:
        """Validate tensor shapes before export."""
        expected_shape = (sample_input.shape[0], self.student.input_dim)
        actual_shape = tuple(sample_input.shape)
        
        if actual_shape != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        
        # Run forward pass to check output shape
        with torch.no_grad():
            output = self.student(sample_input)
        
        if output.shape[-1] != self.student.output_dim:
            return False, f"Output dim mismatch: expected {self.student.output_dim}, got {output.shape[-1]}"
        
        return True, "Shapes validated successfully"


if __name__ == "__main__":
    print("Knowledge distillation module loaded")
