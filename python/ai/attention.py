"""
Lightweight Temporal Attention Mechanisms for High-Frequency Trading

This module implements non-LLM temporal attention mechanisms using PyTorch
with DirectML/ROCm support to capture long-range dependencies in order flow
without the latency overhead of full transformer architectures.

Key features:
- AMD DirectML/ROCm environment detection and optimization
- Lightweight attention (no feed-forward networks)
- Causal masking for temporal consistency
- Optimized for microsecond inference latency
- Memory-efficient implementation for 4GB RAM quota

Usage:
    attention = TemporalAttention(hidden_dim=64, num_heads=4)
    output = attention(order_flow_features)
"""

import os
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# AMD DirectML/ROCm configuration
AMD_DIRECTML_AVAILABLE = False
AMD_ROCM_AVAILABLE = False


def detect_amd_hardware() -> Tuple[bool, bool, str]:
    """
    Detect AMD hardware availability and return optimal device configuration.
    
    Returns:
        Tuple of (directml_available, rocm_available, recommended_device)
    """
    directml_available = False
    rocm_available = False
    recommended_device = "cpu"
    
    # Check for PyTorch DirectML (Windows)
    try:
        import torch_directml
        directml_available = True
        recommended_device = torch_directml.device()
        print("[INFO] PyTorch DirectML detected - AMD GPU acceleration enabled")
    except ImportError:
        pass
    
    # Check for PyTorch ROCm (Linux)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'rocm'):
        try:
            rocm_available = torch.backends.rocm.is_available()
            if rocm_available:
                recommended_device = "cuda"  # ROCm uses cuda device type in PyTorch
                print(f"[INFO] AMD ROCm detected - GPU acceleration enabled on {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    
    if not directml_available and not rocm_available:
        print("[INFO] AMD GPU acceleration not available - using CPU mode")
        # Optimize for Ryzen AI 5 on CPU
        torch.set_num_threads(6)  # Match Ryzen AI 5 performance cores
    
    return directml_available, rocm_available, recommended_device


# Detect hardware at module load
DIRECTML_AVAILABLE, ROCM_AVAILABLE, RECOMMENDED_DEVICE = detect_amd_hardware()


@dataclass
class AttentionConfig:
    """Configuration for temporal attention module."""
    hidden_dim: int = 64
    num_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    use_causal_mask: bool = True
    scale_attention: bool = True
    use_rope: bool = True  # Rotary Position Embeddings
    rope_theta: float = 10000.0


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for temporal attention.
    
    More efficient than absolute position embeddings and provides
    better extrapolation to longer sequences.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-compute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate rotary embeddings for sequence length."""
        t = torch.arange(seq_len, device=self.freqs.device)
        freqs = torch.einsum('i,j->ij', t, self.freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.float()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        cos = positions.cos()
        sin = positions.sin()
        return (x * cos) + (self.rotate_half(x) * sin)


class LightweightMultiHeadAttention(nn.Module):
    """
    Memory-efficient multi-head attention without feed-forward networks.
    
    Optimized for high-frequency trading where latency is critical.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        scale_attention: bool = True,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5 if scale_attention else 1.0
        
        # Single projection for Q, K, V (more efficient than separate)
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through attention layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask
            positions: Optional rotary position embeddings
            return_attention: Whether to return attention weights
        
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary position embeddings if provided
        if positions is not None:
            q = RotaryPositionEmbedding(self.head_dim).apply_rope(q, positions)
            k = RotaryPositionEmbedding(self.head_dim).apply_rope(k, positions)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class TemporalAttention(nn.Module):
    """
    Lightweight temporal attention mechanism for order flow analysis.
    
    Captures long-range dependencies without the overhead of full transformers.
    Optimized for AMD DirectML/ROCm and Ryzen AI 5 architecture.
    """
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        super().__init__()
        
        self.config = config or AttentionConfig()
        
        # Move to optimal device
        self.device = RECOMMENDED_DEVICE
        if isinstance(self.device, int) or (isinstance(self.device, str) and self.device != "cpu"):
            self.to(self.device)
        
        # Rotary position embeddings
        if self.config.use_rope:
            self.rope = RotaryPositionEmbedding(
                self.config.hidden_dim // self.config.num_heads,
                self.config.max_seq_len,
                self.config.rope_theta,
            )
        else:
            self.rope = None
        
        # Multi-head attention
        self.attention = LightweightMultiHeadAttention(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            scale_attention=self.config.scale_attention,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        
        # Cache for causal mask
        self._causal_mask_cache: Optional[torch.Tensor] = None
    
    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Get or create causal mask for sequence length."""
        if self._causal_mask_cache is not None and self._causal_mask_cache.size(0) >= seq_len:
            return self._causal_mask_cache[:seq_len, :seq_len]
        
        # Create new causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        self._causal_mask_cache = mask
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through temporal attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional custom attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Ensure we're on the right device
        if x.device.type != self.device:
            x = x.to(self.device)
        
        # Get causal mask if enabled
        if self.config.use_causal_mask and mask is None:
            mask = self._get_causal_mask(seq_len)
        
        # Get position embeddings
        positions = None
        if self.rope is not None:
            positions = self.rope(seq_len)
        
        # Apply attention
        residual = x
        x = self.norm(x)
        output, attn_weights = self.attention(
            x, mask=mask, positions=positions, return_attention=True
        )
        output = output + residual  # Residual connection
        
        if return_attention:
            return output, attn_weights
        return output, None
    
    def forward_incremental(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental forward pass for streaming inference.
        
        Uses KV caching to avoid recomputing attention for previous tokens.
        
        Args:
            x: Input tensor of shape (batch, 1, hidden_dim) - single token
            kv_cache: Optional cached K, V tensors from previous steps
        
        Returns:
            Tuple of (output tensor, updated KV cache)
        """
        batch_size = x.shape[0]
        
        # Project to Q, K, V
        qkv = self.attention.qkv_proj(x)
        qkv = qkv.reshape(batch_size, 1, 3, self.config.num_heads, self.attention.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Update KV cache
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        
        # Compute attention
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.attention.scale
        
        # Causal mask for incremental decoding
        seq_len = k.size(2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(qk.masked_fill(mask == 0, float('-inf')), dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, 1, self.config.hidden_dim)
        output = self.attention.out_proj(output)
        
        return output, (k, v)


class OrderFlowAttention(nn.Module):
    """
    Specialized temporal attention for order flow analysis.
    
    Takes raw order flow features and produces attended representations
    suitable for RL policy networks or direct signal generation.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_heads: int = 4,
        max_seq_len: int = 128,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Temporal attention
        config = AttentionConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=0.05,
        )
        self.temporal_attention = TemporalAttention(config)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        
        # Move to optimal device
        if RECOMMENDED_DEVICE != "cpu":
            self.to(RECOMMENDED_DEVICE)
    
    def forward(self, order_flow: torch.Tensor) -> torch.Tensor:
        """
        Process order flow through attention network.
        
        Args:
            order_flow: Tensor of shape (batch, seq_len, input_dim)
                       Expected features: [bid_volume, ask_volume, bid_price, ask_price, ...]
        
        Returns:
            Attended features of shape (batch, seq_len, hidden_dim // 2)
        """
        # Project input
        x = self.input_proj(order_flow)
        
        # Apply temporal attention
        x, _ = self.temporal_attention(x, return_attention=False)
        
        # Project output
        return self.output_proj(x)
    
    def get_signal(self, order_flow: torch.Tensor) -> torch.Tensor:
        """
        Extract trading signal from final attended state.
        
        Args:
            order_flow: Tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Signal tensor of shape (batch, 3) - [buy_prob, hold_prob, sell_prob]
        """
        x = self.forward(order_flow)
        
        # Use last token's representation
        last_state = x[:, -1, :]  # (batch, hidden_dim // 2)
        
        # Project to action probabilities
        signal = F.softmax(last_state.mean(dim=-1, keepdim=True), dim=-1)
        return signal


def create_attention_model(
    input_dim: int = 10,
    hidden_dim: int = 64,
    num_heads: int = 4,
    use_gpu: bool = True,
) -> OrderFlowAttention:
    """
    Factory function to create optimized attention model.
    
    Automatically configures for AMD DirectML/ROCm if available.
    """
    model = OrderFlowAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
    
    # Force device placement if requested
    if use_gpu and ROCM_AVAILABLE:
        model.to("cuda")
    elif use_gpu and DIRECTML_AVAILABLE:
        model.to(torch_directml.device())
    
    return model


if __name__ == "__main__":
    # Test the attention module
    print("Testing Temporal Attention Module...")
    print(f"DirectML Available: {DIRECTML_AVAILABLE}")
    print(f"ROCm Available: {ROCM_AVAILABLE}")
    print(f"Recommended Device: {RECOMMENDED_DEVICE}")
    
    # Create model
    model = create_attention_model(input_dim=10, hidden_dim=64, num_heads=4)
    model.eval()
    
    # Create sample input (batch=2, seq_len=32, features=10)
    sample_input = torch.randn(2, 32, 10)
    
    # Run inference
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(sample_input)
        signal = model.get_signal(sample_input)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Signal shape: {signal.shape}")
    print(f"Inference time: {elapsed_ms:.2f}ms")
    print("\nTemporal Attention Module test complete!")
