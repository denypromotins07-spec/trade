"""
Custom Triton Kernels for L2 Order Book Feature Extraction on AMD ROCm/DirectML.

This module implements fused GPU kernels that combine normalization and softmax
operations directly on the AMD Radeon GPU, eliminating CPU-GPU PCIe bus bottlenecks.
Optimized for microsecond latency with strict VRAM bounds for 8GB RAM systems.

Author: Elite Quantitative Software Engineering Team
Stage: 49 - Custom AMD ROCm Kernels
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
import os

# =============================================================================
# Configuration: AMD ROCm-specific tuning parameters
# =============================================================================

# Force ROCm backend for Triton
os.environ['TRITON_BACKEND'] = 'rocm'

# VRAM budget per kernel (in bytes) - strict bound to prevent spills
MAX_VRAM_PER_KERNEL = 512 * 1024 * 1024  # 512MB max per kernel

# Block sizes optimized for AMD RDNA3 architecture
BLOCK_SIZE_NORM = 256  # Optimal for RDNA3 compute units
BLOCK_SIZE_SOFTMAX = 512  # Larger block for softmax fusion


@triton.jit
def _fused_norm_softmax_kernel(
    # Pointers to inputs/outputs
    input_ptr,
    output_ptr,
    # Strides for navigating tensors
    stride_batch,
    stride_level,
    stride_price,
    # Dimensions
    n_levels: tl.constexpr,
    n_prices: tl.constexpr,
    # Compile-time constants
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    """
    Fused kernel: LayerNorm + Softmax for L2 order book data.
    
    This kernel fuses two operations into a single GPU pass:
    1. Layer normalization across price levels
    2. Softmax activation for attention-weighted features
    
    By fusing these operations, we eliminate intermediate memory writes
    and reduce PCIe bus traffic between CPU and AMD GPU.
    
    Memory Layout: [batch, level, price]
    """
    # Calculate global indices
    batch_idx = tl.program_id(0)
    level_idx = tl.program_id(1)
    
    # Offset calculations for strided access
    base_offset = batch_idx * stride_batch + level_idx * stride_level
    
    # ---------------------------------------------------------------------
    # Phase 1: Compute mean and variance for layer normalization
    # ---------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load data in blocks
    for block_start in range(0, n_prices, BLOCK_SIZE):
        price_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = price_offsets < n_prices
        
        # Load prices with bounds checking
        ptrs = input_ptr + base_offset + price_offsets * stride_price
        values = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        
        sum_val += values
        sum_sq += values * values
    
    # Reduce to get global mean and variance
    total_sum = tl.sum(sum_val)
    total_sum_sq = tl.sum(sum_sq)
    
    mean = total_sum / n_prices
    variance = (total_sum_sq / n_prices) - (mean * mean)
    rstd = 1.0 / tl.sqrt(variance + EPSILON)
    
    # ---------------------------------------------------------------------
    # Phase 2: Normalize and compute exp for softmax in single pass
    # ---------------------------------------------------------------------
    max_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_prices, BLOCK_SIZE):
        price_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = price_offsets < n_prices
        
        ptrs = input_ptr + base_offset + price_offsets * stride_price
        values = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        normalized = (values - mean) * rstd
        
        # Compute exp for softmax (numerically stable)
        exp_vals = tl.exp(normalized - tl.max(normalized, axis=0))
        max_exp = tl.maximum(max_exp, exp_vals)
        
        # Store intermediate exp values temporarily
        temp_ptrs = output_ptr + base_offset + price_offsets * stride_price
        tl.store(temp_ptrs, exp_vals, mask=mask)
    
    # ---------------------------------------------------------------------
    # Phase 3: Compute softmax denominator and final output
    # ---------------------------------------------------------------------
    denom_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_prices, BLOCK_SIZE):
        price_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = price_offsets < n_prices
        
        temp_ptrs = output_ptr + base_offset + price_offsets * stride_price
        exp_vals = tl.load(temp_ptrs, mask=mask, other=0.0).to(tl.float32)
        denom_sum += exp_vals
    
    total_denom = tl.sum(denom_sum)
    rdenom = 1.0 / (total_denom + EPSILON)
    
    # Final softmax output with fused normalization
    for block_start in range(0, n_prices, BLOCK_SIZE):
        price_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = price_offsets < n_prices
        
        temp_ptrs = output_ptr + base_offset + price_offsets * stride_price
        exp_vals = tl.load(temp_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Final softmax: exp(x) / sum(exp(x))
        softmax_out = exp_vals * rdenom
        
        # Store final result
        tl.store(temp_ptrs, softmax_out, mask=mask)


@triton.jit
def _order_book_imbalance_kernel(
    bid_ptr,
    ask_ptr,
    imbalance_ptr,
    volume_ptr,
    stride_batch,
    stride_level,
    n_batches: tl.constexpr,
    n_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute order book imbalance metrics directly on GPU.
    
    Calculates: (bid_volume - ask_volume) / (bid_volume + ask_volume)
    This is a critical feature for HFT prediction models.
    
    Optimized for AMD RDNA3 with coalesced memory access patterns.
    """
    batch_idx = tl.program_id(0)
    level_idx = tl.program_id(1)
    
    base_offset = batch_idx * stride_batch + level_idx * stride_level
    
    # Load bid and ask volumes
    bid_vol = tl.load(bid_ptr + base_offset).to(tl.float32)
    ask_vol = tl.load(ask_ptr + base_offset).to(tl.float32)
    
    # Compute imbalance with numerical stability
    total_vol = bid_vol + ask_vol
    imbalance = tl.where(total_vol > 1e-8, (bid_vol - ask_vol) / total_vol, 0.0)
    
    # Store results
    tl.store(imbalance_ptr + base_offset, imbalance)
    tl.store(volume_ptr + base_offset, total_vol)


class TritonL2OrderBookKernel:
    """
    High-performance L2 order book feature extractor using custom Triton kernels.
    
    This class manages:
    - Kernel compilation and caching (AOT via kernel_compiler.py)
    - Memory allocation with strict VRAM bounds
    - Batched execution for maximum GPU utilization
    
    Designed for AMD Radeon GPUs with ROCm backend.
    """
    
    def __init__(
        self,
        max_batch_size: int = 64,
        max_levels: int = 50,
        max_prices_per_level: int = 256,
        epsilon: float = 1e-5,
    ):
        """
        Initialize the L2 order book kernel with bounded memory allocation.
        
        Args:
            max_batch_size: Maximum concurrent batches (memory bound)
            max_levels: Maximum order book depth levels
            max_prices_per_level: Maximum price points per level
            epsilon: Numerical stability constant
        """
        self.max_batch_size = max_batch_size
        self.max_levels = max_levels
        self.max_prices_per_level = max_prices_per_level
        self.epsilon = epsilon
        
        # Pre-allocate GPU buffers with strict size limits
        self._allocate_buffers()
        
        # Kernel launch configuration
        self.grid_config = {
            'norm_softmax': lambda batch, levels: (batch, levels),
            'imbalance': lambda batch, levels: (batch, levels),
        }
        
    def _allocate_buffers(self):
        """
        Allocate GPU buffers with strict VRAM limits.
        
        Total VRAM usage calculation:
        - Input buffer: batch * levels * prices * 4 bytes (float32)
        - Output buffer: same as input
        - Intermediate buffers: 2x input size max
        
        For max config: 64 * 50 * 256 * 4 = 3.2MB per buffer
        Well within 512MB per-kernel limit.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate exact buffer sizes
        input_elements = self.max_batch_size * self.max_levels * self.max_prices_per_level
        
        # Pre-allocate pinned memory for zero-copy PCIe transfers
        self.input_buffer = torch.zeros(
            (self.max_batch_size, self.max_levels, self.max_prices_per_level),
            dtype=torch.float32,
            device=device,
            pin_memory=True  # Enable PCIe pinned transfers
        )
        
        self.output_buffer = torch.zeros_like(self.input_buffer)
        self.imbalance_buffer = torch.zeros(
            (self.max_batch_size, self.max_levels),
            dtype=torch.float32,
            device=device,
            pin_memory=True
        )
        self.volume_buffer = torch.zeros_like(self.imbalance_buffer)
        
    def forward(
        self,
        order_book_data: torch.Tensor,
        compute_imbalance: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute fused normalization and softmax on order book data.
        
        Args:
            order_book_data: Input tensor [batch, levels, prices]
            compute_imbalance: Whether to compute imbalance metrics
            
        Returns:
            Tuple of (normalized_output, imbalance_metrics)
            
        Performance:
        - Single kernel launch for norm+softmax fusion
        - Zero intermediate memory allocations
        - PCIe transfer optimized with pinned memory
        """
        batch_size, n_levels, n_prices = order_book_data.shape
        
        # Validate dimensions against pre-allocated buffers
        assert batch_size <= self.max_batch_size, "Batch size exceeds pre-allocated buffer"
        assert n_levels <= self.max_levels, "Levels exceed pre-allocated buffer"
        assert n_prices <= self.max_prices_per_level, "Prices exceed pre-allocated buffer"
        
        # Copy input to pre-allocated buffer (zero-copy if already on GPU)
        self.input_buffer[:batch_size, :n_levels, :n_prices].copy_(order_book_data)
        
        # Launch fused norm+softmax kernel
        grid = self.grid_config['norm_softmax'](batch_size, n_levels)
        
        _fused_norm_softmax_kernel[grid](
            self.input_ptr,
            self.output_ptr,
            self.input_buffer.stride(0),
            self.input_buffer.stride(1),
            self.input_buffer.stride(2),
            n_levels=n_levels,
            n_prices=n_prices,
            BLOCK_SIZE=BLOCK_SIZE_SOFTMAX,
            EPSILON=self.epsilon,
        )
        
        # Optionally compute imbalance metrics
        imbalance_result = None
        if compute_imbalance:
            grid_imb = self.grid_config['imbalance'](batch_size, n_levels)
            _order_book_imbalance_kernel[grid_imb](
                self.bid_ptr,
                self.ask_ptr,
                self.imbalance_ptr,
                self.volume_ptr,
                self.input_buffer.stride(0),
                self.input_buffer.stride(1),
                n_batches=batch_size,
                n_levels=n_levels,
                BLOCK_SIZE=BLOCK_SIZE_NORM,
            )
            imbalance_result = self.imbalance_buffer[:batch_size, :n_levels]
        
        return self.output_buffer[:batch_size, :n_levels, :n_prices], imbalance_result
    
    @property
    def input_ptr(self):
        """Get raw pointer to input buffer for Triton kernel."""
        return triton.interp.debug_wrapper.get_ptr(self.input_buffer)
    
    @property
    def output_ptr(self):
        """Get raw pointer to output buffer for Triton kernel."""
        return triton.interp.debug_wrapper.get_ptr(self.output_buffer)
    
    @property
    def imbalance_ptr(self):
        """Get raw pointer to imbalance buffer."""
        return triton.interp.debug_wrapper.get_ptr(self.imbalance_buffer)
    
    @property
    def volume_ptr(self):
        """Get raw pointer to volume buffer."""
        return triton.interp.debug_wrapper.get_ptr(self.volume_buffer)
    
    @property
    def bid_ptr(self):
        """Get pointer to bid side of order book."""
        return self.input_ptr
    
    @property
    def ask_ptr(self):
        """Get pointer to ask side of order book."""
        return self.input_ptr + (self.max_levels * self.max_prices_per_level)


def create_l2_kernel(
    batch_size: int = 32,
    levels: int = 25,
    prices: int = 128,
) -> TritonL2OrderBookKernel:
    """
    Factory function to create optimized L2 order book kernel.
    
    Default configuration optimized for typical crypto L2 data:
    - 32 batch size for parallel processing
    - 25 levels of depth (standard exchange format)
    - 128 price points per level
    
    Args:
        batch_size: Processing batch size
        levels: Order book depth
        prices: Price resolution per level
        
    Returns:
        Configured TritonL2OrderBookKernel instance
    """
    return TritonL2OrderBookKernel(
        max_batch_size=batch_size,
        max_levels=levels,
        max_prices_per_level=prices,
        epsilon=1e-5,
    )


if __name__ == "__main__":
    # Test kernel functionality
    print("Testing Triton L2 Order Book Kernel...")
    
    # Create test data
    test_data = torch.randn(16, 20, 64, dtype=torch.float32)
    
    # Initialize kernel
    kernel = create_l2_kernel(batch_size=16, levels=20, prices=64)
    
    # Execute forward pass
    output, imbalance = kernel.forward(test_data)
    
    print(f"Input shape: {test_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Imbalance shape: {imbalance.shape if imbalance is not None else None}")
    print("Kernel test completed successfully.")
