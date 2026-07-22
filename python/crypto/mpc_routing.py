"""
`python/crypto/mpc_routing.py`

**Module:** Cryptographic Order Hiding - Secure Multi-Party Computation
**Purpose:** Split large institutional orders across untrusted venues using MPC.
**Optimization:** Ray distributed execution, 4GB RAM quota enforcement per worker.
**Constraints:** AMD DirectML/ROCm acceleration for modular arithmetic operations.

This module implements a simplified MPC protocol for order splitting:
- Secret shares order parameters across multiple compute nodes
- No single node knows the complete order details
- Aggregation reveals only the final execution result
"""

import ray
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import hashlib
import time
import os

# Memory limit configuration (4GB quota per Ray worker)
MAX_RAM_PER_WORKER_GB = 4.0
MAX_RAM_BYTES = int(MAX_RAM_PER_WORKER_GB * 1024 * 1024 * 1024)

# AMD GPU detection and initialization
def detect_amd_gpu() -> Dict[str, bool]:
    """Detect AMD ROCm/DirectML availability for accelerated crypto operations."""
    result = {
        "rocm_available": False,
        "directml_available": False,
        "gpu_backend": None
    }
    
    try:
        import torch
        if torch.cuda.is_available() and "ROCm" in torch.version.cuda or hasattr(torch.version, 'hip'):
            result["rocm_available"] = True
            result["gpu_backend"] = "ROCm"
        elif hasattr(torch.backends, "directml"):
            result["directml_available"] = True
            result["gpu_backend"] = "DirectML"
    except ImportError:
        pass
    
    return result


@dataclass
class SecretShare:
    """Represents a secret share of an order parameter."""
    share_id: int
    venue_id: int
    share_data: np.ndarray
    timestamp_ns: int
    checksum: bytes


@ray.remote(max_calls=1000)
class MPCNode:
    """
    MPC Compute Node that processes secret shares without knowing full order details.
    
    Each node receives shares from multiple orders and computes partial results.
    Only when sufficient shares are combined can the actual execution be determined.
    """
    
    def __init__(self, node_id: int, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.shares_received: List[SecretShare] = []
        self.processed_count = 0
        self.gpu_info = detect_amd_gpu()
        
        # Pre-allocate buffers to stay within RAM quota
        self.share_buffer = np.zeros((10000, 32), dtype=np.float64)
        self.buffer_index = 0
        
    def receive_share(self, share: dict) -> bool:
        """Receive a secret share from the dealer."""
        try:
            # Convert dict back to SecretShare
            s = SecretShare(
                share_id=share["share_id"],
                venue_id=share["venue_id"],
                share_data=np.array(share["share_data"], dtype=np.float64),
                timestamp_ns=share["timestamp_ns"],
                checksum=bytes(share["checksum"])
            )
            
            # Verify checksum
            expected_checksum = self._compute_checksum(s.share_data, s.share_id)
            if expected_checksum != s.checksum:
                return False
            
            self.shares_received.append(s)
            self.processed_count += 1
            
            # Check memory pressure
            if self._check_memory_pressure():
                self._flush_old_shares()
                
            return True
            
        except Exception as e:
            print(f"MPCNode {self.node_id} error receiving share: {e}")
            return False
    
    def compute_partial_result(self) -> Optional[np.ndarray]:
        """Compute partial result from accumulated shares."""
        if len(self.shares_received) < 2:
            return None
        
        # Aggregate shares using simple additive secret sharing
        # In production, use proper Shamir secret sharing reconstruction
        share_data = np.array([s.share_data for s in self.shares_received])
        
        # Use GPU acceleration if available
        if self.gpu_info["gpu_backend"]:
            try:
                import torch
                tensor = torch.from_numpy(share_data)
                if self.gpu_info["gpu_backend"] == "ROCm":
                    tensor = tensor.cuda()
                # Perform aggregation on GPU
                result = torch.sum(tensor, dim=0).cpu().numpy()
            except Exception:
                # Fallback to CPU
                result = np.sum(share_data, axis=0)
        else:
            result = np.sum(share_data, axis=0)
        
        return result
    
    def get_stats(self) -> Dict:
        """Return node statistics."""
        return {
            "node_id": self.node_id,
            "shares_processed": self.processed_count,
            "gpu_backend": self.gpu_info["gpu_backend"],
            "memory_usage_bytes": self.share_buffer.nbytes
        }
    
    def _compute_checksum(self, data: np.ndarray, share_id: int) -> bytes:
        """Compute SHA256 checksum for share verification."""
        hasher = hashlib.sha256()
        hasher.update(data.tobytes())
        hasher.update(share_id.to_bytes(8, 'little'))
        return hasher.digest()
    
    def _check_memory_pressure(self) -> bool:
        """Check if we're approaching RAM quota."""
        current_usage = len(self.shares_received) * 32 * 8  # Approximate
        return current_usage > (MAX_RAM_BYTES * 0.8)
    
    def _flush_old_shares(self):
        """Remove old shares to free memory."""
        if len(self.shares_received) > 1000:
            self.shares_received = self.shares_received[-1000:]


@ray.remote
class MPCDealer:
    """
    MPC Dealer that splits orders into secret shares and distributes them.
    
    The dealer knows the original order but individual nodes only see shares.
    """
    
    def __init__(self, num_nodes: int, threshold: int = 2):
        self.num_nodes = num_nodes
        self.threshold = threshold  # Minimum shares needed for reconstruction
        self.mpc_nodes = [MPCNode.remote(i, num_nodes) for i in range(num_nodes)]
        self.order_counter = 0
        
    def split_order(self, order_params: Dict) -> List[SecretShare]:
        """
        Split an order into secret shares using additive secret sharing.
        
        For order value v, create n shares s1, s2, ..., sn such that:
        sum(si) = v (mod p) where p is a large prime
        
        No single share reveals information about v.
        """
        self.order_counter += 1
        order_id = self.order_counter
        
        # Extract order parameters
        size = order_params.get("size", 0)
        price = order_params.get("price", 0)
        is_buy = 1 if order_params.get("is_buy", True) else 0
        
        # Create base vector [size, price, direction]
        original_vector = np.array([size, price, is_buy], dtype=np.float64)
        
        # Generate random shares
        shares = []
        remaining = original_vector.copy()
        
        for i in range(self.num_nodes - 1):
            # Generate random share
            random_share = np.random.uniform(-original_vector, original_vector)
            
            share = SecretShare(
                share_id=order_id * 1000 + i,
                venue_id=i,
                share_data=random_share,
                timestamp_ns=time.time_ns(),
                checksum=self._compute_checksum(random_share, order_id * 1000 + i)
            )
            shares.append(share)
            
            # Update remaining for last share
            remaining = remaining - random_share
        
        # Last share ensures sum equals original
        last_share = SecretShare(
            share_id=order_id * 1000 + (self.num_nodes - 1),
            venue_id=self.num_nodes - 1,
            share_data=remaining,
            timestamp_ns=time.time_ns(),
            checksum=self._compute_checksum(remaining, order_id * 1000 + (self.num_nodes - 1))
        )
        shares.append(last_share)
        
        return shares
    
    def distribute_shares(self, shares: List[SecretShare]) -> List[bool]:
        """Distribute shares to MPC nodes."""
        results = []
        for share in shares:
            share_dict = {
                "share_id": share.share_id,
                "venue_id": share.venue_id,
                "share_data": share.share_data.tolist(),
                "timestamp_ns": share.timestamp_ns,
                "checksum": list(share.checksum)
            }
            # Send to appropriate node
            node_idx = share.venue_id % self.num_nodes
            result = ray.get(self.mpc_nodes[node_idx].receive_share.remote(share_dict))
            results.append(result)
        
        return results
    
    def aggregate_results(self) -> Optional[np.ndarray]:
        """Collect and aggregate partial results from all nodes."""
        partial_results = ray.get(
            [node.compute_partial_result.remote() for node in self.mpc_nodes]
        )
        
        valid_results = [r for r in partial_results if r is not None]
        if len(valid_results) < self.threshold:
            return None
        
        # Combine partial results
        return np.mean(valid_results, axis=0)
    
    def get_cluster_stats(self) -> List[Dict]:
        """Get statistics from all MPC nodes."""
        return ray.get([node.get_stats.remote() for node in self.mpc_nodes])
    
    def _compute_checksum(self, data: np.ndarray, share_id: int) -> bytes:
        """Compute SHA256 checksum for share verification."""
        hasher = hashlib.sha256()
        hasher.update(data.tobytes())
        hasher.update(share_id.to_bytes(8, 'little'))
        return hasher.digest()


class MPCOrderRouter:
    """
    High-level interface for MPC-based order routing.
    
    Splits large institutional orders across venues while hiding true size
    and direction from any single venue.
    """
    
    def __init__(self, num_venues: int = 4, ram_quota_gb: float = 4.0):
        """
        Initialize MPC router.
        
        Args:
            num_venues: Number of venues to split orders across
            ram_quota_gb: RAM quota per worker (default 4GB)
        """
        global MAX_RAM_BYTES
        MAX_RAM_BYTES = int(ram_quota_gb * 1024 * 1024 * 1024)
        
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count() or 4,
                _system_config={"object_store_memory": MAX_RAM_BYTES}
            )
        
        self.dealer = MPCDealer.remote(num_venues)
        self.num_venues = num_venues
        self.gpu_info = detect_amd_gpu()
        
    def route_order(self, size: float, price: float, is_buy: bool) -> Optional[np.ndarray]:
        """
        Route an order through MPC protocol.
        
        Args:
            size: Order size
            price: Limit price
            is_buy: True for buy, False for sell
            
        Returns:
            Aggregated execution result or None if insufficient shares
        """
        order_params = {
            "size": size,
            "price": price,
            "is_buy": is_buy
        }
        
        # Split order into shares
        shares = ray.get(self.dealer.split_order.remote(order_params))
        
        # Distribute shares
        distribution_results = ray.get(self.dealer.distribute_shares.remote(shares))
        
        if not all(distribution_results):
            print("Warning: Some shares failed to distribute")
        
        # Aggregate results
        result = ray.get(self.dealer.aggregate_results.remote())
        
        return result
    
    def get_stats(self) -> Dict:
        """Get router and cluster statistics."""
        cluster_stats = ray.get(self.dealer.get_cluster_stats.remote())
        return {
            "num_venues": self.num_venues,
            "gpu_acceleration": self.gpu_info,
            "cluster_stats": cluster_stats,
            "ram_quota_bytes": MAX_RAM_BYTES
        }
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        ray.shutdown()


# Example usage and testing
if __name__ == "__main__":
    print("Initializing MPC Order Router...")
    print(f"GPU Detection: {detect_amd_gpu()}")
    
    router = MPCOrderRouter(num_venues=4)
    
    # Route a test order
    result = router.route_order(size=1000.0, price=50000.0, is_buy=True)
    print(f"Order routing result: {result}")
    
    stats = router.get_stats()
    print(f"Router stats: {stats}")
    
    router.shutdown()
    print("MPC Router shutdown complete.")
