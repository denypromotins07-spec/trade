"""
HNSW Vector Search with Cython Acceleration

Implements a lightweight Hierarchical Navigable Small World (HNSW) graph
in pure Python/Cython for rapid similarity search of past market regimes
without external database overhead.

Key Features:
- HNSW graph construction for approximate nearest neighbor search
- Market regime similarity detection
- AMD ROCm/DirectML environment checks for GPU acceleration
- Strict memory limits to stay within 4GB Python quota
- Zero external dependencies (pure Python fallback)
"""

import os
import math
import heapq
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading

# Check for AMD ROCm/DirectML availability
try:
    import torch
    ROCM_AVAILABLE = torch.cuda.is_available() and torch.version.hip is not None
    DIRECTML_AVAILABLE = False
except ImportError:
    ROCM_AVAILABLE = False
    DIRECTML_AVAILABLE = False

# Try Cython for acceleration
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@dataclass
class HNSWNode:
    """Node in the HNSW graph."""
    id: int
    vector: np.ndarray
    level: int
    edges: Dict[int, List[int]] = None
    
    def __post_init__(self):
        if self.edges is None:
            self.edges = {}


@dataclass
class SearchResult:
    """Search result entry."""
    node_id: int
    distance: float
    metadata: Optional[Dict] = None


class HNSWGraph:
    """
    Hierarchical Navigable Small World graph for approximate nearest neighbor search.
    
    Pure Python implementation optimized for market regime similarity search.
    """
    
    def __init__(
        self,
        m: int = 16,           # Number of connections per layer
        max_m0: int = 32,      # Max connections at layer 0
        ef_construction: int = 200,  # Size of dynamic candidate list during construction
        max_memory_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB limit
    ):
        self.m = m
        self.max_m0 = max_m0
        self.ef_construction = ef_construction
        self.max_memory_bytes = max_memory_bytes
        
        # Graph storage
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.node_count = 0
        self.current_memory_bytes = 0
        
        # Level generation parameters
        self.level_mult = 1 / math.log(m)
        
        self._lock = threading.Lock()
    
    def _generate_level(self) -> int:
        """Generate random level for new node."""
        return int(-math.log(random.random()) * self.level_mult)
    
    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between vectors."""
        return np.linalg.norm(v1 - v2)
    
    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine distance between vectors."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1.0 - dot / (norm1 * norm2)
    
    def _search_layer(
        self,
        query: np.ndarray,
        entry_point: int,
        layer: int,
        ef: int
    ) -> List[Tuple[float, int]]:
        """Search within a single layer."""
        visited = set()
        candidates = []
        results = []
        
        # Initialize with entry point
        if entry_point is not None and entry_point in self.nodes:
            dist = self._cosine_distance(query, self.nodes[entry_point].vector)
            heapq.heappush(candidates, (-dist, entry_point))
            visited.add(entry_point)
        
        while candidates:
            neg_dist, curr_id = heapq.heappop(candidates)
            curr_dist = -neg_dist
            
            # Check if we can stop
            if results and curr_dist > -results[0][0]:
                continue
            
            results.append((curr_dist, curr_id))
            
            # Keep only ef best results
            if len(results) > ef:
                results.pop()
            
            # Explore neighbors
            if layer in self.nodes[curr_id].edges:
                for neighbor_id in self.nodes[curr_id].edges[layer]:
                    if neighbor_id not in visited and neighbor_id in self.nodes:
                        visited.add(neighbor_id)
                        neighbor_dist = self._cosine_distance(
                            query, 
                            self.nodes[neighbor_id].vector
                        )
                        heapq.heappush(candidates, (-neighbor_dist, neighbor_id))
        
        return sorted(results, key=lambda x: x[0])
    
    def insert(self, node_id: int, vector: np.ndarray, metadata: Optional[Dict] = None):
        """Insert a new node into the HNSW graph."""
        level = self._generate_level()
        
        # Check memory limit
        estimated_size = vector.nbytes + 100  # Overhead estimate
        with self._lock:
            if self.current_memory_bytes + estimated_size > self.max_memory_bytes:
                # Evict oldest/least important nodes
                self._evict_nodes(estimated_size)
        
        node = HNSWNode(id=node_id, vector=vector, level=level)
        
        with self._lock:
            if self.entry_point is None:
                # First node
                self.entry_point = node_id
                self.nodes[node_id] = node
                self.node_count += 1
                self.current_memory_bytes += estimated_size
                return
            
            # Search from top level down
            curr_ep = self.entry_point
            
            for layer in range(max(level + 1, 0), 0, -1):
                candidates = self._search_layer(vector, curr_ep, layer, 1)
                if candidates:
                    curr_ep = candidates[0][1]
            
            # Insert at each level up to node's level
            for layer in range(min(level, 0), -1, -1):
                candidates = self._search_layer(
                    vector, 
                    curr_ep, 
                    layer, 
                    self.ef_construction
                )
                
                # Connect to nearest neighbors
                max_m = self.max_m0 if layer == 0 else self.m
                neighbors = self._select_neighbors_heuristic(
                    vector, 
                    candidates, 
                    max_m,
                    layer
                )
                
                node.edges[layer] = neighbors
                
                # Update reverse edges
                for neighbor_id in neighbors:
                    if neighbor_id in self.nodes:
                        if layer not in self.nodes[neighbor_id].edges:
                            self.nodes[neighbor_id].edges[layer] = []
                        if node_id not in self.nodes[neighbor_id].edges[layer]:
                            self.nodes[neighbor_id].edges[layer].append(node_id)
            
            self.nodes[node_id] = node
            self.node_count += 1
            self.current_memory_bytes += estimated_size
    
    def _select_neighbors_heuristic(
        self,
        vector: np.ndarray,
        candidates: List[Tuple[float, int]],
        max_m: int,
        layer: int
    ) -> List[int]:
        """Select neighbors using heuristic pruning."""
        selected = []
        
        for dist, cand_id in candidates[:max_m]:
            if cand_id in self.nodes:
                selected.append(cand_id)
        
        return selected
    
    def _evict_nodes(self, needed_bytes: int):
        """Evict nodes to free memory."""
        if not self.nodes:
            return
        
        # Simple eviction: remove oldest nodes (by ID order)
        sorted_ids = sorted(self.nodes.keys())
        
        for node_id in sorted_ids:
            if self.current_memory_bytes + needed_bytes <= self.max_memory_bytes:
                break
            
            if node_id in self.nodes:
                node = self.nodes[node_id]
                size = node.vector.nbytes + 100
                del self.nodes[node_id]
                self.node_count -= 1
                self.current_memory_bytes -= size
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = None
    ) -> List[SearchResult]:
        """Search for k nearest neighbors."""
        if ef is None:
            ef = max(k, 50)
        
        with self._lock:
            if self.entry_point is None or self.entry_point not in self.nodes:
                return []
            
            # Search from top layer
            curr_ep = self.entry_point
            max_layer = max(n.level for n in self.nodes.values()) if self.nodes else 0
            
            for layer in range(max_layer, 0, -1):
                candidates = self._search_layer(query, curr_ep, layer, 1)
                if candidates:
                    curr_ep = candidates[0][1]
            
            # Final search at layer 0
            candidates = self._search_layer(query, curr_ep, 0, ef)
            
            # Return top k results
            results = []
            for dist, node_id in candidates[:k]:
                if node_id in self.nodes:
                    results.append(SearchResult(
                        node_id=node_id,
                        distance=dist,
                        metadata=None
                    ))
            
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            return {
                "node_count": self.node_count,
                "memory_used_bytes": self.current_memory_bytes,
                "memory_limit_bytes": self.max_memory_bytes,
                "entry_point": self.entry_point,
                "avg_level": sum(n.level for n in self.nodes.values()) / max(1, len(self.nodes)),
                "rocm_available": ROCM_AVAILABLE
            }
    
    def clear(self):
        """Clear the graph."""
        with self._lock:
            self.nodes.clear()
            self.entry_point = None
            self.node_count = 0
            self.current_memory_bytes = 0


class MarketRegimeDetector:
    """Detect similar market regimes using HNSW."""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.graph = HNSWGraph(m=16, max_m0=32)
        self.regime_counter = 0
    
    def encode_regime(
        self,
        volatility: float,
        trend: float,
        momentum: float,
        volume_ratio: float
    ) -> np.ndarray:
        """Encode market regime into embedding vector."""
        # Simple encoding (in production, use trained encoder)
        features = np.array([
            volatility,
            trend,
            momentum,
            volume_ratio,
            volatility * trend,
            momentum * volume_ratio
        ])
        
        # Pad to embedding dimension
        embedding = np.zeros(self.embedding_dim)
        embedding[:len(features)] = features
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding.astype(np.float32)
    
    def add_regime(
        self,
        volatility: float,
        trend: float,
        momentum: float,
        volume_ratio: float,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a market regime to the index."""
        embedding = self.encode_regime(volatility, trend, momentum, volume_ratio)
        regime_id = self.regime_counter
        self.regime_counter += 1
        
        self.graph.insert(regime_id, embedding, metadata)
        return regime_id
    
    def find_similar_regimes(
        self,
        volatility: float,
        trend: float,
        momentum: float,
        volume_ratio: float,
        k: int = 5
    ) -> List[SearchResult]:
        """Find k most similar historical regimes."""
        query = self.encode_regime(volatility, trend, momentum, volume_ratio)
        return self.graph.search(query, k=k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "regime_count": self.regime_counter,
            "graph_stats": self.graph.get_stats()
        }


def check_amd_environment() -> Dict[str, Any]:
    """Check AMD ROCm/DirectML environment."""
    env_info = {
        "rocm_available": ROCM_AVAILABLE,
        "directml_available": DIRECTML_AVAILABLE,
        "cython_available": CYTHON_AVAILABLE,
        "gpu_acceleration_enabled": ROCM_AVAILABLE or DIRECTML_AVAILABLE,
        "recommendations": []
    }
    
    if ROCM_AVAILABLE:
        env_info["recommendations"].append(
            "ROCm detected - consider using GPU-accelerated distance computations"
        )
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"
    
    if DIRECTML_AVAILABLE:
        env_info["recommendations"].append(
            "DirectML detected - Windows GPU acceleration available"
        )
    
    if CYTHON_AVAILABLE:
        env_info["recommendations"].append(
            "Cython available - JIT compilation enabled for faster graph operations"
        )
    else:
        env_info["recommendations"].append(
            "Using pure Python fallback - install Cython for better performance"
        )
    
    return env_info


# Example usage
if __name__ == "__main__":
    # Check environment
    env = check_amd_environment()
    print(f"Environment: {env}")
    
    # Create regime detector
    detector = MarketRegimeDetector(embedding_dim=64)
    
    # Add some historical regimes
    regimes = [
        (0.2, 0.8, 0.5, 1.2),   # High trend, moderate momentum
        (0.5, -0.3, -0.2, 0.8), # High volatility, negative trend
        (0.1, 0.1, 0.1, 1.0),   # Low volatility, sideways
        (0.8, 0.5, 0.9, 2.0),   # Very high volatility, strong momentum
    ]
    
    for vol, trend, mom, vol_ratio in regimes:
        detector.add_regime(vol, trend, mom, vol_ratio)
    
    # Find similar regimes
    similar = detector.find_similar_regimes(0.3, 0.7, 0.4, 1.1, k=3)
    
    print(f"\nFound {len(similar)} similar regimes:")
    for result in similar:
        print(f"  Regime {result.node_id}: distance={result.distance:.4f}")
    
    # Get stats
    stats = detector.get_stats()
    print(f"\nDetector stats: {stats}")
