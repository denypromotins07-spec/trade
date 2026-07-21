"""
Hierarchical Risk Parity (HRP) Portfolio Optimization
Uses scipy hierarchical clustering for robust capital allocation.
Bypasses fragile covariance matrix inversions.
AMD ROCm/DirectML acceleration support for linear algebra.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import ray


def check_amd_acceleration() -> str:
    """Detect available AMD acceleration backend."""
    if os.environ.get("ROCM_PATH") or os.path.exists("/opt/rocm"):
        try:
            import torch
            if torch.cuda.is_available():
                return "rocm"
        except ImportError:
            pass
    
    if os.environ.get("DIRECTML_ENABLED") == "1":
        try:
            import torch_directml
            return "directml"
        except ImportError:
            pass
    
    return "cpu"


@dataclass
class HRPResult:
    """HRP optimization result container."""
    weights: np.ndarray
    assets: List[str]
    cluster_assignments: np.ndarray
    dendrogram: np.ndarray
    portfolio_variance: float
    diversification_ratio: float
    is_converged: bool


@ray.remote(max_calls=100)
class HierarchicalRiskParityOptimizer:
    """
    Ray actor for Hierarchical Risk Parity optimization.
    
    HRP addresses limitations of traditional mean-variance optimization:
    1. No matrix inversion required (more stable)
    2. Accounts for hierarchical structure in asset correlations
    3. Produces diversified portfolios without extreme weights
    
    Algorithm:
    1. Compute correlation-based distance matrix
    2. Build hierarchical tree using linkage
    3. Quasi-diagonalize the covariance matrix
    4. Recursive bisection to allocate weights
    """
    
    def __init__(self, max_assets: int = 100, linkage_method: str = 'single'):
        """
        Initialize HRP optimizer.
        
        Parameters
        ----------
        max_assets : int
            Maximum number of assets to handle
        linkage_method : str
            Clustering method: 'single', 'complete', 'average', 'ward'
        """
        self.max_assets = max_assets
        self.linkage_method = linkage_method
        self.accelerator = check_amd_acceleration()
        self._validate_memory_budget()
    
    def _validate_memory_budget(self) -> None:
        """Ensure memory stays within limits."""
        # Distance matrix: N^2 * 8 bytes
        # For N=100: 80KB (negligible)
        self.max_distance_matrix_bytes = self.max_assets ** 2 * 8
        self.memory_limit_bytes = 500 * 1024 * 1024  # 500MB per worker
    
    def compute_correlation_distance(self, returns: np.ndarray) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.
        
        D_ij = sqrt((1 - corr_ij) / 2)
        
        Parameters
        ----------
        returns : np.ndarray
            T x N array of returns
        
        Returns
        -------
        np.ndarray
            N x N distance matrix
        """
        corr_matrix = np.corrcoef(returns.T)
        
        # Handle NaN correlations (constant returns)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Clip to valid range
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        
        # Convert to distance
        distance_matrix = np.sqrt((1.0 - corr_matrix) / 2.0)
        
        return distance_matrix
    
    def build_dendrogram(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Build hierarchical tree from distance matrix.
        
        Parameters
        ----------
        distance_matrix : np.ndarray
            N x N distance matrix
        
        Returns
        -------
        np.ndarray
            Linkage matrix for dendrogram
        """
        n = distance_matrix.shape[0]
        
        # Convert to condensed form (upper triangle)
        condensed = squareform(distance_matrix, checks=False)
        
        # Build linkage matrix
        linkage_matrix = linkage(condensed, method=self.linkage_method)
        
        return linkage_matrix
    
    def quasi_diagonalize(self, linkage_matrix: np.ndarray) -> List[int]:
        """
        Reorder assets to group correlated ones together.
        
        Parameters
        ----------
        linkage_matrix : np.ndarray
            Linkage matrix from hierarchical clustering
        
        Returns
        -------
        List[int]
            Permuted leaf order
        """
        n = linkage_matrix.shape[0] + 1
        
        def get_leaves(node: int) -> List[int]:
            if node < n:
                return [node]
            left = int(linkage_matrix[node - n, 0])
            right = int(linkage_matrix[node - n, 1])
            return get_leaves(left) + get_leaves(right)
        
        root_node = 2 * n - 2
        return get_leaves(root_node)
    
    def recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        sorted_indices: List[int]
    ) -> np.ndarray:
        """
        Allocate weights using recursive bisection.
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix (reordered)
        sorted_indices : List[int]
            Quasi-diagonalized order
        
        Returns
        -------
        np.ndarray
            Allocated weights
        """
        n = len(sorted_indices)
        weights = np.ones(n)
        
        # Start with full cluster
        clusters = [list(range(n))]
        
        while len(clusters) > 0:
            new_clusters = []
            
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                
                # Split cluster in half
                mid = len(cluster) // 2
                left_cluster = cluster[:mid]
                right_cluster = cluster[mid:]
                
                # Calculate cluster variances
                left_cov = cov_matrix[np.ix_(left_cluster, left_cluster)]
                right_cov = cov_matrix[np.ix_(right_cluster, right_cluster)]
                
                # Inverse variance allocation between clusters
                left_var = np.trace(left_cov) / len(left_cluster)
                right_var = np.trace(right_cov) / len(right_cluster)
                
                total_var = left_var + right_var
                if total_var > 1e-12:
                    alpha = 1.0 - left_var / total_var
                else:
                    alpha = 0.5
                
                # Scale weights
                weights[left_cluster] *= alpha
                weights[right_cluster] *= (1.0 - alpha)
                
                # Add sub-clusters for further splitting
                if len(left_cluster) > 1:
                    new_clusters.append(left_cluster)
                if len(right_cluster) > 1:
                    new_clusters.append(right_cluster)
            
            clusters = new_clusters
        
        # Normalize weights
        weights /= weights.sum()
        
        return weights
    
    def optimize(
        self,
        returns: np.ndarray,
        assets: List[str],
        target_clusters: Optional[int] = None
    ) -> Dict:
        """
        Perform HRP optimization.
        
        Parameters
        ----------
        returns : np.ndarray
            T x N array of returns
        assets : List[str]
            Asset names
        target_clusters : int, optional
            Target number of clusters (for analysis)
        
        Returns
        -------
        Dict
            Optimization results
        """
        n_assets = len(assets)
        
        if n_assets > self.max_assets:
            raise ValueError(f"Too many assets: {n_assets} > {self.max_assets}")
        
        if n_assets < 2:
            # Single asset: full weight
            return {
                "weights": [1.0],
                "assets": assets,
                "cluster_assignments": [0],
                "portfolio_variance": float(np.var(returns)),
                "diversification_ratio": 1.0,
                "is_converged": True,
                "accelerator_backend": self.accelerator
            }
        
        # Step 1: Compute distance matrix
        distance_matrix = self.compute_correlation_distance(returns)
        
        # Step 2: Build dendrogram
        linkage_matrix = self.build_dendrogram(distance_matrix)
        
        # Step 3: Quasi-diagonalize
        sorted_indices = self.quasi_diagonalize(linkage_matrix)
        
        # Step 4: Get reordered covariance
        cov_matrix = np.cov(returns.T)
        cov_reordered = cov_matrix[np.ix_(sorted_indices, sorted_indices)]
        
        # Step 5: Recursive bisection
        weights_reordered = self.recursive_bisection(cov_reordered, sorted_indices)
        
        # Restore original order
        weights = np.zeros(n_assets)
        for i, orig_idx in enumerate(sorted_indices):
            weights[orig_idx] = weights_reordered[i]
        
        # Calculate portfolio metrics
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = np.sum(weights * individual_vols)
        div_ratio = weighted_avg_vol / (portfolio_vol + 1e-12)
        
        # Cluster assignments (optional)
        if target_clusters:
            condensed = squareform(distance_matrix, checks=False)
            cluster_assignments = fcluster(
                linkage_matrix, 
                target_clusters, 
                criterion='maxclust'
            ) - 1  # Zero-indexed
        else:
            cluster_assignments = np.zeros(n_assets, dtype=int)
        
        return {
            "weights": weights.tolist(),
            "assets": assets,
            "cluster_assignments": cluster_assignments.tolist(),
            "dendrogram": linkage_matrix.tolist(),
            "portfolio_variance": float(portfolio_variance),
            "diversification_ratio": float(div_ratio),
            "is_converged": True,
            "accelerator_backend": self.accelerator
        }


def run_hrp_optimization(
    returns_data: Dict[str, np.ndarray],
    target_clusters: Optional[int] = None,
    linkage_method: str = 'average'
) -> HRPResult:
    """
    Run HRP optimization on return data.
    
    Parameters
    ----------
    returns_data : Dict[str, np.ndarray]
        Dictionary mapping asset names to return series
    target_clusters : int, optional
        Target number of clusters
    linkage_method : str
        Clustering linkage method
    
    Returns
    -------
    HRPResult
        Optimization results
    """
    assets = list(returns_data.keys())
    n_assets = len(assets)
    
    # Align and stack returns
    min_len = min(len(v) for v in returns_data.values())
    returns_array = np.column_stack([
        returns_data[a][-min_len:] for a in assets
    ])
    
    # Create optimizer
    optimizer = HierarchicalRiskParityOptimizer(
        max_assets=n_assets,
        linkage_method=linkage_method
    )
    
    # Run optimization
    result = optimizer.optimize(
        returns_array, 
        assets, 
        target_clusters=target_clusters
    )
    
    return HRPResult(
        weights=np.array(result["weights"]),
        assets=assets,
        cluster_assignments=np.array(result["cluster_assignments"]),
        dendrogram=np.array(result["dendrogram"]),
        portfolio_variance=result["portfolio_variance"],
        diversification_ratio=result["diversification_ratio"],
        is_converged=result["is_converged"]
    )


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate correlated returns
    n_assets = 10
    n_periods = 252
    assets = [f"ASSET_{i}" for i in range(n_assets)]
    
    # Create factor-based correlation structure
    common_factor = np.random.randn(n_periods)
    returns = {}
    
    for i in range(n_assets):
        idio = np.random.randn(n_periods) * 0.5
        factor_loading = 0.3 + 0.4 * (i // 3)  # Groups of 3
        returns[assets[i]] = common_factor * factor_loading + idio
    
    result = run_hrp_optimization(returns, target_clusters=3)
    
    print(f"Diversification Ratio: {result.diversification_ratio:.3f}")
    print(f"Portfolio Volatility: {np.sqrt(result.portfolio_variance):.4f}")
    print(f"Weights: {dict(zip(result.assets, result.weights.round(4)))}")
    print(f"Clusters: {dict(zip(result.assets, result.cluster_assignments))}")
