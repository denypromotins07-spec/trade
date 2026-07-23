"""
python/factors/orthogonalization.py

Strict Gram-Schmidt Orthogonalization for Factor Portfolios

Ensures factor portfolios remain completely uncorrelated to prevent hidden
concentration risks. Uses modified Gram-Schmidt with re-orthogonalization
for numerical stability.

Memory Constraint: In-place operations where possible, O(n*k) memory for n assets and k factors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import torch


def check_amd_acceleration() -> Dict[str, bool]:
    """Detect AMD ROCm/DirectML availability."""
    result = {"cuda": torch.cuda.is_available(), "rocm": False, "directml": False, "cpu": True}
    if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
        result["rocm"] = True
    rocm_path = os.environ.get("ROCM_PATH", "")
    if rocm_path and os.path.exists(rocm_path):
        result["rocm"] = True
    if os.name == 'nt':
        try:
            import torch_directml
            result["directml"] = True
        except ImportError:
            pass
    return result


@dataclass
class OrthogonalizationConfig:
    """Configuration for Gram-Schmidt orthogonalization."""
    tolerance: float = 1e-10  # Threshold for linear dependence detection
    max_reortho_iterations: int = 2  # Re-orthogonalization passes
    use_modified_gs: bool = True  # Use modified GS (more stable than classical)
    normalize_output: bool = True  # Return orthonormal vectors


class GramSchmidtOrthogonalizer:
    """
    Numerically stable Gram-Schmidt orthogonalization.
    
    Supports both classical and modified Gram-Schmidt algorithms,
    with optional re-orthogonalization for ill-conditioned inputs.
    """
    
    def __init__(self, config: Optional[OrthogonalizationConfig] = None):
        self.config = config or OrthogonalizationConfig()
        self.acceleration = check_amd_acceleration()
        
    def orthogonalize(
        self, 
        vectors: np.ndarray,
        existing_basis: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orthogonalize a set of vectors against each other and optionally
        against an existing basis.
        
        Args:
            vectors: Matrix of shape (n, k) where each column is a vector
            existing_basis: Optional matrix of shape (n, m) representing
                           already orthogonalized vectors to orthogonalize against
        
        Returns:
            Tuple of (orthonormal_vectors, projection_coefficients)
            - orthonormal_vectors: Shape (n, k) orthonormal basis
            - projection_coefficients: Shape (m+k, k) coefficients showing how much
              of each original/existing basis vector was projected out
        """
        n, k = vectors.shape
        
        if self.config.use_modified_gs:
            return self._modified_gram_schmidt(vectors, existing_basis)
        else:
            return self._classical_gram_schmidt(vectors, existing_basis)
    
    def _modified_gram_schmidt(
        self,
        vectors: np.ndarray,
        existing_basis: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modified Gram-Schmidt with re-orthogonalization.
        More numerically stable than classical GS.
        """
        n, k = vectors.shape
        Q = np.zeros((n, k), dtype=vectors.dtype)
        R = np.zeros((k, k), dtype=vectors.dtype)
        
        # Track coefficients against existing basis
        m = existing_basis.shape[1] if existing_basis is not None else 0
        proj_coeffs = np.zeros((m + k, k), dtype=vectors.dtype)
        
        for j in range(k):
            v = vectors[:, j].copy()
            
            # First orthogonalize against existing basis if provided
            if existing_basis is not None:
                for i in range(m):
                    coeff = np.dot(existing_basis[:, i], v)
                    v = v - coeff * existing_basis[:, i]
                    proj_coeffs[i, j] = coeff
            
            # Then orthogonalize against previously computed Q vectors
            for i in range(j):
                coeff = np.dot(Q[:, i], v)
                v = v - coeff * Q[:, i]
                R[i, j] = coeff
                proj_coeffs[m + i, j] = coeff
            
            # Re-orthogonalization pass (Björck's method)
            for _ in range(self.config.max_reortho_iterations - 1):
                v_orig = v.copy()
                for i in range(j):
                    coeff = np.dot(Q[:, i], v)
                    v = v - coeff * Q[:, i]
                
                # Check if re-ortho made a difference
                if np.linalg.norm(v - v_orig) < self.config.tolerance:
                    break
            
            # Normalize
            norm = np.linalg.norm(v)
            
            if norm < self.config.tolerance:
                # Vector is linearly dependent
                if self.config.normalize_output:
                    Q[:, j] = 0.0
                else:
                    Q[:, j] = v
                R[j, j] = 0.0
            else:
                if self.config.normalize_output:
                    Q[:, j] = v / norm
                    R[j, j] = norm
                else:
                    Q[:, j] = v
                    R[j, j] = 1.0
        
        return Q, proj_coeffs
    
    def _classical_gram_schmidt(
        self,
        vectors: np.ndarray,
        existing_basis: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classical Gram-Schmidt (less stable, but sometimes faster).
        Included for completeness.
        """
        n, k = vectors.shape
        Q = np.zeros((n, k), dtype=vectors.dtype)
        R = np.zeros((k, k), dtype=vectors.dtype)
        
        m = existing_basis.shape[1] if existing_basis is not None else 0
        proj_coeffs = np.zeros((m + k, k), dtype=vectors.dtype)
        
        for j in range(k):
            v = vectors[:, j].copy()
            
            # Orthogonalize against existing basis
            if existing_basis is not None:
                for i in range(m):
                    coeff = np.dot(existing_basis[:, i], v)
                    v = v - coeff * existing_basis[:, i]
                    proj_coeffs[i, j] = coeff
            
            # Orthogonalize against previous Q vectors
            for i in range(j):
                coeff = np.dot(Q[:, i], vectors[:, j])
                v = v - coeff * Q[:, i]
                R[i, j] = coeff
                proj_coeffs[m + i, j] = coeff
            
            # Normalize
            norm = np.linalg.norm(v)
            
            if norm < self.config.tolerance:
                if self.config.normalize_output:
                    Q[:, j] = 0.0
                else:
                    Q[:, j] = v
                R[j, j] = 0.0
            else:
                if self.config.normalize_output:
                    Q[:, j] = v / norm
                    R[j, j] = norm
                else:
                    Q[:, j] = v
                    R[j, j] = 1.0
        
        return Q, proj_coeffs
    
    def verify_orthogonality(self, Q: np.ndarray) -> Dict[str, float]:
        """
        Verify orthogonality of the result.
        
        Returns dict with orthogonality metrics.
        """
        n, k = Q.shape
        
        # Compute Q^T Q
        QtQ = Q.T @ Q
        
        # Should be identity (or diagonal if not normalized)
        off_diag = QtQ - np.diag(np.diag(QtQ))
        diag_vals = np.diag(QtQ)
        
        return {
            'max_off_diagonal': np.max(np.abs(off_diag)),
            'mean_off_diagonal': np.mean(np.abs(off_diag)),
            'min_diagonal': np.min(diag_vals),
            'max_diagonal': np.max(diag_vals),
            'is_orthonormal': (
                np.max(np.abs(off_diag)) < self.config.tolerance and
                np.allclose(diag_vals, 1.0, atol=self.config.tolerance)
            ),
        }


class FactorPortfolioOrthogonalizer:
    """
    Specialized orthogonalizer for factor portfolio construction.
    Ensures factor returns are uncorrelated to prevent hidden concentration.
    """
    
    def __init__(self, config: Optional[OrthogonalizationConfig] = None):
        self.gs = GramSchmidtOrthogonalizer(config)
        self.acceleration = check_amd_acceleration()
        self.factor_names: List[str] = []
        self.current_basis: Optional[np.ndarray] = None
    
    def orthogonalize_factors(
        self,
        factor_returns: np.ndarray,
        factor_names: List[str],
        benchmark: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Orthogonalize factor returns to ensure zero correlation.
        
        Args:
            factor_returns: Matrix of shape (t, k) where t=time, k=factors
            factor_names: Names of each factor
            benchmark: Optional benchmark to orthogonalize against first
        
        Returns:
            Dict mapping factor names to orthogonalized return series
        """
        n_factors = factor_returns.shape[1]
        self.factor_names = factor_names
        
        # Transpose so factors are columns
        factors_T = factor_returns.T  # Shape (k, t)
        
        # Orthogonalize against benchmark first if provided
        if benchmark is not None:
            Q, coeffs = self.gs.orthogonalize(factors_T, benchmark.reshape(-1, 1))
        else:
            Q, coeffs = self.gs.orthogonalize(factors_T)
        
        # Transpose back to (t, k)
        orthogonal_returns = Q.T
        
        # Verify orthogonality
        ortho_check = self.gs.verify_orthogonality(Q)
        if not ortho_check['is_orthonormal']:
            print(f"Warning: Orthogonality check failed: {ortho_check}")
        
        # Return as dict
        result = {}
        for i, name in enumerate(factor_names):
            result[name] = orthogonal_returns[:, i]
        
        # Store basis for future use
        self.current_basis = Q
        
        return result
    
    def project_new_factor(
        self,
        new_factor: np.ndarray,
        name: str
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Project a new factor against existing orthogonal basis.
        Returns the residual (orthogonal component) and projection coefficients.
        """
        if self.current_basis is None:
            return new_factor, {}
        
        Q, coeffs = self.gs.orthogonalize(
            new_factor.reshape(-1, 1),
            self.current_basis
        )
        
        # Extract projection coefficients
        proj_dict = {}
        for i, factor_name in enumerate(self.factor_names):
            if i < len(coeffs):
                proj_dict[factor_name] = float(coeffs[i, 0])
        
        return Q.flatten(), proj_dict


if __name__ == "__main__":
    print("Factor Orthogonalization - AMD Acceleration:", check_amd_acceleration())
    
    # Test Gram-Schmidt
    config = OrthogonalizationConfig()
    gs = GramSchmidtOrthogonalizer(config)
    
    # Create random vectors
    np.random.seed(42)
    vectors = np.random.randn(100, 5)
    
    # Orthogonalize
    Q, R = gs.orthogonalize(vectors)
    
    # Verify
    ortho_check = gs.verify_orthogonality(Q)
    print(f"Orthogonality check: {ortho_check}")
    
    # Test factor portfolio orthogonalization
    fpo = FactorPortfolioOrthogonalizer(config)
    
    # Simulate correlated factor returns
    n_days = 252
    n_factors = 4
    factor_names = ['Momentum', 'Value', 'Quality', 'LowVol']
    
    # Create correlated returns
    base_returns = np.random.randn(n_days, n_factors)
    correlation_matrix = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.3],
        [0.1, 0.2, 0.3, 1.0],
    ])
    
    # Induce correlation
    L = np.linalg.cholesky(correlation_matrix)
    correlated_returns = base_returns @ L.T
    
    # Check initial correlations
    initial_corr = np.corrcoef(correlated_returns.T)
    print(f"\nInitial factor correlations:\n{initial_corr}")
    
    # Orthogonalize
    ortho_factors = fpo.orthogonalize_factors(
        correlated_returns, 
        factor_names
    )
    
    # Stack orthogonalized factors
    ortho_matrix = np.column_stack([ortho_factors[name] for name in factor_names])
    
    # Check final correlations
    final_corr = np.corrcoef(ortho_matrix.T)
    print(f"\nOrthogonalized factor correlations:\n{final_corr}")
    
    # Test projecting new factor
    new_factor = np.random.randn(n_days)
    residual, projections = fpo.project_new_factor(new_factor, 'NewFactor')
    print(f"\nProjections of new factor onto existing factors: {projections}")
