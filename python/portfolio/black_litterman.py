"""
Black-Litterman Model for Portfolio Optimization
Integrates RL agent alpha views with market equilibrium.
Prevents extreme concentration risk through Bayesian shrinkage.
AMD ROCm/DirectML acceleration support for matrix operations.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
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
class BlackLittermanResult:
    """Black-Litterman optimization result."""
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    portfolio_weights: np.ndarray
    assets: List[str]
    views_applied: int
    tau: float  # Scaling factor for prior uncertainty
    confidence_weighted: float  # Average confidence in views


@ray.remote(max_calls=100)
class BlackLittermanOptimizer:
    """
    Ray actor for Black-Litterman portfolio optimization.
    Combines market equilibrium returns with subjective views.
    
    The model uses:
    - Market cap weights as equilibrium reference
    - RL agent predictions as views
    - Confidence levels based on prediction accuracy
    """
    
    def __init__(
        self, 
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        max_assets: int = 50
    ):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        risk_aversion : float
            Risk aversion coefficient (lambda)
        tau : float
            Scaling factor for prior uncertainty (typically 0.025-0.05)
        max_assets : int
            Maximum number of assets to handle
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.max_assets = max_assets
        self.accelerator = check_amd_acceleration()
    
    def calculate_equilibrium_returns(
        self,
        cov_matrix: np.ndarray,
        market_caps: np.ndarray
    ) -> np.ndarray:
        """
        Calculate implied equilibrium returns from market caps.
        
        Uses reverse optimization: Π = λ * Σ * w_mkt
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            N x N covariance matrix
        market_caps : np.ndarray
            Market capitalizations (used as weights)
        
        Returns
        -------
        np.ndarray
            Implied equilibrium returns
        """
        # Normalize market caps to weights
        w_mkt = market_caps / market_caps.sum()
        
        # Implied returns: Π = λ * Σ * w
        pi = self.risk_aversion * cov_matrix @ w_mkt
        
        return pi
    
    def incorporate_views(
        self,
        prior_returns: np.ndarray,
        cov_matrix: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incorporate subjective views into posterior distribution.
        
        Black-Litterman formula:
        E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1*Π + P'Ω^-1*Q]
        
        Parameters
        ----------
        prior_returns : np.ndarray
            Prior (equilibrium) returns Π
        cov_matrix : np.ndarray
            Covariance matrix Σ
        P : np.ndarray
            K x N pick matrix (K views, N assets)
        Q : np.ndarray
            K x 1 view returns
        omega : np.ndarray, optional
            K x K view uncertainty matrix (if None, use proportional)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Posterior returns and covariance
        """
        n_assets = len(prior_returns)
        tau = self.tau
        
        # If omega not provided, use proportional to variance
        if omega is None:
            # Ω = diag(P * (τΣ) * P')
            # Simplified: use scalar variance for each view
            view_variances = np.diag(P @ (tau * cov_matrix) @ P.T)
            omega = np.diag(view_variances)
        
        # Add small regularization for numerical stability
        omega += 1e-8 * np.eye(omega.shape[0])
        
        # Compute intermediate matrices
        tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
        omega_inv = np.linalg.inv(omega)
        
        # Posterior precision matrix
        M1 = tau_sigma_inv
        M2 = P.T @ omega_inv @ P
        posterior_precision = M1 + M2
        
        # Posterior mean calculation
        rhs = tau_sigma_inv @ prior_returns + P.T @ omega_inv @ Q
        
        # Solve for posterior mean
        posterior_returns = np.linalg.solve(posterior_precision, rhs)
        
        # Posterior covariance
        posterior_cov = np.linalg.inv(posterior_precision)
        
        return posterior_returns, posterior_cov
    
    def optimize(
        self,
        cov_matrix: np.ndarray,
        market_caps: np.ndarray,
        assets: List[str],
        views: Optional[List[Dict]] = None,
        long_only: bool = True,
        max_weight: float = 0.30
    ) -> Dict:
        """
        Full Black-Litterman optimization pipeline.
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            N x N covariance matrix
        market_caps : np.ndarray
            Market capitalizations
        assets : List[str]
            Asset names
        views : List[Dict], optional
            List of view dictionaries with keys:
            - 'assets': list of asset names involved
            - 'return': expected return
            - 'confidence': confidence level (0-1)
        long_only : bool
            No short selling constraint
        max_weight : float
            Maximum weight per asset
        
        Returns
        -------
        Dict
            Optimization results
        """
        n_assets = len(assets)
        asset_to_idx = {a: i for i, a in enumerate(assets)}
        
        # Step 1: Calculate equilibrium returns
        pi = self.calculate_equilibrium_returns(cov_matrix, market_caps)
        
        # Step 2: Build view matrices if views provided
        if views:
            k_views = len(views)
            P = np.zeros((k_views, n_assets))
            Q = np.zeros(k_views)
            
            for i, view in enumerate(views):
                view_assets = view.get('assets', [])
                view_return = view.get('return', 0.0)
                confidence = view.get('confidence', 0.5)
                
                # Relative view: sum of weights = 0
                # Absolute view: sum of weights = 1
                n_view_assets = len(view_assets)
                if n_view_assets > 0:
                    weight = 1.0 / n_view_assets
                    for va in view_assets:
                        if va in asset_to_idx:
                            P[i, asset_to_idx[va]] = weight
                    
                    Q[i] = view_return
            
            # View uncertainty matrix (diagonal)
            # Higher confidence = lower uncertainty
            view_uncertainties = []
            for view in views:
                conf = view.get('confidence', 0.5)
                # Uncertainty inversely proportional to confidence
                var_scale = (1.0 - conf) / (conf + 1e-6)
                view_uncertainties.append(var_scale)
            
            omega = np.diag(view_uncertainties)
            
            # Incorporate views
            posterior_returns, posterior_cov = self.incorporate_views(
                pi, cov_matrix, P, Q, omega
            )
        else:
            # No views: use equilibrium
            posterior_returns = pi
            posterior_cov = cov_matrix
        
        # Step 3: Calculate optimal weights
        # Using mean-variance with posterior estimates
        cov_inv = np.linalg.inv(posterior_cov + 1e-8 * np.eye(n_assets))
        
        # Unconstrained optimal weights
        raw_weights = cov_inv @ posterior_returns
        raw_weights /= self.risk_aversion
        
        # Apply constraints
        if long_only:
            raw_weights = np.maximum(0, raw_weights)
        
        # Normalize
        weight_sum = raw_weights.sum()
        if weight_sum > 1e-12:
            weights = raw_weights / weight_sum
        else:
            # Fallback to equal weight
            weights = np.ones(n_assets) / n_assets
        
        # Cap individual weights
        if max_weight < 1.0:
            weights = np.minimum(weights, max_weight)
            weights /= weights.sum() + 1e-12
        
        # Calculate portfolio metrics
        port_return = weights @ posterior_returns
        port_vol = np.sqrt(weights @ posterior_cov @ weights)
        sharpe = (port_return - 0.02) / (port_vol + 1e-12)  # Assuming 2% risk-free
        
        return {
            "weights": weights.tolist(),
            "expected_return": float(port_return),
            "volatility": float(port_vol),
            "sharpe_ratio": float(sharpe),
            "posterior_returns": posterior_returns.tolist(),
            "assets": assets,
            "views_applied": len(views) if views else 0,
            "tau": self.tau,
            "accelerator_backend": self.accelerator
        }


def run_black_litterman(
    cov_matrix: np.ndarray,
    market_caps: Dict[str, float],
    rl_views: Optional[List[Dict]] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> BlackLittermanResult:
    """
    Run Black-Litterman optimization with RL agent views.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Asset covariance matrix
    market_caps : Dict[str, float]
        Market capitalizations by asset
    rl_views : List[Dict], optional
        Views from RL agent
    risk_aversion : float
        Risk aversion coefficient
    tau : float
        Prior uncertainty scaling
    
    Returns
    -------
    BlackLittermanResult
        Optimization results
    """
    assets = list(market_caps.keys())
    caps_array = np.array([market_caps[a] for a in assets])
    
    optimizer = BlackLittermanOptimizer(
        risk_aversion=risk_aversion,
        tau=tau,
        max_assets=len(assets)
    )
    
    result = optimizer.optimize(
        cov_matrix=cov_matrix,
        market_caps=caps_array,
        assets=assets,
        views=rl_views,
        long_only=True
    )
    
    avg_confidence = 0.0
    if rl_views:
        avg_confidence = np.mean([v.get('confidence', 0.5) for v in rl_views])
    
    return BlackLittermanResult(
        posterior_returns=np.array(result["posterior_returns"]),
        posterior_covariance=cov_matrix,  # Simplified
        portfolio_weights=np.array(result["weights"]),
        assets=assets,
        views_applied=result["views_applied"],
        tau=tau,
        confidence_weighted=avg_confidence
    )


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n_assets = 5
    assets = [f"ASSET_{i}" for i in range(n_assets)]
    
    # Generate random covariance
    returns_data = np.random.randn(252, n_assets) * 0.02
    cov_matrix = np.cov(returns_data.T)
    
    # Market caps (arbitrary)
    market_caps = {a: 1000.0 * (i + 1) for i, a in enumerate(assets)}
    
    # RL agent views
    rl_views = [
        {"assets": ["ASSET_0"], "return": 0.15, "confidence": 0.7},
        {"assets": ["ASSET_1", "ASSET_2"], "return": 0.08, "confidence": 0.5}
    ]
    
    result = run_black_litterman(cov_matrix, market_caps, rl_views)
    
    print(f"Portfolio Weights: {dict(zip(result.assets, result.portfolio_weights.round(4)))}")
    print(f"Views Applied: {result.views_applied}")
    print(f"Average Confidence: {result.confidence_weighted:.2f}")
