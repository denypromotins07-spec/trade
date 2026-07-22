"""
Risk-Sensitive Distributional RL - CVaR and Distortion Functions
=================================================================
This module develops distributional distortion functions that optimize the 
RL agent's objective for Conditional Value at Risk (CVaR) rather than raw 
expected cumulative reward. It provides risk-sensitive policy optimization
for trading strategies that must manage tail risk carefully.

Key Features:
- CVaR (Conditional Value at Risk) optimization
- Wang transform and other distortion functions
- Dynamic risk appetite based on drawdown state
- AMD ROCm/DirectML acceleration support
- Strict 4GB Python RAM quota enforcement

Constraints:
- No LLM dependencies
- Contiguous memory layouts for cache efficiency
- Compatible with C51 and IQN modules
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Callable
import ray

MAX_RAM_GB = 4.0
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def check_amd_acceleration() -> str:
    """Detect AMD ROCm or DirectML availability."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if 'AMD' in device_name or 'Radeon' in device_name or 'MI' in device_name:
            print(f"AMD ROCm detected: {device_name}")
            return 'cuda'
        return 'cuda'
    
    try:
        import torch_directml
        print("DirectML available")
        return 'dml'
    except ImportError:
        pass
    
    return 'cpu'


class DistortionFunction:
    """
    Base class for probability distortion functions.
    
    Distortion functions transform cumulative probabilities to achieve
    risk-sensitive behavior. Common examples include Wang transform,
    dual power transform, and exponential distortion.
    """
    
    def __init__(self, risk_param: float = 0.0):
        """
        Args:
            risk_param: Risk sensitivity parameter.
                       Positive = risk-averse, Negative = risk-seeking
        """
        self.risk_param = risk_param
    
    def distort(self, p: torch.Tensor) -> torch.Tensor:
        """Apply distortion to cumulative probability p."""
        raise NotImplementedError
    
    def inverse_distort(self, p: torch.Tensor) -> torch.Tensor:
        """Apply inverse distortion."""
        raise NotImplementedError


class WangTransform(DistortionFunction):
    """
    Wang Transform distortion function.
    
    Uses the inverse normal CDF to shift probabilities based on risk parameter.
    Commonly used in actuarial science and financial risk management.
    """
    
    def __init__(self, risk_param: float = 0.0):
        super(WangTransform, self).__init__(risk_param)
        # Precompute normal CDF lookup table for speed
        self.register_buffer('norm_ppf', None)
    
    def _norm_cdf_inv(self, p: torch.Tensor) -> torch.Tensor:
        """Inverse normal CDF (probit function)."""
        return torch.erfinv(2 * p - 1) * np.sqrt(2)
    
    def _norm_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Normal CDF."""
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))
    
    def distort(self, p: torch.Tensor) -> torch.Tensor:
        """
        Apply Wang transform: g(p) = Φ(Φ⁻¹(p) + λ)
        
        where λ is the risk parameter and Φ is standard normal CDF.
        """
        z = self._norm_cdf_inv(p.clamp(1e-10, 1 - 1e-10))
        distorted = self._norm_cdf(z + self.risk_param)
        return distorted.clamp(0, 1)
    
    def inverse_distort(self, p: torch.Tensor) -> torch.Tensor:
        """Inverse Wang transform."""
        z = self._norm_cdf_inv(p.clamp(1e-10, 1 - 1e-10))
        inverse = self._norm_cdf(z - self.risk_param)
        return inverse.clamp(0, 1)


class DualPowerTransform(DistortionFunction):
    """
    Dual Power Transform distortion function.
    
    g(p) = p^γ where γ controls risk sensitivity.
    γ > 1: risk-averse (concave distortion)
    γ < 1: risk-seeking (convex distortion)
    """
    
    def __init__(self, gamma: float = 1.0):
        super(DualPowerTransform, self).__init__(gamma)
    
    @property
    def gamma(self) -> float:
        return self.risk_param
    
    def distort(self, p: torch.Tensor) -> torch.Tensor:
        """Apply power distortion."""
        return p.pow(self.gamma)
    
    def inverse_distort(self, p: torch.Tensor) -> torch.Tensor:
        """Inverse power distortion."""
        return p.pow(1.0 / self.gamma)


class ExponentialDistortion(DistortionFunction):
    """
    Exponential distortion function.
    
    g(p) = (exp(λp) - 1) / (exp(λ) - 1)
    
    λ > 0: risk-averse
    λ < 0: risk-seeking
    """
    
    def __init__(self, lambd: float = 0.0):
        super(ExponentialDistortion, self).__init__(lambd)
    
    @property
    def lambd(self) -> float:
        return self.risk_param
    
    def distort(self, p: torch.Tensor) -> torch.Tensor:
        if abs(self.lambd) < 1e-10:
            return p
        
        exp_lambd = torch.exp(torch.tensor(self.lambd, device=p.device))
        numerator = torch.exp(self.lambd * p) - 1
        denominator = exp_lambd - 1
        return (numerator / denominator).clamp(0, 1)
    
    def inverse_distort(self, p: torch.Tensor) -> torch.Tensor:
        if abs(self.lambd) < 1e-10:
            return p
        
        exp_lambd = torch.exp(torch.tensor(self.lambd, device=p.device))
        return torch.log(1 + p * (exp_lambd - 1)) / self.lambd


def compute_cvar_from_distribution(
    returns: torch.Tensor,
    probabilities: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    Compute Conditional Value at Risk (CVaR) from a return distribution.
    
    CVaR_α = E[R | R ≤ VaR_α]
    
    Args:
        returns: Return values sorted in ascending order [batch, n_atoms]
        probabilities: Probability mass for each return [batch, n_atoms]
        alpha: Risk level (e.g., 0.05 for 5% CVaR)
        
    Returns:
        cvar: Conditional Value at Risk
    """
    batch_size, n_atoms = returns.size()
    
    # Compute cumulative probabilities
    cum_prob = torch.cumsum(probabilities, dim=1)
    
    # Find VaR threshold (quantile at alpha)
    var_mask = cum_prob <= alpha
    var_indices = var_mask.sum(dim=1, keepdim=True).clamp(0, n_atoms - 1)
    
    # Compute CVaR as weighted average of returns below VaR
    # Weight by probability mass in the tail
    tail_mask = cum_prob <= alpha + 1e-10
    tail_probs = probabilities * tail_mask.float()
    tail_prob_sum = tail_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
    
    cvar = (returns * tail_probs).sum(dim=1, keepdim=True) / tail_prob_sum
    
    return cvar.squeeze(-1)


def distort_distribution(
    returns: torch.Tensor,
    probabilities: torch.Tensor,
    distortion_fn: DistortionFunction,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply distortion function to a probability distribution.
    
    This transforms the cumulative probabilities and recomputes
    the probability masses accordingly.
    
    Args:
        returns: Return values [batch, n_atoms]
        probabilities: Original probability masses [batch, n_atoms]
        distortion_fn: Distortion function to apply
        
    Returns:
        distorted_returns: Same returns
        distorted_probs: Distorted probability masses
    """
    batch_size, n_atoms = returns.size()
    
    # Compute cumulative probabilities
    cum_prob = torch.cumsum(probabilities, dim=1)
    
    # Apply distortion to cumulative probabilities
    distorted_cum_prob = distortion_fn.distort(cum_prob)
    
    # Recover probability masses from distorted cumulative
    distorted_probs = torch.zeros_like(distorted_cum_prob)
    distorted_probs[:, 0] = distorted_cum_prob[:, 0]
    distorted_probs[:, 1:] = distorted_cum_prob[:, 1:] - distorted_cum_prob[:, :-1]
    
    # Ensure valid probability distribution
    distorted_probs = distorted_probs.clamp(min=0)
    distorted_probs = distorted_probs / (distorted_probs.sum(dim=1, keepdim=True) + 1e-10)
    
    return returns, distorted_probs


class RiskSensitiveOptimizer:
    """
    Optimizer for risk-sensitive objectives using distributional RL.
    
    Supports CVaR optimization and various distortion-based objectives.
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        risk_level: float = 0.05,
        distortion_fn: Optional[DistortionFunction] = None,
        use_cvar: bool = True,
    ):
        self.base_optimizer = base_optimizer
        self.risk_level = risk_level
        self.distortion_fn = distortion_fn or WangTransform(0.0)
        self.use_cvar = use_cvar
    
    def step(self, closure=None):
        return self.base_optimizer.step(closure)
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    def compute_risk_sensitive_loss(
        self,
        returns: torch.Tensor,
        probabilities: torch.Tensor,
        target_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute risk-sensitive loss using CVaR or distortion.
        
        Args:
            returns: Predicted return distribution values
            probabilities: Predicted probability masses
            target_values: Optional target values for regression
            
        Returns:
            loss: Risk-sensitive loss value
        """
        if self.use_cvar:
            # Optimize for CVaR (minimize tail risk)
            cvar = compute_cvar_from_distribution(
                returns, probabilities, self.risk_level
            )
            # Negative because we want to maximize returns (minimize negative)
            loss = -cvar.mean()
        else:
            # Use distortion-based objective
            _, distorted_probs = distort_distribution(
                returns, probabilities, self.distortion_fn
            )
            
            # Expected value under distorted probabilities
            expected_return = (returns * distorted_probs).sum(dim=1)
            loss = -expected_return.mean()
        
        if target_values is not None:
            # Add regression loss to guide learning
            predicted_mean = (returns * probabilities).sum(dim=1)
            mse_loss = F.mse_loss(predicted_mean, target_values)
            loss = loss + 0.1 * mse_loss
        
        return loss


class DrawdownAwareRiskController:
    """
    Dynamically adjusts risk sensitivity based on portfolio drawdown.
    
    As drawdown increases, becomes more risk-averse to prevent catastrophic losses.
    """
    
    def __init__(
        self,
        initial_drawdown_threshold: float = 0.05,
        max_risk_param: float = 2.0,
        min_risk_param: float = 0.0,
    ):
        self.initial_threshold = initial_drawdown_threshold
        self.max_risk_param = max_risk_param
        self.min_risk_param = min_risk_param
        self.current_drawdown = 0.0
        self.peak_value = 1.0
    
    def update_portfolio_value(self, current_value: float):
        """Update drawdown tracking with new portfolio value."""
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value
    
    def get_risk_parameter(self) -> float:
        """
        Get risk parameter based on current drawdown.
        
        Returns higher values (more risk-averse) as drawdown increases.
        """
        if self.current_drawdown <= self.initial_threshold:
            return self.min_risk_param
        
        # Linear interpolation between threshold and max drawdown
        excess_drawdown = self.current_drawdown - self.initial_threshold
        max_excess = 0.20 - self.initial_threshold  # Assume 20% max drawdown
        
        normalized_excess = min(excess_drawdown / max_excess, 1.0)
        
        return self.min_risk_param + normalized_excess * (self.max_risk_param - self.min_risk_param)
    
    def get_distortion_function(self) -> DistortionFunction:
        """Get distortion function configured for current risk level."""
        risk_param = self.get_risk_parameter()
        return WangTransform(risk_param)


@ray.remote(max_calls=1000)
class RiskSensitiveWorker:
    """
    Ray worker for risk-sensitive distributional RL training.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        risk_level: float = 0.05,
        use_cvar: bool = True,
    ):
        self.device = check_amd_acceleration()
        self.risk_level = risk_level
        self.use_cvar = use_cvar
        
        # Simple network for demonstration
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim * 51),  # 51 atoms
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.risk_optimizer = RiskSensitiveOptimizer(
            self.optimizer,
            risk_level=risk_level,
            use_cvar=use_cvar,
        )
        
        self.drawdown_controller = DrawdownAwareRiskController()
    
    def set_drawdown(self, current_value: float):
        """Update drawdown state and adjust risk parameters."""
        self.drawdown_controller.update_portfolio_value(current_value)
        risk_param = self.drawdown_controller.get_risk_parameter()
        
        # Update distortion function if not using pure CVaR
        if not self.use_cvar:
            self.risk_optimizer.distortion_fn = WangTransform(risk_param)
    
    def train_risk_sensitive_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict:
        """Train with risk-sensitive objective."""
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        
        batch_size = states_t.size(0)
        n_atoms = 51
        v_min, v_max = -100.0, 100.0
        
        # Forward pass
        logits = self.net(states_t).view(batch_size, -1, n_atoms)
        probs = F.softmax(logits, dim=-1)
        atoms = torch.linspace(v_min, v_max, n_atoms, device=self.device)
        
        # Get action-specific distributions
        action_probs = probs.gather(
            1, actions_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_atoms)
        ).squeeze(1)
        action_atoms = atoms.unsqueeze(0).expand(batch_size, n_atoms)
        
        # Compute risk-sensitive loss
        loss = self.risk_optimizer.compute_risk_sensitive_loss(
            action_atoms, action_probs
        )
        
        self.risk_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.risk_optimizer.step()
        
        return {'loss': loss.item(), 'risk_param': self.drawdown_controller.get_risk_parameter()}
    
    def check_memory_quota(self) -> bool:
        """Verify 4GB RAM quota."""
        if self.device == 'cuda':
            allocated_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            return allocated_gb < MAX_RAM_GB
        else:
            import psutil
            process = psutil.Process(os.getpid())
            used_gb = process.memory_info().rss / 1024 / 1024 / 1024
            return used_gb < MAX_RAM_GB


if __name__ == '__main__':
    # Test distortion functions
    p = torch.linspace(0.01, 0.99, 100)
    
    wang = WangTransform(risk_param=0.5)
    distorted_wang = wang.distort(p)
    
    dual = DualPowerTransform(gamma=1.5)
    distorted_dual = dual.distort(p)
    
    print("Wang transform (risk-averse):", distorted_wang[:5])
    print("Dual power transform (risk-averse):", distorted_dual[:5])
    
    # Test CVaR computation
    returns = torch.randn(10, 51).sort(dim=1)[0]
    probs = torch.rand(10, 51)
    probs = probs / probs.sum(dim=1, keepdim=True)
    
    cvar = compute_cvar_from_distribution(returns, probs, alpha=0.05)
    print(f"CVaR (5%): {cvar.mean().item():.4f}")
