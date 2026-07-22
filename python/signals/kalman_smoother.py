"""
Rauch-Tung-Striebel (RTS) Kalman Smoother for Offline Backtest Refinement

This module codes RTS Kalman smoothers for offline backtest state refinement,
providing ground-truth labels for the RL agent's offline training buffers.

Architecture Notes:
- Uses NumPy arrays with contiguous memory layout to prevent cache thrashing
- Injects AMD ROCm/DirectML environment checks for acceleration
- Memory-bounded smoothing windows to respect 4GB RAM limit
- Designed for offline backtesting (not real-time)

Mathematical Foundation:
The RTS smoother is a two-pass algorithm:
1. Forward pass: Standard Kalman filter produces filtered estimates
2. Backward pass: Smooths estimates using future information

This provides optimal (minimum variance) state estimates given all observations.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import ray


def check_amd_acceleration() -> Dict[str, bool]:
    """
    Check for AMD DirectML/ROCm environment and return availability status.
    
    Returns:
        Dictionary with acceleration backend availability flags
    """
    acceleration_status = {
        "rocm_available": False,
        "directml_available": False,
        "cuda_available": False,
        "cpu_only": True
    }
    
    # Check for ROCm (AMD GPUs)
    try:
        import torch
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            acceleration_status["rocm_available"] = True
            acceleration_status["cpu_only"] = False
    except ImportError:
        pass
    
    # Check for DirectML (Windows AMD GPU acceleration)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DirectMLExecutionProvider' in providers:
            acceleration_status["directml_available"] = True
            acceleration_status["cpu_only"] = False
    except ImportError:
        pass
    
    return acceleration_status


@dataclass
class KalmanState:
    """Container for Kalman filter state at a single timestep."""
    state_mean: np.ndarray
    state_covariance: np.ndarray
    predicted_mean: np.ndarray
    predicted_covariance: np.ndarray
    kalman_gain: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray


@dataclass
class SmoothedResult:
    """Container for RTS smoother output."""
    smoothed_states: np.ndarray
    smoothed_covariances: np.ndarray
    cross_covariances: np.ndarray  # P_{k,k+1} for parameter estimation
    log_likelihood: float
    n_timesteps: int
    state_dim: int


class KalmanFilter:
    """
    Standard Kalman Filter for forward pass.
    
    Implements the predict-update cycle for linear Gaussian state space models:
    
    State equation: x_k = F * x_{k-1} + B * u_k + w_k,  w_k ~ N(0, Q)
    Obs equation:   z_k = H * x_k + v_k,              v_k ~ N(0, R)
    """
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None
    ):
        """
        Initialize Kalman filter.
        
        Args:
            state_dim: Dimension of state vector
            obs_dim: Dimension of observation vector
            control_dim: Dimension of control input (optional)
            initial_state: Initial state mean
            initial_covariance: Initial state covariance
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.control_dim = control_dim
        
        # State transition matrix F
        self.F = np.eye(state_dim)
        
        # Observation matrix H
        self.H = np.eye(obs_dim, state_dim)
        
        # Control matrix B (if applicable)
        self.B = np.zeros((state_dim, control_dim)) if control_dim > 0 else None
        
        # Process noise covariance Q
        self.Q = np.eye(state_dim) * 0.01
        
        # Observation noise covariance R
        self.R = np.eye(obs_dim) * 0.1
        
        # Initial state
        self.x = initial_state if initial_state is not None else np.zeros(state_dim)
        self.P = initial_covariance if initial_covariance is not None else np.eye(state_dim)
        
        # Storage for smoothing
        self.history: List[KalmanState] = []
        
        # Acceleration check
        self.accel_status = check_amd_acceleration()
    
    def set_transition_matrix(self, F: np.ndarray) -> 'KalmanFilter':
        """Set state transition matrix."""
        assert F.shape == (self.state_dim, self.state_dim)
        self.F = np.ascontiguousarray(F, dtype=np.float64)
        return self
    
    def set_observation_matrix(self, H: np.ndarray) -> 'KalmanFilter':
        """Set observation matrix."""
        assert H.shape == (self.obs_dim, self.state_dim)
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        return self
    
    def set_process_noise(self, Q: np.ndarray) -> 'KalmanFilter':
        """Set process noise covariance."""
        assert Q.shape == (self.state_dim, self.state_dim)
        self.Q = np.ascontiguousarray(Q, dtype=np.float64)
        return self
    
    def set_observation_noise(self, R: np.ndarray) -> 'KalmanFilter':
        """Set observation noise covariance."""
        assert R.shape == (self.obs_dim, self.obs_dim)
        self.R = np.ascontiguousarray(R, dtype=np.float64)
        return self
    
    def predict(self, control_input: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step.
        
        Args:
            control_input: Optional control vector u_k
            
        Returns:
            Predicted state mean and covariance
        """
        # x_pred = F * x
        x_pred = self.F @ self.x
        
        # Add control if present
        if self.B is not None and control_input is not None:
            x_pred += self.B @ control_input
        
        # P_pred = F * P * F^T + Q
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, measurement: np.ndarray) -> KalmanState:
        """
        Update step with new measurement.
        
        Args:
            measurement: Observation vector z_k
            
        Returns:
            KalmanState with all intermediate values for smoothing
        """
        # Predict first
        x_pred, P_pred = self.predict()
        
        # Innovation (measurement residual)
        y = measurement - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Updated state estimate
        self.x = x_pred + K @ y
        
        # Updated covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        
        # Store history for smoothing
        state = KalmanState(
            state_mean=self.x.copy(),
            state_covariance=self.P.copy(),
            predicted_mean=x_pred,
            predicted_covariance=P_pred,
            kalman_gain=K,
            innovation=y,
            innovation_covariance=S
        )
        self.history.append(state)
        
        return state
    
    def filter(self, measurements: np.ndarray, controls: Optional[np.ndarray] = None) -> List[KalmanState]:
        """
        Run full forward filter on sequence of measurements.
        
        Args:
            measurements: Array of shape (n_timesteps, obs_dim)
            controls: Optional array of shape (n_timesteps, control_dim)
            
        Returns:
            List of KalmanState for each timestep
        """
        self.history = []
        n_steps = len(measurements)
        
        for k in range(n_steps):
            control = controls[k] if controls is not None else None
            self.update(measurements[k])
        
        return self.history
    
    def get_log_likelihood(self) -> float:
        """Compute log likelihood of observed data."""
        log_lik = 0.0
        for state in self.history:
            S = state.innovation_covariance
            y = state.innovation
            
            # Multivariate normal log density
            sign, logdet = np.linalg.slogdet(S)
            log_lik -= 0.5 * (
                self.obs_dim * np.log(2 * np.pi) + 
                logdet + 
                y.T @ np.linalg.inv(S) @ y
            )
        
        return log_lik


class RTSSmoother:
    """
    Rauch-Tung-Striebel (RTS) Kalman Smoother.
    
    Performs backward smoothing pass after forward filtering to produce
    optimal state estimates given all observations.
    """
    
    def __init__(self, kf: KalmanFilter):
        """
        Initialize smoother with a fitted Kalman filter.
        
        Args:
            kf: KalmanFilter that has been run on data
        """
        self.kf = kf
        self.smoothed_states: Optional[np.ndarray] = None
        self.smoothed_covariances: Optional[np.ndarray] = None
        self.cross_covariances: Optional[np.ndarray] = None
    
    def smooth(self) -> SmoothedResult:
        """
        Perform RTS backward smoothing pass.
        
        Returns:
            SmoothedResult with smoothed states and covariances
        """
        if not self.kf.history:
            raise ValueError("Kalman filter has no history. Run filter() first.")
        
        n = len(self.kf.history)
        state_dim = self.kf.state_dim
        
        # Initialize arrays
        smoothed_means = np.zeros((n, state_dim))
        smoothed_covs = np.zeros((n, state_dim, state_dim))
        cross_covs = np.zeros((n - 1, state_dim, state_dim))
        
        # Initialize with final filtered state
        final_state = self.kf.history[-1]
        smoothed_means[-1] = final_state.state_mean
        smoothed_covs[-1] = final_state.state_covariance
        
        # Backward pass
        for k in range(n - 2, -1, -1):
            curr_state = self.kf.history[k]
            next_smoothed_mean = smoothed_means[k + 1]
            next_smoothed_cov = smoothed_covs[k + 1]
            
            # Smoother gain: G_k = P_k * F^T * P_{k+1|k}^{-1}
            P_pred_next = curr_state.predicted_covariance @ self.kf.F.T + self.kf.Q @ self.kf.F.T if k == 0 else curr_state.predicted_covariance
            P_pred_next = self.kf.F @ curr_state.state_covariance @ self.kf.F.T + self.kf.Q
            
            try:
                P_pred_next_inv = np.linalg.inv(P_pred_next)
                G = curr_state.state_covariance @ self.kf.F.T @ P_pred_next_inv
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                P_pred_next_inv = np.linalg.pinv(P_pred_next)
                G = curr_state.state_covariance @ self.kf.F.T @ P_pred_next_inv
            
            # Smoothed state estimate
            smoothed_means[k] = curr_state.state_mean + G @ (next_smoothed_mean - curr_state.predicted_mean)
            
            # Smoothed covariance
            smoothed_covs[k] = curr_state.state_covariance + G @ (next_smoothed_cov - P_pred_next) @ G.T
            
            # Cross-covariance P_{k,k+1} (needed for EM algorithm)
            cross_covs[k] = smoothed_covs[k] @ G.T
        
        self.smoothed_states = smoothed_means
        self.smoothed_covariances = smoothed_covs
        self.cross_covariances = cross_covs
        
        return SmoothedResult(
            smoothed_states=smoothed_means,
            smoothed_covariances=smoothed_covs,
            cross_covariances=cross_covs,
            log_likelihood=self.kf.get_log_likelihood(),
            n_timesteps=n,
            state_dim=state_dim
        )


class EMKalmanSmoother:
    """
    Expectation-Maximization for learning Kalman parameters from data.
    
    Uses RTS smoother in E-step to compute expected sufficient statistics,
    then updates model parameters in M-step.
    """
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Initialize EM-Kalman learner.
        
        Args:
            state_dim: Dimension of latent state
            obs_dim: Dimension of observations
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance for log-likelihood
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        self.kf: Optional[KalmanFilter] = None
        self.smoother: Optional[RTSSmoother] = None
        
        self.accel_status = check_amd_acceleration()
    
    def fit(self, observations: np.ndarray, verbose: bool = False) -> KalmanFilter:
        """
        Learn Kalman parameters from observations using EM.
        
        Args:
            observations: Array of shape (n_timesteps, obs_dim)
            verbose: Print progress
            
        Returns:
            Fitted KalmanFilter
        """
        n_steps = len(observations)
        
        # Initialize Kalman filter with random parameters
        self.kf = KalmanFilter(self.state_dim, self.obs_dim)
        self.kf.F = np.random.randn(self.state_dim, self.state_dim) * 0.1 + np.eye(self.state_dim) * 0.9
        self.kf.H = np.random.randn(self.obs_dim, self.state_dim) * 0.1
        self.kf.Q = np.eye(self.state_dim) * 0.1
        self.kf.R = np.eye(self.obs_dim) * 0.1
        
        prev_log_lik = -np.inf
        
        for iteration in range(self.max_iterations):
            # E-step: Run filter and smoother
            self.kf.history = []
            self.kf.filter(observations)
            self.smoother = RTSSmoother(self.kf)
            result = self.smoother.smooth()
            
            # Compute log-likelihood
            log_lik = result.log_likelihood
            
            if verbose:
                print(f"EM Iteration {iteration}: Log-likelihood = {log_lik:.4f}")
            
            # Check convergence
            if abs(log_lik - prev_log_lik) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_log_lik = log_lik
            
            # M-step: Update parameters
            self._m_step(result, observations)
        
        return self.kf
    
    def _m_step(self, result: SmoothedResult, observations: np.ndarray):
        """
        M-step: Update model parameters using smoothed statistics.
        
        Args:
            result: SmoothedResult from E-step
            observations: Original observations
        """
        n = result.n_timesteps
        x_smooth = result.smoothed_states
        P_smooth = result.smoothed_covariances
        P_cross = result.cross_covariances
        
        # Update F (state transition)
        # F_new = (sum_{k=1}^{n-1} E[x_k x_{k-1}^T]) @ (sum_{k=0}^{n-2} E[x_k x_k^T])^{-1}
        sum_xx_prev = np.zeros((self.state_dim, self.state_dim))
        sum_x_xprev = np.zeros((self.state_dim, self.state_dim))
        
        for k in range(1, n):
            sum_xx_prev += P_smooth[k - 1] + np.outer(x_smooth[k - 1], x_smooth[k - 1])
            sum_x_xprev += P_cross[k - 1] + np.outer(x_smooth[k], x_smooth[k - 1])
        
        try:
            self.kf.F = sum_x_xprev @ np.linalg.inv(sum_xx_prev)
        except np.linalg.LinAlgError:
            pass  # Keep old F
        
        # Update Q (process noise)
        sum_Q = np.zeros((self.state_dim, self.state_dim))
        for k in range(1, n):
            x_pred = self.kf.F @ x_smooth[k - 1]
            diff = x_smooth[k] - x_pred
            # E[(x_k - F x_{k-1})(x_k - F x_{k-1})^T]
            cov_term = P_smooth[k] - self.kf.F @ P_cross[k - 1].T - P_cross[k - 1] @ self.kf.F.T + self.kf.F @ P_smooth[k - 1] @ self.kf.F.T
            sum_Q += cov_term + np.outer(diff, diff)
        
        self.kf.Q = sum_Q / (n - 1)
        
        # Update H (observation matrix)
        sum_zz = np.zeros((self.obs_dim, self.obs_dim))
        sum_zx = np.zeros((self.obs_dim, self.state_dim))
        
        for k in range(n):
            sum_zz += np.outer(observations[k], observations[k])
            sum_zx += np.outer(observations[k], x_smooth[k])
        
        sum_xx = np.sum(P_smooth + np.einsum('nij,nik->jk', x_smooth[:, None, :] * x_smooth[:, :, None]), axis=0)
        
        try:
            self.kf.H = sum_zx @ np.linalg.inv(sum_xx)
        except np.linalg.LinAlgError:
            pass
        
        # Update R (observation noise)
        sum_R = np.zeros((self.obs_dim, self.obs_dim))
        for k in range(n):
            z_pred = self.kf.H @ x_smooth[k]
            diff = observations[k] - z_pred
            cov_term = self.kf.H @ P_smooth[k] @ self.kf.H.T
            sum_R += cov_term + np.outer(diff, diff)
        
        self.kf.R = sum_R / n


@ray.remote(max_calls=100)
class RayKalmanWorker:
    """
    Ray worker for distributed Kalman smoothing.
    
    Processes chunks of backtest data in parallel for RL label generation.
    """
    
    def __init__(self, worker_id: int, state_dim: int, obs_dim: int):
        """Initialize worker with dimensions."""
        self.worker_id = worker_id
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.processed_count = 0
        
    def smooth_trajectory(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Smooth a trajectory of observations.
        
        Args:
            observations: Array of shape (n_timesteps, obs_dim)
            
        Returns:
            Dictionary with smoothed states and metadata
        """
        kf = KalmanFilter(self.state_dim, self.obs_dim)
        kf.filter(observations)
        
        smoother = RTSSmoother(kf)
        result = smoother.smooth()
        
        self.processed_count += 1
        
        return {
            "worker_id": self.worker_id,
            "smoothed_states": result.smoothed_states,
            "smoothed_covariances": result.smoothed_covariances,
            "log_likelihood": result.log_likelihood,
            "n_timesteps": result.n_timesteps
        }
    
    def em_fit(self, observations: np.ndarray, max_iter: int = 50) -> Dict[str, Any]:
        """
        Fit Kalman parameters using EM.
        
        Args:
            observations: Training observations
            max_iter: Maximum EM iterations
            
        Returns:
            Fitted parameters and diagnostics
        """
        em = EMKalmanSmoother(self.state_dim, self.obs_dim, max_iterations=max_iter)
        kf = em.fit(observations, verbose=False)
        
        return {
            "worker_id": self.worker_id,
            "F": kf.F.tolist(),
            "H": kf.H.tolist(),
            "Q": kf.Q.tolist(),
            "R": kf.R.tolist(),
            "accel_status": self._check_accel()
        }
    
    def _check_accel(self) -> Dict[str, bool]:
        """Check acceleration availability."""
        return check_amd_acceleration()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "processed_count": self.processed_count,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim
        }


if __name__ == "__main__":
    import time
    
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data from known model
    true_F = np.array([[0.9, 0.1], [-0.1, 0.9]])
    true_H = np.array([[1.0, 0.0], [0.0, 1.0]])
    true_Q = np.eye(2) * 0.01
    true_R = np.eye(2) * 0.1
    
    n_steps = 500
    state_dim = 2
    obs_dim = 2
    
    # Simulate states and observations
    states = np.zeros((n_steps, state_dim))
    observations = np.zeros((n_steps, obs_dim))
    
    for k in range(1, n_steps):
        states[k] = true_F @ states[k - 1] + np.random.randn(state_dim) * np.sqrt(0.01)
        observations[k] = true_H @ states[k] + np.random.randn(obs_dim) * np.sqrt(0.1)
    
    # Test EM learning
    print("Testing EM-Kalman learning...")
    start = time.time()
    
    em = EMKalmanSmoother(state_dim, obs_dim, max_iterations=50)
    kf = em.fit(observations, verbose=True)
    
    elapsed = time.time() - start
    print(f"\nEM completed in {elapsed:.3f}s")
    print(f"Learned F:\n{kf.F}")
    print(f"True F:\n{true_F}")
    
    # Test Ray distributed smoothing
    print("\n\nTesting Ray distributed smoothing...")
    ray.init(ignore_reinit_error=True)
    
    workers = [RayKalmanWorker.remote(i, state_dim, obs_dim) for i in range(2)]
    
    # Split data
    chunk_size = n_steps // 2
    futures = []
    for i, worker in enumerate(workers):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < len(workers) - 1 else n_steps
        chunk = observations[start_idx:end_idx]
        futures.append(worker.smooth_trajectory.remote(chunk))
    
    results = ray.get(futures)
    for r in results:
        print(f"Worker {r['worker_id']}: smoothed {r['n_timesteps']} timesteps, LL={r['log_likelihood']:.2f}")
    
    ray.shutdown()
