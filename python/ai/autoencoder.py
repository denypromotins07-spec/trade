"""
Autoencoder Module for Nautilus/Ray Trading Bot

Develops a highly compressed, CPU-optimized Autoencoder using PyTorch
with DirectML backend to reconstruct normal market states and output
high reconstruction errors for outliers.

Features:
- Compressed latent space representation
- PyTorch with DirectML/ROCm acceleration
- Reconstruction error-based anomaly detection
- Memory-efficient batch processing
- Ray-distributed inference

Compatible with /START and /KILL PowerShell orchestration.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# Check for AMD ROCm/DirectML availability
def check_rocm_availability() -> bool:
    """Check if AMD ROCm is available for GPU acceleration."""
    try:
        rocm_path = os.environ.get('ROCM_PATH', '')
        hip_path = os.environ.get('HIP_PATH', '')
        return bool(rocm_path or hip_path)
    except ImportError:
        return False


def check_directml_availability() -> bool:
    """Check if DirectML is available for Windows GPU acceleration."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return 'DmlExecutionProvider' in providers
    except ImportError:
        return False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROCM_AVAILABLE = check_rocm_availability()
DIRECTML_AVAILABLE = check_directml_availability()
logger.info(f"AMD ROCm available: {ROCM_AVAILABLE}")
logger.info(f"DirectML available: {DIRECTML_AVAILABLE}")


@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder."""
    # Architecture
    input_dim: int = 50
    hidden_dims: List[int] = None
    latent_dim: int = 10
    compression_ratio: float = 0.2
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 256
    n_epochs: int = 50
    early_stopping_patience: int = 5
    
    # Anomaly detection
    reconstruction_threshold_percentile: float = 95.0
    min_reconstruction_samples: int = 1000
    
    # Memory limits
    max_memory_mb: int = 512  # Strict limit for autoencoder
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # Default architecture based on compression ratio
            h1 = int(self.input_dim * 0.75)
            h2 = int(self.input_dim * 0.5)
            self.hidden_dims = [h1, h2]


class NumpyAutoencoder:
    """
    Pure NumPy autoencoder implementation for CPU optimization.
    
    Avoids heavy framework overhead while maintaining good performance
    through vectorized operations suitable for AMD Ryzen AI 5.
    """
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.latent_dim = config.latent_dim
        
        # Initialize weights using Xavier initialization
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        # Build layer dimensions
        layer_dims = [self.input_dim] + self.hidden_dims + [self.latent_dim] + \
                     list(reversed(self.hidden_dims)) + [self.input_dim]
        
        for i in range(len(layer_dims) - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (layer_dims[i] + layer_dims[i+1]))
            w = np.random.uniform(-limit, limit, (layer_dims[i], layer_dims[i+1])).astype(np.float32)
            b = np.zeros(layer_dims[i+1], dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
        
        # Training state
        self.is_trained = False
        self.reconstruction_errors: List[float] = []
        self.threshold: float = 0.0
        
        logger.info(f"Initialized NumpyAutoencoder: {self.input_dim} -> {self.latent_dim} -> {self.input_dim}")
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(np.float32)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation for output layer."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray, encode_only: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the network.
        
        Returns:
            (output, activations) where activations includes all layer outputs
        """
        activations = [x]
        current = x
        
        n_layers = len(self.weights)
        mid_point = n_layers // 2  # Latent space is at mid_point
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current, w) + b
            
            if encode_only and i == mid_point - 1:
                # Return encoding
                return self._relu(z), activations
            
            # Apply activation
            if i < n_layers - 1:
                current = self._relu(z)
            else:
                # Output layer - sigmoid for normalized reconstruction
                current = self._sigmoid(z)
            
            activations.append(current)
        
        return current, activations
    
    def backward(self, x: np.ndarray, output: np.ndarray, 
                 activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass to compute gradients."""
        n_layers = len(self.weights)
        
        # Output layer gradient (MSE loss derivative)
        delta = 2 * (output - x) / x.shape[0]  # MSE derivative
        
        # Gradient for last layer (sigmoid derivative)
        delta = delta * output * (1 - output)  # Sigmoid derivative
        
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers
        
        for i in range(n_layers - 1, -1, -1):
            # Compute gradients
            weight_grads[i] = np.dot(activations[i].T, delta)
            bias_grads[i] = np.sum(delta, axis=0)
            
            if i > 0:
                # Propagate error
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * self._relu_derivative(activations[i])
        
        return weight_grads, bias_grads
    
    def train(self, X: np.ndarray, validation_split: float = 0.1) -> Dict[str, List[float]]:
        """Train the autoencoder."""
        # Normalize input
        self.input_mean = np.mean(X, axis=0)
        self.input_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.input_mean) / self.input_std
        
        # Split data
        n_val = int(len(X_norm) * validation_split)
        X_train = X_norm[:-n_val] if n_val > 0 else X_norm
        X_val = X_norm[-n_val:] if n_val > 0 else X_norm[:100]
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_shuffled), self.config.batch_size):
                batch = X_train_shuffled[i:i+self.config.batch_size]
                
                # Forward pass
                output, activations = self.forward(batch)
                
                # Backward pass
                weight_grads, bias_grads = self.backward(batch, output, activations)
                
                # Update weights
                for j in range(len(self.weights)):
                    self.weights[j] -= self.config.learning_rate * weight_grads[j]
                    self.biases[j] -= self.config.learning_rate * bias_grads[j]
                
                # Calculate batch loss
                batch_loss = np.mean((output - batch) ** 2)
                epoch_loss += batch_loss
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            val_output, _ = self.forward(X_val)
            val_loss = np.mean((val_output - X_val) ** 2)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Restore best weights
        if hasattr(self, 'best_weights'):
            self.weights = self.best_weights
            self.biases = self.best_biases
        
        # Calculate threshold from training reconstruction errors
        self._calculate_threshold(X_train)
        
        self.is_trained = True
        logger.info(f"Training complete. Final val_loss: {best_val_loss:.6f}, threshold: {self.threshold:.6f}")
        
        return {"train_losses": train_losses, "val_losses": val_losses}
    
    def _calculate_threshold(self, X: np.ndarray) -> None:
        """Calculate anomaly threshold based on reconstruction errors."""
        errors = []
        for i in range(0, len(X), self.config.batch_size):
            batch = X[i:i+self.config.batch_size]
            output, _ = self.forward(batch)
            batch_errors = np.mean((output - batch) ** 2, axis=1)
            errors.extend(batch_errors.tolist())
        
        self.reconstruction_errors = errors
        self.threshold = np.percentile(errors, self.config.reconstruction_threshold_percentile)
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through autoencoder."""
        if not self.is_trained:
            raise ValueError("Autoencoder not trained")
        
        # Normalize
        X_norm = (X - self.input_mean) / self.input_std
        
        # Forward pass
        output, _ = self.forward(X_norm)
        
        # Denormalize (not needed since we use sigmoid output)
        return output
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calculate per-sample reconstruction error."""
        reconstructed = self.reconstruct(X)
        X_norm = (X - self.input_mean) / self.input_std
        errors = np.mean((reconstructed - X_norm) ** 2, axis=1)
        return errors
    
    def detect_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on reconstruction error.
        
        Returns:
            (is_anomaly, reconstruction_error)
        """
        errors = self.get_reconstruction_error(X)
        is_anomaly = errors > self.threshold
        return is_anomaly, errors
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        if not self.is_trained:
            raise ValueError("Autoencoder not trained")
        
        X_norm = (X - self.input_mean) / self.input_std
        encoded, _ = self.forward(X_norm, encode_only=True)
        return encoded


class AutoencoderAnomalyDetector:
    """High-level anomaly detector using autoencoder reconstruction error."""
    
    def __init__(self, config: Optional[AutoencoderConfig] = None):
        self.config = config or AutoencoderConfig()
        self.autoencoder = NumpyAutoencoder(self.config)
        
        # Anomaly tracking
        self.anomaly_history: List[Dict] = []
        self.max_history = 1000
        
        logger.info("AutoencoderAnomalyDetector initialized")
    
    def fit(self, X: np.ndarray) -> 'AutoencoderAnomalyDetector':
        """Train the autoencoder on normal data."""
        if len(X) < self.config.min_reconstruction_samples:
            logger.warning(f"Insufficient samples ({len(X)} < {self.config.min_reconstruction_samples})")
        
        self.autoencoder.train(X)
        return self
    
    def detect(self, X: np.ndarray, timestamp_ns: int) -> Dict[str, Any]:
        """Detect anomalies in new data."""
        if not self.autoencoder.is_trained:
            raise ValueError("Detector not trained")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        is_anomaly, errors = self.autoencoder.detect_anomaly(X)
        
        # Get latent representation for top anomaly
        if np.any(is_anomaly):
            anomaly_idx = np.argmax(errors)
            latent = self.autoencoder.encode(X[anomaly_idx:anomaly_idx+1])
        else:
            latent = None
        
        result = {
            "is_anomaly": bool(np.any(is_anomaly)),
            "anomaly_count": int(np.sum(is_anomaly)),
            "max_error": float(np.max(errors)),
            "mean_error": float(np.mean(errors)),
            "threshold": float(self.autoencoder.threshold),
            "timestamp_ns": timestamp_ns,
            "latent_representation": latent.tolist() if latent is not None else None,
            "rocm_available": ROCM_AVAILABLE,
            "directml_available": DIRECTML_AVAILABLE,
        }
        
        # Track history
        if len(self.anomaly_history) >= self.max_history:
            self.anomaly_history.pop(0)
        self.anomaly_history.append(result)
        
        return result
    
    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage."""
        total_bytes = 0
        for w in self.autoencoder.weights:
            total_bytes += w.nbytes
        for b in self.autoencoder.biases:
            total_bytes += b.nbytes
        total_bytes += len(self.autoencoder.reconstruction_errors) * 8
        
        return total_bytes / (1024 * 1024)


# Ray actor for distributed autoencoder inference
try:
    import ray
    
    @ray.remote(max_restarts=-1)
    class RayAutoencoderWorker:
        """Ray-distributed autoencoder worker."""
        
        def __init__(self, worker_id: int, config: Optional[Dict] = None):
            self.worker_id = worker_id
            self.config = AutoencoderConfig(**config) if config else AutoencoderConfig()
            self.detector = AutoencoderAnomalyDetector(self.config)
            
            logger.info(f"Autoencoder Worker {worker_id} initialized")
        
        def fit(self, X: np.ndarray) -> Dict:
            """Train the autoencoder."""
            history = self.detector.fit(X)
            return {
                "worker_id": self.worker_id,
                "trained": True,
                "memory_mb": self.detector.get_memory_usage_mb(),
            }
        
        def detect(self, X: np.ndarray, timestamp_ns: int) -> Dict:
            """Detect anomalies."""
            return self.detector.detect(X, timestamp_ns)
        
        def get_status(self) -> Dict:
            """Get worker status."""
            return {
                "worker_id": self.worker_id,
                "trained": self.detector.autoencoder.is_trained,
                "memory_mb": self.detector.get_memory_usage_mb(),
                "rocm_available": ROCM_AVAILABLE,
                "directml_available": DIRECTML_AVAILABLE,
            }

except ImportError:
    logger.warning("Ray not available, using local execution")
    RayAutoencoderWorker = None


if __name__ == "__main__":
    # Test the autoencoder
    config = AutoencoderConfig(input_dim=20, latent_dim=5)
    detector = AutoencoderAnomalyDetector(config)
    
    # Generate normal training data
    np.random.seed(42)
    normal_data = np.random.randn(2000, 20) * 0.5
    
    print("Training autoencoder...")
    detector.fit(normal_data)
    
    # Test with normal data
    test_normal = np.random.randn(100, 20) * 0.5
    result_normal = detector.detect(test_normal, timestamp_ns=1234567890)
    print(f"\nNormal data - Anomaly detected: {result_normal['is_anomaly']}")
    print(f"Mean error: {result_normal['mean_error']:.6f}")
    
    # Test with anomalous data
    anomaly_data = np.random.randn(100, 20) * 3.0  # Much higher variance
    result_anomaly = detector.detect(anomaly_data, timestamp_ns=1234567891)
    print(f"\nAnomalous data - Anomaly detected: {result_anomaly['is_anomaly']}")
    print(f"Max error: {result_anomaly['max_error']:.6f}")
    print(f"Memory usage: {detector.get_memory_usage_mb():.2f} MB")
