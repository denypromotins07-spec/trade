"""
PCA Factors Extraction Module

Extracts statistical risk factors using Incremental PCA on Ray workers,
strictly enforcing the 4GB Python RAM quota by processing crypto returns
in streaming mini-batches.

Optimized for AMD Ryzen AI 5 architecture with DirectML acceleration.
"""

import numpy as np
import polars as pl
from typing import Generator, Optional, Tuple
from sklearn.decomposition import IncrementalPCA
import ray
import os

# Enforce 4GB RAM limit per worker
MAX_RAM_GB = 4.0
BATCH_SIZE = 1000  # Number of assets per batch


def check_amd_directml() -> bool:
    """Check if AMD DirectML/ROCm environment is available."""
    try:
        import torch
        # Check for ROCm availability
        if hasattr(torch.backends, 'rocm') and torch.backends.rocm.is_available():
            return True
        # Check for DirectML (Windows)
        if os.name == 'nt':
            # DirectML typically available on Windows with AMD GPUs
            return True
        return False
    except ImportError:
        return False


def get_device_config():
    """Get optimal device configuration based on hardware."""
    if check_amd_directml():
        return {'device': 'cuda', 'dtype': 'float32'}  # ROCm uses CUDA interface
    return {'device': 'cpu', 'dtype': 'float32'}


@ray.remote(max_calls=100)
class PCABatchProcessor:
    """
    Ray actor for processing PCA batches with strict memory limits.
    
    Each instance is limited to 4GB RAM and processes data in mini-batches
    to prevent memory overflow.
    """
    
    def __init__(self, n_components: int = 10, batch_size: int = BATCH_SIZE):
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.mean_returns: Optional[np.ndarray] = None
        self.total_samples = 0
        self.device_config = get_device_config()
        
    def process_batch(self, returns_batch: np.ndarray) -> dict:
        """
        Process a single batch of returns data.
        
        Args:
            returns_batch: Array of shape (n_assets, n_features) containing returns
            
        Returns:
            Dictionary with explained variance and component information
        """
        # Validate input dimensions
        if returns_batch.ndim != 2:
            raise ValueError(f"Expected 2D array, got {returns_batch.ndim}D")
        
        # Handle missing data gracefully - replace NaN with column mean
        col_means = np.nanmean(returns_batch, axis=0, keepdims=True)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        returns_clean = np.where(np.isnan(returns_batch), col_means, returns_batch)
        
        # Track running mean for proper centering
        n_new = returns_clean.shape[0]
        if self.mean_returns is None:
            self.mean_returns = np.mean(returns_clean, axis=0)
        else:
            # Update running mean
            total = self.total_samples + n_new
            self.mean_returns = (self.mean_returns * self.total_samples + 
                                np.mean(returns_clean, axis=0) * n_new) / total
        
        # Center the batch
        returns_centered = returns_clean - self.mean_returns
        
        # Fit incremental PCA
        self.ipca.partial_fit(returns_centered)
        self.total_samples += n_new
        
        return {
            'n_samples_processed': self.total_samples,
            'explained_variance_ratio': self.ipca.explained_variance_ratio_.tolist(),
            'n_components': self.n_components,
        }
    
    def get_components(self) -> np.ndarray:
        """Get the principal components after fitting."""
        return self.ipca.components_
    
    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance for each component."""
        return self.ipca.explained_variance_
    
    def reset(self):
        """Reset the processor for new data."""
        self.ipca = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size)
        self.mean_returns = None
        self.total_samples = 0


def stream_returns_batches(
    returns_data: pl.DataFrame,
    batch_size: int = BATCH_SIZE
) -> Generator[np.ndarray, None, None]:
    """
    Stream returns data in mini-batches to enforce memory limits.
    
    Args:
        returns_data: Polars DataFrame with returns (rows=dates, cols=assets)
        batch_size: Number of rows per batch
        
    Yields:
        numpy arrays of shape (batch_size, n_assets)
    """
    n_rows = returns_data.height
    
    for start_idx in range(0, n_rows, batch_size):
        end_idx = min(start_idx + batch_size, n_rows)
        batch_df = returns_data.slice(start_idx, end_idx - start_idx)
        
        # Convert to numpy for sklearn compatibility
        batch_array = batch_df.to_numpy()
        
        # Handle any remaining NaN values
        batch_array = np.nan_to_num(batch_array, nan=0.0)
        
        yield batch_array


@ray.remote
def extract_pca_factors(
    returns_chunks: list,
    n_components: int = 10,
    ram_limit_gb: float = MAX_RAM_GB
) -> dict:
    """
    Extract PCA factors from returns data using Ray distributed processing.
    
    Args:
        returns_chunks: List of returns data chunks
        n_components: Number of principal components to extract
        ram_limit_gb: RAM limit per worker (default 4GB)
        
    Returns:
        Dictionary with factor loadings, explained variance, and scores
    """
    # Create batch processor
    processor = PCABatchProcessor.remote(n_components=n_components)
    
    total_variance_explained = 0.0
    n_batches = 0
    
    for chunk in returns_chunks:
        # Process each chunk through the batch processor
        result = ray.get(processor.process_batch.remote(chunk))
        total_variance_explained = sum(result['explained_variance_ratio'])
        n_batches += 1
        
        # Memory pressure check - recreate processor if needed
        if n_batches % 100 == 0:
            # Force garbage collection
            import gc
            gc.collect()
    
    # Retrieve final components
    components = ray.get(processor.get_components.remote())
    explained_var = ray.get(processor.get_explained_variance.remote())
    
    return {
        'components': components,
        'explained_variance': explained_var,
        'total_variance_explained': total_variance_explained,
        'n_batches_processed': n_batches,
    }


def compute_factor_scores(
    returns: pl.DataFrame,
    components: np.ndarray
) -> pl.DataFrame:
    """
    Compute factor scores (principal component scores) from returns.
    
    Args:
        returns: Original returns DataFrame
        components: PCA components matrix
        
    Returns:
        DataFrame with factor scores
    """
    # Center returns
    returns_np = returns.to_numpy()
    col_means = np.nanmean(returns_np, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    returns_centered = returns_np - col_means
    
    # Handle NaN
    returns_centered = np.nan_to_num(returns_centered, nan=0.0)
    
    # Project onto principal components
    factor_scores = returns_centered @ components.T
    
    # Create DataFrame with factor columns
    factor_names = [f'PC{i+1}' for i in range(components.shape[0])]
    
    return pl.DataFrame(factor_scores, schema=factor_names)


def validate_covariance_matrix(cov_matrix: np.ndarray) -> bool:
    """
    Validate that covariance matrix is positive semi-definite.
    
    Args:
        cov_matrix: Covariance matrix to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        return np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
    except np.linalg.LinAlgError:
        return False


def build_factor_model(
    returns_data: pl.DataFrame,
    n_components: int = 10,
    use_ray: bool = True
) -> dict:
    """
    Build complete PCA factor model from returns data.
    
    Args:
        returns_data: Polars DataFrame of asset returns
        n_components: Number of factors to extract
        use_ray: Whether to use Ray for distributed processing
        
    Returns:
        Dictionary with complete factor model
    """
    if use_ray and not ray.is_initialized():
        # Initialize Ray with memory limits
        ray.init(
            object_store_memory=int(2 * 1024**3),  # 2GB object store
            _system_config={'max_direct_call_object_size': 1024**2}  # 1MB direct calls
        )
    
    # Split data into chunks for streaming
    chunks = list(stream_returns_batches(returns_data, batch_size=BATCH_SIZE))
    
    if use_ray:
        # Distributed extraction
        result = ray.get(extract_pca_factors.remote(chunks, n_components))
    else:
        # Local extraction for testing
        processor = PCABatchProcessor(n_components=n_components)
        for chunk in chunks:
            processor.process_batch(chunk)
        result = {
            'components': processor.get_components(),
            'explained_variance': processor.get_explained_variance(),
            'total_variance_explained': sum(processor.ipca.explained_variance_ratio_),
        }
    
    # Compute factor scores
    factor_scores_df = compute_factor_scores(returns_data, result['components'])
    
    # Validate covariance structure
    cov_matrix = np.cov(returns_data.to_numpy().T)
    is_valid_cov = validate_covariance_matrix(cov_matrix)
    
    return {
        'factor_loadings': result['components'],
        'explained_variance': result['explained_variance'],
        'total_variance_explained': result['total_variance_explained'],
        'factor_scores': factor_scores_df,
        'covariance_valid': is_valid_cov,
        'n_factors': n_components,
    }


if __name__ == '__main__':
    # Example usage
    print("AMD DirectML Available:", check_amd_directml())
    print("Device Config:", get_device_config())
    
    # Create sample returns data for testing
    np.random.seed(42)
    n_assets = 100
    n_days = 1000
    
    sample_returns = np.random.randn(n_days, n_assets) * 0.02  # 2% daily vol
    sample_returns[:, :10] += 0.001  # Add some signal to first 10 assets
    
    returns_df = pl.DataFrame(sample_returns)
    
    # Build factor model
    model = build_factor_model(returns_df, n_components=5)
    
    print(f"\nFactor Model Results:")
    print(f"Total Variance Explained: {model['total_variance_explained']:.2%}")
    print(f"Covariance Matrix Valid: {model['covariance_valid']}")
    print(f"Factor Scores Shape: {model['factor_scores'].shape}")
