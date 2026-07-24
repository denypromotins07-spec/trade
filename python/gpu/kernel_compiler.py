"""
Stage 62: AI & Pipeline Audit - File 18/20
Module: python/gpu/kernel_compiler.py
Focus: AOT Caching, Stale Binary Loading Prevention
Constraints: 4GB RAM Quota, AMD ROCm Compatibility

AUDIT FIXES APPLIED:
- Fixed AOT caching with version validation
- Prevented stale binary loading on driver updates
- Added cache invalidation on driver changes
"""

from __future__ import annotations
import torch
import hashlib
import os
import pickle
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KernelCompiler:
    """
    GPU kernel compiler with AOT caching.
    FIX: Validates cache against driver version to prevent stale binaries.
    """
    
    def __init__(self, cache_dir: str = "./kernel_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get driver version for cache validation
        self._driver_version = self._get_driver_version()
        self._cache: Dict[str, Any] = {}
        
    def _get_driver_version(self) -> str:
        """Get GPU driver version for cache validation."""
        try:
            if torch.cuda.is_available():
                # Try to get CUDA/ROCm version
                if hasattr(torch.version, 'cuda'):
                    return f"cuda_{torch.version.cuda}"
                elif hasattr(torch.version, 'hip'):
                    return f"rocm_{torch.version.hip}"
        except Exception:
            pass
        return "unknown"
    
    def _compute_cache_key(self, source_code: str, config: Dict[str, Any]) -> str:
        """Compute cache key from source and config."""
        content = source_code + str(sorted(config.items()))
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{hash_val}_{self._driver_version}"
    
    def compile_and_cache(
        self, 
        source_code: str, 
        config: Dict[str, Any],
        compile_fn
    ) -> Any:
        """Compile kernel with AOT caching."""
        cache_key = self._compute_cache_key(source_code, config)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Check cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Validate driver version matches
                if cached_data.get('driver_version') == self._driver_version:
                    logger.info(f"Loaded cached kernel: {cache_key}")
                    return cached_data['kernel']
                else:
                    logger.warning(f"Cache stale (driver mismatch). Recompiling.")
                    os.remove(cache_path)
                    
            except Exception as e:
                logger.error(f"Cache load failed: {e}")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        # Compile
        logger.info(f"Compiling kernel: {cache_key}")
        kernel = compile_fn(source_code, config)
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'kernel': kernel,
                    'driver_version': self._driver_version,
                    'config': config
                }, f)
            logger.info(f"Cached kernel: {cache_path}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        
        return kernel
    
    def clear_cache(self) -> None:
        """Clear all cached kernels."""
        try:
            for f in os.listdir(self.cache_dir):
                if f.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, f))
            logger.info("Cleared kernel cache")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


if __name__ == "__main__":
    print("Kernel compiler module loaded")
