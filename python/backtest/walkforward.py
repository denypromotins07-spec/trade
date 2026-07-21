"""
`python/backtest/walkforward.py`

**Walk-Forward Optimization Engine**

Implements Ray-distributed walk-forward analysis to prevent overfitting.
Dynamically retrains the RL agent on rolling out-of-sample windows.

Optimization Strategy:
- Uses Ray for distributed parallel training across multiple time windows
- Strict memory management to stay within 4GB Python quota
- AMD ROCm/DirectML checks for GPU-accelerated training
- Prevents look-ahead bias through strict temporal separation
"""

import asyncio
import os
import ray
from ray import tune
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import polars as pl

# Check for AMD ROCm/DirectML availability
AMD_GPU_AVAILABLE = (
    os.environ.get("ROCM_PATH") is not None or
    os.environ.get("DIRECTML_ENABLED") == "1"
)


class WalkForwardMode(Enum):
    """Walk-forward analysis modes"""
    EXPANDING = "expanding"  # Training window grows, test window slides
    ROLLING = "rolling"      # Both windows slide together
    ANCHORED = "anchored"    # Fixed start date, expanding training


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization"""
    mode: WalkForwardMode = WalkForwardMode.ROLLING
    train_window_days: int = 90
    test_window_days: int = 30
    step_days: int = 7  # How much to shift each iteration
    min_train_samples: int = 1000
    num_iterations: Optional[int] = None  # Auto-calculated if None
    overlap_allowed: bool = False  # Prevent data leakage


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward iteration"""
    iteration: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    model_params: Dict[str, Any]
    oos_degradation: float  # Performance drop from train to test


@dataclass
class WalkForwardSummary:
    """Aggregated results from all iterations"""
    total_iterations: int
    avg_train_return: float
    avg_test_return: float
    avg_oos_degradation: float
    std_test_return: float
    max_drawdown_avg: float
    sharpe_ratio_avg: float
    robustness_score: float  # 0-1, higher = more robust
    parameter_stability: Dict[str, float]  # Std dev of optimal params


@ray.remote
class WalkForwardWorker:
    """
    Ray actor for parallel walk-forward iteration execution.
    
    Each worker handles one or more iterations independently.
    """
    
    def __init__(self, gpu_id: Optional[int] = None):
        self.gpu_id = gpu_id
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Configure GPU resources if available"""
        if AMD_GPU_AVAILABLE and self.gpu_id is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = str(self.gpu_id)
            print(f"Worker configured with GPU {self.gpu_id}")
    
    def run_iteration(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_config: Dict,
        model_class: str,
    ) -> Dict[str, Any]:
        """
        Execute a single walk-forward iteration.
        
        Args:
            train_data: Training dataset (features + returns)
            test_data: Out-of-sample test dataset
            train_config: Model training configuration
            model_class: Name of model class to instantiate
            
        Returns:
            Dictionary with metrics and optimal parameters
        """
        # Import here to avoid loading ML libs unless needed
        from python.ai.agent import TradingAgent
        
        # Train model
        agent = TradingAgent(config=train_config)
        train_result = agent.train_on_data(train_data)
        
        # Evaluate on test data
        test_result = agent.evaluate_on_data(test_data)
        
        # Calculate OOS degradation
        train_return = train_result.get("total_return", 0)
        test_return = test_result.get("total_return", 0)
        
        if abs(train_return) > 1e-6:
            oos_degradation = (train_return - test_return) / abs(train_return)
        else:
            oos_degradation = 0.0
        
        return {
            "train_metrics": train_result,
            "test_metrics": test_result,
            "model_params": agent.get_params(),
            "oos_degradation": oos_degradation,
        }


class WalkForwardOptimizer:
    """
    Main walk-forward optimization engine.
    
    Distributes iterations across Ray workers for parallel execution.
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        num_workers: int = 4,
        memory_limit_gb: float = 4.0,
    ):
        self.config = config
        self.num_workers = num_workers
        self.memory_limit_gb = memory_limit_gb
        self.results: List[WalkForwardResult] = []
        self.workers: List[ray.actor.ActorHandle] = []
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(
                num_cpus=num_workers,
                _system_config={"object_store_memory": int(memory_limit_gb * 1e9 / 2)},
            )
    
    def create_time_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[tuple]:
        """
        Generate train/test window pairs based on configuration.
        
        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        current_start = start_date
        
        iteration = 0
        max_iterations = self.config.num_iterations or 100
        
        while current_start < end_date and iteration < max_iterations:
            # Calculate train window
            train_end = current_start + timedelta(days=self.config.train_window_days)
            
            if train_end > end_date:
                break
            
            # Calculate test window
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_window_days)
            
            if test_end > end_date:
                break
            
            windows.append((current_start, train_end, test_start, test_end))
            
            # Move to next window
            current_start += timedelta(days=self.config.step_days)
            iteration += 1
        
        return windows
    
    def prepare_data_splits(
        self,
        full_data: pl.DataFrame,
        windows: List[tuple]
    ) -> List[Dict]:
        """
        Split data into train/test sets for each window.
        
        Enforces strict temporal separation to prevent look-ahead bias.
        """
        splits = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Filter data using Polars (efficient datetime filtering)
            train_mask = (
                (pl.col("timestamp") >= train_start) & 
                (pl.col("timestamp") < train_end)
            )
            test_mask = (
                (pl.col("timestamp") >= test_start) & 
                (pl.col("timestamp") < test_end)
            )
            
            train_data = full_data.filter(train_mask)
            test_data = full_data.filter(test_mask)
            
            if len(train_data) < self.config.min_train_samples:
                continue
            
            if len(test_data) < 10:  # Minimum test samples
                continue
            
            splits.append({
                "iteration": i,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_data": train_data.to_numpy(),
                "test_data": test_data.to_numpy(),
            })
        
        return splits
    
    async def run_optimization(
        self,
        data: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        base_config: Dict,
        model_class: str = "PPO",
    ) -> WalkForwardSummary:
        """
        Run complete walk-forward optimization.
        
        Args:
            data: Full historical dataset
            start_date: Analysis start date
            end_date: Analysis end date
            base_config: Base model configuration
            model_class: RL algorithm class name
            
        Returns:
            Aggregated summary of all iterations
        """
        # Create time windows
        windows = self.create_time_windows(start_date, end_date)
        print(f"Created {len(windows)} walk-forward windows")
        
        # Prepare data splits
        splits = self.prepare_data_splits(data, windows)
        print(f"Prepared {len(splits)} valid data splits")
        
        if not splits:
            return WalkForwardSummary(
                total_iterations=0,
                avg_train_return=0,
                avg_test_return=0,
                avg_oos_degradation=0,
                std_test_return=0,
                max_drawdown_avg=0,
                sharpe_ratio_avg=0,
                robustness_score=0,
                parameter_stability={},
            )
        
        # Create workers
        self.workers = [
            WalkForwardWorker.remote(gpu_id=i % 4 if AMD_GPU_AVAILABLE else None)
            for i in range(min(self.num_workers, len(splits)))
        ]
        
        # Distribute work across workers
        futures = []
        for i, split in enumerate(splits):
            worker = self.workers[i % len(self.workers)]
            future = worker.run_iteration.remote(
                split["train_data"],
                split["test_data"],
                base_config,
                model_class,
            )
            futures.append((i, split, future))
        
        # Collect results
        results = []
        for idx, split, future in futures:
            try:
                result = await asyncio.wrap_future(future)
                
                results.append(WalkForwardResult(
                    iteration=idx,
                    train_start=split["train_start"],
                    train_end=split["train_end"],
                    test_start=split["test_start"],
                    test_end=split["test_end"],
                    train_metrics=result["train_metrics"],
                    test_metrics=result["test_metrics"],
                    model_params=result["model_params"],
                    oos_degradation=result["oos_degradation"],
                ))
            except Exception as e:
                print(f"Iteration {idx} failed: {e}")
        
        self.results = results
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _aggregate_results(
        self, 
        results: List[WalkForwardResult]
    ) -> WalkForwardSummary:
        """Aggregate individual results into summary statistics"""
        if not results:
            return WalkForwardSummary(
                total_iterations=0,
                avg_train_return=0,
                avg_test_return=0,
                avg_oos_degradation=0,
                std_test_return=0,
                max_drawdown_avg=0,
                sharpe_ratio_avg=0,
                robustness_score=0,
                parameter_stability={},
            )
        
        # Extract metrics
        train_returns = [r.train_metrics.get("total_return", 0) for r in results]
        test_returns = [r.test_metrics.get("total_return", 0) for r in results]
        oos_degradations = [r.oos_degradation for r in results]
        sharpe_ratios = [r.test_metrics.get("sharpe_ratio", 0) for r in results]
        max_drawdowns = [r.test_metrics.get("max_drawdown", 0) for r in results]
        
        # Calculate parameter stability
        param_values = {}
        for r in results:
            for key, value in r.model_params.items():
                if isinstance(value, (int, float)):
                    if key not in param_values:
                        param_values[key] = []
                    param_values[key].append(value)
        
        param_stability = {
            k: float(np.std(v)) if len(v) > 1 else 0.0
            for k, v in param_values.items()
        }
        
        # Robustness score: combination of low OOS degradation and consistent returns
        avg_oos = np.mean(oos_degradations)
        std_test = np.std(test_returns)
        robustness = max(0, 1 - avg_oos - (std_test / (np.mean(abs(test_returns)) + 1e-6)))
        
        return WalkForwardSummary(
            total_iterations=len(results),
            avg_train_return=float(np.mean(train_returns)),
            avg_test_return=float(np.mean(test_returns)),
            avg_oos_degradation=float(avg_oos),
            std_test_return=float(std_test),
            max_drawdown_avg=float(np.mean(max_drawdowns)),
            sharpe_ratio_avg=float(np.mean(sharpe_ratios)),
            robustness_score=float(robustness),
            parameter_stability=param_stability,
        )
    
    def get_best_parameters(self, metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Find best parameters based on out-of-sample performance.
        
        Args:
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
            
        Returns:
            Dictionary of optimal parameters
        """
        if not self.results:
            return {}
        
        # Sort by test metric
        sorted_results = sorted(
            self.results,
            key=lambda r: r.test_metrics.get(metric, 0),
            reverse=True
        )
        
        # Return parameters from best performing iteration
        return sorted_results[0].model_params
    
    def cleanup(self):
        """Clean up Ray resources"""
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []


async def main():
    """Example usage of walk-forward optimizer"""
    # Create sample data (in production, load from backtest harness)
    dates = pl.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2024, 1, 1),
        interval="1h",
        eager=True
    )
    
    data = pl.DataFrame({
        "timestamp": dates,
        "returns": np.random.randn(len(dates)) * 0.001,
        "features": np.random.randn(len(dates), 10).tolist(),
    })
    
    config = WalkForwardConfig(
        mode=WalkForwardMode.ROLLING,
        train_window_days=60,
        test_window_days=14,
        step_days=7,
    )
    
    optimizer = WalkForwardOptimizer(config=config, num_workers=4)
    
    summary = await optimizer.run_optimization(
        data=data,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        base_config={"learning_rate": 0.001},
        model_class="PPO",
    )
    
    print("\n=== Walk-Forward Summary ===")
    print(f"Iterations: {summary.total_iterations}")
    print(f"Avg Train Return: {summary.avg_train_return:.4f}")
    print(f"Avg Test Return: {summary.avg_test_return:.4f}")
    print(f"OOS Degradation: {summary.avg_oos_degradation:.4f}")
    print(f"Robustness Score: {summary.robustness_score:.4f}")
    print(f"Parameter Stability: {summary.parameter_stability}")
    
    best_params = optimizer.get_best_parameters()
    print(f"\nBest Parameters: {best_params}")
    
    optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
