"""
Parameter Sweeper Module for Nautilus/Ray Trading Bot

Implements an automated parameter sweeper that runs continuous micro-backtests
on recent 1-hour windows to find optimal hyperparameters for the current market regime.

Features:
- Ray-distributed parameter search
- Continuous micro-backtesting on rolling windows
- Adaptive parameter space refinement
- Memory-efficient batch processing
- AMD ROCm/DirectML environment checks

Compatible with /START and /KILL PowerShell orchestration.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import deque
import time

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
class ParameterRange:
    """Defines a parameter search range."""
    name: str
    min_val: float
    max_val: float
    step_size: Optional[float] = None
    log_scale: bool = False
    
    def get_values(self, n_samples: int = 10) -> np.ndarray:
        """Generate parameter values to test."""
        if self.log_scale:
            log_min = np.log(max(self.min_val, 1e-10))
            log_max = np.log(self.max_val)
            values = np.exp(np.linspace(log_min, log_max, n_samples))
        else:
            if self.step_size is not None:
                values = np.arange(self.min_val, self.max_val + self.step_size, self.step_size)
            else:
                values = np.linspace(self.min_val, self.max_val, n_samples)
        
        return values


@dataclass
class BacktestResult:
    """Result from a single backtest run."""
    parameters: Dict[str, float]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    window_start: int
    window_end: int
    execution_time_ms: float


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    # Window settings
    window_duration_hours: int = 1
    window_step_minutes: int = 15
    min_window_size: int = 100  # Minimum bars per window
    
    # Parameter search
    n_samples_per_param: int = 5
    max_combinations_per_window: int = 50
    
    # Performance thresholds
    min_sharpe_threshold: float = 0.5
    min_trades_threshold: int = 10
    
    # Memory limits
    max_memory_mb: int = 512
    
    # Strategy-specific parameters to sweep
    parameter_ranges: List[ParameterRange] = None
    
    def __post_init__(self):
        if self.parameter_ranges is None:
            self.parameter_ranges = [
                ParameterRange("entry_threshold", 0.001, 0.01, log_scale=True),
                ParameterRange("exit_threshold", 0.001, 0.01, log_scale=True),
                ParameterRange("stop_loss_pct", 0.01, 0.10),
                ParameterRange("take_profit_pct", 0.02, 0.20),
            ]


class MicroBacktester:
    """
    Fast micro-backtester for 1-hour windows.
    
    Optimized for speed with minimal allocations.
    """
    
    def __init__(self, config: SweepConfig):
        self.config = config
        self.results_history: deque = deque(maxlen=1000)
    
    def run_backtest(self, data: np.ndarray, parameters: Dict[str, float],
                     window_start: int, window_end: int) -> BacktestResult:
        """
        Run a quick backtest on the given window.
        
        Args:
            data: OHLCV data array (n_rows, 6) where columns are [time, open, high, low, close, volume]
            parameters: Strategy parameters
            window_start: Start index of window
            window_end: End index of window
        
        Returns:
            BacktestResult with performance metrics
        """
        start_time = time.time()
        
        window_data = data[window_start:window_end]
        
        if len(window_data) < self.config.min_window_size:
            return self._empty_result(parameters, window_start, window_end)
        
        # Extract prices
        closes = window_data[:, 4]  # Close price column
        highs = window_data[:, 2]   # High price column
        lows = window_data[:, 3]    # Low price column
        
        # Simplified strategy simulation
        entry_threshold = parameters.get("entry_threshold", 0.005)
        exit_threshold = parameters.get("exit_threshold", 0.003)
        stop_loss = parameters.get("stop_loss_pct", 0.05)
        take_profit = parameters.get("take_profit_pct", 0.10)
        
        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        
        # Generate signals (simplified momentum strategy)
        position = 0
        entry_price = 0.0
        trades = []
        pnl = []
        
        for i in range(1, len(closes)):
            ret = returns[i - 1]
            
            if position == 0:
                # Entry logic
                if abs(ret) > entry_threshold:
                    position = 1 if ret > 0 else -1
                    entry_price = closes[i]
            
            elif position > 0:
                # Long position exit logic
                current_pnl = (closes[i] - entry_price) / entry_price
                
                if current_pnl >= take_profit or current_pnl <= -stop_loss or abs(ret) < exit_threshold:
                    trades.append(current_pnl)
                    pnl.append(current_pnl)
                    position = 0
            
            elif position < 0:
                # Short position exit logic
                current_pnl = (entry_price - closes[i]) / entry_price
                
                if current_pnl >= take_profit or current_pnl <= -stop_loss or abs(ret) < exit_threshold:
                    trades.append(current_pnl)
                    pnl.append(current_pnl)
                    position = 0
        
        # Close any open position at end
        if position != 0:
            final_pnl = (closes[-1] - entry_price) / entry_price * position
            trades.append(final_pnl)
            pnl.append(final_pnl)
        
        # Calculate metrics
        if len(trades) == 0:
            return self._empty_result(parameters, window_start, window_end)
        
        total_return = sum(trades)
        n_trades = len(trades)
        win_rate = sum(1 for t in trades if t > 0) / n_trades if n_trades > 0 else 0
        
        # Sharpe ratio (annualized, assuming 1-hour bars)
        if len(pnl) > 1:
            mean_pnl = np.mean(pnl)
            std_pnl = np.std(pnl)
            sharpe = (mean_pnl / std_pnl * np.sqrt(252 * 24)) if std_pnl > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Profit factor
        gross_profit = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        execution_time = (time.time() - start_time) * 1000
        
        result = BacktestResult(
            parameters=parameters,
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            n_trades=n_trades,
            window_start=window_start,
            window_end=window_end,
            execution_time_ms=execution_time,
        )
        
        self.results_history.append(result)
        return result
    
    def _empty_result(self, parameters: Dict[str, float], 
                      window_start: int, window_end: int) -> BacktestResult:
        """Return empty result for invalid windows."""
        return BacktestResult(
            parameters=parameters,
            sharpe_ratio=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            n_trades=0,
            window_start=window_start,
            window_end=window_end,
            execution_time_ms=0.0,
        )


class ParameterSweeper:
    """
    Main parameter sweeper orchestrator.
    
    Continuously searches for optimal parameters on rolling windows.
    """
    
    def __init__(self, config: Optional[SweepConfig] = None):
        self.config = config or SweepConfig()
        self.backtester = MicroBacktester(self.config)
        
        # Best parameters found
        self.best_parameters: Dict[str, float] = {}
        self.best_score: float = 0.0
        
        # Parameter history
        self.parameter_history: List[Dict] = []
        
        logger.info("ParameterSweeper initialized")
    
    def generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """Generate parameter combinations to test."""
        all_params = []
        
        # Generate grid of parameter values
        param_values = {}
        for pr in self.config.parameter_ranges:
            param_values[pr.name] = pr.get_values(self.config.n_samples_per_param)
        
        # Create combinations (limit to max)
        import itertools
        keys = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[k] for k in keys]))
        
        # Random sample if too many
        if len(combinations) > self.config.max_combinations_per_window:
            indices = np.random.choice(len(combinations), 
                                       self.config.max_combinations_per_window, 
                                       replace=False)
            combinations = [combinations[i] for i in indices]
        
        for combo in combinations:
            all_params.append(dict(zip(keys, combo)))
        
        return all_params
    
    def sweep_window(self, data: np.ndarray, window_start: int, 
                     window_end: int) -> Dict[str, Any]:
        """Run parameter sweep on a single window."""
        combinations = self.generate_parameter_combinations()
        results = []
        
        for params in combinations:
            result = self.backtester.run_backtest(data, params, window_start, window_end)
            
            # Filter poor results
            if result.n_trades >= self.config.min_trades_threshold:
                results.append(result)
        
        if not results:
            return {
                "success": False,
                "best_params": {},
                "best_score": 0.0,
                "n_tested": len(combinations),
            }
        
        # Find best by Sharpe ratio
        best = max(results, key=lambda r: r.sharpe_ratio)
        
        # Update global best
        if best.sharpe_ratio > self.best_score:
            self.best_score = best.sharpe_ratio
            self.best_parameters = best.parameters.copy()
        
        return {
            "success": True,
            "best_params": best.parameters,
            "best_score": best.sharpe_ratio,
            "best_result": {
                "sharpe": best.sharpe_ratio,
                "return": best.total_return,
                "drawdown": best.max_drawdown,
                "win_rate": best.win_rate,
                "n_trades": best.n_trades,
            },
            "n_tested": len(results),
        }
    
    def continuous_sweep(self, data: np.ndarray) -> Dict[str, Any]:
        """Run continuous sweep across all rolling windows."""
        n_bars = len(data)
        window_bars = self.config.window_duration_hours * 60  # Assuming 1-minute bars
        step_bars = self.config.window_step_minutes
        
        all_results = []
        
        # Slide window across data
        for start in range(0, n_bars - window_bars, step_bars):
            end = start + window_bars
            result = self.sweep_window(data, start, end)
            all_results.append(result)
        
        # Aggregate results
        successful = [r for r in all_results if r["success"]]
        
        if not successful:
            return {
                "status": "no_valid_results",
                "windows_tested": len(all_results),
            }
        
        # Find most consistent parameters
        param_scores: Dict[str, List[float]] = {}
        for r in successful:
            for param, value in r["best_params"].items():
                if param not in param_scores:
                    param_scores[param] = []
                param_scores[param].append((value, r["best_score"]))
        
        # Weighted average of best parameters
        consensus_params = {}
        for param, scores in param_scores.items():
            total_weight = sum(s[1] for s in scores)
            if total_weight > 0:
                consensus_params[param] = sum(v * w for v, w in scores) / total_weight
        
        return {
            "status": "success",
            "windows_tested": len(all_results),
            "successful_windows": len(successful),
            "consensus_params": consensus_params,
            "global_best_params": self.best_parameters,
            "global_best_score": self.best_score,
        }
    
    def get_current_best(self) -> Dict[str, float]:
        """Get current best parameters."""
        return self.best_parameters.copy()


# Ray actor for distributed parameter sweeping
try:
    import ray
    
    @ray.remote(max_restarts=-1)
    class RayParameterSweeper:
        """Ray-distributed parameter sweeper worker."""
        
        def __init__(self, worker_id: int, config: Optional[Dict] = None):
            self.worker_id = worker_id
            self.config = SweepConfig(**config) if config else SweepConfig()
            self.sweeper = ParameterSweeper(self.config)
            
            logger.info(f"ParameterSweeper Worker {worker_id} initialized")
        
        def sweep_batch(self, data: np.ndarray, windows: List[Tuple[int, int]]) -> List[Dict]:
            """Sweep parameters on specified windows."""
            results = []
            for start, end in windows:
                result = self.sweeper.sweep_window(data, start, end)
                results.append(result)
            return results
        
        def get_best_params(self) -> Dict[str, float]:
            """Get current best parameters."""
            return self.sweeper.get_current_best()
        
        def get_status(self) -> Dict:
            """Get worker status."""
            return {
                "worker_id": self.worker_id,
                "best_score": self.sweeper.best_score,
                "best_params": self.sweeper.best_parameters,
                "rocm_available": ROCM_AVAILABLE,
                "directml_available": DIRECTML_AVAILABLE,
            }

except ImportError:
    logger.warning("Ray not available, using local execution")
    RayParameterSweeper = None


if __name__ == "__main__":
    # Test the parameter sweeper
    config = SweepConfig(
        window_duration_hours=1,
        n_samples_per_param=3,
        max_combinations_per_window=20,
    )
    sweeper = ParameterSweeper(config)
    
    # Generate synthetic OHLCV data (1-minute bars for 24 hours)
    np.random.seed(42)
    n_bars = 24 * 60
    base_price = 50000
    
    times = np.arange(n_bars)
    opens = base_price + np.cumsum(np.random.randn(n_bars) * 10)
    highs = opens + np.abs(np.random.randn(n_bars) * 5)
    lows = opens - np.abs(np.random.randn(n_bars) * 5)
    closes = opens + np.random.randn(n_bars) * 5
    volumes = np.random.randint(100, 1000, n_bars)
    
    data = np.column_stack([times, opens, highs, lows, closes, volumes])
    
    print("Running parameter sweep...")
    result = sweeper.continuous_sweep(data)
    
    print(f"\nStatus: {result['status']}")
    print(f"Windows tested: {result.get('windows_tested', 'N/A')}")
    print(f"Successful windows: {result.get('successful_windows', 'N/A')}")
    
    if result.get('consensus_params'):
        print("\nConsensus Parameters:")
        for param, value in result['consensus_params'].items():
            print(f"  {param}: {value:.6f}")
    
    print(f"\nGlobal Best Score (Sharpe): {sweeper.best_score:.4f}")
    print(f"Global Best Params: {sweeper.best_parameters}")
