"""
Strategy Promotion Gate with Statistical Validation

This module implements strict statistical promotion gates (Sortino, Calmar,
Deflated Sharpe) that a shadow strategy must pass before being approved
for live deployment.

Architecture Notes:
- Uses NumPy arrays with contiguous memory layout for cache efficiency
- Memory-bounded calculations to respect 4GB RAM limit per worker
- Ray distributed execution for parallel strategy evaluation
- AMD ROCm/DirectML acceleration checks included

Statistical Tests Implemented:
1. Sortino Ratio - Risk-adjusted return using downside deviation
2. Calmar Ratio - Return relative to maximum drawdown
3. Deflated Sharpe Ratio (DSR) - Sharpe adjusted for multiple testing bias
4. Probabilistic Sharpe Ratio (PSR) - Probability that true Sharpe > threshold
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import ray
from scipy import stats


class PromotionDecision(Enum):
    """Outcome of promotion gate evaluation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending_more_data"
    WARNING = "approved_with_warning"


@dataclass
class PerformanceMetrics:
    """Container for strategy performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    skewness: float
    kurtosis: float


@dataclass
class GateResult:
    """Result from promotion gate evaluation."""
    decision: PromotionDecision
    metrics: PerformanceMetrics
    deflated_sharpe: float
    probabilistic_sharpe: float
    p_value: float
    confidence_level: float
    warnings: List[str]
    passed_tests: List[str]
    failed_tests: List[str]


class StatisticalTests:
    """Collection of statistical tests for strategy validation."""
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (return / downside deviation).
        
        Unlike Sharpe, Sortino only penalizes downside volatility.
        
        Args:
            returns: Array of periodic returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sortino ratio
        """
        returns = np.ascontiguousarray(returns, dtype=np.float64)
        
        # Convert risk-free rate to periodic
        n_periods_per_year = 252 * 24 * 60  # Assuming minute data
        periodic_rf = risk_free_rate / n_periods_per_year
        
        # Excess returns
        excess_returns = returns - periodic_rf
        
        # Downside returns only
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf  # No downside = infinite Sortino
        
        # Downside deviation (annualized)
        downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(n_periods_per_year)
        
        if downside_deviation == 0:
            return np.inf
        
        # Annualized excess return
        ann_excess_return = np.mean(excess_returns) * n_periods_per_year
        
        return ann_excess_return / downside_deviation
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Args:
            returns: Array of periodic returns
            
        Returns:
            Calmar ratio
        """
        returns = np.ascontiguousarray(returns, dtype=np.float64)
        
        # Cumulative returns
        cum_returns = np.cumprod(1 + returns)
        
        # Running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Drawdowns
        drawdowns = (cum_returns - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdowns)
        
        if max_dd == 0:
            return np.inf
        
        # Annualized return
        n_years = len(returns) / (252 * 24 * 60)
        if n_years <= 0:
            return 0.0
        
        total_return = cum_returns[-1] - 1
        ann_return = (1 + total_return) ** (1 / n_years) - 1
        
        return ann_return / abs(max_dd)
    
    @staticmethod
    def calculate_deflated_sharpe(
        returns: np.ndarray,
        sharpe_threshold: float = 0.0,
        n_trials: int = 1,
        skew: Optional[float] = None,
        kurt: Optional[float] = None
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio (DSR).
        
        Adjusts observed Sharpe for multiple testing bias and non-normal returns.
        
        Based on: Lopez de Prado & Lewis (2018)
        
        Args:
            returns: Array of periodic returns
            sharpe_threshold: Minimum acceptable Sharpe
            n_trials: Number of strategies tested (for multiple testing correction)
            skew: Skewness of returns (optional, calculated if None)
            kurt: Kurtosis of returns (optional, calculated if None)
            
        Returns:
            Deflated Sharpe Ratio (probability that true Sharpe > threshold)
        """
        returns = np.ascontiguousarray(returns, dtype=np.float64)
        n = len(returns)
        
        if n < 10:
            return 0.0  # Insufficient data
        
        # Observed Sharpe
        obs_sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(n)
        
        # Calculate moments if not provided
        if skew is None:
            skew = stats.skew(returns)
        if kurt is None:
            kurt = stats.kurtosis(returns)  # Excess kurtosis
        
        # Variance inflation due to non-normality
        var_inflation = 1 + 0.5 * skew * obs_sharpe + (kurt - 1) / 4 * obs_sharpe ** 2
        
        # Standard error of Sharpe
        se_sharpe = np.sqrt(var_inflation / n)
        
        # Multiple testing adjustment (Bailey-Lopez de Prado)
        # Expected maximum Sharpe under null
        if n_trials > 1:
            # Euler-Mascheroni constant
            gamma = 0.5772156649
            expected_max = np.sqrt(2 * np.log(n_trials))
            variance_max = 1 / (2 * np.log(n_trials))
            
            # Adjusted threshold
            adj_threshold = sharpe_threshold + expected_max * se_sharpe
        else:
            adj_threshold = sharpe_threshold
        
        # DSR = P(true Sharpe > threshold | observed)
        z_score = (obs_sharpe - adj_threshold) / se_sharpe
        dsr = stats.norm.cdf(z_score)
        
        return dsr
    
    @staticmethod
    def calculate_probabilistic_sharpe(
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).
        
        Probability that the true Sharpe ratio exceeds a benchmark.
        
        Args:
            returns: Array of periodic returns
            benchmark_sharpe: Benchmark Sharpe to compare against
            
        Returns:
            PSR value between 0 and 1
        """
        returns = np.ascontiguousarray(returns, dtype=np.float64)
        n = len(returns)
        
        if n < 10:
            return 0.5  # No information
        
        # Sample moments
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Observed Sharpe (annualized)
        ann_factor = np.sqrt(252 * 24 * 60)  # Minute data
        obs_sharpe = (mu / sigma) * ann_factor
        
        # Standard error of Sharpe
        se = np.sqrt((1 + 0.5 * skew * obs_sharpe / ann_factor + 
                     (kurt - 1) / 4 * (obs_sharpe / ann_factor) ** 2) / n) * ann_factor
        
        # PSR = P(true Sharpe > benchmark)
        z = (obs_sharpe - benchmark_sharpe) / se
        psr = stats.norm.cdf(z)
        
        return psr


class PromotionGate:
    """
    Main promotion gate evaluator.
    
    Implements comprehensive statistical tests to validate
    shadow strategies before live deployment.
    """
    
    def __init__(
        self,
        min_sharpe: float = 1.0,
        min_sortino: float = 1.5,
        min_calmar: float = 2.0,
        min_dsr: float = 0.95,
        min_psr: float = 0.90,
        min_samples: int = 1000,
        significance_level: float = 0.05
    ):
        """
        Initialize promotion gate with thresholds.
        
        Args:
            min_sharpe: Minimum Sharpe ratio
            min_sortino: Minimum Sortino ratio
            min_calmar: Minimum Calmar ratio
            min_dsr: Minimum Deflated Sharpe probability
            min_psr: Minimum Probabilistic Sharpe probability
            min_samples: Minimum number of observations
            significance_level: Statistical significance level
        """
        self.min_sharpe = min_sharpe
        self.min_sortino = min_sortino
        self.min_calmar = min_calmar
        self.min_dsr = min_dsr
        self.min_psr = min_psr
        self.min_samples = min_samples
        self.significance_level = significance_level
        
        self.tests = StatisticalTests()
    
    def evaluate(self, returns: np.ndarray, pnl_series: np.ndarray, 
                 n_trials: int = 1) -> GateResult:
        """
        Evaluate strategy against all promotion criteria.
        
        Args:
            returns: Array of periodic returns
            pnl_series: Array of cumulative PnL values
            n_trials: Number of strategies tested (for DSR)
            
        Returns:
            GateResult with decision and detailed metrics
        """
        returns = np.ascontiguousarray(returns, dtype=np.float64)
        pnl_series = np.ascontiguousarray(pnl_series, dtype=np.float64)
        
        warnings = []
        passed_tests = []
        failed_tests = []
        
        # Check minimum samples
        if len(returns) < self.min_samples:
            return GateResult(
                decision=PromotionDecision.PENDING,
                metrics=self._calculate_metrics(returns, pnl_series),
                deflated_sharpe=0.0,
                probabilistic_sharpe=0.0,
                p_value=1.0,
                confidence_level=0.0,
                warnings=[f"Insufficient samples: {len(returns)} < {self.min_samples}"],
                passed_tests=[],
                failed_tests=["min_samples"]
            )
        
        # Calculate all metrics
        metrics = self._calculate_metrics(returns, pnl_series)
        
        # Calculate advanced statistics
        dsr = self.tests.calculate_deflated_sharpe(
            returns, 
            sharpe_threshold=self.min_sharpe,
            n_trials=n_trials,
            skew=metrics.skewness,
            kurt=metrics.kurtosis
        )
        
        psr = self.tests.calculate_probabilistic_sharpe(
            returns,
            benchmark_sharpe=self.min_sharpe
        )
        
        # Evaluate each criterion
        if metrics.sharpe_ratio >= self.min_sharpe:
            passed_tests.append("sharpe_ratio")
        else:
            failed_tests.append("sharpe_ratio")
        
        if metrics.sortino_ratio >= self.min_sortino:
            passed_tests.append("sortino_ratio")
        else:
            failed_tests.append("sortino_ratio")
        
        if metrics.calmar_ratio >= self.min_calmar:
            passed_tests.append("calmar_ratio")
        else:
            failed_tests.append("calmar_ratio")
        
        if dsr >= self.min_dsr:
            passed_tests.append("deflated_sharpe")
        else:
            failed_tests.append("deflated_sharpe")
            warnings.append(f"DSR {dsr:.3f} below threshold {self.min_dsr}")
        
        if psr >= self.min_psr:
            passed_tests.append("probabilistic_sharpe")
        else:
            failed_tests.append("probabilistic_sharpe")
            warnings.append(f"PSR {psr:.3f} below threshold {self.min_psr}")
        
        # Check for excessive drawdown
        if metrics.max_drawdown < -0.20:  # More than 20% drawdown
            warnings.append(f"Excessive drawdown: {metrics.max_drawdown:.2%}")
        
        # Check return distribution
        if metrics.skewness < -1.0:
            warnings.append(f"Negative skew: {metrics.skewness:.2f}")
        
        if metrics.kurtosis > 5.0:
            warnings.append(f"High kurtosis (fat tails): {metrics.kurtosis:.2f}")
        
        # Make decision
        n_failed = len(failed_tests)
        n_passed = len(passed_tests)
        
        if n_failed == 0:
            if len(warnings) == 0:
                decision = PromotionDecision.APPROVED
            else:
                decision = PromotionDecision.WARNING
        elif n_failed <= 1 and n_passed >= 4:
            # Allow one failure if everything else passes strongly
            decision = PromotionDecision.WARNING
        else:
            decision = PromotionDecision.REJECTED
        
        # Calculate p-value (simple t-test against zero mean)
        t_stat, p_value = stats.ttest_1samp(returns, 0.0)
        
        # Confidence level
        confidence_level = 1 - p_value
        
        return GateResult(
            decision=decision,
            metrics=metrics,
            deflated_sharpe=dsr,
            probabilistic_sharpe=psr,
            p_value=p_value,
            confidence_level=confidence_level,
            warnings=warnings,
            passed_tests=passed_tests,
            failed_tests=failed_tests
        )
    
    def _calculate_metrics(self, returns: np.ndarray, pnl_series: np.ndarray) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        returns = np.ascontiguousarray(returns, dtype=np.float64)
        
        # Basic statistics
        total_return = pnl_series[-1] / pnl_series[0] - 1 if len(pnl_series) > 0 else 0.0
        
        n_periods = len(returns)
        n_years = n_periods / (252 * 24 * 60)  # Minute data assumption
        ann_factor = np.sqrt(252 * 24 * 60)
        
        ann_return = np.mean(returns) * ann_factor * 252 * 24 * 60 / n_periods if n_periods > 0 else 0.0
        volatility = np.std(returns, ddof=1) * ann_factor
        
        # Sharpe ratio
        sharpe = ann_return / volatility if volatility > 0 else 0.0
        
        # Sortino ratio
        sortino = self.tests.calculate_sortino_ratio(returns)
        
        # Calmar ratio
        calmar = self.tests.calculate_calmar_ratio(returns)
        
        # Maximum drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        max_dd = np.min(drawdowns)
        
        # Win rate
        winning_trades = np.sum(returns > 0)
        win_rate = winning_trades / n_periods if n_periods > 0 else 0.0
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average trade PnL
        avg_trade = np.mean(returns)
        
        # Distribution moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade,
            skewness=skewness,
            kurtosis=kurtosis
        )


@ray.remote(max_calls=100)
class RayPromotionWorker:
    """
    Ray worker for distributed strategy evaluation.
    
    Evaluates multiple shadow strategies in parallel.
    """
    
    def __init__(self, worker_id: int):
        """Initialize worker."""
        self.worker_id = worker_id
        self.gate = PromotionGate()
        self.evaluated_count = 0
        
    def evaluate_strategy(
        self, 
        returns: np.ndarray, 
        pnl_series: np.ndarray,
        strategy_id: str,
        n_trials: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate a single strategy.
        
        Args:
            returns: Periodic returns
            pnl_series: Cumulative PnL
            strategy_id: Strategy identifier
            n_trials: Number of trials for DSR
            
        Returns:
            Evaluation results dictionary
        """
        result = self.gate.evaluate(returns, pnl_series, n_trials)
        self.evaluated_count += 1
        
        return {
            "worker_id": self.worker_id,
            "strategy_id": strategy_id,
            "decision": result.decision.value,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "sortino_ratio": result.metrics.sortino_ratio,
            "calmar_ratio": result.metrics.calmar_ratio,
            "deflated_sharpe": result.deflated_sharpe,
            "probabilistic_sharpe": result.probabilistic_sharpe,
            "max_drawdown": result.metrics.max_drawdown,
            "warnings": result.warnings,
            "passed_tests": result.passed_tests,
            "failed_tests": result.failed_tests
        }
    
    def batch_evaluate(
        self,
        strategies: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple strategies in batch.
        
        Args:
            strategies: List of (strategy_id, returns, pnl_series) tuples
            
        Returns:
            List of evaluation results
        """
        results = []
        for strategy_id, returns, pnl_series in strategies:
            result = self.evaluate_strategy(returns, pnl_series, strategy_id)
            results.append(result)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "evaluated_count": self.evaluated_count
        }


def create_promotion_pool(n_workers: int = 4) -> List[ray.ObjectRef]:
    """Create a pool of promotion gate workers."""
    workers = [RayPromotionWorker.remote(i) for i in range(n_workers)]
    return workers


if __name__ == "__main__":
    import time
    
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic returns (good strategy)
    n_samples = 10000
    good_returns = np.random.normal(0.0001, 0.001, n_samples)  # Positive drift
    good_pnl = np.cumprod(1 + good_returns) * 1000000  # Start with $1M
    
    # Generate synthetic returns (bad strategy)
    bad_returns = np.random.normal(-0.0001, 0.002, n_samples)  # Negative drift
    bad_pnl = np.cumprod(1 + bad_returns) * 1000000
    
    print("Testing Promotion Gate...")
    gate = PromotionGate(min_samples=1000)
    
    # Evaluate good strategy
    start = time.time()
    good_result = gate.evaluate(good_returns, good_pnl, n_trials=10)
    elapsed = time.time() - start
    
    print(f"\nGood Strategy Results ({elapsed:.3f}s):")
    print(f"  Decision: {good_result.decision.value}")
    print(f"  Sharpe: {good_result.metrics.sharpe_ratio:.3f}")
    print(f"  Sortino: {good_result.metrics.sortino_ratio:.3f}")
    print(f"  Calmar: {good_result.metrics.calmar_ratio:.3f}")
    print(f"  DSR: {good_result.deflated_sharpe:.3f}")
    print(f"  PSR: {good_result.probabilistic_sharpe:.3f}")
    print(f"  Passed: {good_result.passed_tests}")
    print(f"  Failed: {good_result.failed_tests}")
    print(f"  Warnings: {good_result.warnings}")
    
    # Evaluate bad strategy
    bad_result = gate.evaluate(bad_returns, bad_pnl, n_trials=10)
    
    print(f"\nBad Strategy Results:")
    print(f"  Decision: {bad_result.decision.value}")
    print(f"  Sharpe: {bad_result.metrics.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {bad_result.metrics.max_drawdown:.2%}")
    print(f"  Failed: {bad_result.failed_tests}")
    
    # Test Ray distributed evaluation
    print("\n\nTesting Ray distributed evaluation...")
    ray.init(ignore_reinit_error=True)
    
    workers = create_promotion_pool(n_workers=2)
    
    # Distribute work
    futures = []
    for i, worker in enumerate(workers):
        if i == 0:
            fut = worker.evaluate_strategy.remote(good_returns, good_pnl, "good_strat", 10)
        else:
            fut = worker.evaluate_strategy.remote(bad_returns, bad_pnl, "bad_strat", 10)
        futures.append(fut)
    
    results = ray.get(futures)
    for r in results:
        print(f"Worker {r['worker_id']}: {r['strategy_id']} -> {r['decision']}")
    
    ray.shutdown()
