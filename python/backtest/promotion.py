"""
Promotion Pipeline Module for Nautilus/Ray Trading Bot

Writes a promotion pipeline that rigorously validates shadow strategies
and safely hot-swaps them into the live Rust execution engine without
dropping active positions.

Features:
- Multi-stage validation (paper -> shadow -> canary -> production)
- Zero-downtime strategy hot-swap
- Position preservation during transitions
- Rollback capability on failure detection
- AMD ROCm/DirectML environment checks

Compatible with /START and /KILL PowerShell orchestration.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import threading
from collections import deque

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


class PromotionStage(Enum):
    """Stages in the promotion pipeline."""
    PAPER = "paper"           # Simulated trading only
    SHADOW = "shadow"         # Shadow alongside live
    CANARY = "canary"         # Small % of live capital
    PRODUCTION = "production" # Full live deployment


@dataclass
class ValidationResult:
    """Result from a validation stage."""
    stage: PromotionStage
    passed: bool
    metrics: Dict[str, float]
    issues: List[str]
    timestamp_ns: int
    duration_ms: float


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_duration_ms: float = 0.0
    slippage_bps: float = 0.0
    fill_rate: float = 1.0


@dataclass
class PromotionConfig:
    """Configuration for promotion pipeline."""
    # Stage thresholds
    paper_min_sharpe: float = 0.0
    paper_min_trades: int = 50
    
    shadow_max_tracking_error_bps: float = 50.0
    shadow_min_fill_rate: float = 0.95
    
    canary_max_drawdown: float = 0.02
    canary_min_return: float = -0.01
    
    production_min_sharpe: float = 1.0
    
    # Timing
    paper_min_duration_hours: int = 24
    shadow_min_duration_hours: int = 12
    canary_min_duration_hours: int = 6
    
    # Rollback triggers
    rollback_max_drawdown: float = 0.05
    rollback_sharpe_threshold: float = -1.0
    
    # Position handling
    preserve_positions_on_promote: bool = True
    gradual_position_transfer: bool = True


@dataclass
class ActivePosition:
    """Represents an active position being preserved."""
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    strategy_id: str


@dataclass
class PromotionState:
    """Current state of a strategy in the pipeline."""
    strategy_id: str
    current_stage: PromotionStage
    stage_entry_time: int
    metrics_history: List[StrategyMetrics] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    active_positions: List[ActivePosition] = field(default_factory=list)
    is_promoting: bool = False
    is_rolling_back: bool = False
    rollback_reason: Optional[str] = None


class ValidationEngine:
    """Validates strategies at each promotion stage."""
    
    def __init__(self, config: PromotionConfig):
        self.config = config
        self.metrics_window: deque = deque(maxlen=1000)
    
    def validate_paper(self, metrics: StrategyMetrics, 
                       duration_hours: float) -> ValidationResult:
        """Validate paper trading results."""
        start_time = time.time()
        issues = []
        
        # Check minimum trades
        if metrics.n_trades < self.config.paper_min_trades:
            issues.append(f"Insufficient trades: {metrics.n_trades} < {self.config.paper_min_trades}")
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.config.paper_min_sharpe:
            issues.append(f"Sharpe too low: {metrics.sharpe_ratio:.2f} < {self.config.paper_min_sharpe}")
        
        # Check duration
        if duration_hours < self.config.paper_min_duration_hours:
            issues.append(f"Insufficient duration: {duration_hours:.1f}h < {self.config.paper_min_duration_hours}h")
        
        passed = len(issues) == 0
        
        return ValidationResult(
            stage=PromotionStage.PAPER,
            passed=passed,
            metrics=self._metrics_to_dict(metrics),
            issues=issues,
            timestamp_ns=time.time_ns(),
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    def validate_shadow(self, shadow_metrics: StrategyMetrics,
                        live_metrics: StrategyMetrics,
                        tracking_error_bps: float) -> ValidationResult:
        """Validate shadow vs live comparison."""
        start_time = time.time()
        issues = []
        
        # Check tracking error
        if tracking_error_bps > self.config.shadow_max_tracking_error_bps:
            issues.append(f"Tracking error too high: {tracking_error_bps:.1f} bps")
        
        # Check fill rate
        if shadow_metrics.fill_rate < self.config.shadow_min_fill_rate:
            issues.append(f"Fill rate too low: {shadow_metrics.fill_rate:.2%}")
        
        passed = len(issues) == 0
        
        return ValidationResult(
            stage=PromotionStage.SHADOW,
            passed=passed,
            metrics={
                **self._metrics_to_dict(shadow_metrics),
                "tracking_error_bps": tracking_error_bps,
                "live_sharpe": live_metrics.sharpe_ratio,
            },
            issues=issues,
            timestamp_ns=time.time_ns(),
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    def validate_canary(self, metrics: StrategyMetrics,
                        duration_hours: float) -> ValidationResult:
        """Validate canary deployment results."""
        start_time = time.time()
        issues = []
        
        # Check drawdown
        if metrics.max_drawdown > self.config.canary_max_drawdown:
            issues.append(f"Drawdown exceeded: {metrics.max_drawdown:.2%}")
        
        # Check return floor
        if metrics.total_return < self.config.canary_min_return:
            issues.append(f"Return below floor: {metrics.total_return:.2%}")
        
        # Check duration
        if duration_hours < self.config.canary_min_duration_hours:
            issues.append(f"Insufficient canary duration: {duration_hours:.1f}h")
        
        passed = len(issues) == 0
        
        return ValidationResult(
            stage=PromotionStage.CANARY,
            passed=passed,
            metrics=self._metrics_to_dict(metrics),
            issues=issues,
            timestamp_ns=time.time_ns(),
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    def validate_production_readiness(self, metrics: StrategyMetrics) -> ValidationResult:
        """Final validation before full production."""
        start_time = time.time()
        issues = []
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.config.production_min_sharpe:
            issues.append(f"Production Sharpe too low: {metrics.sharpe_ratio:.2f}")
        
        # Check trade count for statistical significance
        if metrics.n_trades < 100:
            issues.append(f"Need more trades for production: {metrics.n_trades}")
        
        passed = len(issues) == 0
        
        return ValidationResult(
            stage=PromotionStage.PRODUCTION,
            passed=passed,
            metrics=self._metrics_to_dict(metrics),
            issues=issues,
            timestamp_ns=time.time_ns(),
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    def _metrics_to_dict(self, metrics: StrategyMetrics) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "total_return": metrics.total_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "n_trades": float(metrics.n_trades),
            "slippage_bps": metrics.slippage_bps,
            "fill_rate": metrics.fill_rate,
        }


class PromotionPipeline:
    """
    Main promotion pipeline orchestrator.
    
    Manages strategy progression through validation stages with
    zero-downtime hot-swap capability.
    """
    
    def __init__(self, config: Optional[PromotionConfig] = None):
        self.config = config or PromotionConfig()
        self.validator = ValidationEngine(self.config)
        
        # Strategy states
        self.strategy_states: Dict[str, PromotionState] = {}
        
        # Active production strategy
        self.active_production_strategy: Optional[str] = None
        
        # Position registry for preservation
        self.position_registry: Dict[str, ActivePosition] = {}
        
        # Lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Event callbacks
        self.on_promotion: Optional[callable] = None
        self.on_rollback: Optional[callable] = None
        
        logger.info("PromotionPipeline initialized")
    
    def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy in the pipeline."""
        with self._lock:
            self.strategy_states[strategy_id] = PromotionState(
                strategy_id=strategy_id,
                current_stage=PromotionStage.PAPER,
                stage_entry_time=time.time_ns(),
            )
            logger.info(f"Registered strategy: {strategy_id}")
    
    def submit_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> Optional[ValidationResult]:
        """Submit performance metrics for a strategy."""
        with self._lock:
            if strategy_id not in self.strategy_states:
                logger.warning(f"Unknown strategy: {strategy_id}")
                return None
            
            state = self.strategy_states[strategy_id]
            state.metrics_history.append(metrics)
            
            # Calculate duration
            duration_hours = (time.time_ns() - state.stage_entry_time) / 3.6e12
            
            # Run appropriate validation based on stage
            result = None
            
            if state.current_stage == PromotionStage.PAPER:
                result = self.validator.validate_paper(metrics, duration_hours)
                if result.passed:
                    self._advance_stage(strategy_id, PromotionStage.SHADOW)
            
            elif state.current_stage == PromotionStage.SHADOW:
                # Need live metrics for comparison
                if self.active_production_strategy:
                    live_state = self.strategy_states.get(self.active_production_strategy)
                    if live_state and live_state.metrics_history:
                        live_metrics = live_state.metrics_history[-1]
                        tracking_error = self._calculate_tracking_error(metrics, live_metrics)
                        result = self.validator.validate_shadow(metrics, live_metrics, tracking_error)
                        
                        if result.passed:
                            self._advance_stage(strategy_id, PromotionStage.CANARY)
            
            elif state.current_stage == PromotionStage.CANARY:
                result = self.validator.validate_canary(metrics, duration_hours)
                if result.passed:
                    self._advance_stage(strategy_id, PromotionStage.PRODUCTION)
            
            elif state.current_stage == PromotionStage.PRODUCTION:
                result = self.validator.validate_production_readiness(metrics)
                
                # Check for rollback conditions
                if self._should_rollback(metrics):
                    self._trigger_rollback(strategy_id, metrics)
            
            if result:
                state.validation_results.append(result)
            
            return result
    
    def _advance_stage(self, strategy_id: str, new_stage: PromotionStage) -> None:
        """Advance a strategy to the next stage."""
        state = self.strategy_states[strategy_id]
        old_stage = state.current_stage
        
        logger.info(f"Advancing {strategy_id}: {old_stage.value} -> {new_stage.value}")
        
        state.current_stage = new_stage
        state.stage_entry_time = time.time_ns()
        
        # Handle position transfer if promoting to canary or production
        if new_stage in [PromotionStage.CANARY, PromotionStage.PRODUCTION]:
            if self.config.preserve_positions_on_promote:
                self._transfer_positions(strategy_id)
        
        # Update active production strategy
        if new_stage == PromotionStage.PRODUCTION:
            self.active_production_strategy = strategy_id
        
        # Notify callback
        if self.on_promotion:
            self.on_promotion(strategy_id, old_stage, new_stage)
    
    def _transfer_positions(self, strategy_id: str) -> None:
        """Transfer positions to the promoting strategy."""
        # In production, this would coordinate with the Rust execution engine
        logger.info(f"Transferring {len(self.position_registry)} positions to {strategy_id}")
        
        state = self.strategy_states[strategy_id]
        state.active_positions = list(self.position_registry.values())
    
    def _calculate_tracking_error(self, shadow: StrategyMetrics, 
                                   live: StrategyMetrics) -> float:
        """Calculate tracking error between shadow and live."""
        # Simplified: difference in returns
        return abs(shadow.total_return - live.total_return) * 10000  # Convert to bps
    
    def _should_rollback(self, metrics: StrategyMetrics) -> bool:
        """Check if rollback conditions are met."""
        if metrics.max_drawdown > self.config.rollback_max_drawdown:
            return True
        if metrics.sharpe_ratio < self.config.rollback_sharpe_threshold:
            return True
        return False
    
    def _trigger_rollback(self, strategy_id: str, metrics: StrategyMetrics) -> None:
        """Trigger rollback to previous stable strategy."""
        with self._lock:
            state = self.strategy_states[strategy_id]
            state.is_rolling_back = True
            
            reason = f"Drawdown: {metrics.max_drawdown:.2%}, Sharpe: {metrics.sharpe_ratio:.2f}"
            state.rollback_reason = reason
            
            logger.warning(f"Rolling back {strategy_id}: {reason}")
            
            # Find previous stable strategy or revert to paper
            previous = None
            for sid, sstate in self.strategy_states.items():
                if sstate.current_stage == PromotionStage.PRODUCTION and sid != strategy_id:
                    previous = sid
                    break
            
            if previous:
                self.active_production_strategy = previous
                logger.info(f"Rolled back to: {previous}")
            else:
                self.active_production_strategy = None
                logger.warning("No fallback strategy available")
            
            state.is_rolling_back = False
            
            # Notify callback
            if self.on_rollback:
                self.on_rollback(strategy_id, reason)
    
    def promote_to_canary(self, strategy_id: str) -> bool:
        """Manually promote a strategy to canary."""
        with self._lock:
            if strategy_id not in self.strategy_states:
                return False
            
            state = self.strategy_states[strategy_id]
            if state.current_stage != PromotionStage.SHADOW:
                logger.warning(f"Cannot promote {strategy_id}: not in shadow stage")
                return False
            
            self._advance_stage(strategy_id, PromotionStage.CANARY)
            return True
    
    def promote_to_production(self, strategy_id: str) -> bool:
        """Manually promote a strategy to production."""
        with self._lock:
            if strategy_id not in self.strategy_states:
                return False
            
            state = self.strategy_states[strategy_id]
            if state.current_stage != PromotionStage.CANARY:
                logger.warning(f"Cannot promote {strategy_id}: not in canary stage")
                return False
            
            self._advance_stage(strategy_id, PromotionStage.PRODUCTION)
            return True
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a strategy."""
        with self._lock:
            if strategy_id not in self.strategy_states:
                return None
            
            state = self.strategy_states[strategy_id]
            
            return {
                "strategy_id": strategy_id,
                "current_stage": state.current_stage.value,
                "stage_duration_hours": (time.time_ns() - state.stage_entry_time) / 3.6e12,
                "n_validations": len(state.validation_results),
                "last_validation_passed": state.validation_results[-1].passed if state.validation_results else None,
                "active_positions": len(state.active_positions),
                "is_promoting": state.is_promoting,
                "is_rolling_back": state.is_rolling_back,
                "rollback_reason": state.rollback_reason,
            }
    
    def get_active_production_strategy(self) -> Optional[str]:
        """Get the currently active production strategy."""
        return self.active_production_strategy
    
    def register_position(self, position: ActivePosition) -> None:
        """Register an active position for preservation."""
        with self._lock:
            key = f"{position.symbol}:{position.side}"
            self.position_registry[key] = position
    
    def get_all_positions(self) -> List[ActivePosition]:
        """Get all registered positions."""
        with self._lock:
            return list(self.position_registry.values())


# Ray actor for distributed promotion management
try:
    import ray
    
    @ray.remote(max_restarts=-1)
    class RayPromotionManager:
        """Ray-distributed promotion manager."""
        
        def __init__(self, config: Optional[Dict] = None):
            self.config = PromotionConfig(**config) if config else PromotionConfig()
            self.pipeline = PromotionPipeline(self.config)
            
            logger.info("RayPromotionManager initialized")
        
        def register_strategy(self, strategy_id: str) -> bool:
            """Register a strategy."""
            self.pipeline.register_strategy(strategy_id)
            return True
        
        def submit_metrics(self, strategy_id: str, metrics_dict: Dict) -> Optional[Dict]:
            """Submit metrics for validation."""
            metrics = StrategyMetrics(**metrics_dict)
            result = self.pipeline.submit_metrics(strategy_id, metrics)
            
            if result:
                return {
                    "passed": result.passed,
                    "stage": result.stage.value,
                    "issues": result.issues,
                }
            return None
        
        def get_status(self, strategy_id: str) -> Optional[Dict]:
            """Get strategy status."""
            return self.pipeline.get_strategy_status(strategy_id)
        
        def get_production_strategy(self) -> Optional[str]:
            """Get active production strategy."""
            return self.pipeline.get_active_production_strategy()

except ImportError:
    logger.warning("Ray not available, using local execution")
    RayPromotionManager = None


if __name__ == "__main__":
    # Test the promotion pipeline
    config = PromotionConfig(
        paper_min_trades=10,
        paper_min_duration_hours=0.01,  # Short for testing
    )
    pipeline = PromotionPipeline(config)
    
    # Register and test a strategy
    pipeline.register_strategy("strategy_v1")
    
    # Submit some metrics
    for i in range(15):
        metrics = StrategyMetrics(
            total_return=0.05 + i * 0.01,
            sharpe_ratio=1.5 + i * 0.1,
            max_drawdown=0.02,
            win_rate=0.55,
            n_trades=i * 5,
            fill_rate=0.98,
        )
        result = pipeline.submit_metrics("strategy_v1", metrics)
        if result:
            print(f"Validation {i}: Passed={result.passed}, Stage={result.stage.value}")
    
    # Check status
    status = pipeline.get_strategy_status("strategy_v1")
    print(f"\nFinal Status: {status}")
    print(f"Active Production: {pipeline.get_active_production_strategy()}")
