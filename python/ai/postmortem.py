"""
Automated Trade Post-Mortem Analyzer
Identifies root causes of losses and formulates strategic pivots.
Writes self-corrections directly into the SOUL.md ledger.
AMD ROCm/DirectML acceleration support for analysis operations.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


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


class LossCategory(Enum):
    """Categories of trading losses."""
    TIMING_ERROR = "timing_error"
    SIZE_ERROR = "size_error"
    MARKET_REGIME = "market_regime"
    STRATEGY_FAILURE = "strategy_failure"
    EXECUTION_SLIPPAGE = "execution_slippage"
    STOP_HUNT = "stop_hunt"
    TREND_EXHAUSTION = "trend_exhaustion"
    UNKNOWN = "unknown"


@dataclass
class TradeAnalysis:
    """Detailed analysis of a single trade."""
    trade_id: str
    timestamp: str
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    strategy: str
    is_win: bool
    
    # Analysis results
    loss_category: Optional[LossCategory] = None
    root_cause: Optional[str] = None
    confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    suggested_mutations: List[str] = field(default_factory=list)
    
    # Market context at time of trade
    market_regime: Optional[str] = None
    volatility_percentile: Optional[float] = None
    volume_profile: Optional[str] = None


class PostMortemAnalyzer:
    """
    Automated trade analyzer for identifying loss patterns.
    
    Analyzes trades to determine:
    - Root cause of losses
    - Pattern recognition across multiple trades
    - Strategic recommendations
    - Hyperparameter mutation suggestions
    """
    
    def __init__(self, lookback_trades: int = 100):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        lookback_trades : int
            Number of trades to consider for pattern analysis
        """
        self.lookback_trades = lookback_trades
        self.accelerator = check_amd_acceleration()
        
        # Trade history storage
        self.trade_history: List[TradeAnalysis] = []
        
        # Pattern statistics
        self.pattern_stats: Dict[str, Dict] = {
            "by_strategy": {},
            "by_regime": {},
            "by_hour": {},
            "by_asset": {},
        }
        
        # Mutation recommendations queue
        self.mutation_queue: List[Dict] = []
    
    def analyze_trade(
        self,
        trade_data: Dict[str, Any],
        market_context: Optional[Dict] = None
    ) -> TradeAnalysis:
        """
        Perform post-mortem analysis on a completed trade.
        
        Parameters
        ----------
        trade_data : Dict
            Trade execution details
        market_context : Dict, optional
            Market conditions at trade time
        
        Returns
        -------
        TradeAnalysis
            Detailed analysis with root cause and recommendations
        """
        # Create base analysis
        analysis = TradeAnalysis(
            trade_id=trade_data.get("trade_id", "UNKNOWN"),
            timestamp=trade_data.get("timestamp", ""),
            asset=trade_data.get("asset", ""),
            direction=trade_data.get("direction", ""),
            entry_price=trade_data.get("entry_price", 0.0),
            exit_price=trade_data.get("exit_price", 0.0),
            size=trade_data.get("size", 0.0),
            pnl=trade_data.get("pnl", 0.0),
            pnl_percent=trade_data.get("pnl_percent", 0.0),
            strategy=trade_data.get("strategy", ""),
            is_win=trade_data.get("pnl", 0.0) > 0,
            market_regime=market_context.get("regime") if market_context else None,
            volatility_percentile=market_context.get("vol_percentile") if market_context else None,
        )
        
        # Analyze losing trades in detail
        if not analysis.is_win:
            self._analyze_loss(analysis, trade_data, market_context)
        else:
            self._analyze_win(analysis, trade_data, market_context)
        
        # Store in history
        self.trade_history.append(analysis)
        if len(self.trade_history) > self.lookback_trades:
            self.trade_history.pop(0)
        
        # Update pattern statistics
        self._update_pattern_stats(analysis)
        
        return analysis
    
    def _analyze_loss(
        self,
        analysis: TradeAnalysis,
        trade_data: Dict,
        market_context: Optional[Dict]
    ) -> None:
        """Analyze a losing trade to determine root cause."""
        
        # Calculate key metrics
        max_adverse_excursion = trade_data.get("max_adverse_excursion", 0.0)
        max_favorable_excursion = trade_data.get("max_favorable_excursion", 0.0)
        hold_time = trade_data.get("hold_time_minutes", 0)
        slippage = trade_data.get("slippage_bps", 0.0)
        
        # Determine loss category
        categories_scores = {}
        
        # Timing error: entered just before reversal
        if max_favorable_excursion > abs(analysis.pnl_percent) * 2:
            categories_scores[LossCategory.TIMING_ERROR] = 0.7
        
        # Size error: position too large for volatility
        if analysis.volatility_percentile and analysis.volatility_percentile > 80:
            if abs(analysis.pnl_percent) > 5.0:
                categories_scores[LossCategory.SIZE_ERROR] = 0.6
        
        # Market regime mismatch
        if analysis.market_regime:
            strategy_regime_map = {
                "trend_following": ["uptrend", "downtrend"],
                "mean_reversion": ["ranging", "low_volatility"],
                "breakout": ["expansion", "high_volatility"],
            }
            expected_regimes = strategy_regime_map.get(analysis.strategy, [])
            if analysis.market_regime not in expected_regimes:
                categories_scores[LossCategory.MARKET_REGIME] = 0.8
        
        # Execution slippage
        if slippage > 10:  # More than 10 bps
            categories_scores[LossCategory.EXECUTION_SLIPPAGE] = 0.5 + slippage / 100
        
        # Stop hunt: price hit stop then reversed
        if max_adverse_excursion < -abs(analysis.pnl_percent) * 0.9:
            if trade_data.get("exit_reason") == "stop_loss":
                categories_scores[LossCategory.STOP_HUNT] = 0.6
        
        # Trend exhaustion: late entry in mature trend
        if trade_data.get("trend_age_bars", 0) > 20:
            if analysis.direction == "long" and trade_data.get("rsi", 50) > 70:
                categories_scores[LossCategory.TREND_EXHAUSTION] = 0.7
        
        # Select most likely category
        if categories_scores:
            best_category = max(categories_scores.items(), key=lambda x: x[1])
            analysis.loss_category = best_category[0]
            analysis.confidence = best_category[1]
        else:
            analysis.loss_category = LossCategory.UNKNOWN
            analysis.confidence = 0.3
        
        # Generate root cause explanation
        analysis.root_cause = self._generate_root_cause(analysis, trade_data)
        
        # Identify contributing factors
        analysis.contributing_factors = self._identify_contributing_factors(
            analysis, trade_data, market_context
        )
        
        # Generate lessons learned
        analysis.lessons_learned = self._generate_lessons(analysis)
        
        # Suggest mutations
        analysis.suggested_mutations = self._suggest_mutations(analysis, trade_data)
    
    def _analyze_win(
        self,
        analysis: TradeAnalysis,
        trade_data: Dict,
        market_context: Optional[Dict]
    ) -> None:
        """Analyze a winning trade for positive patterns."""
        
        # Check if win was lucky or skill-based
        max_adverse = trade_data.get("max_adverse_excursion", 0.0)
        
        if max_adverse < -analysis.pnl_percent * 0.5:
            analysis.lessons_learned.append(
                "Trade showed significant drawdown before profit - consider tighter entry criteria"
            )
        else:
            analysis.lessons_learned.append(
                "Clean trade execution - replicate this setup"
            )
    
    def _generate_root_cause(
        self,
        analysis: TradeAnalysis,
        trade_data: Dict
    ) -> str:
        """Generate human-readable root cause explanation."""
        
        category_explanations = {
            LossCategory.TIMING_ERROR: (
                "Entry timing was suboptimal. Price moved favorably initially but reversed "
                "before target was reached. Consider waiting for additional confirmation."
            ),
            LossCategory.SIZE_ERROR: (
                "Position size was too large relative to market volatility. "
                "Normal price fluctuations triggered premature exit."
            ),
            LossCategory.MARKET_REGIME: (
                f"Strategy '{analysis.strategy}' is mismatched with current market regime "
                f"'{analysis.market_regime}'. Consider switching to regime-appropriate strategy."
            ),
            LossCategory.STRATEGY_FAILURE: (
                "Strategy signal failed to produce expected outcome. This may indicate "
                "structural market changes or degraded edge."
            ),
            LossCategory.EXECUTION_SLIPPAGE: (
                "Significant slippage eroded potential profits. Consider using limit orders "
                "or reducing order size during low liquidity periods."
            ),
            LossCategory.STOP_HUNT: (
                "Price action suggests stop-hunting behavior. Consider wider stops or "
                "placing stops at less obvious levels."
            ),
            LossCategory.TREND_EXHAUSTION: (
                "Entry occurred late in trend lifecycle. Momentum indicators showed "
                "divergence before entry."
            ),
            LossCategory.UNKNOWN: (
                "Loss pattern does not match known categories. May be random noise "
                "or novel market condition."
            ),
        }
        
        return category_explanations.get(analysis.loss_category, "Unknown cause")
    
    def _identify_contributing_factors(
        self,
        analysis: TradeAnalysis,
        trade_data: Dict,
        market_context: Optional[Dict]
    ) -> List[str]:
        """Identify factors that contributed to the outcome."""
        factors = []
        
        # Time-based factors
        hour = datetime.fromisoformat(analysis.timestamp).hour if analysis.timestamp else 0
        if hour in [0, 1, 2, 22, 23]:
            factors.append("Low liquidity overnight session")
        
        # Volatility factors
        if analysis.volatility_percentile and analysis.volatility_percentile > 90:
            factors.append("Extreme volatility environment")
        elif analysis.volatility_percentile and analysis.volatility_percentile < 10:
            factors.append("Very low volatility - prone to fakeouts")
        
        # Asset-specific factors
        if trade_data.get("spread_bps", 0) > 5:
            factors.append("Wide bid-ask spread increased transaction costs")
        
        # Strategy-specific factors
        if analysis.strategy == "breakout" and trade_data.get("volume_ratio", 1.0) < 0.5:
            factors.append("Breakout lacked volume confirmation")
        
        return factors
    
    def _generate_lessons(self, analysis: TradeAnalysis) -> List[str]:
        """Generate actionable lessons from the trade."""
        lessons = []
        
        if analysis.loss_category == LossCategory.TIMING_ERROR:
            lessons.append("Wait for pullback after initial breakout before entering")
            lessons.append("Use limit orders instead of market orders for better entry control")
        
        elif analysis.loss_category == LossCategory.SIZE_ERROR:
            lessons.append("Reduce position size by 25% during high volatility regimes")
            lessons.append("Implement volatility-adjusted position sizing")
        
        elif analysis.loss_category == LossCategory.MARKET_REGIME:
            lessons.append(f"Avoid {analysis.strategy} strategy in {analysis.market_regime} regime")
            lessons.append("Add regime filter to strategy entry conditions")
        
        elif analysis.loss_category == LossCategory.STOP_HUNT:
            lessons.append("Place stops below recent swing lows rather than fixed percentages")
            lessons.append("Consider using mental stops with manual execution")
        
        elif analysis.loss_category == LossCategory.TREND_EXHAUSTION:
            lessons.append("Check RSI divergence before entering late-stage trends")
            lessons.append("Reduce profit targets when trend age exceeds 20 bars")
        
        return lessons
    
    def _suggest_mutations(
        self,
        analysis: TradeAnalysis,
        trade_data: Dict
    ) -> List[str]:
        """Suggest hyperparameter mutations based on analysis."""
        mutations = []
        
        if analysis.loss_category == LossCategory.SIZE_ERROR:
            mutations.append("DECREASE position_size_pct by 20%")
            mutations.append("INCREASE max_drawdown_pct tolerance check sensitivity")
        
        elif analysis.loss_category == LossCategory.TIMING_ERROR:
            mutations.append("INCREASE entry_threshold by 0.5 standard deviations")
            mutations.append("ADD confirmation candle requirement")
        
        elif analysis.loss_category == LossCategory.STOP_HUNT:
            mutations.append("INCREASE stop_loss_pct by 30%")
            mutations.append("IMPLEMENT ATR-based dynamic stop placement")
        
        elif analysis.loss_category == LossCategory.MARKET_REGIME:
            mutations.append("ADD regime_filter to strategy configuration")
            mutations.append("DECREASE kelly_fraction during unfavorable regimes")
        
        # Add mutation to queue
        for mutation in mutations:
            self.mutation_queue.append({
                "trade_id": analysis.trade_id,
                "mutation": mutation,
                "priority": analysis.confidence,
                "timestamp": analysis.timestamp,
            })
        
        return mutations
    
    def _update_pattern_stats(self, analysis: TradeAnalysis) -> None:
        """Update running statistics for pattern detection."""
        
        # By strategy
        strategy = analysis.strategy
        if strategy not in self.pattern_stats["by_strategy"]:
            self.pattern_stats["by_strategy"][strategy] = {
                "wins": 0, "losses": 0, "total_pnl": 0.0
            }
        
        stats = self.pattern_stats["by_strategy"][strategy]
        if analysis.is_win:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["total_pnl"] += analysis.pnl
        
        # By regime
        if analysis.market_regime:
            regime = analysis.market_regime
            if regime not in self.pattern_stats["by_regime"]:
                self.pattern_stats["by_regime"][regime] = {"trades": 0, "win_rate": 0.0}
            self.pattern_stats["by_regime"][regime]["trades"] += 1
    
    def get_aggregate_insights(self) -> Dict[str, Any]:
        """Generate aggregate insights from trade history."""
        if not self.trade_history:
            return {"error": "No trades analyzed"}
        
        total_trades = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t.is_win)
        losses = total_trades - wins
        total_pnl = sum(t.pnl for t in self.trade_history)
        
        # Loss category distribution
        loss_categories = {}
        for t in self.trade_history:
            if not t.is_win and t.loss_category:
                cat = t.loss_category.value
                loss_categories[cat] = loss_categories.get(cat, 0) + 1
        
        # Strategy performance
        strategy_perf = {}
        for strategy, stats in self.pattern_stats["by_strategy"].items():
            total = stats["wins"] + stats["losses"]
            if total > 0:
                strategy_perf[strategy] = {
                    "win_rate": stats["wins"] / total,
                    "avg_pnl": stats["total_pnl"] / total,
                    "total_trades": total,
                }
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0.0,
            "total_pnl": total_pnl,
            "loss_category_distribution": loss_categories,
            "strategy_performance": strategy_perf,
            "pending_mutations": len(self.mutation_queue),
        }
    
    def generate_soul_entry(self, analysis: TradeAnalysis) -> str:
        """Generate formatted entry for SOUL.md ledger."""
        result_icon = "✅" if analysis.is_win else "❌"
        
        md = f"""
## {result_icon} Trade Analysis: {analysis.trade_id}

**Asset:** {analysis.asset} | **Direction:** {analysis.direction} | **Strategy:** {analysis.strategy}

| Metric | Value |
|--------|-------|
| Entry | ${analysis.entry_price:.8f} |
| Exit | ${analysis.exit_price:.8f} |
| PnL | ${analysis.pnl:.2f} ({analysis.pnl_percent:.2f}%) |

"""
        
        if not analysis.is_win and analysis.root_cause:
            md += f"""
### Root Cause: {analysis.loss_category.value if analysis.loss_category else 'Unknown'}

{analysis.root_cause}

"""
        
        if analysis.contributing_factors:
            md += "### Contributing Factors\n"
            for factor in analysis.contributing_factors:
                md += f"- {factor}\n"
            md += "\n"
        
        if analysis.lessons_learned:
            md += "### Lessons Learned\n"
            for lesson in analysis.lessons_learned:
                md += f"- {lesson}\n"
            md += "\n"
        
        if analysis.suggested_mutations:
            md += "### Suggested Mutations\n"
            for mutation in analysis.suggested_mutations:
                md += f"🔄 {mutation}\n"
            md += "\n"
        
        md += "---\n"
        
        return md
    
    def get_pending_mutations(self) -> List[Dict]:
        """Get queued mutation recommendations."""
        return sorted(self.mutation_queue, key=lambda x: x.get("priority", 0), reverse=True)
    
    def clear_mutation_queue(self) -> None:
        """Clear processed mutations from queue."""
        self.mutation_queue.clear()


if __name__ == "__main__":
    # Example usage
    analyzer = PostMortemAnalyzer()
    
    # Simulate a losing trade
    trade_data = {
        "trade_id": "T001",
        "timestamp": "2024-01-15T14:30:00",
        "asset": "BTCUSDT",
        "direction": "Long",
        "entry_price": 42000.0,
        "exit_price": 41500.0,
        "size": 0.5,
        "pnl": -250.0,
        "pnl_percent": -1.19,
        "strategy": "trend_following",
        "max_adverse_excursion": -1.5,
        "max_favorable_excursion": 0.3,
        "hold_time_minutes": 45,
        "slippage_bps": 5,
        "exit_reason": "stop_loss",
        "trend_age_bars": 25,
        "rsi": 72,
    }
    
    market_context = {
        "regime": "ranging",
        "vol_percentile": 45,
    }
    
    analysis = analyzer.analyze_trade(trade_data, market_context)
    
    print(f"Loss Category: {analysis.loss_category}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"\nRoot Cause:\n{analysis.root_cause}")
    print(f"\nLessons:")
    for lesson in analysis.lessons_learned:
        print(f"  - {lesson}")
    print(f"\nSuggested Mutations:")
    for mutation in analysis.suggested_mutations:
        print(f"  🔄 {mutation}")
    
    print("\n--- SOUL.md Entry ---")
    print(analyzer.generate_soul_entry(analysis))
