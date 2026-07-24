"""
continuous_learner.py - SOUL.md Autonomous Learning Engine
Stage 54: Nautilus/Ray Crypto Trading Bot
Background Ray worker that continuously reads trade post-mortems and updates SOUL.md
Optimized for 4GB RAM quota, AMD DirectML/ROCm GPU acceleration
"""

import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading
import queue

import ray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enforce 4GB RAM limit for this process
try:
    import resource
    FOUR_GB = 4 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (FOUR_GB, FOUR_GB))
    logger.info(f"RAM limit set to 4GB")
except (ImportError, ValueError):
    logger.warning("Could not set RAM limit (Windows or unavailable)")


@dataclass
class TradePostMortem:
    """Represents a completed trade analysis for learning"""
    trade_id: str
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    strategy_name: str
    mistake_category: Optional[str] = None
    mistake_severity: float = 0.0  # 0.0 to 1.0
    lessons_learned: List[str] = field(default_factory=list)
    avoidance_rules: List[Dict[str, Any]] = field(default_factory=list)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradePostMortem':
        return cls(**data)


@dataclass
class AvoidanceRule:
    """A rule to avoid repeating mistakes"""
    rule_id: str
    pattern_hash: str
    condition: str
    penalty_multiplier: float
    created_at: str
    violation_count: int = 0
    last_violation: Optional[str] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MistakePatternDetector:
    """Detects repeated mistake patterns using GPU-accelerated inference"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.pattern_cache: Dict[str, List[TradePostMortem]] = defaultdict(list)
        self.severity_threshold = 0.5
        
        # Try to initialize GPU acceleration (AMD DirectML/ROCm)
        self._init_gpu_acceleration()
    
    def _init_gpu_acceleration(self):
        """Initialize GPU acceleration for pattern detection"""
        try:
            # Try ROCm for AMD GPUs
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.device = 'cuda'
                        logger.info("ROCm/CUDA acceleration enabled")
                    else:
                        self.device = 'cpu'
                        logger.info("GPU not available, using CPU")
                except ImportError:
                    self.device = 'cpu'
                    logger.info("PyTorch not available, using CPU-only mode")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
            self.device = 'cpu'
    
    def compute_pattern_hash(self, mortem: TradePostMortem) -> str:
        """Compute a hash of the mistake pattern for quick lookup"""
        pattern_data = {
            'symbol': mortem.symbol,
            'strategy': mortem.strategy_name,
            'mistake_category': mortem.mistake_category,
            'market_regime': mortem.market_conditions.get('regime', 'unknown'),
            'timeframe': mortem.market_conditions.get('timeframe', 'unknown'),
        }
        pattern_json = json.dumps(pattern_data, sort_keys=True)
        return hashlib.sha256(pattern_json.encode()).hexdigest()[:16]
    
    def add_mortem(self, mortem: TradePostMortem) -> Optional[AvoidanceRule]:
        """Add a post-mortem and check for repeated patterns"""
        pattern_hash = self.compute_pattern_hash(mortem)
        self.pattern_cache[pattern_hash].append(mortem)
        
        # Check if this is a repeated mistake
        occurrences = len(self.pattern_cache[pattern_hash])
        
        if occurrences >= 3:  # Pattern confirmed after 3 occurrences
            avg_severity = sum(m.mistake_severity for m in self.pattern_cache[pattern_hash]) / occurrences
            
            if avg_severity >= self.severity_threshold:
                # Create avoidance rule
                rule = self._create_avoidance_rule(
                    pattern_hash,
                    self.pattern_cache[pattern_hash],
                    avg_severity
                )
                logger.warning(f"Repeated mistake pattern detected! Rule created: {rule.rule_id}")
                return rule
        
        return None
    
    def _create_avoidance_rule(
        self,
        pattern_hash: str,
        mortems: List[TradePostMortem],
        avg_severity: float
    ) -> AvoidanceRule:
        """Create an avoidance rule from repeated mistakes"""
        first_mortem = mortems[0]
        
        # Generate condition string based on pattern
        condition_parts = [
            f"symbol == '{first_mortem.symbol}'",
            f"strategy == '{first_mortem.strategy_name}'",
        ]
        
        if first_mortem.mistake_category:
            condition_parts.append(f"category == '{first_mortem.mistake_category}'")
        
        condition = ' AND '.join(condition_parts)
        
        # Penalty scales with severity and occurrence count
        penalty = min(10.0, 1.0 + (avg_severity * len(mortems)))
        
        return AvoidanceRule(
            rule_id=f"AR-{pattern_hash}",
            pattern_hash=pattern_hash,
            condition=condition,
            penalty_multiplier=penalty,
            created_at=datetime.utcnow().isoformat(),
        )


@ray.remote(num_cpus=2, max_calls=1)
class ContinuousLearner:
    """Ray actor for continuous learning from trade post-mortems"""
    
    def __init__(self, soul_md_path: str, ram_limit_gb: float = 4.0):
        self.soul_md_path = Path(soul_md_path)
        self.ram_limit_bytes = int(ram_limit_gb * 1024 * 1024 * 1024)
        
        self.pattern_detector = MistakePatternDetector(use_gpu=True)
        self.avoidance_rules: Dict[str, AvoidanceRule] = {}
        self.mortem_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        self.running = False
        self.stats = {
            'mortems_processed': 0,
            'rules_created': 0,
            'patterns_detected': 0,
            'gpu_acceleration': self.pattern_detector.device == 'cuda',
        }
        
        logger.info(f"ContinuousLearner initialized (RAM limit: {ram_limit_gb}GB)")
    
    def submit_post_mortem(self, mortem_dict: Dict) -> bool:
        """Submit a trade post-mortem for analysis"""
        try:
            mortem = TradePostMortem.from_dict(mortem_dict)
            
            if self.mortem_queue.full():
                logger.warning("Post-mortem queue full, dropping oldest")
                try:
                    self.mortem_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.mortem_queue.put(mortem)
            return True
        except Exception as e:
            logger.error(f"Failed to submit post-mortem: {e}")
            return False
    
    def process_pending_mortems(self) -> int:
        """Process all pending post-mortems in the queue"""
        processed = 0
        
        while not self.mortem_queue.empty():
            try:
                mortem = self.mortem_queue.get_nowait()
                
                # Detect patterns and potentially create avoidance rules
                new_rule = self.pattern_detector.add_mortem(mortem)
                
                if new_rule:
                    self.avoidance_rules[new_rule.rule_id] = new_rule
                    self.stats['rules_created'] += 1
                    
                    # Update SOUL.md immediately for critical rules
                    if new_rule.penalty_multiplier >= 5.0:
                        self._write_to_soul_md(mortem, new_rule)
                
                self.stats['mortems_processed'] += 1
                processed += 1
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing post-mortem: {e}")
        
        return processed
    
    def _write_to_soul_md(self, mortem: TradePostMortem, rule: AvoidanceRule):
        """Write critical learning to SOUL.md ledger"""
        try:
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            
            entry = f"""
## Critical Avoidance Rule Added: {timestamp}

**Rule ID**: {rule.rule_id}
**Pattern Hash**: {rule.pattern_hash}
**Penalty Multiplier**: {rule.penalty_multiplier}x

### Triggering Trade
- **Symbol**: {mortem.symbol}
- **Strategy**: {mortem.strategy_name}
- **PnL**: {mortem.pnl:.2f} ({mortem.pnl_percent:.2f}%)
- **Mistake Category**: {mortem.mistake_category}

### Avoidance Condition
```
{rule.condition}
```

### Lessons Learned
{chr(10).join('- ' + lesson for lesson in mortem.lessons_learned)}

---
"""
            
            # Ensure directory exists
            self.soul_md_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to SOUL.md
            mode = 'a' if self.soul_md_path.exists() else 'w'
            with open(self.soul_md_path, mode, encoding='utf-8') as f:
                if mode == 'w':
                    f.write("# SOUL.md - Autonomous Learning Ledger\n\n")
                    f.write("*This file contains learned avoidance rules and trading wisdom.*\n\n")
                f.write(entry)
            
            logger.info(f"Written to SOUL.md: Rule {rule.rule_id}")
            
        except Exception as e:
            logger.error(f"Failed to write to SOUL.md: {e}")
    
    def get_active_rules(self) -> List[Dict]:
        """Get all active avoidance rules for penalty injection"""
        return [
            rule.to_dict() 
            for rule in self.avoidance_rules.values() 
            if rule.is_active
        ]
    
    def get_stats(self) -> Dict:
        """Get learner statistics"""
        return {
            **self.stats,
            'active_rules': len([r for r in self.avoidance_rules.values() if r.is_active]),
            'queue_size': self.mortem_queue.qsize(),
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    def run_learning_loop(self, interval_seconds: float = 5.0):
        """Run the continuous learning loop"""
        self.running = True
        logger.info("Starting continuous learning loop")
        
        while self.running:
            try:
                processed = self.process_pending_mortems()
                
                if processed > 0:
                    logger.info(f"Processed {processed} post-mortems")
                
                # Periodic stats logging
                if self.stats['mortems_processed'] % 100 == 0:
                    logger.info(f"Learner stats: {self.get_stats()}")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Learning loop interrupted")
                break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(interval_seconds)
        
        self.running = False
        logger.info("Learning loop stopped")
    
    def stop(self):
        """Stop the learning loop"""
        self.running = False


def start_learner():
    """Entry point for starting the continuous learner"""
    # Get base directory
    base_dir = Path(__file__).parent.parent.parent
    soul_md_path = base_dir / "SOUL.md"
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            num_cpus=4,
            object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB object store
            _system_config={"max_direct_call_object_size": 1024 * 1024}
        )
        logger.info("Ray initialized for continuous learner")
    
    # Create and start learner
    learner = ContinuousLearner.remote(str(soul_md_path), ram_limit_gb=4.0)
    
    # Start learning loop in background thread
    def run_loop():
        ray.get(learner.run_learning_loop.remote(interval_seconds=5.0))
    
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    
    logger.info("Continuous learner started")
    return learner


if __name__ == "__main__":
    # Run as standalone process
    learner = start_learner()
    
    # Keep alive
    try:
        while True:
            time.sleep(60)
            stats = ray.get(learner.get_stats.remote())
            logger.info(f"Stats: {stats}")
    except KeyboardInterrupt:
        ray.get(learner.stop.remote())
        logger.info("Continuous learner shutdown complete")
