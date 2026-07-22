"""
Git-based SOUL.md Version Control System

Implements an automated, background Git-like version control system for SOUL.md,
committing strategy mutations and post-mortems asynchronously without blocking
the hot path. Uses lock-free append-only logs before batching into Git commits.

Features:
- Lock-free append-only logging
- Async Git commit batching
- Strategy mutation tracking
- Post-mortem documentation
- Non-blocking operation on hot path
"""

import os
import hashlib
import json
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
from queue import Queue, Empty
import subprocess


@dataclass
class SoulEntry:
    """Single entry in the SOUL.md log"""
    timestamp_ns: int
    entry_type: str  # 'mutation', 'post_mortem', 'metric', 'config'
    strategy_id: str
    content_hash: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SoulEntry':
        return cls(**data)


class LockFreeAppendLog:
    """
    Lock-free append-only log for SOUL.md entries.
    Uses atomic file operations for thread safety.
    """
    
    def __init__(self, log_path: str, max_entries_per_batch: int = 100):
        self.log_path = Path(log_path)
        self.max_entries_per_batch = max_entries_per_batch
        
        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffer for pending entries
        self.pending_entries: deque = deque()
        self.pending_lock = threading.Lock()
        
        # Statistics
        self.total_appends = 0
        self.total_commits = 0
    
    def append(self, entry: SoulEntry) -> bool:
        """
        Append entry to log (lock-free for reads, locked for writes).
        Returns True if entry was added successfully.
        """
        with self.pending_lock:
            self.pending_entries.append(entry)
            self.total_appends += 1
            
            # Check if we should trigger a batch commit
            if len(self.pending_entries) >= self.max_entries_per_batch:
                return True  # Signal that batch is ready
        
        return False
    
    def get_pending_batch(self) -> List[SoulEntry]:
        """Get pending entries for batch commit."""
        with self.pending_lock:
            batch_size = min(len(self.pending_entries), self.max_entries_per_batch)
            batch = []
            for _ in range(batch_size):
                if self.pending_entries:
                    batch.append(self.pending_entries.popleft())
            return batch
    
    def persist_entry(self, entry: SoulEntry) -> bool:
        """Persist single entry to log file atomically."""
        try:
            entry_json = json.dumps(entry.to_dict()) + '\n'
            
            # Atomic append using 'a' mode
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(entry_json)
                f.flush()
                os.fsync(f.fileno())  # Ensure written to disk
            
            return True
        except Exception as e:
            print(f"Error persisting entry: {e}")
            return False
    
    def read_entries(self, limit: int = 1000) -> List[SoulEntry]:
        """Read entries from log file."""
        entries = []
        
        if not self.log_path.exists():
            return entries
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        entries.append(SoulEntry.from_dict(data))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading log: {e}")
        
        return entries


class GitSoulVersionControl:
    """
    Git-based version control for SOUL.md strategy documentation.
    Runs asynchronously to avoid blocking the hot path.
    """
    
    def __init__(
        self,
        soul_dir: str = "./soul",
        git_remote: Optional[str] = None,
        commit_batch_size: int = 50,
        auto_commit_interval_sec: float = 60.0,
    ):
        self.soul_dir = Path(soul_dir)
        self.git_remote = git_remote
        self.commit_batch_size = commit_batch_size
        self.auto_commit_interval_sec = auto_commit_interval_sec
        
        # Initialize directories
        self.soul_dir.mkdir(parents=True, exist_ok=True)
        (self.soul_dir / "strategies").mkdir(exist_ok=True)
        (self.soul_dir / "post_mortems").mkdir(exist_ok=True)
        (self.soul_dir / "logs").mkdir(exist_ok=True)
        
        # Append-only log
        self.append_log = LockFreeAppendLog(
            self.soul_dir / "logs" / "soul_log.jsonl",
            max_entries_per_batch=commit_batch_size,
        )
        
        # Background commit thread
        self.commit_queue: Queue = Queue()
        self.commit_thread: Optional[threading.Thread] = None
        self.shutdown_flag = threading.Event()
        
        # Strategy registry (in-memory cache)
        self.strategies: Dict[str, StrategyRecord] = {}
        
        # Statistics
        self.stats = {
            'commits_made': 0,
            'entries_logged': 0,
            'last_commit_time': None,
        }
    
    def start_background_processor(self):
        """Start background thread for async commits."""
        if self.commit_thread is not None and self.commit_thread.is_alive():
            return
        
        self.commit_thread = threading.Thread(
            target=self._background_commit_loop,
            daemon=True,
            name="SoulGitCommitThread",
        )
        self.commit_thread.start()
    
    def stop_background_processor(self):
        """Stop background processor gracefully."""
        self.shutdown_flag.set()
        if self.commit_thread:
            self.commit_thread.join(timeout=5.0)
    
    def _background_commit_loop(self):
        """Background loop for processing commits."""
        last_commit_time = time.time()
        
        while not self.shutdown_flag.is_set():
            try:
                # Check if batch commit is needed
                current_time = time.time()
                time_since_commit = current_time - last_commit_time
                
                # Get pending entries
                pending = self.append_log.get_pending_batch()
                
                if pending and (
                    len(pending) >= self.commit_batch_size or
                    time_since_commit >= self.auto_commit_interval_sec
                ):
                    self._create_git_commit(pending)
                    last_commit_time = current_time
                    self.stats['commits_made'] += 1
                    self.stats['last_commit_time'] = datetime.now().isoformat()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Background commit error: {e}")
                time.sleep(1.0)
    
    def record_mutation(
        self,
        strategy_id: str,
        mutation_type: str,
        old_config: Dict,
        new_config: Dict,
        reason: str = "",
    ) -> str:
        """Record a strategy mutation."""
        content = json.dumps({
            'type': 'mutation',
            'mutation_type': mutation_type,
            'old_config': old_config,
            'new_config': new_config,
            'reason': reason,
        })
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        entry = SoulEntry(
            timestamp_ns=time.time_ns(),
            entry_type='mutation',
            strategy_id=strategy_id,
            content_hash=content_hash,
            content=content,
            metadata={
                'mutation_type': mutation_type,
                'reason': reason,
            },
        )
        
        self.append_log.append(entry)
        self.stats['entries_logged'] += 1
        
        # Update strategy record
        if strategy_id in self.strategies:
            self.strategies[strategy_id].mutations.append(entry)
        
        return content_hash
    
    def record_post_mortem(
        self,
        strategy_id: str,
        incident_type: str,
        description: str,
        root_cause: str,
        remediation: str,
        metrics_before: Dict,
        metrics_after: Dict,
    ) -> str:
        """Record a post-mortem analysis."""
        content = json.dumps({
            'type': 'post_mortem',
            'incident_type': incident_type,
            'description': description,
            'root_cause': root_cause,
            'remediation': remediation,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
        })
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        entry = SoulEntry(
            timestamp_ns=time.time_ns(),
            entry_type='post_mortem',
            strategy_id=strategy_id,
            content_hash=content_hash,
            content=content,
            metadata={
                'incident_type': incident_type,
                'severity': 'high',
            },
        )
        
        self.append_log.append(entry)
        self.stats['entries_logged'] += 1
        
        # Save post-mortem to file
        pm_file = self.soul_dir / "post_mortems" / f"{strategy_id}_{content_hash}.md"
        self._write_post_mortem_file(pm_file, entry)
        
        return content_hash
    
    def _write_post_mortem_file(self, path: Path, entry: SoulEntry):
        """Write post-mortem to markdown file."""
        content = json.loads(entry.content)
        
        md_content = f"""# Post-Mortem: {entry.strategy_id}

## Incident Information
- **Timestamp**: {datetime.fromtimestamp(entry.timestamp_ns / 1e9).isoformat()}
- **Type**: {content.get('incident_type', 'Unknown')}
- **Content Hash**: {entry.content_hash}

## Description
{content.get('description', 'No description provided.')}

## Root Cause
{content.get('root_cause', 'Root cause not determined.')}

## Remediation
{content.get('remediation', 'No remediation specified.')}

## Metrics Comparison

### Before Incident
```json
{json.dumps(content.get('metrics_before', {}), indent=2)}
```

### After Remediation
```json
{json.dumps(content.get('metrics_after', {}), indent=2)}
```

---
*Generated automatically by SOUL.md Version Control System*
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _create_git_commit(self, entries: List[SoulEntry]):
        """Create Git commit for batch of entries."""
        try:
            # Write entries to log file
            for entry in entries:
                self.append_log.persist_entry(entry)
            
            # Initialize git repo if needed
            if not (self.soul_dir / ".git").exists():
                self._init_git_repo()
            
            # Stage changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.soul_dir,
                capture_output=True,
                timeout=30,
            )
            
            # Create commit
            commit_msg = f"Soul update: {len(entries)} entries\n\n"
            for entry in entries[:5]:  # Include first 5 entries in message
                commit_msg += f"- [{entry.entry_type}] {entry.strategy_id}: {entry.content_hash}\n"
            
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.soul_dir,
                capture_output=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                # Push to remote if configured
                if self.git_remote:
                    subprocess.run(
                        ["git", "push", self.git_remote, "main"],
                        cwd=self.soul_dir,
                        capture_output=True,
                        timeout=60,
                    )
            
        except subprocess.TimeoutExpired:
            print("Git commit timed out")
        except Exception as e:
            print(f"Git commit error: {e}")
    
    def _init_git_repo(self):
        """Initialize Git repository."""
        try:
            subprocess.run(
                ["git", "init"],
                cwd=self.soul_dir,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "checkout", "-b", "main"],
                cwd=self.soul_dir,
                capture_output=True,
                timeout=10,
            )
        except Exception as e:
            print(f"Git init error: {e}")
    
    def register_strategy(self, strategy_id: str, config: Dict, initial_hash: str):
        """Register a new strategy."""
        self.strategies[strategy_id] = StrategyRecord(
            strategy_id=strategy_id,
            config=config,
            initial_hash=initial_hash,
            created_at=time.time_ns(),
        )
    
    def get_strategy_history(self, strategy_id: str) -> List[SoulEntry]:
        """Get history of mutations for a strategy."""
        entries = self.append_log.read_entries(limit=10000)
        return [e for e in entries if e.strategy_id == strategy_id]
    
    def get_statistics(self) -> Dict:
        """Get version control statistics."""
        return {
            **self.stats,
            'registered_strategies': len(self.strategies),
            'pending_entries': len(self.append_log.pending_entries),
        }


@dataclass
class StrategyRecord:
    """Record of a registered strategy."""
    strategy_id: str
    config: Dict
    initial_hash: str
    created_at: int
    mutations: List[SoulEntry] = field(default_factory=list)


# Example usage
if __name__ == "__main__":
    # Initialize version control
    vcs = GitSoulVersionControl(
        soul_dir="./soul_test",
        commit_batch_size=10,
        auto_commit_interval_sec=30.0,
    )
    
    # Start background processor
    vcs.start_background_processor()
    
    # Register a strategy
    vcs.register_strategy(
        strategy_id="test_strategy_001",
        config={"threshold": 0.5, "lookback": 100},
        initial_hash="abc123",
    )
    
    # Record a mutation
    vcs.record_mutation(
        strategy_id="test_strategy_001",
        mutation_type="parameter_change",
        old_config={"threshold": 0.5},
        new_config={"threshold": 0.6},
        reason="Performance optimization based on backtest",
    )
    
    # Record a post-mortem
    vcs.record_post_mortem(
        strategy_id="test_strategy_001",
        incident_type="drawdown_exceeded",
        description="Strategy exceeded max drawdown during flash crash",
        root_cause="Insufficient stop-loss mechanism",
        remediation="Added dynamic stop-loss based on volatility",
        metrics_before={"drawdown": 0.15, "sharpe": 1.2},
        metrics_after={"drawdown": 0.08, "sharpe": 1.5},
    )
    
    # Wait for background processing
    time.sleep(2)
    
    # Get statistics
    print("Statistics:", vcs.get_statistics())
    
    # Get strategy history
    history = vcs.get_strategy_history("test_strategy_001")
    print(f"History entries: {len(history)}")
    
    # Cleanup
    vcs.stop_background_processor()
