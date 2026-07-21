# SOUL.md - Self-Learning Optimized Universal Ledger
# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - AUTONOMOUS MEMORY & LEARNING SYSTEM
# =============================================================================
# 
# This document serves as the persistent memory ledger for the trading bot's
# self-learning architecture. All trade post-mortems, strategy mutations, and
# risk parameter adjustments are recorded here for continuous improvement.
#
# Architecture Version: 1.0.0
# Target Hardware: AMD Ryzen AI 5 (Windows)
# Memory Constraint: 8GB Total System RAM
# Latency Budget: <100μs hot path execution
# =============================================================================

## SYSTEM ARCHITECTURE OVERVIEW

### Core Components
```
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON AI CLUSTER (Ray)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ RL Brain    │  │ Feature     │  │ Walk-Forward Trainer    │  │
│  │ (PPO/A2C)   │  │ Extractor   │  │ (Historical Tick Data)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                    Memory Cap: 4GB                               │
└─────────────────────────────────────────────────────────────────┘
                              ↕ MPSC Channels (Lock-Free)
┌─────────────────────────────────────────────────────────────────┐
│                   RUST EXECUTION ENGINE                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Market Data │  │ Risk        │  │ Order Execution         │  │
│  │ Ingestion   │  │ Manager     │  │ (Binance Futures API)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                    Memory Cap: 4GB                               │
│                    Latency: <10μs per event                      │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Layout Specification
| Component          | Allocated Memory | Type        | Priority |
|--------------------|------------------|-------------|----------|
| Rust Heap          | 2.5 GB           | Stack/Heap  | Critical |
| Rust Channel Buffers| 512 MB          | Pre-allocated| Critical |
| Ray Object Store   | 2.0 GB           | Shared Mem  | High     |
| Python Workers     | 2.0 GB           | Heap        | Medium   |
| OS Reserve         | 1.0 GB           | System      | Low      |

## RISK PARAMETERS (Current State)

### Position Sizing Rules
- **Maximum Position Size**: $10,000 USD notional per symbol
- **Maximum Leverage**: 10x isolated margin
- **Maximum Concurrent Positions**: 5 symbols
- **Position Rebalancing Threshold**: 5% drift from target allocation

### Stop-Loss Configuration
- **Hard Stop-Loss**: 2.0% from entry price
- **Trailing Stop**: Activates at 3% profit, trails by 1.5%
- **Time-Based Exit**: Close position after 4 hours if no movement >0.5%

### Daily Loss Limits
- **Maximum Daily Loss**: $500 USD (hard kill switch)
- **Drawdown Throttle**: Reduce position size by 50% after 50% of daily limit
- **Cool-Down Period**: 1 hour trading suspension after any daily loss >$300

### Order Execution Constraints
- **Maximum Open Orders**: 10 per symbol
- **Order Timeout**: 500ms before cancellation and retry
- **Minimum Order Size**: $10 USD (Binance Futures minimum)
- **Slippage Tolerance**: 0.05% maximum acceptable slippage

## STRATEGY MUTATION LOG

### Template for Future Autonomous Updates
```markdown
#### Mutation #____ - [DATE]
**Trigger Condition**: [Describe market condition or performance metric that triggered mutation]

**Previous Parameters**:
- Parameter A: value_x
- Parameter B: value_y

**New Parameters**:
- Parameter A: value_x_prime
- Parameter B: value_y_prime

**Expected Impact**: [Quantitative prediction of performance change]

**Validation Method**: [Walk-forward test results, backtest metrics]

**Approval Status**: [Pending | Approved | Rejected]
**Approved By**: [Autonomous RL Agent | Human Supervisor]
```

## TRADE POST-MORTEM TEMPLATE

### Trade ID: [SYMBOL]_[ENTRY_TIMESTAMP]_[DIRECTION]

#### Execution Summary
| Field              | Value                                    |
|--------------------|------------------------------------------|
| Symbol             | [e.g., BTCUSDT]                          |
| Entry Price        | [Fill on execution]                      |
| Exit Price         | [Fill on closure]                        |
| Direction          | LONG / SHORT                             |
| Position Size      | [USD notional]                           |
| Leverage           | [Isolated margin multiplier]             |
| Entry Timestamp    | [ISO 8601 UTC]                           |
| Exit Timestamp     | [ISO 8601 UTC]                           |
| Duration           | [Seconds]                                |

#### Performance Metrics
| Metric             | Value                                    |
|--------------------|------------------------------------------|
| P&L (USD)          | [Realized profit/loss]                   |
| P&L (%)            | [Percentage return]                      |
| Max Drawdown       | [Peak-to-trough decline during trade]    |
| Max Favorable Excursion | [Highest unrealized profit]         |
| Slippage (bps)     | [Execution quality metric]               |
| Commission (USD)   | [Binance fees paid]                      |

#### Decision Analysis
**Entry Signal Source**: [AI Model / Technical Indicator / Manual]
- Model Confidence: [0.0 - 1.0]
- Feature Importance Snapshot: [Top 5 features contributing to signal]

**Exit Reason**: [Take Profit / Stop Loss / Time Exit / Emergency Kill]
- Exit Signal Latency: [Microseconds from signal to order submission]
- Fill Quality: [Maker/Taker ratio, partial fills]

#### Lessons Learned
[Autonomous analysis of what went well, what failed, and proposed adjustments]

#### Strategy Adjustments Proposed
[List specific parameter changes or logic modifications based on this trade]

---

## AUTONOMOUS LEARNING CYCLE STATUS

### Last Training Run
- **Timestamp**: [ISO 8601 UTC]
- **Episodes Completed**: [Number]
- **Reward Metric**: [Sharpe Ratio / Sortino / Custom]
- **Model Checkpoint**: [Path to saved model weights]

### Next Scheduled Training
- **Trigger**: [Time-based / Performance degradation / New data threshold]
- **Estimated Start**: [ISO 8601 UTC]
- **Data Window**: [Start date] to [End date]

### Feature Engineering Queue
[List of pending feature additions or modifications scheduled for next iteration]

---

## EMERGENCY PROTOCOLS

### Kill Switch Activation Conditions
1. Daily loss exceeds $500 USD
2. Network latency spikes above 100ms for >10 consecutive seconds
3. Binance API returns repeated 429/503 errors (>5 in 1 minute)
4. Memory usage exceeds 7.5GB (93% of 8GB cap)
5. Rust engine panics or Ray cluster loses >50% of workers

### Graceful Shutdown Sequence
1. Cancel all open orders via Binance API (with 3 retries)
2. Persist current state to `SOUL.md` and checkpoint files
3. Flush mmap log buffers to disk
4. Release Ray cluster resources
5. Force garbage collection in Python runtime
6. Zero out sensitive memory regions in Rust engine

### Recovery Procedure
1. Load last known good state from checkpoint
2. Validate integrity of `SOUL.md` entries
3. Reconnect to Binance WebSocket with exponential backoff
4. Restart Ray cluster with reduced worker count (fallback mode)
5. Resume trading only after all health checks pass

---

*This document is auto-updated by the autonomous learning system. 
Manual edits should be clearly marked with [MANUAL_OVERRIDE] tags.*
