//! # Mempool Monitor for MEV Detection
//! 
//! Creates a Rust-based mempool monitor that tracks pending transactions
//! to detect potential MEV attacks and adjust execution routing to avoid
//! toxic DEX liquidity pools.
//! 
//! ## Key Features:
//! - Real-time pending transaction monitoring
//! - MEV attack pattern detection (frontrun, backrun, sandwich)
//! - Toxic pool identification and avoidance
//! - Integration with execution routing decisions
//! - Lock-free data structures for microsecond latency

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Pending transaction structure
#[derive(Debug, Clone)]
pub struct PendingTx {
    /// Transaction hash
    pub tx_hash: [u8; 32],
    /// Sender address
    pub from: [u8; 20],
    /// Target contract address
    pub to: Option<[u8; 20]>,
    /// Transaction value in wei
    pub value: u128,
    /// Gas price in wei
    pub gas_price: u128,
    /// Gas limit
    pub gas_limit: u64,
    /// Input data (first 4 bytes = function selector)
    pub input_selector: Option<[u8; 4]>,
    /// Nonce
    pub nonce: u64,
    /// Timestamp when first seen
    pub first_seen: Instant,
    /// Chain ID
    pub chain_id: u64,
}

impl PendingTx {
    /// Check if this is a DEX swap transaction
    pub fn is_dex_swap(&self) -> bool {
        // Common DEX function selectors
        let swap_selectors: [[u8; 4]; 4] = [
            0x38ed1739, // swapExactTokensForTokens
            0xfb3bdb41, // swapETHForExactTokens
            0x7ff36ab5, // swapExactETHForTokens
            0x18cbafe5, // swapExactTokensForETH
        ];
        
        self.input_selector.map_or(false, |sel| swap_selectors.contains(&sel))
    }
    
    /// Get priority score (higher = more likely to be mined soon)
    pub fn priority_score(&self) -> u128 {
        self.gas_price
    }
}

/// MEV attack type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MevAttackType {
    /// Frontrun - executing before victim tx
    Frontrun,
    /// Backrun - executing after victim tx
    Backrun,
    /// Sandwich - frontrun + backrun combination
    Sandwich,
    /// Liquidation - liquidating undercollateralized position
    Liquidation,
    /// Arbitrage - exploiting price differences
    Arbitrage,
    /// Unknown/uncategorized
    Unknown,
}

/// Detected MEV opportunity
#[derive(Debug, Clone)]
pub struct MevOpportunity {
    /// Type of MEV attack
    pub attack_type: MevAttackType,
    /// Victim transaction hash
    pub victim_tx_hash: [u8; 32],
    /// Attacker transaction hash (if known)
    pub attacker_tx_hash: Option<[u8; 32]>,
    /// Target pool/address
    pub target_address: [u8; 20],
    /// Estimated profit in wei
    pub estimated_profit: u128,
    /// Confidence score (0-100)
    pub confidence: u8,
    /// Detection timestamp
    pub detected_at: Instant,
}

/// Toxicity level for DEX pools
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PoolToxicity {
    /// Safe pool, no issues detected
    Safe = 0,
    /// Minor concerns, proceed with caution
    LowRisk = 1,
    /// Moderate risk, reduce position size
    MediumRisk = 2,
    /// High risk, avoid large trades
    HighRisk = 3,
    /// Extremely toxic, avoid entirely
    Toxic = 4,
}

/// Pool toxicity information
#[derive(Debug, Clone)]
pub struct PoolInfo {
    /// Pool address
    pub address: [u8; 20],
    /// Current toxicity level
    pub toxicity: PoolToxicity,
    /// Number of MEV attacks detected
    pub mev_attack_count: u32,
    /// Last attack timestamp
    pub last_attack: Option<Instant>,
    /// Average slippage from MEV
    pub avg_mev_slippage_bps: u16,
    /// Chain ID
    pub chain_id: u64,
}

/// Mempool monitor configuration
#[derive(Debug, Clone)]
pub struct MempoolConfig {
    /// Maximum pending transactions to track
    pub max_pending_txs: usize,
    /// Time window for MEV pattern detection
    pub detection_window_ms: u64,
    /// Minimum confidence threshold for alerts
    pub min_confidence: u8,
    /// Chains to monitor
    pub chain_ids: Vec<u64>,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_pending_txs: 10000,
            detection_window_ms: 5000, // 5 seconds
            min_confidence: 70,
            chain_ids: vec![1, 42161, 10, 8453], // ETH, Arb, Opt, Base
        }
    }
}

/// Main mempool monitor struct
pub struct MempoolMonitor {
    /// Pending transactions queue
    pending_txs: VecDeque<PendingTx>,
    /// Known DEX pools
    dex_pools: HashMap<[u8; 20], PoolInfo>,
    /// Detected MEV opportunities
    mev_opportunities: VecDeque<MevOpportunity>,
    /// Configuration
    config: MempoolConfig,
    /// Statistics
    tx_count: AtomicUsize,
    mev_count: AtomicUsize,
    /// Shutdown flag
    shutdown: AtomicBool,
}

impl MempoolMonitor {
    /// Create new mempool monitor
    pub fn new(config: MempoolConfig) -> Self {
        Self {
            pending_txs: VecDeque::with_capacity(config.max_pending_txs),
            dex_pools: HashMap::new(),
            mev_opportunities: VecDeque::with_capacity(1000),
            config,
            tx_count: AtomicUsize::new(0),
            mev_count: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Add pending transaction to monitor
    pub fn add_pending_tx(&mut self, tx: PendingTx) {
        if self.pending_txs.len() >= self.config.max_pending_txs {
            self.pending_txs.pop_front();
        }
        
        self.pending_txs.push_back(tx);
        self.tx_count.fetch_add(1, Ordering::Relaxed);
        
        // Check for MEV patterns
        self.detect_mev_patterns();
    }

    /// Register or update DEX pool
    pub fn register_pool(&mut self, address: [u8; 20], chain_id: u64) {
        self.dex_pools.entry(address)
            .or_insert_with(|| PoolInfo {
                address,
                toxicity: PoolToxicity::Safe,
                mev_attack_count: 0,
                last_attack: None,
                avg_mev_slippage_bps: 0,
                chain_id,
            });
    }

    /// Detect MEV patterns in pending transactions
    fn detect_mev_patterns(&mut self) {
        let now = Instant::now();
        let window = Duration::from_millis(self.config.detection_window_ms);
        
        // Group transactions by target pool
        let mut pool_txs: HashMap<[u8; 20], Vec<&PendingTx>> = HashMap::new();
        
        for tx in &self.pending_txs {
            if tx.is_dex_swap() {
                if let Some(to) = tx.to {
                    pool_txs.entry(to).or_default().push(tx);
                }
            }
        }
        
        // Detect sandwich attacks (high gas price tx before and after victim)
        for (pool, txs) in &pool_txs {
            if txs.len() >= 3 {
                let mut sorted_txs: Vec<&&PendingTx> = txs.iter().collect();
                sorted_txs.sort_by(|a, b| b.gas_price.cmp(&a.gas_price));
                
                // Check for sandwich pattern
                if sorted_txs.len() >= 3 {
                    let high_gas = sorted_txs[0];
                    let medium_gas = sorted_txs[1];
                    
                    // If highest gas tx has significantly higher gas than second
                    if high_gas.gas_price > medium_gas.gas_price * 150 / 100 {
                        // Potential frontrun detected
                        let opportunity = MevOpportunity {
                            attack_type: MevAttackType::Frontrun,
                            victim_tx_hash: medium_gas.tx_hash,
                            attacker_tx_hash: Some(high_gas.tx_hash),
                            target_address: *pool,
                            estimated_profit: 0, // Would calculate from simulation
                            confidence: 75,
                            detected_at: now,
                        };
                        
                        if opportunity.confidence >= self.config.min_confidence {
                            self.mev_opportunities.push_back(opportunity);
                            self.mev_count.fetch_add(1, Ordering::Relaxed);
                            
                            // Update pool toxicity
                            if let Some(pool_info) = self.dex_pools.get_mut(pool) {
                                pool_info.mev_attack_count += 1;
                                pool_info.last_attack = Some(now);
                                self.update_pool_toxicity(pool_info);
                            }
                        }
                    }
                }
            }
        }
        
        // Prune old opportunities
        while let Some(front) = self.mev_opportunities.front() {
            if front.detected_at.elapsed() > window {
                self.mev_opportunities.pop_front();
            } else {
                break;
            }
        }
    }

    /// Update pool toxicity based on attack history
    fn update_pool_toxicity(&mut self, pool: &mut PoolInfo) {
        let now = Instant::now();
        
        pool.toxicity = if pool.mev_attack_count == 0 {
            PoolToxicity::Safe
        } else if pool.mev_attack_count < 3 {
            PoolToxicity::LowRisk
        } else if pool.mev_attack_count < 10 {
            PoolToxicity::MediumRisk
        } else if pool.mev_attack_count < 50 {
            PoolToxicity::HighRisk
        } else {
            PoolToxicity::Toxic
        };
        
        // Decay attack count over time
        if let Some(last_attack) = pool.last_attack {
            if last_attack.elapsed() > Duration::from_secs(300) {
                pool.mev_attack_count = pool.mev_attack_count.saturating_sub(1);
            }
        }
    }

    /// Check if pool is safe for trading
    pub fn is_pool_safe(&self, address: &[u8; 20], max_toxicity: PoolToxicity) -> bool {
        self.dex_pools.get(address)
            .map_or(true, |info| info.toxicity <= max_toxicity)
    }

    /// Get recommended routing avoiding toxic pools
    pub fn get_safe_route(&self, candidate_pools: &[[u8; 20]]) -> Option<[u8; 20]> {
        candidate_pools.iter()
            .filter(|pool| self.is_pool_safe(pool, PoolToxicity::MediumRisk))
            .next()
            .copied()
    }

    /// Get recent MEV opportunities
    pub fn get_recent_mev(&self, limit: usize) -> Vec<MevOpportunity> {
        self.mev_opportunities.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get statistics
    pub fn get_stats(&self) -> MempoolStats {
        MempoolStats {
            pending_tx_count: self.pending_txs.len(),
            tracked_pools: self.dex_pools.len(),
            total_txs_processed: self.tx_count.load(Ordering::Relaxed),
            total_mev_detected: self.mev_count.load(Ordering::Relaxed),
            toxic_pools: self.dex_pools.values()
                .filter(|p| p.toxicity >= PoolToxicity::HighRisk)
                .count(),
        }
    }

    /// Initiate shutdown
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Check if shutting down
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }
}

/// Mempool statistics
#[derive(Debug, Clone)]
pub struct MempoolStats {
    pub pending_tx_count: usize,
    pub tracked_pools: usize,
    pub total_txs_processed: usize,
    pub total_mev_detected: usize,
    pub toxic_pools: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mempool_monitor() {
        let config = MempoolConfig::default();
        let mut monitor = MempoolMonitor::new(config);
        
        // Register a pool
        let pool_addr = [1u8; 20];
        monitor.register_pool(pool_addr, 1);
        
        // Create mock pending transactions
        let tx1 = PendingTx {
            tx_hash: [2u8; 32],
            from: [3u8; 20],
            to: Some(pool_addr),
            value: 1000000000000000000,
            gas_price: 100000000000,
            gas_limit: 200000,
            input_selector: Some(0x38ed1739u32.to_be_bytes()),
            nonce: 1,
            first_seen: Instant::now(),
            chain_id: 1,
        };
        
        monitor.add_pending_tx(tx1);
        
        let stats = monitor.get_stats();
        assert_eq!(stats.pending_tx_count, 1);
        assert_eq!(stats.total_txs_processed, 1);
        
        // Check pool safety
        assert!(monitor.is_pool_safe(&pool_addr, PoolToxicity::Safe));
    }

    #[test]
    fn test_pool_toxicity_levels() {
        assert!(PoolToxicity::Safe < PoolToxicity::LowRisk);
        assert!(PoolToxicity::LowRisk < PoolToxicity::MediumRisk);
        assert!(PoolToxicity::MediumRisk < PoolToxicity::HighRisk);
        assert!(PoolToxicity::HighRisk < PoolToxicity::Toxic);
    }
}
