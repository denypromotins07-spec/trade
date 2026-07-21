//! Cross-Chain & DEX Aggregation - Chapter 3
//! File 8: mempool.rs
//! 
//! Creates a Rust-based mempool monitor that tracks pending transactions
//! to detect potential MEV attacks and adjust execution routing to avoid
//! toxic DEX liquidity pools. Optimized for microsecond latency.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use dashmap::DashMap;

/// Pending transaction in mempool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTransaction {
    pub tx_hash: String,
    pub from_address: String,
    pub to_address: String,
    pub value_wei: u128,
    pub gas_price_gwei: u64,
    pub gas_limit: u64,
    pub nonce: u64,
    pub input_data: Vec<u8>,
    pub timestamp_ns: u64,
    pub chain_id: u64,
}

/// Detected MEV opportunity/threat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MEVOpportunity {
    pub mev_type: MEVType,
    pub confidence: f64,
    pub expected_profit_wei: u128,
    pub target_tx_hash: String,
    pub recommended_action: String,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MEVType {
    /// Front-running opportunity
    FrontRun,
    /// Back-running opportunity  
    BackRun,
    /// Sandwich attack detection
    SandwichAttack,
    /// Arbitrage opportunity
    Arbitrage,
    /// Liquidation opportunity
    Liquidation,
    /// Toxic pool warning
    ToxicPool,
}

/// Mempool transaction analyzer
pub struct MempoolMonitor {
    /// Pending transactions by hash
    pending_txs: DashMap<String, PendingTransaction>,
    /// Transactions by target address (for DEX contracts)
    txs_by_target: DashMap<String, Vec<String>>,
    /// Known DEX router addresses
    dex_routers: DashMap<String, DexInfo>,
    /// Detected MEV opportunities queue
    mev_queue: crossbeam_queue::SegQueue<MEVOpportunity>,
    /// Configuration
    min_gas_price_gwei: u64,
    sandwich_detection_threshold: f64,
    /// Statistics
    txs_processed: AtomicU64,
    mev_detected: AtomicU64,
    /// Active monitoring flag
    is_monitoring: AtomicBool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexInfo {
    pub name: String,
    pub chain_id: u64,
    pub is_toxic: bool,
    toxicity_score: f64,
}

impl MempoolMonitor {
    /// Create new mempool monitor
    pub fn new(min_gas_price_gwei: u64, sandwich_threshold: f64) -> Self {
        let mut monitor = Self {
            pending_txs: DashMap::with_capacity(10000),
            txs_by_target: DashMap::with_capacity(1000),
            dex_routers: DashMap::new(),
            mev_queue: crossbeam_queue::SegQueue::new(),
            min_gas_price_gwei,
            sandwich_detection_threshold: sandwich_threshold,
            txs_processed: AtomicU64::new(0),
            mev_detected: AtomicU64::new(0),
            is_monitoring: AtomicBool::new(true),
        };
        
        // Register known DEX routers
        monitor.register_known_dex_routers();
        monitor
    }

    /// Register known DEX router addresses
    fn register_known_dex_routers(&mut self) {
        // Uniswap V2/V3 routers
        self.dex_routers.insert(
            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D".to_string(),
            DexInfo { name: "UniswapV2".to_string(), chain_id: 1, is_toxic: false, toxicity_score: 0.0 },
        );
        self.dex_routers.insert(
            "0xE592427A0AEce92De3Edee1F18E0157C05861564".to_string(),
            DexInfo { name: "UniswapV3".to_string(), chain_id: 1, is_toxic: false, toxicity_score: 0.0 },
        );
        // PancakeSwap
        self.dex_routers.insert(
            "0x10ED43C718714eb63d5aA57B78B54704E256024E".to_string(),
            DexInfo { name: "PancakeSwap".to_string(), chain_id: 56, is_toxic: false, toxicity_score: 0.0 },
        );
    }

    /// Process a new pending transaction from mempool
    pub fn process_pending_tx(&self, tx: PendingTransaction) {
        if !self.is_monitoring.load(Ordering::Relaxed) {
            return;
        }

        // Filter by minimum gas price (potential MEV transactions have high gas)
        if tx.gas_price_gwei < self.min_gas_price_gwei {
            return;
        }

        let tx_hash = tx.tx_hash.clone();
        let to_addr = tx.to_address.clone();

        // Store pending transaction
        self.pending_txs.insert(tx_hash.clone(), tx);

        // Index by target address
        self.txs_by_target
            .entry(to_addr.clone())
            .or_insert_with(Vec::new)
            .push(tx_hash.clone());

        self.txs_processed.fetch_add(1, Ordering::Relaxed);

        // Check if targeting DEX router
        if let Some(dex_info) = self.dex_routers.get(&to_addr) {
            self.analyze_for_mev(&tx, &dex_info);
        }

        // Cleanup old transactions periodically
        if self.txs_processed.load(Ordering::Relaxed) % 1000 == 0 {
            self.cleanup_old_transactions();
        }
    }

    /// Analyze transaction for MEV patterns
    fn analyze_for_mev(&self, tx: &PendingTransaction, dex_info: &DexInfo) {
        let tx_hash = &tx.tx_hash;
        let input_data = &tx.input_data;

        // Decode swap function calls
        if let Some(swap_params) = self.decode_swap_input(input_data) {
            // Check for sandwich attack pattern
            self.detect_sandwich_attack(tx, &swap_params);
            
            // Check for front-running opportunity
            self.detect_frontrun_opportunity(tx, &swap_params);
            
            // Check for back-running opportunity
            self.detect_backrun_opportunity(tx, &swap_params);
        }

        // Check for toxic pool interactions
        if dex_info.is_toxic || dex_info.toxicity_score > 0.5 {
            let opportunity = MEVOpportunity {
                mev_type: MEVType::ToxicPool,
                confidence: dex_info.toxicity_score,
                expected_profit_wei: 0,
                target_tx_hash: tx_hash.clone(),
                recommended_action: format!("Avoid interaction with toxic pool via {}", dex_info.name),
                timestamp_ns: tx.timestamp_ns,
            };
            self.mev_queue.push(opportunity);
            self.mev_detected.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decode swap input data (simplified - would need full ABI decoding in production)
    fn decode_swap_input(&self, input_data: &[u8]) -> Option<SwapParams> {
        if input_data.len() < 68 {
            return None;
        }

        // Check for common swap function selectors
        let selector = &input_data[0..4];
        
        // swapExactTokensForTokens: 0x38ed1739
        // swapTokensForExactTokens: 0x8803dbee
        // exactInputSingle (V3): 0xdb3e2198
        
        if selector == [0x38, 0xed, 0x17, 0x39] || 
           selector == [0x88, 0x03, 0xdb, 0xee] {
            // Parse amount and path (simplified)
            let amount_in = u128::from_be_bytes([0; 16]); // Would parse from actual data
            
            return Some(SwapParams {
                amount_in,
                amount_out_min: 0,
                is_exact_input: selector == [0x38, 0xed, 0x17, 0x39],
            });
        }

        None
    }

    /// Detect sandwich attack patterns
    fn detect_sandwich_attack(&self, tx: &PendingTransaction, swap: &SwapParams) {
        let to_addr = &tx.to_address;
        
        // Look for similar transactions targeting same DEX
        if let Some(tx_hashes) = self.txs_by_target.get(to_addr) {
            let similar_txs: Vec<_> = tx_hashes.iter()
                .filter(|h| *h != &tx.tx_hash)
                .filter_map(|h| self.pending_txs.get(h))
                .filter(|other| {
                    // Similar gas price indicates potential sandwich
                    (other.gas_price_gwei as i64 - tx.gas_price_gwei as i64).abs() < 10
                })
                .collect();

            if similar_txs.len() >= 2 {
                // Potential sandwich detected
                let confidence = 0.7 + (similar_txs.len() as f64 * 0.05).min(0.3);
                
                let opportunity = MEVOpportunity {
                    mev_type: MEVType::SandwichAttack,
                    confidence,
                    expected_profit_wei: 0, // Would calculate based on swap size
                    target_tx_hash: tx.tx_hash.clone(),
                    recommended_action: "High probability sandwich attack - consider delaying execution".to_string(),
                    timestamp_ns: tx.timestamp_ns,
                };
                
                self.mev_queue.push(opportunity);
                self.mev_detected.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Detect front-running opportunities
    fn detect_frontrun_opportunity(&self, tx: &PendingTransaction, _swap: &SwapParams) {
        // Check if this tx has unusually high gas price (potential frontrunner)
        let avg_gas = self.calculate_average_gas_price();
        
        if tx.gas_price_gwei > (avg_gas * 2.0) as u64 {
            let opportunity = MEVOpportunity {
                mev_type: MEVType::FrontRun,
                confidence: 0.6,
                expected_profit_wei: 0,
                target_tx_hash: tx.tx_hash.clone(),
                recommended_action: "Transaction may be front-run - increase gas or use private RPC".to_string(),
                timestamp_ns: tx.timestamp_ns,
            };
            
            self.mev_queue.push(opportunity);
        }
    }

    /// Detect back-running opportunities
    fn detect_backrun_opportunity(&self, tx: &PendingTransaction, swap: &SwapParams) {
        // Large swaps create back-running arbitrage opportunities
        if swap.amount_in > 1_000_000_000_000_000_000_000u128 { // > 1000 tokens
            let opportunity = MEVOpportunity {
                mev_type: MEVType::BackRun,
                confidence: 0.5,
                expected_profit_wei: swap.amount_in / 1000, // Estimate 0.1% profit
                target_tx_hash: tx.tx_hash.clone(),
                recommended_action: "Large swap creates back-run opportunity".to_string(),
                timestamp_ns: tx.timestamp_ns,
            };
            
            self.mev_queue.push(opportunity);
        }
    }

    /// Calculate average gas price across pending transactions
    fn calculate_average_gas_price(&self) -> f64 {
        let mut total = 0u64;
        let mut count = 0u64;
        
        for entry in self.pending_txs.iter() {
            total += entry.value().gas_price_gwei;
            count += 1;
        }
        
        if count == 0 {
            return 0.0;
        }
        
        total as f64 / count as f64
    }

    /// Poll detected MEV opportunities
    pub fn poll_mev_opportunities(&self) -> Vec<MEVOpportunity> {
        let mut opportunities = Vec::new();
        while let Ok(opp) = self.mev_queue.pop() {
            opportunities.push(opp);
        }
        opportunities
    }

    /// Get pending transaction count
    pub fn get_pending_count(&self) -> usize {
        self.pending_txs.len()
    }

    /// Check if pool/router is toxic
    pub fn is_toxic_pool(&self, address: &str) -> bool {
        if let Some(dex) = self.dex_routers.get(address) {
            return dex.is_toxic || dex.toxicity_score > 0.5;
        }
        false
    }

    /// Mark a pool as toxic
    pub fn mark_pool_toxic(&self, address: &str, reason: &str) {
        self.dex_routers.entry(address.to_string())
            .or_insert_with(|| DexInfo {
                name: reason.to_string(),
                chain_id: 0,
                is_toxic: true,
                toxicity_score: 1.0,
            })
            .is_toxic = true;
    }

    /// Cleanup old processed transactions
    fn cleanup_old_transactions(&self) {
        // Remove transactions older than 30 seconds
        let cutoff_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let threshold = 30_000_000_000; // 30 seconds

        let mut to_remove = Vec::new();
        for entry in self.pending_txs.iter() {
            if cutoff_ns > entry.value().timestamp_ns + threshold {
                to_remove.push(entry.key().clone());
            }
        }

        for hash in to_remove {
            self.pending_txs.remove(&hash);
        }
    }

    /// Start/stop monitoring
    pub fn set_monitoring(&self, enabled: bool) {
        self.is_monitoring.store(enabled, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_statistics(&self) -> MempoolStats {
        MempoolStats {
            pending_transactions: self.pending_txs.len(),
            total_processed: self.txs_processed.load(Ordering::Relaxed),
            mev_opportunities_detected: self.mev_detected.load(Ordering::Relaxed),
            registered_dex_routers: self.dex_routers.len(),
            is_monitoring: self.is_monitoring.load(Ordering::Relaxed),
        }
    }
}

/// Swap parameters decoded from transaction input
#[derive(Debug, Clone)]
pub struct SwapParams {
    pub amount_in: u128,
    pub amount_out_min: u128,
    pub is_exact_input: bool,
}

/// Mempool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolStats {
    pub pending_transactions: usize,
    pub total_processed: u64,
    pub mev_opportunities_detected: u64,
    pub registered_dex_routers: usize,
    pub is_monitoring: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mempool_basic() {
        let monitor = MempoolMonitor::new(50, 0.7);
        
        // Create a test transaction
        let tx = PendingTransaction {
            tx_hash: "0xtest123".to_string(),
            from_address: "0xSender".to_string(),
            to_address: "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D".to_string(),
            value_wei: 0,
            gas_price_gwei: 100,
            gas_limit: 200000,
            nonce: 1,
            input_data: vec![0x38, 0xed, 0x17, 0x39], // swap selector
            timestamp_ns: 1000000,
            chain_id: 1,
        };
        
        monitor.process_pending_tx(tx);
        
        assert_eq!(monitor.get_pending_count(), 1);
        
        let stats = monitor.get_statistics();
        assert_eq!(stats.total_processed, 1);
    }
}
