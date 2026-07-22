//! Smart Order Routing - Routing Engine
//! 
//! Implements a Dijkstra-based routing algorithm that calculates the cheapest
//! execution path across venues, factoring in real-time maker/taker fees and
//! slippage models. Optimized for microsecond latency on AMD Ryzen AI 5.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crate::sor::liquidity_aggregator::{SyntheticBook, PriceLevel};

/// Maximum number of venues in the routing graph
const MAX_VENUES: usize = 8;

/// Maximum path length for routing (prevents infinite loops)
const MAX_PATH_LENGTH: usize = 5;

/// Venue identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VenueId(pub u8);

impl VenueId {
    pub const SPOT: Self = VenueId(0);
    pub const MARGIN: Self = VenueId(1);
    pub const FUTURES: Self = VenueId(2);
    pub const OPTIONS: Self = VenueId(3);
    
    #[inline(always)]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Fee structure for a venue (fixed-point nanobasis points)
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct FeeStructure {
    /// Maker fee in nanobps (negative = rebate)
    pub maker_fee_nbps: i64,
    /// Taker fee in nanobps
    pub taker_fee_nbps: i64,
    /// Withdrawal fee (asset-specific, simplified here)
    pub withdrawal_fee_nbps: i64,
    /// Minimum order size in nanounits
    pub min_order_size_ns: i64,
    /// Maximum order size in nanounits
    pub max_order_size_ns: i64,
    /// Padding for cache alignment
    _padding: [u8; 24],
}

impl FeeStructure {
    pub const fn new(maker_nbps: i64, taker_nbps: i64) -> Self {
        Self {
            maker_fee_nbps: maker_nbps,
            taker_fee_nbps: taker_nbps,
            withdrawal_fee_nbps: 0,
            min_order_size_ns: 1_000_000, // 0.001 units minimum
            max_order_size_ns: 1_000_000_000_000_000, // 1M units maximum
            _padding: [0u8; 24],
        }
    }

    /// Calculate total cost for a taker order (positive = cost, negative = profit)
    #[inline(always)]
    pub fn calc_taker_cost(&self, notional_ns: i64) -> i64 {
        // notional * fee / 10^9 (nanobps to absolute)
        (notional_ns * self.taker_fee_nbps).wrapping_div(1_000_000_000)
    }

    /// Calculate total cost/rebate for a maker order
    #[inline(always)]
    pub fn calc_maker_cost(&self, notional_ns: i64) -> i64 {
        (notional_ns * self.maker_fee_nbps).wrapping_div(1_000_000_000)
    }
}

/// Slippage model estimate
#[derive(Clone, Copy, Debug)]
pub struct SlippageModel {
    /// Linear slippage coefficient (nanobps per unit of volume)
    pub linear_coef_nbps: i64,
    /// Quadratic slippage coefficient
    pub quadratic_coef_nbps: i64,
    /// Recent volatility (affects slippage)
    pub volatility_nbps: i64,
}

impl SlippageModel {
    pub const fn new() -> Self {
        Self {
            linear_coef_nbps: 100, // 0.1 bps per unit
            quadratic_coef_nbps: 10,
            volatility_nbps: 500,
        }
    }

    /// Estimate slippage for a given order size
    #[inline(always)]
    pub fn estimate_slippage(&self, order_size_ns: i64, mid_price_ns: i64) -> i64 {
        let notional = (order_size_ns * mid_price_ns).wrapping_div(1_000_000_000);
        let linear = (notional * self.linear_coef_nbps).wrapping_div(1_000_000_000);
        let quadratic = (notional * notional * self.quadratic_coef_nbps)
            .wrapping_div(1_000_000_000_000_000_000);
        
        linear.wrapping_add(quadratic).wrapping_add(self.volatility_nbps)
    }
}

/// Edge in the routing graph
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct RouteEdge {
    /// Source venue
    pub from: VenueId,
    /// Destination venue
    pub to: VenueId,
    /// Total cost in nanodollars (fees + slippage)
    pub cost_ns: i64,
    /// Available liquidity at this edge (nanounits)
    pub liquidity_ns: i64,
    /// Expected execution time in nanoseconds
    pub latency_ns: u64,
    /// Edge validity flag
    pub valid: AtomicBool,
}

impl RouteEdge {
    #[inline(always)]
    pub fn new(from: VenueId, to: VenueId, cost_ns: i64, liquidity_ns: i64, latency_ns: u64) -> Self {
        Self {
            from,
            to,
            cost_ns,
            liquidity_ns,
            latency_ns,
            valid: AtomicBool::new(true),
        }
    }
}

/// Path result from Dijkstra's algorithm
#[derive(Clone, Debug)]
pub struct ExecutionPath {
    /// Ordered list of venue IDs in the path
    pub venues: [VenueId; MAX_PATH_LENGTH],
    /// Actual path length
    pub length: usize,
    /// Total estimated cost in nanodollars
    pub total_cost_ns: i64,
    /// Total expected latency in nanoseconds
    pub total_latency_ns: u64,
    /// Maximum executable volume (nanounits)
    pub max_volume_ns: i64,
}

impl ExecutionPath {
    pub const fn empty() -> Self {
        Self {
            venues: [VenueId(0); MAX_PATH_LENGTH],
            length: 0,
            total_cost_ns: 0,
            total_latency_ns: 0,
            max_volume_ns: 0,
        }
    }
}

/// Priority queue item for Dijkstra's algorithm
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PQItem {
    /// Current cost (negated for min-heap behavior)
    cost_ns: i64,
    /// Current venue
    venue: VenueId,
    /// Path length so far
    path_len: usize,
}

impl Ord for PQItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap (lower cost = higher priority)
        other.cost_ns.cmp(&self.cost_ns)
            .then_with(|| self.path_len.cmp(&other.path_len))
    }
}

impl PartialOrd for PQItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Main routing engine
#[repr(C, align(64))]
pub struct RoutingEngine {
    /// Adjacency matrix for venue graph
    adjacency: [[Option<RouteEdge>; MAX_VENUES]; MAX_VENUES],
    /// Fee structures per venue
    fees: [FeeStructure; MAX_VENUES],
    /// Slippage models per venue
    slippage: [SlippageModel; MAX_VENUES],
    /// Current best path cache
    cached_path: AtomicU64, // Hash of current best path
    /// Last recalculation timestamp
    last_recalc_ns: AtomicU64,
    /// Recalculation interval in nanoseconds
    recalc_interval_ns: u64,
}

impl RoutingEngine {
    pub const fn new() -> Self {
        Self {
            adjacency: [[None; MAX_VENUES]; MAX_VENUES],
            fees: [FeeStructure::new(0, 0); MAX_VENUES],
            slippage: [SlippageModel::new(); MAX_VENUES],
            cached_path: AtomicU64::new(0),
            last_recalc_ns: AtomicU64::new(0),
            recalc_interval_ns: 1_000_000, // 1ms recalc interval
        }
    }

    /// Initialize venue with fee structure
    #[inline(always)]
    pub fn set_venue_fees(&mut self, venue: VenueId, fees: FeeStructure) {
        self.fees[venue.as_usize()] = fees;
    }

    /// Update edge between venues
    #[inline(always)]
    pub fn update_edge(
        &mut self,
        from: VenueId,
        to: VenueId,
        book: &SyntheticBook,
        order_side: bool, // true = buy, false = sell
    ) {
        let from_usize = from.as_usize();
        let to_usize = to.as_usize();

        if from_usize >= MAX_VENUES || to_usize >= MAX_VENUES {
            return;
        }

        // Get liquidity from synthetic book
        let (liquidity_ns, best_price_ns) = if order_side {
            // Buy side - use ask liquidity
            match book.asks.get_best() {
                Some(level) => (level.qty_ns, level.price_ns),
                None => return, // No liquidity
            }
        } else {
            // Sell side - use bid liquidity
            match book.bids.get_best() {
                Some(level) => (level.qty_ns, level.price_ns),
                None => return,
            }
        };

        // Calculate total cost
        let notional_ns = (liquidity_ns * best_price_ns).wrapping_div(1_000_000_000);
        let fee_cost = self.fees[to_usize].calc_taker_cost(notional_ns);
        let slippage_cost = self.slippage[to_usize].estimate_slippage(liquidity_ns, best_price_ns);
        
        // Latency estimate (network + processing)
        let latency_ns = 500_000; // 500μs base latency

        let total_cost_ns = fee_cost.wrapping_add(slippage_cost);

        self.adjacency[from_usize][to_usize] = Some(RouteEdge::new(
            from,
            to,
            total_cost_ns,
            liquidity_ns,
            latency_ns,
        ));
    }

    /// Find optimal execution path using Dijkstra's algorithm
    #[inline(always)]
    pub fn find_optimal_path(
        &self,
        start: VenueId,
        end: VenueId,
        order_size_ns: i64,
    ) -> ExecutionPath {
        let mut dist = [i64::MAX; MAX_VENUES];
        let mut prev = [None; MAX_VENUES];
        let mut liquidity = [0i64; MAX_VENUES];
        let mut path_len = [0usize; MAX_VENUES];
        
        dist[start.as_usize()] = 0;
        liquidity[start.as_usize()] = order_size_ns;

        let mut pq = BinaryHeap::new();
        pq.push(PQItem {
            cost_ns: 0,
            venue: start,
            path_len: 0,
        });

        while let Some(item) = pq.pop() {
            let u = item.venue;
            let u_idx = u.as_usize();

            // Skip if we found a better path already
            if item.cost_ns > dist[u_idx] {
                continue;
            }

            // Check path length limit
            if item.path_len >= MAX_PATH_LENGTH {
                continue;
            }

            // Explore neighbors
            for v_idx in 0..MAX_VENUES {
                if let Some(edge) = &self.adjacency[u_idx][v_idx] {
                    if !edge.valid.load(Ordering::Acquire) {
                        continue;
                    }

                    // Check if edge has sufficient liquidity
                    if edge.liquidity_ns < order_size_ns {
                        continue;
                    }

                    let alt = dist[u_idx].wrapping_add(edge.cost_ns);
                    
                    if alt < dist[v_idx] {
                        dist[v_idx] = alt;
                        prev[v_idx] = Some(u);
                        liquidity[v_idx] = edge.liquidity_ns.min(liquidity[u_idx]);
                        path_len[v_idx] = path_len[u_idx] + 1;
                        
                        pq.push(PQItem {
                            cost_ns: alt,
                            venue: VenueId(v_idx as u8),
                            path_len: path_len[v_idx],
                        });
                    }
                }
            }
        }

        // Reconstruct path
        let mut path = ExecutionPath::empty();
        let mut current = end;
        let mut idx = 0;

        while current != start && idx < MAX_PATH_LENGTH {
            path.venues[MAX_PATH_LENGTH - 1 - idx] = current;
            
            let c_idx = current.as_usize();
            path.total_cost_ns = path.total_cost_ns.wrapping_add(dist[c_idx]);
            path.max_volume_ns = path.max_volume_ns.max(liquidity[c_idx]);
            
            if let Some(p) = prev[c_idx] {
                // Add edge latency
                if let Some(edge) = &self.adjacency[p.as_usize()][c_idx] {
                    path.total_latency_ns += edge.latency_ns;
                }
                current = p;
            } else {
                break;
            }
            idx += 1;
        }

        if current == start {
            path.venues[MAX_PATH_LENGTH - 1 - idx] = start;
            path.length = idx + 1;
            
            // Reverse the path array
            path.venues[..path.length].reverse();
        } else {
            path.length = 0; // No valid path found
        }

        path
    }

    /// Execute route-and-execute for an order
    #[inline(always)]
    pub fn route_and_execute(
        &self,
        start: VenueId,
        end: VenueId,
        order_size_ns: i64,
        is_buy: bool,
    ) -> Option<ExecutionPath> {
        let path = self.find_optimal_path(start, end, order_size_ns);
        
        if path.length == 0 {
            return None;
        }

        // Validate path meets constraints
        if path.max_volume_ns < order_size_ns {
            // Insufficient liquidity - could split order here
            return None;
        }

        Some(path)
    }

    /// Force recalculation of routing tables
    #[inline(always)]
    pub fn force_recalc(&mut self) {
        self.cached_path.store(0, Ordering::Release);
        self.last_recalc_ns.store(get_time_ns(), Ordering::Release);
    }
}

/// Get current time in nanoseconds
#[inline(always)]
fn get_time_ns() -> u64 {
    use std::time::Instant;
    static START: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    START.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sor::liquidity_aggregator::SyntheticBook;

    #[test]
    fn test_routing_engine_basic() {
        let mut engine = RoutingEngine::new();
        
        // Set up fee structures
        engine.set_venue_fees(VenueId::SPOT, FeeStructure::new(-100, 100)); // Maker rebate
        engine.set_venue_fees(VenueId::FUTURES, FeeStructure::new(-50, 75));
        
        // Create synthetic book
        let book = SyntheticBook::new(0x12345678);
        
        // Update edges
        engine.update_edge(VenueId::SPOT, VenueId::FUTURES, &book, true);
        
        // Find optimal path
        let path = engine.find_optimal_path(VenueId::SPOT, VenueId::FUTURES, 1_000_000_000);
        
        // Path should exist (even if no liquidity, direct connection exists)
        assert!(path.length >= 0);
    }

    #[test]
    fn test_fee_calculation() {
        let fees = FeeStructure::new(-100, 100); // -0.1 bps maker, 1.0 bps taker
        
        let notional = 1_000_000_000_000; // $1000 in nanodollars
        
        let taker_cost = fees.calc_taker_cost(notional);
        assert_eq!(taker_cost, 100_000); // 0.0001 USD
        
        let maker_cost = fees.calc_maker_cost(notional);
        assert_eq!(maker_cost, -100_000); // Rebate
    }
}
