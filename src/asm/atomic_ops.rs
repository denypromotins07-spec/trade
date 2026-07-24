//! src/asm/atomic_ops.rs
//!
//! Stage 51: Custom Lock-Free Atomic Operations for AMD Zen Architecture
//!
//! Implements 128-bit Compare-And-Swap using `lock cmpxchg16b` instruction
//! for atomic state transitions in the high-frequency trading engine.
//! Optimized for AMD Zen 4/Zen 5 with strict memory ordering guarantees.
//!
//! Critical for lock-free order book updates and position tracking.

#![feature(asm_sym)]
#![feature(asm_experimental_arch)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::mem;

/// 128-bit atomic value for lock-free operations
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct AtomicU128 {
    low: AtomicU64,
    high: AtomicU64,
}

/// 128-bit value for CAS operations
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U128 {
    pub low: u64,
    pub high: u64,
}

impl U128 {
    #[inline(always)]
    pub const fn new(high: u64, low: u64) -> Self {
        Self { low, high }
    }

    #[inline(always)]
    pub const fn from_u128(val: u128) -> Self {
        Self {
            low: val as u64,
            high: (val >> 64) as u64,
        }
    }

    #[inline(always)]
    pub const fn to_u128(&self) -> u128 {
        ((self.high as u128) << 64) | (self.low as u128)
    }
}

impl AtomicU128 {
    /// Create a new 128-bit atomic value
    #[inline(always)]
    pub const fn new(val: U128) -> Self {
        Self {
            low: AtomicU64::new(val.low),
            high: AtomicU64::new(val.high),
        }
    }

    /// Load the current value with specified ordering
    ///
    /// # Safety
    /// This operation is not truly atomic on x86 without CMPXCHG16B.
    /// Use only for reading approximate values where torn reads are acceptable.
    #[inline(always)]
    pub fn load_approx(&self, order: Ordering) -> U128 {
        let low = self.low.load(order);
        let high = self.high.load(order);
        U128 { low, high }
    }

    /// Atomic Compare-And-Swap for 128-bit values using CMPXCHG16B
    ///
    /// This is the only way to achieve true atomicity for 128-bit operations
    /// on x86_64. Requires CPU support for CMPXCHG16B (checked at runtime).
    ///
    /// # Arguments
    /// * `current` - Expected current value
    /// * `new_val` - Value to store if current matches
    /// * `success` - Memory ordering for successful CAS
    /// * `failure` - Memory ordering for failed CAS
    ///
    /// # Returns
    /// Ok(U128) with actual value if successful, Err(U128) with actual value if failed
    ///
    /// # Safety
    /// - Requires CMPXCHG16B CPU feature
    /// - Value must be 16-byte aligned
    #[inline(always)]
    pub unsafe fn compare_exchange_128(
        &self,
        current: U128,
        new_val: U128,
        success: Ordering,
        failure: Ordering,
    ) -> Result<U128, U128> {
        // Runtime check for CMPXCHG16B support
        if !is_x86_feature_detected!("cmpxchg16b") {
            return Err(self.load_approx(Ordering::SeqCst));
        }

        let mut expected_low = current.low;
        let mut expected_high = current.high;
        let desired_low = new_val.low;
        let desired_high = new_val.high;
        let mut result_low: u64;
        let mut result_high: u64;
        let mut success_flag: u8;

        // Determine memory ordering constraints
        let (acquire, release) = match (success, failure) {
            (Ordering::SeqCst, _) | (_, Ordering::SeqCst) => ("aq", "rl"),
            (Ordering::Acquire, _) | (_, Ordering::Acquire) => ("aq", ""),
            (Ordering::Release, _) | (_, Ordering::Release) => ("", "rl"),
            _ => ("", ""),
        };

        // Inline assembly for CMPXCHG16B
        // This is the critical path for lock-free state transitions
        std::arch::asm!(
            "lock cmpxchg16b [{ptr}]",
            "sete {success}",
            ptr = in(reg) &self.low as *const AtomicU64 as *mut u8,
            inout("rax") expected_low => result_low,
            inout("rdx") expected_high => result_high,
            inout("rbx") desired_low => _,
            inout("rcx") desired_high => _,
            outlate("cc") success_flag,
            options(preserves_flags, nostack)
        );

        let actual = U128 {
            low: result_low,
            high: result_high,
        };

        if success_flag == 1 {
            Ok(actual)
        } else {
            Err(actual)
        }
    }

    /// Atomic exchange (swap) operation for 128-bit values
    ///
    /// Uses a loop with CAS to achieve atomic swap semantics.
    #[inline(always)]
    pub fn swap(&self, val: U128, order: Ordering) -> U128 {
        let mut current = self.load_approx(order);

        loop {
            match unsafe {
                self.compare_exchange_128(current, val, order, order)
            } {
                Ok(_) => return current,
                Err(actual) => current = actual,
            }
        }
    }

    /// Store a new value using CAS loop
    #[inline(always)]
    pub fn store(&self, val: U128, order: Ordering) {
        let mut current = self.load_approx(order);

        loop {
            match unsafe {
                self.compare_exchange_128(current, val, order, order)
            } {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Lock-free counter using 128-bit atomics for high-throughput scenarios
///
/// Stores both count and timestamp in a single atomic operation to prevent
/// race conditions in order matching and trade execution.
#[repr(C, align(16))]
pub struct LockFreeCounter128 {
    data: AtomicU128,
}

impl LockFreeCounter128 {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: AtomicU128::new(U128::new(0, 0)),
        }
    }

    /// Increment counter and return previous value atomically
    ///
    /// High 64 bits: counter value
    /// Low 64 bits: monotonic timestamp (TSC cycles)
    #[inline(always)]
    pub unsafe fn increment_with_timestamp(&self) -> U128 {
        let mut current = self.data.load_approx(Ordering::Relaxed);

        loop {
            // Get current TSC for timestamp
            let tsc = get_tsc();
            let new_val = U128::new(current.high + 1, tsc);

            match self.data.compare_exchange_128(current, new_val, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return current,
                Err(actual) => current = actual,
            }
        }
    }

    /// Get current counter value
    #[inline(always)]
    pub fn get(&self) -> U128 {
        self.data.load_approx(Ordering::Acquire)
    }
}

impl Default for LockFreeCounter128 {
    fn default() -> Self {
        Self::new()
    }
}

/// Read Time Stamp Counter using RDTSCP instruction
///
/// Provides serializing read that prevents reordering with surrounding instructions.
#[inline(always)]
fn get_tsc() -> u64 {
    let lo: u32;
    let hi: u32;

    unsafe {
        std::arch::asm!(
            "rdtscp",
            out("rax") lo,
            out("rdx") hi,
            out("rcx") _, // IA32_TSC_AUX
            options(nomem, nostack, preserves_flags)
        );
    }

    ((hi as u64) << 32) | (lo as u64)
}

/// Order book state stored atomically for lock-free updates
///
/// Encodes multiple fields in a single 128-bit word:
/// - Bits 0-31: Best bid price (scaled)
/// - Bits 32-63: Best ask price (scaled)
/// - Bits 64-95: Last trade volume
/// - Bits 96-127: Sequence number
#[repr(C, align(16))]
pub struct AtomicOrderBookState {
    state: AtomicU128,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct OrderBookState {
    pub best_bid: u32,      // Price scaled by 10000
    pub best_ask: u32,      // Price scaled by 10000
    pub last_volume: u32,   // Volume scaled by 1000000
    pub sequence: u32,      // Monotonic sequence number
}

impl OrderBookState {
    #[inline(always)]
    pub fn to_u128(&self) -> U128 {
        U128::new(
            ((self.last_volume as u64) << 32) | (self.sequence as u64),
            ((self.best_ask as u64) << 32) | (self.best_bid as u64),
        )
    }

    #[inline(always)]
    pub fn from_u128(val: U128) -> Self {
        let low = val.low;
        let high = val.high;

        Self {
            best_bid: (low & 0xFFFFFFFF) as u32,
            best_ask: ((low >> 32) & 0xFFFFFFFF) as u32,
            last_volume: (high & 0xFFFFFFFF) as u32,
            sequence: ((high >> 32) & 0xFFFFFFFF) as u32,
        }
    }
}

impl AtomicOrderBookState {
    #[inline(always)]
    pub const fn new(state: OrderBookState) -> Self {
        Self {
            state: AtomicU128::new(state.to_u128()),
        }
    }

    /// Atomically update order book state if sequence number is greater
    ///
    /// Prevents stale updates from overwriting newer state.
    #[inline(always)]
    pub unsafe fn update_if_newer(&self, new_state: OrderBookState) -> Result<OrderBookState, OrderBookState> {
        let mut current_u128 = self.state.load_approx(Ordering::Relaxed);
        let mut current = OrderBookState::from_u128(current_u128);

        loop {
            if new_state.sequence <= current.sequence {
                return Err(current);
            }

            let new_u128 = new_state.to_u128();

            match self.state.compare_exchange_128(
                current_u128,
                new_u128,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => Ok(new_state),
                Err(actual_u128) => {
                    current_u128 = actual_u128;
                    current = OrderBookState::from_u128(actual_u128);
                }
            }
        }
    }

    /// Get current state
    #[inline(always)]
    pub fn get(&self) -> OrderBookState {
        OrderBookState::from_u128(self.state.load_approx(Ordering::Acquire))
    }
}

/// Runtime CPU feature detection for CMPXCHG16B
pub struct AtomicCapabilities {
    pub has_cmpxchg16b: bool,
}

impl AtomicCapabilities {
    pub fn detect() -> Self {
        Self {
            has_cmpxchg16b: is_x86_feature_detected!("cmpxchg16b"),
        }
    }

    /// Panics if CMPXCHG16B is not available (required for 128-bit atomics)
    pub fn require_cmpxchg16b() {
        if !is_x86_feature_detected!("cmpxchg16b") {
            panic!("CMPXCHG16B instruction required for 128-bit atomic operations");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u128_conversion() {
        let original = 0x123456789ABCDEF0123456789ABCDEF0u128;
        let u128_struct = U128::from_u128(original);
        let converted = u128_struct.to_u128();
        assert_eq!(original, converted);
    }

    #[test]
    fn test_atomic_counter() {
        let counter = LockFreeCounter128::new();
        
        unsafe {
            let prev1 = counter.increment_with_timestamp();
            let prev2 = counter.increment_with_timestamp();
            
            assert_eq!(prev1.high, 0);
            assert_eq!(prev2.high, 1);
        }
    }

    #[test]
    fn test_order_book_state_encoding() {
        let state = OrderBookState {
            best_bid: 500000,  // $50.0000
            best_ask: 500100,  // $50.0100
            last_volume: 1000000, // 1.0
            sequence: 42,
        };

        let encoded = state.to_u128();
        let decoded = OrderBookState::from_u128(encoded);

        assert_eq!(state.best_bid, decoded.best_bid);
        assert_eq!(state.best_ask, decoded.best_ask);
        assert_eq!(state.last_volume, decoded.last_volume);
        assert_eq!(state.sequence, decoded.sequence);
    }

    #[test]
    fn test_cpu_feature_detection() {
        let caps = AtomicCapabilities::detect();
        println!("CMPXCHG16B supported: {}", caps.has_cmpxchg16b);
        
        // Most modern x86_64 CPUs support this
        // Test will skip gracefully if not available
        if caps.has_cmpxchg16b {
            AtomicCapabilities::require_cmpxchg16b(); // Should not panic
        }
    }
}
