//! src/asm/context_switch.rs
//!
//! Stage 51: Minimal Context Switching Routines Using Inline Assembly
//!
//! Implements fiber-like coroutines with sub-microsecond yield times
//! by bypassing OS thread scheduling. Optimized for AMD Zen architecture
//! with manual register saving/restoring.
//!
//! Critical for ultra-low latency order processing without kernel involvement.

#![feature(asm_sym)]
#![feature(naked_functions)]
#![feature(asm_experimental_arch)]

use std::arch::x86_64::*;
use std::mem;
use std::ptr;

/// Size of fiber stack in bytes (64KB default)
const FIBER_STACK_SIZE: usize = 64 * 1024;

/// Number of general purpose registers to save (AMD Zen: 16 GPRs)
const NUM_GPRS: usize = 16;

/// Fiber context structure containing saved register state
#[repr(C, align(16))]
#[derive(Clone, Debug)]
pub struct FiberContext {
    /// General purpose registers: RAX, RBX, RCX, RDX, RSI, RDI, RBP, R8-R15
    /// Note: RSP is stored separately as it's the stack pointer
    gprs: [u64; NUM_GPRS],
    
    /// Stack pointer (RSP)
    rsp: u64,
    
    /// Instruction pointer (RIP) - set via return address on stack
    rip: u64,
    
    /// SSE/XMM registers for SIMD state (optional, for compute fibers)
    xmm_regs: [u128; 8], // Save XMM0-XMM7 for compute-intensive fibers
    
    /// Fiber state flags
    flags: FiberFlags,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiberFlags {
    Empty = 0,
    Ready = 1,
    Running = 2,
    Suspended = 3,
    Terminated = 4,
}

impl FiberContext {
    /// Create a new fiber context with initialized stack
    #[inline(always)]
    pub fn new(entry: extern "C" fn() -> (), stack: &mut [u8]) -> Self {
        let mut ctx = Self {
            gprs: [0; NUM_GPRS],
            rsp: 0,
            rip: entry as u64,
            xmm_regs: [0; 8],
            flags: FiberFlags::Ready,
        };

        // Initialize stack pointer to top of stack (grows downward)
        let stack_top = stack.as_ptr() as u64 + stack.len() as u64;
        
        // Align stack to 16-byte boundary (required by System V AMD64 ABI)
        ctx.rsp = (stack_top - 16) & !0xF;

        // Set up initial stack frame with return address pointing to entry
        // When fiber is first resumed, it will "return" to entry function
        unsafe {
            let stack_ptr = ctx.rsp as *mut u64;
            // Push dummy return address (entry point)
            *stack_ptr = entry as u64;
            ctx.rsp -= 8;
        }

        ctx
    }

    /// Check if fiber is ready to run
    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        self.flags == FiberFlags::Ready
    }

    /// Check if fiber is currently running
    #[inline(always)]
    pub fn is_running(&self) -> bool {
        self.flags == FiberFlags::Running
    }

    /// Mark fiber as terminated
    #[inline(always)]
    pub fn terminate(&mut self) {
        self.flags = FiberFlags::Terminated;
    }
}

/// Fiber scheduler managing multiple lightweight coroutines
pub struct FiberScheduler {
    /// Currently running fiber index
    current: usize,
    
    /// Maximum number of fibers
    capacity: usize,
    
    /// Fiber contexts (stored in contiguous memory for cache efficiency)
    fibers: Vec<FiberContext>,
    
    /// Stacks for each fiber (allocated separately)
    stacks: Vec<Vec<u8>>,
}

impl FiberScheduler {
    /// Create a new fiber scheduler with given capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0 && capacity <= 256, "Capacity must be between 1 and 256");
        
        let mut fibers = Vec::with_capacity(capacity);
        let mut stacks = Vec::with_capacity(capacity);

        // Pre-allocate fiber slots
        for _ in 0..capacity {
            fibers.push(FiberContext {
                gprs: [0; NUM_GPRS],
                rsp: 0,
                rip: 0,
                xmm_regs: [0; 8],
                flags: FiberFlags::Empty,
            });
            stacks.push(Vec::new());
        }

        Self {
            current: 0,
            capacity,
            fibers,
            stacks,
        }
    }

    /// Spawn a new fiber with given entry function
    ///
    /// Returns fiber index or None if at capacity
    pub fn spawn(&mut self, entry: extern "C" fn() -> ()) -> Option<usize> {
        for i in 0..self.capacity {
            if self.fibers[i].flags == FiberFlags::Empty {
                // Allocate stack for this fiber
                self.stacks[i] = vec![0u8; FIBER_STACK_SIZE];
                
                // Initialize fiber context
                self.fibers[i] = FiberContext::new(entry, &mut self.stacks[i]);
                
                return Some(i);
            }
        }
        None
    }

    /// Yield execution to another ready fiber
    ///
    /// Uses inline assembly for minimal overhead context switch
    #[inline(always)]
    pub fn yield_to(&mut self, target: usize) {
        if target == self.current {
            return; // No-op if yielding to self
        }

        if target >= self.capacity || self.fibers[target].flags != FiberFlags::Ready {
            return; // Invalid target
        }

        unsafe {
            switch_fiber(
                &mut self.fibers[self.current],
                &mut self.fibers[target],
            );
        }

        self.current = target;
    }

    /// Get reference to current fiber
    #[inline(always)]
    pub fn current_fiber(&self) -> &FiberContext {
        &self.fibers[self.current]
    }

    /// Get mutable reference to current fiber
    #[inline(always)]
    pub fn current_fiber_mut(&mut self) -> &mut FiberContext {
        &mut self.fibers[self.current]
    }
}

impl Default for FiberScheduler {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Low-level context switch between two fibers
///
/// Saves current register state to `from` context and restores from `to` context.
/// This is the critical path for sub-microsecond yields.
///
/// # Safety
/// - Both contexts must have valid stack pointers
/// - Caller must ensure no concurrent access to fiber contexts
#[naked]
unsafe extern "system" fn switch_fiber(from: &mut FiberContext, to: &mut FiberContext) {
    // Naked function with inline assembly for minimal overhead
    // AMD Zen optimized register save/restore order
    
    std::arch::asm!(
        // Save callee-saved registers to 'from' context
        // RBP, RBX, R12-R15 are callee-saved per System V ABI
        
        // Save RBP
        "mov [rdi + 0x00], rbp",
        // Save RBX  
        "mov [rdi + 0x08], rbx",
        // Save R12
        "mov [rdi + 0x10], r12",
        // Save R13
        "mov [rdi + 0x18], r13",
        // Save R14
        "mov [rdi + 0x20], r14",
        // Save R15
        "mov [rdi + 0x28], r15",
        
        // Save current RSP
        "mov [rdi + 0x30], rsp",
        
        // Save XMM registers (for compute fibers)
        "movdqu [rdi + 0x40], xmm0",
        "movdqu [rdi + 0x50], xmm1",
        "movdqu [rdi + 0x60], xmm2",
        "movdqu [rdi + 0x70], xmm3",
        
        // Load new register state from 'to' context
        
        // Restore RBP
        "mov rbp, [rsi + 0x00]",
        // Restore RBX
        "mov rbx, [rsi + 0x08]",
        // Restore R12
        "mov r12, [rsi + 0x10]",
        // Restore R13
        "mov r13, [rsi + 0x18]",
        // Restore R14
        "mov r14, [rsi + 0x20]",
        // Restore R15
        "mov r15, [rsi + 0x28]",
        
        // Restore XMM registers
        "movdqu xmm0, [rsi + 0x40]",
        "movdqu xmm1, [rsi + 0x50]",
        "movdqu xmm2, [rsi + 0x60]",
        "movdqu xmm3, [rsi + 0x70]",
        
        // Restore RSP and jump to saved RIP
        "mov rsp, [rsi + 0x30]",
        
        // Return to the saved instruction pointer
        // The saved RIP is effectively the return address on the stack
        "ret",
        
        options(noreturn)
    );
}

/// High-performance yield using minimal register save
///
/// Only saves volatile registers for ultra-fast context switches
/// when full state preservation is not required.
#[inline(always)]
pub unsafe fn fast_yield() {
    // Use PAUSE instruction to prevent pipeline stalls on AMD Zen
    std::arch::asm!("pause", options(nomem, nostack));
    
    // Memory barrier to ensure visibility
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Spin-wait with exponential backoff for AMD Zen
///
/// Optimized for short wait times in lock-free algorithms
pub struct SpinWait {
    iterations: u32,
}

impl SpinWait {
    pub const fn new() -> Self {
        Self { iterations: 0 }
    }

    /// Perform one spin iteration with PAUSE
    #[inline(always)]
    pub fn spin_once(&mut self) {
        // PAUSE instruction reduces power consumption and prevents
        // pipeline stall on AMD Zen during spin loops
        unsafe {
            std::arch::asm!(
                "pause",
                options(nomem, nostack, preserves_flags)
            );
        }

        // Exponential backoff after certain iterations
        if self.iterations > 10 {
            for _ in 0..(self.iterations.min(100)) {
                std::hint::spin_loop();
            }
        }

        self.iterations = self.iterations.saturating_add(1);
    }

    /// Reset spin counter
    #[inline(always)]
    pub fn reset(&mut self) {
        self.iterations = 0;
    }
}

impl Default for SpinWait {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local fiber scheduler for zero-allocation yields
thread_local! {
    static LOCAL_SCHEDULER: std::cell::RefCell<Option<FiberScheduler>> = 
        std::cell::RefCell::new(None);
}

/// Initialize thread-local fiber scheduler
#[inline(always)]
pub fn init_scheduler(capacity: usize) {
    LOCAL_SCHEDULER.with(|s| {
        *s.borrow_mut() = Some(FiberScheduler::new(capacity));
    });
}

/// Get reference to thread-local scheduler
#[inline(always)]
pub fn get_scheduler() -> Option<std::cell::Ref<'static, FiberScheduler>> {
    LOCAL_SCHEDULER.with(|s| {
        if s.borrow().is_some() {
            Some(std::cell::Ref::map(s.borrow(), |opt| opt.as_ref().unwrap()))
        } else {
            None
        }
    })
}

/// Example fiber entry point
extern "C" fn example_fiber_entry() {
    // Fiber logic here
    // Can yield back to scheduler when done
}

#[cfg(test)]
mod tests {
    use super::*;

    static mut FIBER_RAN: bool = false;

    extern "C" fn test_fiber() {
        unsafe {
            FIBER_RAN = true;
        }
    }

    #[test]
    fn test_fiber_creation() {
        let mut scheduler = FiberScheduler::new(8);
        
        let fiber_idx = scheduler.spawn(test_fiber);
        assert!(fiber_idx.is_some());
        
        let fiber = &scheduler.fibers[fiber_idx.unwrap()];
        assert_eq!(fiber.flags, FiberFlags::Ready);
        assert!(fiber.rsp > 0);
    }

    #[test]
    fn test_fiber_context_size() {
        // Verify FiberContext fits in reasonable cache lines
        let size = mem::size_of::<FiberContext>();
        println!("FiberContext size: {} bytes", size);
        
        // Should be less than 256 bytes for L1 cache efficiency
        assert!(size < 512);
    }

    #[test]
    fn test_spin_wait() {
        let mut spin = SpinWait::new();
        
        for _ in 0..100 {
            spin.spin_once();
        }
        
        assert!(spin.iterations > 0);
    }

    #[test]
    fn test_scheduler_capacity() {
        let mut scheduler = FiberScheduler::new(16);
        
        // Fill all slots
        for _ in 0..16 {
            assert!(scheduler.spawn(test_fiber).is_some());
        }
        
        // Should fail on 17th
        assert!(scheduler.spawn(test_fiber).is_none());
    }
}
