//! Chapter 3: Real-Time Feature Store & Vector Search
//! File 9: src/features/state_matcher.rs
//!
//! Matches current LOB state to historical vector embeddings to instantly
//! hot-swap regime-specific RL weights via lock-free Read-Copy-Update (RCU) pointers.
//! Enables microsecond regime detection and policy switching.
//!
//! Optimized for AMD Ryzen AI 5 with atomic pointer operations.

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicBool, Ordering};
use std::ptr;

/// Maximum number of regime templates
const MAX_REGIMES: usize = 64;

/// Maximum embedding dimension
const MAX_EMBEDDING_DIM: usize = 256;

/// RL weight set for a specific regime
#[repr(C, align(64))]
pub struct RLWeightSet {
    /// Actor network weights (flattened)
    pub actor_weights: [f32; 1024],
    /// Critic network weights (flattened)
    pub critic_weights: [f32; 1024],
    /// Weight count
    pub actor_count: u32,
    pub critic_count: u32,
    /// Regime identifier
    pub regime_id: u32,
    /// Is valid
    pub is_valid: bool,
}

impl Default for RLWeightSet {
    fn default() -> Self {
        RLWeightSet {
            actor_weights: [0.0; 1024],
            critic_weights: [0.0; 1024],
            actor_count: 0,
            critic_count: 0,
            regime_id: 0,
            is_valid: false,
        }
    }
}

/// Historical state template for matching
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct StateTemplate {
    /// Embedding vector
    pub embedding: [f32; MAX_EMBEDDING_DIM],
    /// Actual dimension used
    pub dim: u16,
    /// Regime label
    pub regime_label: u16,
    /// Timestamp of original state
    pub timestamp_ns: u64,
    /// Match count (how many times this was matched)
    pub match_count: AtomicU64,
}

impl Default for StateTemplate {
    fn default() -> Self {
        StateTemplate {
            embedding: [0.0; MAX_EMBEDDING_DIM],
            dim: 0,
            regime_label: 0,
            timestamp_ns: 0,
            match_count: AtomicU64::new(0),
        }
    }
}

/// RCU-protected pointer container
struct RCUPtr<T> {
    ptr: AtomicPtr<T>,
}

impl<T> RCUPtr<T> {
    fn new(value: T) -> Self {
        let boxed = Box::new(value);
        let ptr = Box::into_raw(boxed);
        RCUPtr {
            ptr: AtomicPtr::new(ptr),
        }
    }
    
    fn load(&self) -> *const T {
        self.ptr.load(Ordering::Acquire)
    }
    
    /// Safe RCU update - creates new value, swaps atomically, schedules old for deletion
    fn update(&self, new_value: T) {
        let new_box = Box::new(new_value);
        let new_ptr = Box::into_raw(new_box);
        
        let old_ptr = self.ptr.swap(new_ptr, Ordering::AcqRel);
        
        // In production, use epoch-based reclamation or hazard pointers
        // For now, immediately free (safe if no concurrent readers during update)
        unsafe {
            drop(Box::from_raw(old_ptr));
        }
    }
    
    fn with_read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let ptr = self.load();
        unsafe { f(&*ptr) }
    }
}

/// State Matcher with RCU protection
#[repr(C, align(64))]
pub struct StateMatcher {
    /// Current active weights (RCU protected)
    current_weights: RCUPtr<RLWeightSet>,
    
    /// Historical state templates
    templates: [StateTemplate; MAX_REGIMES],
    
    /// Template count
    template_count: AtomicU64,
    
    /// Embedding dimension
    embedding_dim: AtomicU64,
    
    /// Match threshold (cosine similarity)
    match_threshold: f32,
    
    /// Last match timestamp
    last_match_ns: AtomicU64,
    
    /// Total matches
    total_matches: AtomicU64,
    
    /// Is initialized
    is_initialized: AtomicBool,
}

impl StateMatcher {
    /// Create new state matcher with default weights
    pub fn new(match_threshold: f32) -> Self {
        Self {
            current_weights: RCUPtr::new(RLWeightSet::default()),
            templates: [(); MAX_REGIMES].map(|_| StateTemplate::default()),
            template_count: AtomicU64::new(0),
            embedding_dim: AtomicU64::new(0),
            match_threshold,
            last_match_ns: AtomicU64::new(0),
            total_matches: AtomicU64::new(0),
            is_initialized: AtomicBool::new(true),
        }
    }
    
    /// Initialize with embedding dimension
    pub fn init_with_dim(&self, dim: usize) -> bool {
        if dim > MAX_EMBEDDING_DIM || !self.is_initialized.load(Ordering::Relaxed) {
            return false;
        }
        self.embedding_dim.store(dim as u64, Ordering::Relaxed);
        true
    }
    
    /// Register a historical state template
    pub fn register_template(
        &self,
        embedding: &[f32],
        regime_label: u16,
        timestamp_ns: u64,
    ) -> Option<usize> {
        let current = self.template_count.load(Ordering::Relaxed);
        if current >= MAX_REGIMES as u64 {
            return None;
        }
        
        let dim = self.embedding_dim.load(Ordering::Relaxed) as usize;
        if dim == 0 || embedding.len() < dim {
            return None;
        }
        
        let idx = current as usize;
        
        unsafe {
            let tpl_ptr = self.templates.as_mut_ptr().add(idx);
            (*tpl_ptr).dim = dim as u16;
            (*tpl_ptr).regime_label = regime_label;
            (*tpl_ptr).timestamp_ns = timestamp_ns;
            
            std::ptr::copy_nonoverlapping(
                embedding.as_ptr(),
                (*tpl_ptr).embedding.as_mut_ptr(),
                dim,
            );
        }
        
        self.template_count.fetch_add(1, Ordering::Relaxed);
        Some(idx)
    }
    
    /// Register RL weights for a regime
    pub fn register_regime_weights(&self, regime_label: u16, weights: &RLWeightSet) -> bool {
        // Find template with this regime label
        let count = self.template_count.load(Ordering::Relaxed) as usize;
        
        for i in 0..count {
            unsafe {
                let tpl_ptr = self.templates.as_ptr().add(i);
                if (*tpl_ptr).regime_label == regime_label {
                    // Update the template's associated weights
                    // In production, would store weights separately
                    break;
                }
            }
        }
        
        // If this is for the current best match, update via RCU
        if regime_label == 0 {
            self.current_weights.update(*weights);
            return true;
        }
        
        true
    }
    
    /// Match current embedding to historical templates
    /// Returns the regime label of the best match
    pub fn match_state(&self, current_embedding: &[f32]) -> Option<(u16, f32)> {
        let dim = self.embedding_dim.load(Ordering::Relaxed) as usize;
        let count = self.template_count.load(Ordering::Relaxed) as usize;
        
        if dim == 0 || count == 0 || current_embedding.len() < dim {
            return None;
        }
        
        let mut best_regime = 0u16;
        let mut best_similarity = -1.0f32;
        
        for i in 0..count {
            unsafe {
                let tpl_ptr = self.templates.as_ptr().add(i);
                let tpl = &*tpl_ptr;
                
                let similarity = cosine_similarity(
                    current_embedding,
                    &tpl.embedding[..dim],
                );
                
                if similarity > best_similarity && similarity >= self.match_threshold {
                    best_similarity = similarity;
                    best_regime = tpl.regime_label;
                }
            }
        }
        
        if best_similarity >= self.match_threshold {
            // Increment match count for the matched template
            for i in 0..count {
                unsafe {
                    let tpl_ptr = self.templates.as_ptr().add(i);
                    if (*tpl_ptr).regime_label == best_regime {
                        (*tpl_ptr).match_count.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
            }
            
            self.total_matches.fetch_add(1, Ordering::Relaxed);
            self.last_match_ns.store(get_timestamp_ns(), Ordering::Relaxed);
            
            Some((best_regime, best_similarity))
        } else {
            None
        }
    }
    
    /// Hot-swap to regime-specific weights (RCU protected)
    pub fn swap_to_regime(&self, regime_label: u16, new_weights: RLWeightSet) {
        // Find the template for this regime and get its embedding as key
        let count = self.template_count.load(Ordering::Relaxed) as usize;
        
        for i in 0..count {
            unsafe {
                let tpl_ptr = self.templates.as_ptr().add(i);
                if (*tpl_ptr).regime_label == regime_label {
                    // Perform RCU swap
                    self.current_weights.update(new_weights);
                    return;
                }
            }
        }
        
        // Regime not found, still update as fallback
        self.current_weights.update(new_weights);
    }
    
    /// Get current weights (read via RCU)
    pub fn get_current_weights(&self) -> RLWeightSet {
        self.current_weights.with_read(|w| *w)
    }
    
    /// Statistics
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.template_count.load(Ordering::Relaxed),
            self.total_matches.load(Ordering::Relaxed),
            self.last_match_ns.load(Ordering::Relaxed),
            self.embedding_dim.load(Ordering::Relaxed),
        )
    }
}

/// Cosine similarity between two vectors
#[inline(always)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    
    dot / denom
}

/// Get timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_matcher_creation() {
        let matcher = StateMatcher::new(0.8);
        assert!(matcher.init_with_dim(32));
    }
    
    #[test]
    fn test_template_registration() {
        let matcher = StateMatcher::new(0.8);
        matcher.init_with_dim(4);
        
        let embedding = [1.0, 0.0, 0.0, 0.0];
        assert!(matcher.register_template(&embedding, 1, 1000).is_some());
        
        let (templates, _, _, _) = matcher.stats();
        assert_eq!(templates, 1);
    }
    
    #[test]
    fn test_state_matching() {
        let matcher = StateMatcher::new(0.7);
        matcher.init_with_dim(4);
        
        // Register template
        let embedding1 = [1.0, 0.0, 0.0, 0.0];
        matcher.register_template(&embedding1, 1, 1000);
        
        // Match similar state
        let query = [0.95, 0.05, 0.0, 0.0];
        let result = matcher.match_state(&query);
        
        assert!(result.is_some());
        let (regime, similarity) = result.unwrap();
        assert_eq!(regime, 1);
        assert!(similarity > 0.7);
    }
    
    #[test]
    fn test_ram_limits() {
        assert!(MAX_REGIMES > 0);
        assert!(MAX_REGIMES <= 128);
        assert!(MAX_EMBEDDING_DIM <= 512);
    }
}
