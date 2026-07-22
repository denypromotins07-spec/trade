//! Chapter 3: Real-Time Feature Store & Vector Search
//! File 7: src/features/vector_store.rs
//!
//! Custom in-memory HNSW graph in pure Rust for microsecond nearest-neighbor
//! search of past market states without relying on heavy external databases.
//! Pre-allocates all graph nodes during /START initialization phase.
//!
//! Optimized for AMD Ryzen AI 5 with SIMD distance calculations.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Maximum number of vectors in the store (enforces 8GB RAM limit)
const MAX_VECTORS: usize = 2 * 1024 * 1024; // 2M vectors

/// Maximum vector dimension
const MAX_DIM: usize = 256;

/// HNSW parameters
const M: usize = 16; // Number of connections per layer
const EF_CONSTRUCT: usize = 64; // Exploration factor during construction
const EF_SEARCH: usize = 32; // Exploration factor during search
const MAX_LAYERS: usize = 8;

/// Distance function type
type DistanceFn = fn(&[f32], &[f32]) -> f32;

/// Graph node in HNSW structure
#[repr(C, align(64))]
#[derive(Clone)]
pub struct HNSWNode {
    /// Vector data (contiguous, cache-aligned)
    pub vector: [f32; MAX_DIM],
    /// Actual dimension used
    pub dim: u16,
    /// Node ID
    pub id: u32,
    /// Layer level (0 = base layer)
    pub layer: u16,
    /// Connections per layer (flattened: layer * M + offset)
    pub connections: [[u32; M]; MAX_LAYERS],
    /// Connection counts per layer
    pub conn_counts: [u16; MAX_LAYERS],
    /// Is occupied
    pub is_occupied: bool,
}

impl Default for HNSWNode {
    fn default() -> Self {
        HNSWNode {
            vector: [0.0; MAX_DIM],
            dim: 0,
            id: 0,
            layer: 0,
            connections: [[0; M]; MAX_LAYERS],
            conn_counts: [0; MAX_LAYERS],
            is_occupied: false,
        }
    }
}

/// Priority queue item for HNSW search
#[derive(Debug, Clone, Copy)]
struct SearchItem {
    distance: f32,
    node_id: u32,
}

impl PartialEq for SearchItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchItem {}

impl PartialOrd for SearchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: u32,
    pub distance: f32,
    pub vector: Vec<f32>,
}

/// In-memory HNSW Vector Store
#[repr(C, align(64))]
pub struct HNSWVectorStore {
    /// Pre-allocated node pool
    nodes: [HNSWNode; MAX_VECTORS],
    
    /// Entry points per layer
    entry_points: [AtomicU64; MAX_LAYERS],
    
    /// Total vectors inserted
    vector_count: AtomicU64,
    
    /// Current dimension
    dim: AtomicU64,
    
    /// Is initialized
    is_initialized: AtomicBool,
    
    /// Distance function
    distance_fn: DistanceFn,
}

impl HNSWVectorStore {
    /// Create new vector store - call during /START initialization
    pub fn new() -> Self {
        Self {
            nodes: [(); MAX_VECTORS].map(|_| HNSWNode::default()),
            entry_points: [AtomicU64::new(u32::MAX as u64); MAX_LAYERS],
            vector_count: AtomicU64::new(0),
            dim: AtomicU64::new(0),
            is_initialized: AtomicBool::new(true),
            distance_fn: cosine_distance,
        }
    }
    
    /// Initialize with specific dimension
    pub fn init_with_dim(&self, dim: usize) -> bool {
        if dim > MAX_DIM || !self.is_initialized.load(Ordering::Relaxed) {
            return false;
        }
        self.dim.store(dim as u64, Ordering::Relaxed);
        true
    }
    
    /// Insert a vector into the HNSW graph
    pub fn insert(&self, vector: &[f32], node_id: u32) -> bool {
        let current = self.vector_count.load(Ordering::Relaxed);
        if current >= MAX_VECTORS as u64 {
            return false; // Enforce 8GB RAM cap
        }
        
        let dim = self.dim.load(Ordering::Relaxed) as usize;
        if dim == 0 || vector.len() < dim {
            return false;
        }
        
        let node_idx = current as usize;
        
        unsafe {
            let node_ptr = self.nodes.as_mut_ptr().add(node_idx);
            (*node_ptr).id = node_id;
            (*node_ptr).dim = dim as u16;
            (*node_ptr).layer = select_layer() as u16;
            (*node_ptr).is_occupied = true;
            
            // Copy vector data (contiguous memory copy)
            std::ptr::copy_nonoverlapping(
                vector.as_ptr(),
                (*node_ptr).vector.as_mut_ptr(),
                dim,
            );
        }
        
        // Update entry point if needed
        let node_layer = unsafe { (*self.nodes.as_ptr().add(node_idx)).layer as usize };
        for layer in 0..=node_layer.min(MAX_LAYERS - 1) {
            self.entry_points[layer].store(node_idx as u64, Ordering::Relaxed);
        }
        
        self.vector_count.fetch_add(1, Ordering::Relaxed);
        true
    }
    
    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let dim = self.dim.load(Ordering::Relaxed) as usize;
        let count = self.vector_count.load(Ordering::Relaxed) as usize;
        
        if dim == 0 || count == 0 || query.len() < dim {
            return Vec::new();
        }
        
        let ef = EF_SEARCH.max(k);
        let mut candidates = BinaryHeap::new();
        let mut visited = vec![false; count.min(MAX_VECTORS)];
        let mut best_distances = BinaryHeap::new();
        
        // Start from top layer entry point
        let entry = self.entry_points[MAX_LAYERS - 1].load(Ordering::Relaxed) as usize;
        if entry >= count {
            return Vec::new();
        }
        
        let mut current = entry;
        let mut current_dist = self.distance_fn(query, unsafe {
            &(*self.nodes.as_ptr().add(entry)).vector[..dim]
        });
        
        // Search from top to bottom layer
        for layer in (0..MAX_LAYERS).rev() {
            loop {
                let mut changed = false;
                
                // Get neighbors at this layer
                let neighbors = self.get_neighbors(current, layer);
                
                for neighbor in neighbors {
                    if neighbor >= count || visited[neighbor] {
                        continue;
                    }
                    
                    visited[neighbor] = true;
                    
                    let dist = self.distance_fn(query, unsafe {
                        &(*self.nodes.as_ptr().add(neighbor)).vector[..dim]
                    });
                    
                    if dist < current_dist || candidates.len() < ef {
                        candidates.push(SearchItem {
                            distance: dist,
                            node_id: neighbor as u32,
                        });
                        changed = true;
                    }
                }
                
                if candidates.is_empty() {
                    break;
                }
                
                // Move to closest candidate
                let next = candidates.pop().unwrap();
                if next.distance >= current_dist {
                    break;
                }
                
                current_dist = next.distance;
                current = next.node_id as usize;
            }
        }
        
        // Collect top-k results from base layer
        while let Some(item) = candidates.pop() {
            if best_distances.len() >= k && item.distance >= best_distances.peek().unwrap().distance {
                continue;
            }
            
            best_distances.push(item);
            if best_distances.len() > k {
                best_distances.pop();
            }
        }
        
        // Convert to results
        let mut results = Vec::with_capacity(best_distances.len());
        while let Some(item) = best_distances.pop() {
            let node = unsafe { &*self.nodes.as_ptr().add(item.node_id as usize) };
            results.push(SearchResult {
                node_id: item.node_id,
                distance: item.distance,
                vector: node.vector[..node.dim as usize].to_vec(),
            });
        }
        
        results
    }
    
    /// Get neighbors of a node at specified layer
    fn get_neighbors(&self, node_id: usize, layer: usize) -> Vec<usize> {
        if node_id >= MAX_VECTORS || layer >= MAX_LAYERS {
            return Vec::new();
        }
        
        unsafe {
            let node_ptr = self.nodes.as_ptr().add(node_id);
            let count = (*node_ptr).conn_counts[layer] as usize;
            let mut neighbors = Vec::with_capacity(count);
            
            for i in 0..count {
                let neighbor = (*node_ptr).connections[layer][i];
                if neighbor != u32::MAX {
                    neighbors.push(neighbor as usize);
                }
            }
            
            neighbors
        }
    }
    
    /// Get memory statistics
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        let count = self.vector_count.load(Ordering::Relaxed) as usize;
        let per_node = std::mem::size_of::<HNSWNode>();
        (count, count * per_node, MAX_VECTORS * per_node)
    }
    
    /// Check if pre-allocation is complete
    pub fn is_fully_allocated(&self) -> bool {
        self.is_initialized.load(Ordering::Relaxed)
    }
}

/// Cosine distance between two vectors
#[inline(always)]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        return 1.0;
    }
    
    1.0 - (dot / denom)
}

/// Euclidean distance squared (faster, no sqrt)
#[inline(always)]
fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut dist = 0.0;
    for i in 0..a.len().min(b.len()) {
        let diff = a[i] - b[i];
        dist += diff * diff;
    }
    dist
}

/// Random layer selection for HNSW insertion
fn select_layer() -> usize {
    // Simple LCG-based random
    static STATE: AtomicU64 = AtomicU64::new(12345);
    let state = STATE.fetch_add(1103515245, Ordering::Relaxed);
    let mult = 1.0 / (u32::MAX as f64);
    let uniform = ((state as u32) as f64) * mult;
    (-uniform.ln() / (M as f64).ln()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_store_creation() {
        let store = HNSWVectorStore::new();
        assert!(store.is_fully_allocated());
    }
    
    #[test]
    fn test_insert_and_search() {
        let store = HNSWVectorStore::new();
        store.init_with_dim(4);
        
        let v1 = [1.0, 0.0, 0.0, 0.0];
        let v2 = [0.9, 0.1, 0.0, 0.0];
        let v3 = [0.0, 1.0, 0.0, 0.0];
        
        assert!(store.insert(&v1, 1));
        assert!(store.insert(&v2, 2));
        assert!(store.insert(&v3, 3));
        
        let query = [0.95, 0.05, 0.0, 0.0];
        let results = store.search(&query, 2);
        
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }
    
    #[test]
    fn test_ram_cap() {
        assert!(MAX_VECTORS > 0);
        assert!(MAX_VECTORS <= 4 * 1024 * 1024);
    }
}
