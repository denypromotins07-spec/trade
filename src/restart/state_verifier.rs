//! State Verifier for Shadow Process Memory Layout Validation
//! 
//! This module cryptographically verifies that the shadow process memory
//! layout matches the primary process exactly before routing inbound
//! network traffic to the new binary.
//! 
//! Uses SHA-256 hashing with Merkle tree verification for efficient
//! large-state validation. Optimized for microsecond-level verification.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Chunk size for Merkle tree (256KB)
const MERKLE_CHUNK_SIZE: usize = 256 * 1024;
/// Maximum verification time budget (milliseconds)
const MAX_VERIFICATION_TIME_MS: u64 = 100;
/// Number of hash rounds for key derivation
const HASH_ROUNDS: usize = 3;

/// Verification result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    Success { checksum: String, verified_at: Instant },
    Mismatch { expected: String, actual: String, first_diff_offset: usize },
    Timeout { elapsed_ms: u64 },
    Error { message: String },
}

/// Memory region descriptor
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub name: String,
    pub base_address: u64,
    pub size: usize,
    pub is_critical: bool,
    pub checksum: Option<String>,
}

/// Merkle tree node
#[derive(Debug, Clone)]
pub struct MerkleNode {
    pub hash: [u8; 32],
    pub left: Option<Box<MerkleNode>>,
    pub right: Option<Box<MerkleNode>>,
    pub offset: usize,
    pub size: usize,
}

/// State Verifier - Main verification engine
pub struct StateVerifier {
    /// Expected root hash from primary
    expected_root_hash: parking_lot::RwLock<Option<[u8; 32]>>,
    /// Memory regions to verify
    memory_regions: parking_lot::Mutex<Vec<MemoryRegion>>,
    /// Merkle tree root
    merkle_root: parking_lot::Mutex<Option<MerkleNode>>,
    /// Verification statistics
    total_verifications: AtomicU64,
    successful_verifications: AtomicU64,
    failed_verifications: AtomicU64,
    /// Running flag
    is_running: Arc<AtomicBool>,
}

impl StateVerifier {
    /// Create new state verifier
    pub fn new() -> Self {
        Self {
            expected_root_hash: parking_lot::RwLock::new(None),
            memory_regions: parking_lot::Mutex::new(Vec::new()),
            merkle_root: parking_lot::Mutex::new(None),
            total_verifications: AtomicU64::new(0),
            successful_verifications: AtomicU64::new(0),
            failed_verifications: AtomicU64::new(0),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Add memory region for verification
    pub fn add_memory_region(&self, region: MemoryRegion) {
        let mut regions = self.memory_regions.lock();
        regions.push(region);
    }

    /// Build Merkle tree from memory regions
    pub fn build_merkle_tree(&self, data: &[u8]) -> MerkleNode {
        // Split data into chunks
        let chunks: Vec<&[u8]> = data.chunks(MERKLE_CHUNK_SIZE).collect();
        
        // Create leaf nodes
        let mut nodes: Vec<MerkleNode> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let hash = Self::hash_chunk(chunk);
                MerkleNode {
                    hash,
                    left: None,
                    right: None,
                    offset: i * MERKLE_CHUNK_SIZE,
                    size: chunk.len(),
                }
            })
            .collect();

        // Build tree bottom-up
        while nodes.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in nodes.chunks(2) {
                if chunk.len() == 2 {
                    let left = chunk[0].clone();
                    let right = chunk[1].clone();
                    
                    // Combine hashes
                    let mut combined = Vec::with_capacity(64);
                    combined.extend_from_slice(&left.hash);
                    combined.extend_from_slice(&right.hash);
                    
                    let parent_hash = Self::hash_data(&combined);
                    
                    next_level.push(MerkleNode {
                        hash: parent_hash,
                        left: Some(Box::new(left)),
                        right: Some(Box::new(right)),
                        offset: left.offset,
                        size: left.size + right.size,
                    });
                } else {
                    next_level.push(chunk[0].clone());
                }
            }
            
            nodes = next_level;
        }

        nodes.into_iter().next().unwrap_or_else(|| MerkleNode {
            hash: [0u8; 32],
            left: None,
            right: None,
            offset: 0,
            size: 0,
        })
    }

    /// Hash a data chunk using SHA-256
    fn hash_chunk(data: &[u8]) -> [u8; 32] {
        Self::hash_data(data)
    }

    /// Hash arbitrary data
    fn hash_data(data: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Set expected root hash from primary process
    pub fn set_expected_root_hash(&self, hash: [u8; 32]) {
        *self.expected_root_hash.write() = Some(hash);
    }

    /// Get expected root hash as hex string
    pub fn get_expected_root_hex(&self) -> Option<String> {
        self.expected_root_hash.read().map(|h| hex::encode(h.as_ref()))
    }

    /// Verify shadow state against expected hash
    pub fn verify_state(&self, shadow_data: &[u8]) -> VerificationResult {
        self.total_verifications.fetch_add(1, Ordering::Relaxed);
        
        let start = Instant::now();
        
        // Check timeout
        if start.elapsed() > Duration::from_millis(MAX_VERIFICATION_TIME_MS) {
            self.failed_verifications.fetch_add(1, Ordering::Relaxed);
            return VerificationResult::Timeout {
                elapsed_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Build Merkle tree for shadow data
        let shadow_root = self.build_merkle_tree(shadow_data);
        
        // Compare with expected
        match *self.expected_root_hash.read() {
            Some(expected) => {
                if shadow_root.hash == expected {
                    self.successful_verifications.fetch_add(1, Ordering::Relaxed);
                    VerificationResult::Success {
                        checksum: hex::encode(shadow_root.hash.as_ref()),
                        verified_at: Instant::now(),
                    }
                } else {
                    self.failed_verifications.fetch_add(1, Ordering::Relaxed);
                    
                    // Find first differing chunk
                    let first_diff = self.find_first_difference(shadow_data);
                    
                    VerificationResult::Mismatch {
                        expected: hex::encode(expected.as_ref()),
                        actual: hex::encode(shadow_root.hash.as_ref()),
                        first_diff_offset: first_diff,
                    }
                }
            }
            None => VerificationResult::Error {
                message: "No expected hash set".to_string(),
            },
        }
    }

    /// Find first byte difference between expected and actual
    fn find_first_difference(&self, actual_data: &[u8]) -> usize {
        // This would compare against stored expected data
        // For now, return 0 as placeholder
        0
    }

    /// Generate cryptographic proof for specific memory region
    pub fn generate_proof(&self, region_name: &str, data: &[u8]) -> Option<Proof> {
        let regions = self.memory_regions.lock();
        let region = regions.iter().find(|r| r.name == region_name)?;
        
        let root = self.build_merkle_tree(data);
        
        Some(Proof {
            region_name: region.name.clone(),
            root_hash: root.hash,
            proof_path: self.generate_proof_path(&root, 0),
            timestamp: Instant::now(),
        })
    }

    /// Generate Merkle proof path for a leaf
    fn generate_proof_path(&self, node: &MerkleNode, target_offset: usize) -> Vec<ProofStep> {
        let mut path = Vec::new();
        
        if let (Some(left), Some(right)) = (&node.left, &node.right) {
            if target_offset < left.offset + left.size {
                path.push(ProofStep {
                    sibling_hash: right.hash,
                    is_left: false,
                });
                let mut child_path = self.generate_proof_path(left, target_offset);
                path.append(&mut child_path);
            } else {
                path.push(ProofStep {
                    sibling_hash: left.hash,
                    is_left: true,
                });
                let mut child_path = self.generate_proof_path(right, target_offset);
                path.append(&mut child_path);
            }
        }
        
        path
    }

    /// Verify proof against root hash
    pub fn verify_proof(&self, proof: &Proof, leaf_data: &[u8], root_hash: &[u8; 32]) -> bool {
        let mut current_hash = Self::hash_chunk(leaf_data);
        
        for step in &proof.proof_path {
            let mut combined = Vec::with_capacity(64);
            if step.is_left {
                combined.extend_from_slice(&step.sibling_hash);
                combined.extend_from_slice(&current_hash);
            } else {
                combined.extend_from_slice(&current_hash);
                combined.extend_from_slice(&step.sibling_hash);
            }
            current_hash = Self::hash_data(&combined);
        }
        
        current_hash == *root_hash
    }

    /// Get verification statistics
    pub fn get_stats(&self) -> VerificationStats {
        VerificationStats {
            total: self.total_verifications.load(Ordering::Relaxed),
            successful: self.successful_verifications.load(Ordering::Relaxed),
            failed: self.failed_verifications.load(Ordering::Relaxed),
        }
    }

    /// Export memory layout for debugging
    pub fn export_layout_json(&self) -> String {
        let regions = self.memory_regions.lock();
        let mut json = String::from("[\n");
        
        for (i, region) in regions.iter().enumerate() {
            json.push_str(&format!(
                "  {{\"name\": \"{}\", \"base\": {}, \"size\": {}, \"critical\": {}}}{}",
                region.name,
                region.base_address,
                region.size,
                region.is_critical,
                if i < regions.len() - 1 { "," } else { "" }
            ));
        }
        
        json.push_str("\n]");
        json
    }
}

impl Default for StateVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Merkle proof structure
#[derive(Debug, Clone)]
pub struct Proof {
    pub region_name: String,
    pub root_hash: [u8; 32],
    pub proof_path: Vec<ProofStep>,
    pub timestamp: Instant,
}

/// Single step in Merkle proof
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub sibling_hash: [u8; 32],
    pub is_left: bool,
}

/// Verification statistics
#[derive(Debug, Clone)]
pub struct VerificationStats {
    pub total: u64,
    pub successful: u64,
    pub failed: u64,
}

/// Global state verifier instance
pub static GLOBAL_STATE_VERIFIER: parking_lot::OnceCell<Arc<StateVerifier>> = parking_lot::OnceCell::new();

/// Initialize global state verifier
pub fn init_global_verifier() -> Arc<StateVerifier> {
    let verifier = Arc::new(StateVerifier::new());
    GLOBAL_STATE_VERIFIER.get_or_init(|| verifier.clone());
    verifier
}

/// Get global state verifier
pub fn get_global_verifier() -> Option<Arc<StateVerifier>> {
    GLOBAL_STATE_VERIFIER.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_creation() {
        let verifier = StateVerifier::new();
        let stats = verifier.get_stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_merkle_tree_building() {
        let verifier = StateVerifier::new();
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let root = verifier.build_merkle_tree(&data);
        
        assert!(!root.hash.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_verification_success() {
        let verifier = StateVerifier::new();
        let data = vec![1u8, 2, 3, 4, 5];
        
        let root = verifier.build_merkle_tree(&data);
        verifier.set_expected_root_hash(root.hash);
        
        let result = verifier.verify_state(&data);
        assert!(matches!(result, VerificationResult::Success { .. }));
    }

    #[test]
    fn test_verification_mismatch() {
        let verifier = StateVerifier::new();
        let data1 = vec![1u8, 2, 3, 4, 5];
        let data2 = vec![1u8, 2, 3, 4, 6]; // Different
        
        let root = verifier.build_merkle_tree(&data1);
        verifier.set_expected_root_hash(root.hash);
        
        let result = verifier.verify_state(&data2);
        assert!(matches!(result, VerificationResult::Mismatch { .. }));
    }
}
