//! Build Information Module - Stage 54
//! 
//! Compile-time environment injection embedding:
//! - Exact Git commit hash
//! - Rust compiler version  
//! - PGO optimization status
//! - AMD CPU target architecture
//! 
//! This data is used for strict version tracking in the SOUL.md ledger
//! and enables precise binary identification during production debugging.

use std::fmt;

// =============================================================================
// COMPILE-TIME CONSTANTS (Injected by build.rs via vergen)
// =============================================================================

/// Git commit hash of the current build (short form, 7 characters)
pub const GIT_COMMIT_SHORT: &str = env!("VERGEN_GIT_SHA_SHORT");

/// Git commit hash of the current build (full form, 40 characters)
pub const GIT_COMMIT_FULL: &str = env!("VERGEN_GIT_SHA");

/// Git branch name at build time
pub const GIT_BRANCH: &str = env!("VERGEN_GIT_BRANCH");

/// Git dirty flag - true if working directory had uncommitted changes
pub const GIT_DIRTY: bool = env!("VERGEN_GIT_DIRTY") == "true";

/// Rust compiler version string (e.g., "1.75.0")
pub const RUST_VERSION: &str = env!("VERGEN_RUSTC_SEMVER");

/// Rust compiler host target (e.g., "x86_64-unknown-linux-gnu")
pub const RUST_TARGET: &str = env!("VERGEN_RUSTC_HOST_TRIPLE");

/// Rust compiler channel (stable/beta/nightly)
pub const RUST_CHANNEL: &str = env!("VERGEN_RUSTC_CHANNEL");

/// Build timestamp in RFC3339 format
pub const BUILD_TIMESTAMP: &str = env!("VERGEN_BUILD_TIMESTAMP");

/// Build date in YYYY-MM-DD format
pub const BUILD_DATE: &str = env!("VERGEN_BUILD_DATE");

/// Cargo package version from Cargo.toml
pub const CARGO_VERSION: &str = env!("CARGO_PKG_VERSION");

/// PGO optimization status - determined at compile time
pub const PGO_ENABLED: bool = cfg!(profile = "pgo-instrument") || cfg!(profile = "release");

/// LTO (Link-Time Optimization) status
pub const LTO_ENABLED: bool = cfg!(target_feature = "lto");

/// Panic strategy - "abort" for minimal binary size
pub const PANIC_STRATEGY: &str = if cfg!(panic = "abort") { "abort" } else { "unwind" };

/// Target CPU architecture (AMD Ryzen AI 5 = znver4)
pub const TARGET_CPU: &str = env!("TARGET_CPU");

/// Whether this is a debug build
pub const IS_DEBUG: bool = cfg!(debug_assertions);

// =============================================================================
// BUILD INFO STRUCTURE
// =============================================================================

/// Complete build information structure for serialization and display
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildInfo {
    /// Short git commit hash (7 chars)
    pub git_commit_short: &'static str,
    /// Full git commit hash (40 chars)
    pub git_commit_full: &'static str,
    /// Git branch name
    pub git_branch: &'static str,
    /// Whether working directory was dirty at build time
    pub git_dirty: bool,
    /// Rust compiler version
    pub rust_version: &'static str,
    /// Rust compiler target triple
    pub rust_target: &'static str,
    /// Rust compiler channel
    pub rust_channel: &'static str,
    /// Build timestamp (RFC3339)
    pub build_timestamp: &'static str,
    /// Build date (YYYY-MM-DD)
    pub build_date: &'static str,
    /// Cargo package version
    pub cargo_version: &'static str,
    /// PGO enabled flag
    pub pgo_enabled: bool,
    /// LTO enabled flag
    pub lto_enabled: bool,
    /// Panic strategy
    pub panic_strategy: &'static str,
    /// Target CPU architecture
    pub target_cpu: &'static str,
    /// Debug build flag
    pub is_debug: bool,
}

impl BuildInfo {
    /// Returns the canonical build info singleton
    #[inline(always)]
    pub const fn get() -> Self {
        Self {
            git_commit_short: GIT_COMMIT_SHORT,
            git_commit_full: GIT_COMMIT_FULL,
            git_branch: GIT_BRANCH,
            git_dirty: GIT_DIRTY,
            rust_version: RUST_VERSION,
            rust_target: RUST_TARGET,
            rust_channel: RUST_CHANNEL,
            build_timestamp: BUILD_TIMESTAMP,
            build_date: BUILD_DATE,
            cargo_version: CARGO_VERSION,
            pgo_enabled: PGO_ENABLED,
            lto_enabled: LTO_ENABLED,
            panic_strategy: PANIC_STRATEGY,
            target_cpu: TARGET_CPU,
            is_debug: IS_DEBUG,
        }
    }

    /// Generates the SOUL.md ledger entry for this build
    /// Format: `STAGE54|{commit}|{version}|{timestamp}|{pgo}|{cpu}`
    #[inline]
    pub fn to_soul_ledger_entry(&self) -> String {
        format!(
            "STAGE54|{}|{}|{}|PGO={}|LTO={}|CPU={}",
            self.git_commit_short,
            self.cargo_version,
            self.build_timestamp,
            if self.pgo_enabled { "YES" } else { "NO" },
            if self.lto_enabled { "YES" } else { "NO" },
            self.target_cpu
        )
    }

    /// Returns a compact identifier for logging: `{version}-{commit_short}`
    #[inline(always)]
    pub fn compact_id(&self) -> String {
        format!("{}-{}", self.cargo_version, self.git_commit_short)
    }

    /// Checks if this build is production-ready
    /// Production builds must: not be debug, have PGO, use abort panic
    #[inline(always)]
    pub fn is_production_ready(&self) -> bool {
        !self.is_debug && self.pgo_enabled && self.panic_strategy == "abort"
    }

    /// Validates build configuration matches expected production settings
    /// Returns Ok(()) if valid, Err with description if not
    pub fn validate_production_config(&self) -> Result<(), &'static str> {
        if self.is_debug {
            return Err("Debug builds are not allowed in production");
        }
        
        if !self.pgo_enabled {
            return Err("PGO must be enabled for production builds");
        }
        
        if self.panic_strategy != "abort" {
            return Err("Panic strategy must be 'abort' for production");
        }
        
        if self.git_dirty {
            return Err("Production builds must be from clean git state");
        }
        
        // Verify AMD Ryzen AI 5 target
        if !self.target_cpu.contains("znver4") && !self.target_cpu.contains("native") {
            return Err("Target CPU must be znver4 (AMD Ryzen AI 5) or native");
        }
        
        Ok(())
    }
}

impl fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           NAUTILUS/RAY HFT BOT - BUILD INFO             ║")?;
        writeln!(f, "║                    STAGE 54 - PGO                       ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Version:      {:<42} ║", self.cargo_version)?;
        writeln!(f, "║ Git Commit:   {:<42} ║", self.git_commit_full)?;
        writeln!(f, "║ Git Branch:   {:<42} ║", self.git_branch)?;
        writeln!(f, "║ Git Dirty:    {:<42} ║", if self.git_dirty { "YES (WARNING)" } else { "No" })?;
        writeln!(f, "╠══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Rust Ver:     {:<42} ║", self.rust_version)?;
        writeln!(f, "║ Rust Target:  {:<42} ║", self.rust_target)?;
        writeln!(f, "║ Rust Channel: {:<42} ║", self.rust_channel)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Build Date:   {:<42} ║", self.build_date)?;
        writeln!(f, "║ Build Time:   {:<42} ║", self.build_timestamp)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ PGO Enabled:  {:<42} ║", if self.pgo_enabled { "YES" } else { "NO" })?;
        writeln!(f, "║ LTO Enabled:  {:<42} ║", if self.lto_enabled { "YES (Fat)" } else { "NO" })?;
        writeln!(f, "║ Panic Strat:  {:<42} ║", self.panic_strategy)?;
        writeln!(f, "║ Target CPU:   {:<42} ║", self.target_cpu)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Production Ready: {:<38} ║", if self.is_production_ready() { "YES ✓" } else { "NO ✗" })?;
        writeln!(f, "╚══════════════════════════════════════════════════════════╝")
    }
}

// =============================================================================
// SERDE SERIALIZATION FOR CQRS EVENT LOGGING
// =============================================================================

impl serde::Serialize for BuildInfo {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        
        let mut state = serializer.serialize_struct("BuildInfo", 16)?;
        state.serialize_field("git_commit_short", &self.git_commit_short)?;
        state.serialize_field("git_commit_full", &self.git_commit_full)?;
        state.serialize_field("git_branch", &self.git_branch)?;
        state.serialize_field("git_dirty", &self.git_dirty)?;
        state.serialize_field("rust_version", &self.rust_version)?;
        state.serialize_field("rust_target", &self.rust_target)?;
        state.serialize_field("rust_channel", &self.rust_channel)?;
        state.serialize_field("build_timestamp", &self.build_timestamp)?;
        state.serialize_field("build_date", &self.build_date)?;
        state.serialize_field("cargo_version", &self.cargo_version)?;
        state.serialize_field("pgo_enabled", &self.pgo_enabled)?;
        state.serialize_field("lto_enabled", &self.lto_enabled)?;
        state.serialize_field("panic_strategy", &self.panic_strategy)?;
        state.serialize_field("target_cpu", &self.target_cpu)?;
        state.serialize_field("is_debug", &self.is_debug)?;
        state.serialize_field("production_ready", &self.is_production_ready())?;
        state.end()
    }
}

#[cfg(feature = "python-ffi")]
impl pyo3::IntoPy<pyo3::PyObject> for BuildInfo {
    fn into_py(self, py: pyo3::Python) -> pyo3::PyObject {
        use pyo3::types::PyDict;
        
        let dict = PyDict::new(py);
        dict.set_item("git_commit_short", self.git_commit_short).unwrap();
        dict.set_item("git_commit_full", self.git_commit_full).unwrap();
        dict.set_item("git_branch", self.git_branch).unwrap();
        dict.set_item("git_dirty", self.git_dirty).unwrap();
        dict.set_item("rust_version", self.rust_version).unwrap();
        dict.set_item("rust_target", self.rust_target).unwrap();
        dict.set_item("rust_channel", self.rust_channel).unwrap();
        dict.set_item("build_timestamp", self.build_timestamp).unwrap();
        dict.set_item("build_date", self.build_date).unwrap();
        dict.set_item("cargo_version", self.cargo_version).unwrap();
        dict.set_item("pgo_enabled", self.pgo_enabled).unwrap();
        dict.set_item("lto_enabled", self.lto_enabled).unwrap();
        dict.set_item("panic_strategy", self.panic_strategy).unwrap();
        dict.set_item("target_cpu", self.target_cpu).unwrap();
        dict.set_item("is_debug", self.is_debug).unwrap();
        dict.set_item("production_ready", self.is_production_ready()).unwrap();
        dict.into()
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_info_get() {
        let info = BuildInfo::get();
        assert!(!info.git_commit_short.is_empty());
        assert!(!info.rust_version.is_empty());
        assert!(!info.cargo_version.is_empty());
    }

    #[test]
    fn test_compact_id_format() {
        let info = BuildInfo::get();
        let id = info.compact_id();
        assert!(id.contains(info.cargo_version));
        assert!(id.contains(info.git_commit_short));
    }

    #[test]
    fn test_soul_ledger_entry_format() {
        let info = BuildInfo::get();
        let entry = info.to_soul_ledger_entry();
        assert!(entry.starts_with("STAGE54|"));
        assert!(entry.contains("|PGO="));
        assert!(entry.contains("|LTO="));
        assert!(entry.contains("|CPU="));
    }

    #[test]
    fn test_git_commit_length() {
        let info = BuildInfo::get();
        assert_eq!(info.git_commit_short.len(), 7, "Short commit should be 7 chars");
        assert_eq!(info.git_commit_full.len(), 40, "Full commit should be 40 chars");
    }
}
