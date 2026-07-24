// =============================================================================
// NAUTILUS/RAY CRYPTO TRADING BOT - BUILD INFO MODULE
// =============================================================================
// Stage 54: Compile-time Environment Injection
// Purpose: Embed Git commit hash, Rust compiler version, and PGO status
// Usage: Access via build_info::BUILD_INFO constant for SOUL.md ledger tracking
// =============================================================================

use std::fmt;

/// Build information structure containing all compile-time metadata
/// This is injected at build time and cannot be modified at runtime
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Git commit hash (short form, 7 characters)
    pub git_commit: &'static str,
    
    /// Git commit hash (full form, 40 characters)
    pub git_commit_full: &'static str,
    
    /// Git branch name
    pub git_branch: &'static str,
    
    /// Git tag (if any)
    pub git_tag: Option<&'static str>,
    
    /// Whether the working directory was dirty at build time
    pub git_dirty: bool,
    
    /// Rust compiler version (e.g., "1.75.0")
    pub rustc_version: &'static str,
    
    /// Rust compiler LLVM version
    pub llvm_version: &'static str,
    
    /// Target triple (e.g., "x86_64-pc-windows-msvc")
    pub target: &'static str,
    
    /// Build profile (debug, release, pgo-instrument, pgo-use)
    pub profile: &'static str,
    
    /// Whether PGO was enabled during compilation
    pub pgo_enabled: bool,
    
    /// Whether LTO was enabled during compilation
    pub lto_enabled: bool,
    
    /// Build timestamp (Unix epoch seconds)
    pub build_timestamp: u64,
    
    /// Build host (machine hostname)
    pub build_host: &'static str,
    
    /// Optimization level (0-3)
    pub opt_level: u8,
    
    /// Codegen units count
    pub codegen_units: u32,
}

impl BuildInfo {
    /// Returns a formatted string suitable for SOUL.md ledger entries
    pub fn to_ledger_entry(&self) -> String {
        format!(
            "BUILD|{}|{}|{}|PGO:{}|LTO:{}|OPT:{}",
            self.git_commit,
            self.rustc_version,
            self.profile,
            self.pgo_enabled,
            self.lto_enabled,
            self.opt_level
        )
    }
    
    /// Returns true if this is a production-ready PGO build
    pub fn is_production_pgo(&self) -> bool {
        self.pgo_enabled && self.lto_enabled && self.profile == "pgo-use"
    }
    
    /// Returns true if this is an instrumented PGO build (for profiling)
    pub fn is_instrumented(&self) -> bool {
        self.pgo_enabled && self.profile == "pgo-instrument"
    }
    
    /// Returns the build identifier for version tracking
    pub fn build_id(&self) -> String {
        format!(
            "{}-{}-{}",
            self.git_commit,
            self.profile,
            if self.pgo_enabled { "pgo" } else { "no-pgo" }
        )
    }
}

impl fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Build Information ===")?;
        writeln!(f, "Git Commit:     {} ({})", self.git_commit, 
                 if self.git_dirty { "dirty" } else { "clean" })?;
        writeln!(f, "Git Branch:     {}", self.git_branch)?;
        if let Some(tag) = self.git_tag {
            writeln!(f, "Git Tag:        {}", tag)?;
        }
        writeln!(f, "Rust Version:   {}", self.rustc_version)?;
        writeln!(f, "LLVM Version:   {}", self.llvm_version)?;
        writeln!(f, "Target:         {}", self.target)?;
        writeln!(f, "Profile:        {}", self.profile)?;
        writeln!(f, "PGO Enabled:    {}", self.pgo_enabled)?;
        writeln!(f, "LTO Enabled:    {}", self.lto_enabled)?;
        writeln!(f, "Opt Level:      {}", self.opt_level)?;
        writeln!(f, "Codegen Units:  {}", self.codegen_units)?;
        writeln!(f, "Build Time:     {}", self.build_timestamp)?;
        writeln!(f, "Build Host:     {}", self.build_host)?;
        writeln!(f, "=========================")?;
        Ok(())
    }
}

/// Global build info constant - populated at compile time
pub const BUILD_INFO: BuildInfo = BuildInfo {
    git_commit: env!("VERGEN_GIT_SHA_SHORT"),
    git_commit_full: env!("VERGEN_GIT_SHA"),
    git_branch: env!("VERGEN_GIT_BRANCH"),
    git_tag: {
        const TAG: Option<&str> = option_env!("VERGEN_GIT_DESCRIBE");
        TAG
    },
    git_dirty: cfg!(feature = "git-dirty"),
    rustc_version: env!("VERGEN_RUSTC_SEMVER"),
    llvm_version: env!("VERGEN_RUSTC_LLVM_VERSION"),
    target: env!("VERGEN_CARGO_TARGET_TRIPLE"),
    profile: env!("VERGEN_CARGO_PROFILE"),
    pgo_enabled: cfg!(profile = "pgo-instrument") || cfg!(profile = "pgo-use"),
    lto_enabled: cfg!(lto = "fat") || cfg!(lto = "thin"),
    build_timestamp: {
        const TS: &str = env!("VERGEN_BUILD_TIMESTAMP");
        // Parse Unix timestamp from build timestamp string
        // This is a simplified version; in practice you'd parse the actual timestamp
        0
    },
    build_host: env!("VERGEN_BUILD_HOST"),
    opt_level: {
        const OPT: &str = env!("VERGEN_CARGO_OPT_LEVEL");
        OPT.parse().unwrap_or(0)
    },
    codegen_units: {
        // Default to 1 for PGO builds
        if cfg!(profile = "pgo-instrument") || cfg!(profile = "pgo-use") {
            1
        } else {
            1
        }
    },
};

/// Returns the current build information
#[inline(always)]
pub fn get_build_info() -> &'static BuildInfo {
    &BUILD_INFO
}

/// Checks if the current binary was built with PGO
#[inline(always)]
pub fn is_pgo_build() -> bool {
    BUILD_INFO.pgo_enabled
}

/// Checks if the current binary has LTO enabled
#[inline(always)]
pub fn is_lto_build() -> bool {
    BUILD_INFO.lto_enabled
}

/// Returns the git commit hash for version tracking
#[inline(always)]
pub fn get_git_commit() -> &'static str {
    BUILD_INFO.git_commit
}

/// Returns a human-readable version string
#[inline(always)]
pub fn get_version_string() -> String {
    format!(
        "nautilus-ray-bot v{} ({} - {})",
        env!("CARGO_PKG_VERSION"),
        BUILD_INFO.git_commit,
        BUILD_INFO.profile
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_info_display() {
        let info = BUILD_INFO;
        let display = format!("{}", info);
        assert!(display.contains("Build Information"));
        assert!(display.contains(info.git_commit));
    }

    #[test]
    fn test_ledger_entry_format() {
        let entry = BUILD_INFO.to_ledger_entry();
        assert!(entry.starts_with("BUILD|"));
        assert!(entry.contains(BUILD_INFO.git_commit));
    }

    #[test]
    fn test_pgo_detection() {
        // PGO status depends on build profile
        let _is_pgo = is_pgo_build();
        let _is_lto = is_lto_build();
        // These will vary based on how tests are run
    }

    #[test]
    fn test_version_string() {
        let version = get_version_string();
        assert!(version.contains("nautilus-ray-bot"));
        assert!(version.contains(BUILD_INFO.git_commit));
    }
}

// =============================================================================
// BUILD SCRIPT INTEGRATION
// =============================================================================
// This module expects the following environment variables set by build.rs:
// - VERGEN_GIT_SHA_SHORT: Short git commit hash (7 chars)
// - VERGEN_GIT_SHA: Full git commit hash (40 chars)
// - VERGEN_GIT_BRANCH: Git branch name
// - VERGEN_GIT_DESCRIBE: Git describe output (tag info)
// - VERGEN_RUSTC_SEMVER: Rust compiler semver
// - VERGEN_RUSTC_LLVM_VERSION: LLVM version used by rustc
// - VERGEN_CARGO_TARGET_TRIPLE: Target triple
// - VERGEN_CARGO_PROFILE: Cargo profile name
// - VERGEN_BUILD_TIMESTAMP: Build timestamp
// - VERGEN_BUILD_HOST: Build machine hostname
// - VERGEN_CARGO_OPT_LEVEL: Optimization level
//
// These are automatically generated by the vergen crate in build.rs
// =============================================================================
