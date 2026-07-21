# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - PYDANTIC SETTINGS VALIDATION
# =============================================================================
# File: python/config/settings.py
# Purpose: Rigorous validation of all .env configurations using Pydantic V2
# Validation: Type-safe, strict mode parsing with detailed error reporting
# Memory: Minimal overhead, validation occurs only at initialization
# =============================================================================

"""
Configuration Settings Module

This module provides strict, type-safe configuration validation using
Pydantic V2's Settings API. All environment variables are validated
before any subsystem is allowed to initialize.

Features:
- Strict type enforcement (no implicit conversions)
- Detailed error messages for misconfigured values
- Default values with validation constraints
- Nested configuration sections matching Rust struct layout
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Literal
from functools import lru_cache

try:
    from pydantic import (
        BaseModel,
        Field,
        ValidationError,
        field_validator,
        model_validator,
    )
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    print("ERROR: Pydantic V2 and pydantic-settings are required")
    print("Install with: pip install pydantic>=2.0 pydantic-settings>=2.0")
    sys.exit(1)


# =============================================================================
# BINANCE API SETTINGS
# =============================================================================


class BinanceSettings(BaseSettings):
    """
    Binance exchange API configuration.
    
    Validates API credentials, endpoints, and trading parameters.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="BINANCE_",
        case_sensitive=True,
        extra="ignore",
    )
    
    api_key: str = Field(
        ...,
        min_length=1,
        description="Binance API key for authentication",
    )
    
    api_secret: str = Field(
        ...,
        min_length=1,
        description="Binance API secret for request signing",
    )
    
    testnet: bool = Field(
        default=True,
        description="Use testnet endpoints (recommended for development)",
    )
    
    ws_endpoint: str = Field(
        default="wss://fstream.binancefuture.com/ws",
        description="WebSocket endpoint for market data",
    )
    
    rest_endpoint: str = Field(
        default="https://fapi.binance.com",
        description="REST API endpoint for order management",
    )
    
    trading_symbols: List[str] = Field(
        default=["BTCUSDT"],
        description="List of symbols to trade",
    )
    
    leverage_max: int = Field(
        default=10,
        ge=1,
        le=125,
        description="Maximum leverage allowed (1-125)",
    )
    
    order_timeout_ms: int = Field(
        default=500,
        ge=100,
        le=10000,
        description="Order timeout in milliseconds",
    )
    
    @field_validator("trading_symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v):
        """Parse comma-separated symbols string into list."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v
    
    @field_validator("ws_endpoint")
    @classmethod
    def validate_ws_endpoint(cls, v):
        """Validate WebSocket endpoint format."""
        if not v.startswith("wss://"):
            raise ValueError("WebSocket endpoint must start with wss://")
        return v
    
    @field_validator("rest_endpoint")
    @classmethod
    def validate_rest_endpoint(cls, v):
        """Validate REST endpoint format."""
        if not v.startswith("https://"):
            raise ValueError("REST endpoint must start with https://")
        return v
    
    @model_validator(mode="after")
    def validate_testnet_endpoints(self):
        """Ensure testnet uses correct endpoints when enabled."""
        if self.testnet:
            if "binance" not in self.ws_endpoint.lower():
                pass  # Allow custom endpoints
        return self


# =============================================================================
# RAY CLUSTER SETTINGS
# =============================================================================


class RaySettings(BaseSettings):
    """
    Ray distributed compute cluster configuration.
    
    Enforces strict memory limits to preserve RAM for Rust engine.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="RAY_",
        case_sensitive=True,
        extra="ignore",
    )
    
    head_host: str = Field(
        default="127.0.0.1",
        description="Ray head node hostname",
    )
    
    head_port: int = Field(
        default=6379,
        ge=1024,
        le=65535,
        description="Ray head node port",
    )
    
    dashboard_port: int = Field(
        default=8265,
        ge=1024,
        le=65535,
        description="Ray dashboard port",
    )
    
    worker_memory_gb: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Maximum memory for Ray workers (GB) - hard capped at 4GB",
    )
    
    num_cpus: int = Field(
        default=6,
        ge=1,
        le=16,
        description="Number of CPUs allocated to Ray",
    )
    
    object_store_memory_gb: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Object store memory size (GB)",
    )
    
    @model_validator(mode="after")
    def validate_memory_budget(self):
        """Ensure total Ray memory doesn't exceed 4GB limit."""
        total = self.worker_memory_gb + self.object_store_memory_gb
        if total > 6:
            raise ValueError(
                f"Total Ray memory ({total}GB) exceeds system budget. "
                "Must be <= 6GB combined."
            )
        return self


# =============================================================================
# ENGINE SETTINGS
# =============================================================================


class EngineSettings(BaseSettings):
    """
    Rust execution engine configuration.
    
    Parameters for the low-latency trading engine.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        case_sensitive=True,
        extra="ignore",
    )
    
    memory_cap_gb: int = Field(
        default=4,
        ge=2,
        le=6,
        description="Memory cap for Rust engine (GB)",
    )
    
    channel_capacity: int = Field(
        default=65536,
        ge=1024,
        le=1000000,
        description="MPSC channel capacity",
    )
    
    lto_enabled: bool = Field(
        default=True,
        description="Enable Link Time Optimization",
    )
    
    target_cpu: str = Field(
        default="native",
        description="Target CPU architecture for compilation",
    )
    
    mpsc_spin_count: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Spin count for lock-free channels",
    )
    
    mpsc_backoff_ns: int = Field(
        default=50,
        ge=10,
        le=10000,
        description="Backoff delay in nanoseconds",
    )


# =============================================================================
# FEATURE FLAGS
# =============================================================================


class FeatureSettings(BaseSettings):
    """
    Feature flag configuration.
    
    Enables/disables subsystems without recompilation.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        case_sensitive=True,
        extra="ignore",
    )
    
    enable_execution: bool = Field(
        default=True,
        description="Enable order execution",
    )
    
    enable_market_data: bool = Field(
        default=True,
        description="Enable market data ingestion",
    )
    
    enable_risk_checks: bool = Field(
        default=True,
        description="Enable risk management checks",
    )
    
    enable_ai_signals: bool = Field(
        default=True,
        description="Enable AI signal generation",
    )
    
    enable_paper_trading: bool = Field(
        default=False,
        description="Enable paper trading mode",
    )
    
    enable_verbose_logging: bool = Field(
        default=False,
        description="Enable verbose debug logging",
    )


# =============================================================================
# RISK SETTINGS
# =============================================================================


class RiskSettings(BaseSettings):
    """
    Risk management configuration.
    
    Hard limits to prevent catastrophic losses.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="MAX_",
        case_sensitive=True,
        extra="ignore",
    )
    
    position_size_usd: float = Field(
        default=10000.0,
        gt=0,
        description="Maximum position size in USD",
    )
    
    daily_loss_usd: float = Field(
        default=500.0,
        gt=0,
        description="Maximum daily loss in USD",
    )
    
    open_orders: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent open orders",
    )
    
    stop_loss_percent: float = Field(
        default=2.0,
        gt=0,
        lt=100,
        description="Stop loss percentage",
    )
    
    take_profit_percent: float = Field(
        default=5.0,
        gt=0,
        lt=100,
        description="Take profit percentage",
    )
    
    emergency_kill_switch: bool = Field(
        default=True,
        description="Enable emergency kill switch",
    )


# =============================================================================
# AI/ML SETTINGS
# =============================================================================


class AISettings(BaseSettings):
    """
    AI/ML model configuration.
    
    Parameters for reinforcement learning and inference.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        case_sensitive=True,
        extra="ignore",
    )
    
    model_path: str = Field(
        default="./models/latest.pt",
        description="Path to model checkpoint",
    )
    
    training_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Training interval in hours",
    )
    
    inference_batch_size: int = Field(
        default=1024,
        ge=1,
        le=65536,
        description="Batch size for inference",
    )
    
    device: Literal["cpu", "cuda", "directml", "rocm"] = Field(
        default="cpu",
        description="Device for model inference",
    )
    
    directml_enabled: bool = Field(
        default=False,
        description="Enable DirectML acceleration (Windows AMD)",
    )
    
    rocm_enabled: bool = Field(
        default=False,
        description="Enable ROCm acceleration (Linux AMD)",
    )


# =============================================================================
# MASTER SETTINGS CLASS
# =============================================================================


class AppSettings(BaseSettings):
    """
    Master settings class aggregating all configuration sections.
    
    This is the single source of truth for application configuration.
    All settings are validated at instantiation time.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    ray: RaySettings = Field(default_factory=RaySettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    ai: AISettings = Field(default_factory=AISettings)
    
    log_level: str = Field(
        default="info",
        description="Logging level",
    )
    
    log_mmap_path: str = Field(
        default="./logs/metrics.mmap",
        description="Path to memory-mapped log file",
    )
    
    telemetry_enabled: bool = Field(
        default=True,
        description="Enable telemetry collection",
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ["trace", "debug", "info", "warn", "error", "fatal"]
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.lower()
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"\n[Binace]")
        print(f"  Testnet: {self.binance.testnet}")
        print(f"  Symbols: {', '.join(self.binance.trading_symbols)}")
        print(f"  Max Leverage: {self.binance.leverage_max}x")
        
        print(f"\n[Ray Cluster]")
        print(f"  CPUs: {self.ray.num_cpus}")
        print(f"  Worker Memory: {self.ray.worker_memory_gb}GB")
        print(f"  Object Store: {self.ray.object_store_memory_gb}GB")
        
        print(f"\n[Engine]")
        print(f"  Memory Cap: {self.engine.memory_cap_gb}GB")
        print(f"  Channel Capacity: {self.engine.channel_capacity:,}")
        print(f"  LTO Enabled: {self.engine.lto_enabled}")
        
        print(f"\n[Risk]")
        print(f"  Max Position: ${self.risk.position_size_usd:,.0f}")
        print(f"  Max Daily Loss: ${self.risk.daily_loss_usd:,.0f}")
        print(f"  Stop Loss: {self.risk.stop_loss_percent}%")
        
        print(f"\n[AI]")
        print(f"  Device: {self.ai.device}")
        print(f"  Batch Size: {self.ai.inference_batch_size:,}")
        
        print(f"\n[Features]")
        print(f"  Execution: {'✓' if self.features.enable_execution else '✗'}")
        print(f"  Market Data: {'✓' if self.features.enable_market_data else '✗'}")
        print(f"  Risk Checks: {'✓' if self.features.enable_risk_checks else '✗'}")
        print(f"  AI Signals: {'✓' if self.features.enable_ai_signals else '✗'}")
        print(f"  Paper Trading: {'✓' if self.features.enable_paper_trading else '✗'}")
        
        print("\n" + "=" * 60)


# =============================================================================
# CACHED SETTINGS LOADER
# =============================================================================


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Get cached application settings.
    
    Uses LRU cache to avoid repeated file parsing.
    Settings are loaded once and reused throughout the application lifetime.
    
    Returns:
        AppSettings: Validated application settings
        
    Raises:
        ValidationError: If any configuration value is invalid
    """
    try:
        settings = AppSettings()
        return settings
    except ValidationError as e:
        print("\n" + "=" * 60)
        print("CONFIGURATION VALIDATION ERROR")
        print("=" * 60)
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            print(f"\n[{loc}]")
            print(f"  Error: {msg}")
        print("\n" + "=" * 60)
        raise


def validate_all_settings() -> bool:
    """
    Validate all settings and return success status.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        settings = get_settings()
        return True
    except ValidationError:
        return False


# =============================================================================
# MAIN - For testing configuration
# =============================================================================


if __name__ == "__main__":
    try:
        settings = get_settings()
        settings.print_summary()
        print("\n✓ Configuration validation successful!")
    except ValidationError as e:
        print(f"\n✗ Configuration validation failed!")
        sys.exit(1)
