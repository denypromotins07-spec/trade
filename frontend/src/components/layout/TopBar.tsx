'use client';

import React, { memo, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useGlobalStore, selectSystemStatus, selectGpuMetrics, selectLatencyMetrics, selectIsTradingEnabled } from '@/store/global';

// ============================================================================
// STICKY TOP NAVIGATION BAR
// Houses master /START and /KILL toggles, global PnL tickers,
// latency ping visualizers, and AMD DirectML/ROCm GPU metrics
// ============================================================================

interface TopBarProps {
  onSystemStart?: () => void;
  onSystemKill?: () => void;
}

// ==========================================================================
// LATENCY PING VISUALIZER
// Real-time WebSocket and Rust core latency display
// ==========================================================================

const LatencyVisualizer = memo<{}>(() => {
  const latencyMetrics = useGlobalStore(selectLatencyMetrics);
  
  // Determine latency status color
  const getLatencyColor = (ms: number): string => {
    if (ms < 20) return 'text-status-success';
    if (ms < 50) return 'text-status-warning';
    return 'text-status-error';
  };
  
  const getLatencyLabel = (ms: number): string => {
    if (ms < 10) return 'EXCELLENT';
    if (ms < 30) return 'GOOD';
    if (ms < 100) return 'FAIR';
    return 'HIGH';
  };
  
  // Signal strength bars based on latency
  const signalBars = useMemo(() => {
    const maxBars = 5;
    let activeBars = maxBars;
    
    if (latencyMetrics.wsPingMs > 100) activeBars = 1;
    else if (latencyMetrics.wsPingMs > 50) activeBars = 2;
    else if (latencyMetrics.wsPingMs > 30) activeBars = 3;
    else if (latencyMetrics.wsPingMs > 15) activeBars = 4;
    
    return Array.from({ length: maxBars }).map((_, i) => ({
      active: i < activeBars,
      height: `${(i + 1) * 4}px`,
    }));
  }, [latencyMetrics.wsPingMs]);
  
  return (
    <div className="flex items-center gap-6">
      {/* WebSocket Ping */}
      <div className="flex items-center gap-2" aria-label={`WebSocket latency: ${latencyMetrics.wsPingMs}ms`}>
        <div className="flex items-end gap-1 h-4">
          {signalBars.map((bar, i) => (
            <div
              key={i}
              className={`w-1 rounded-sm transition-all duration-300 ${
                bar.active 
                  ? getLatencyColor(latencyMetrics.wsPingMs).replace('text-', 'bg-')
                  : 'bg-obsidian-300'
              }`}
              style={{ 
                height: bar.height,
                willChange: 'background-color',
              }}
            />
          ))}
        </div>
        <span className={`text-xs font-mono font-bold ${getLatencyColor(latencyMetrics.wsPingMs)}`}>
          {latencyMetrics.wsPingMs.toFixed(0)}ms
        </span>
      </div>
      
      {/* Rust Core Latency */}
      <div className="flex items-center gap-2" aria-label={`Rust core latency: ${latencyMetrics.rustCoreLatencyMs}ms`}>
        <svg className="w-4 h-4 text-neon-magenta" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        <span className={`text-xs font-mono ${getLatencyColor(latencyMetrics.rustCoreLatencyMs)}`}>
          {latencyMetrics.rustCoreLatencyMs.toFixed(1)}ms
        </span>
        <span className="text-xs text-gray-500 hidden xl:inline">{getLatencyLabel(latencyMetrics.rustCoreLatencyMs)}</span>
      </div>
    </div>
  );
});

LatencyVisualizer.displayName = 'LatencyVisualizer';

// ==========================================================================
// GPU METRICS DISPLAY
// AMD DirectML/ROCm utilization visualization
// ==========================================================================

const GpuMetricsDisplay = memo<{}>(() => {
  const gpuMetrics = useGlobalStore(selectGpuMetrics);
  
  if (!gpuMetrics) {
    return (
      <div className="flex items-center gap-2 text-gray-500">
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
        </svg>
        <span className="text-xs font-mono">GPU: OFFLINE</span>
      </div>
    );
  }
  
  const getUtilizationColor = (util: number): string => {
    if (util < 50) return 'text-status-success';
    if (util < 80) return 'text-status-warning';
    return 'text-neon-cyan';
  };
  
  // Progress bar segments for GPU utilization
  const segments = 20;
  const activeSegments = Math.floor((gpuMetrics.utilization / 100) * segments);
  
  return (
    <div className="flex items-center gap-4">
      {/* AMD Badge */}
      <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-obsidian-200 border border-glass-border">
        <span className="text-[10px] font-bold text-red-500">AMD</span>
        {gpuMetrics.directmlEnabled && (
          <span className="text-[8px] text-gray-400">DirectML</span>
        )}
        {gpuMetrics.rocmEnabled && (
          <span className="text-[8px] text-gray-400">ROCm</span>
        )}
      </div>
      
      {/* GPU Utilization Bar */}
      <div className="flex flex-col gap-1 min-w-[120px]" aria-label={`GPU utilization: ${gpuMetrics.utilization}%`}>
        <div className="flex justify-between text-xs">
          <span className="text-gray-400">GPU</span>
          <span className={`font-mono font-bold ${getUtilizationColor(gpuMetrics.utilization)}`}>
            {gpuMetrics.utilization.toFixed(0)}%
          </span>
        </div>
        
        <div className="flex gap-0.5 h-1.5">
          {Array.from({ length: segments }).map((_, i) => (
            <div
              key={i}
              className={`flex-1 rounded-sm transition-all duration-200 ${
                i < activeSegments
                  ? i < 10 ? 'bg-status-success'
                    : i < 15 ? 'bg-status-warning'
                    : 'bg-neon-cyan animate-pulse-fast'
                  : 'bg-obsidian-300'
              }`}
              style={{ willChange: 'background-color' }}
            />
          ))}
        </div>
      </div>
      
      {/* GPU Memory & Temp */}
      <div className="hidden lg:flex flex-col gap-1 text-xs font-mono">
        <div className="text-gray-400">
          VRAM: <span className="text-white">{gpuMetrics.memoryUsed.toFixed(0)}MB</span>
        </div>
        <div className="text-gray-400">
          TEMP: <span className={`${gpuMetrics.temperature > 80 ? 'text-status-error' : 'text-status-success'}`}>
            {gpuMetrics.temperature.toFixed(0)}°C
          </span>
        </div>
      </div>
    </div>
  );
});

GpuMetricsDisplay.displayName = 'GpuMetricsDisplay';

// ==========================================================================
// PNL TICKER DISPLAY
// Global profit/loss with animated updates
// ==========================================================================

const PnlTicker = memo<{}>(() => {
  // Mock PnL data - would come from store in real implementation
  const totalPnl = useMemo(() => 12847.53, []);
  const dailyPnl = useMemo(() => 1243.21, []);
  const pnlPercentage = useMemo(() => 4.23, []);
  
  const isPositive = totalPnl >= 0;
  
  return (
    <div className="flex items-center gap-4">
      {/* Total PnL */}
      <div className="flex flex-col">
        <span className="text-[10px] text-gray-500 uppercase tracking-wider">Total P&L</span>
        <motion.span
          key={totalPnl}
          initial={{ color: isPositive ? '#00ff9d' : '#ff2a6d' }}
          animate={{ color: '#ffffff' }}
          transition={{ duration: 0.3 }}
          className={`text-lg font-display font-bold ${isPositive ? 'text-status-success' : 'text-status-error'}`}
        >
          ${Math.abs(totalPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </motion.span>
      </div>
      
      {/* Daily PnL */}
      <div className="hidden md:flex flex-col pl-4 border-l border-glass-border">
        <span className="text-[10px] text-gray-500 uppercase tracking-wider">24h</span>
        <span className={`text-sm font-mono font-bold ${dailyPnl >= 0 ? 'text-status-success' : 'text-status-error'}`}>
          {dailyPnl >= 0 ? '+' : ''}{dailyPnl.toFixed(2)} ({pnlPercentage.toFixed(2)}%)
        </span>
      </div>
    </div>
  );
});

PnlTicker.displayName = 'PnlTicker';

// ==========================================================================
// MASTER CONTROL BUTTONS
// /START and /KILL execution toggles with PowerShell compatibility
// ==========================================================================

const MasterControls = memo<{
  systemStatus: ReturnType<typeof selectSystemStatus>;
  isTradingEnabled: boolean;
  onStart?: () => void;
  onKill?: () => void;
}>(({ systemStatus, isTradingEnabled, onStart, onKill }) => {
  const isRunning = systemStatus === 'RUNNING';
  const isStopping = systemStatus === 'STOPPING';
  
  const handleStart = useCallback(() => {
    console.log('[TopBar] Initiating /START command');
    // In production, this would trigger PowerShell orchestration
    onStart?.();
  }, [onStart]);
  
  const handleKill = useCallback(() => {
    console.log('[TopBar] Initiating /KILL command');
    // In production, this would trigger immediate PowerShell kill
    onKill?.();
  }, [onKill]);
  
  return (
    <div className="flex items-center gap-3">
      {/* START Button */}
      <motion.button
        onClick={handleStart}
        disabled={isRunning || isStopping}
        className={`
          relative px-6 py-2.5 rounded-lg font-display font-bold text-sm
          transition-all duration-200 overflow-hidden
          ${isRunning || isStopping
            ? 'bg-obsidian-300 text-gray-500 cursor-not-allowed'
            : 'bg-status-success text-black hover:shadow-[0_0_20px_rgba(0,255,157,0.5)]'
          }
        `}
        whileHover={!isRunning && !isStopping ? { scale: 1.02 } : {}}
        whileTap={!isRunning && !isStopping ? { scale: 0.98 } : {}}
        aria-label="Start trading system"
      >
        <span className="relative z-10 flex items-center gap-2">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
          /START
        </span>
        
        {/* Glow effect when enabled */}
        {!isRunning && !isStopping && (
          <div className="absolute inset-0 bg-gradient-to-r from-status-success to-emerald-400 opacity-0 hover:opacity-100 transition-opacity" />
        )}
      </motion.button>
      
      {/* KILL Button */}
      <motion.button
        onClick={handleKill}
        disabled={isStopping || systemStatus === 'KILLED'}
        className={`
          relative px-6 py-2.5 rounded-lg font-display font-bold text-sm
          transition-all duration-200 overflow-hidden
          ${isStopping || systemStatus === 'KILLED'
            ? 'bg-obsidian-300 text-gray-500 cursor-not-allowed'
            : 'bg-status-error text-white hover:shadow-[0_0_20px_rgba(255,42,109,0.5)]'
          }
        `}
        whileHover={!isStopping && systemStatus !== 'KILLED' ? { scale: 1.02 } : {}}
        whileTap={!isStopping && systemStatus !== 'KILLED' ? { scale: 0.98 } : {}}
        aria-label="Kill trading system"
      >
        <span className="relative z-10 flex items-center gap-2">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M6 6h12v12H6z" />
          </svg>
          /KILL
        </span>
        
        {/* Pulsing effect when stopping */}
        {isStopping && (
          <motion.div
            className="absolute inset-0 bg-status-error"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
        )}
      </motion.button>
    </div>
  );
});

MasterControls.displayName = 'MasterControls';

// ==========================================================================
// MAIN TOPBAR COMPONENT
// ==========================================================================

export const TopBar = memo<TopBarProps>(function TopBar({ onSystemStart, onSystemKill }) {
  const systemStatus = useGlobalStore(selectSystemStatus);
  const isTradingEnabled = useGlobalStore(selectIsTradingEnabled);
  
  // Handle system start command
  const handleStart = useCallback(() => {
    // This would integrate with PowerShell orchestration backend
    console.log('[TopBar] /START command triggered');
    onSystemStart?.();
  }, [onSystemStart]);
  
  // Handle system kill command
  const handleKill = useCallback(() => {
    // This would integrate with PowerShell orchestration backend
    console.log('[TopBar] /KILL command triggered');
    onSystemKill?.();
  }, [onSystemKill]);
  
  return (
    <motion.header
      className="fixed top-0 left-0 right-0 h-16 bg-obsidian-100/95 backdrop-blur-xl border-b border-glass-border z-60"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      style={{ willChange: 'transform' }}
    >
      <div className="h-full px-6 flex items-center justify-between">
        {/* Left Section - Controls */}
        <div className="flex items-center gap-6">
          <MasterControls
            systemStatus={systemStatus}
            isTradingEnabled={isTradingEnabled}
            onStart={handleStart}
            onKill={handleKill}
          />
        </div>
        
        {/* Center Section - PnL Ticker */}
        <div className="absolute left-1/2 -translate-x-1/2">
          <PnlTicker />
        </div>
        
        {/* Right Section - Metrics */}
        <div className="flex items-center gap-6">
          <GpuMetricsDisplay />
          <LatencyVisualizer />
          
          {/* Connection Status Dot */}
          <div className="hidden sm:flex items-center gap-2 pl-4 border-l border-glass-border">
            <div className={`w-2 h-2 rounded-full ${
              systemStatus === 'RUNNING' 
                ? 'bg-status-success shadow-[0_0_8px_#00ff9d]' 
                : systemStatus === 'ERROR'
                ? 'bg-status-error animate-pulse-fast'
                : 'bg-gray-500'
            }`} />
            <span className="text-xs font-mono text-gray-400">
              {systemStatus}
            </span>
          </div>
        </div>
      </div>
    </motion.header>
  );
});

TopBar.displayName = 'TopBar';

export default TopBar;
