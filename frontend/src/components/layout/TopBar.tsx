'use client';

import React, { memo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useGlobalStore, selectMasterControl, selectSystemHealth } from '@/store/global';

/**
 * TopBar Component
 * 
 * Features:
 * - Sticky top navigation bar
 * - Master /START and /KILL execution toggles (PowerShell orchestration compatible)
 * - Global PnL tickers
 * - Latency ping visualizers connected to Rust core
 * - GPU utilization metrics (AMD DirectML/ROCm context)
 * - Hardware-accelerated animations
 */
export const TopBar = memo(function TopBar() {
  // Zustand selectors with shallow equality for optimal performance
  const masterControl = useGlobalStore(selectMasterControl);
  const systemHealth = useGlobalStore(selectSystemHealth);
  
  // Store actions
  const setMasterStart = useGlobalStore((state) => state.setMasterStart);
  const setMasterKill = useGlobalStore((state) => state.setMasterKill);

  /**
   * Handle START command - triggers PowerShell orchestration
   */
  const handleStart = useCallback(() => {
    console.log('[TopBar] /START command issued');
    setMasterStart();
    // In production, this would trigger a fetch/POST to the backend
    // to execute the PowerShell /START script
  }, [setMasterStart]);

  /**
   * Handle KILL command - triggers PowerShell orchestration
   */
  const handleKill = useCallback(() => {
    console.log('[TopBar] /KILL command issued');
    setMasterKill();
    // In production, this would trigger a fetch/POST to the backend
    // to execute the PowerShell /KILL script
  }, [setMasterKill]);

  /**
   * Format latency display with color coding
   */
  const getLatencyColor = (latency: number): string => {
    if (latency < 10) return 'text-neon-green';
    if (latency < 50) return 'text-neon-cyan';
    if (latency < 100) return 'text-neon-amber';
    return 'text-neon-red';
  };

  /**
   * Format GPU memory display
   */
  const formatGpuMemory = (mb: number): string => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  };

  /**
   * Calculate global PnL (simulated - would come from backend)
   */
  const globalPnl = useGlobalStore((state) => 
    state.strategies.reduce((sum, s) => sum + s.pnl, 0)
  );

  const pnlColor = globalPnl >= 0 ? 'text-neon-green' : 'text-neon-red';
  const pnlSign = globalPnl >= 0 ? '+' : '';

  return (
    <motion.header
      className="fixed top-0 left-0 right-0 h-16 glass-panel-strong border-b border-white/10 z-[var(--z-topbar)]"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{
        type: 'spring',
        stiffness: 300,
        damping: 30,
      }}
      style={{
        willChange: 'transform',
        transform: 'translateZ(0)',
      }}
    >
      <div className="flex items-center justify-between h-full px-6">
        {/* Left Section - Master Controls */}
        <div className="flex items-center gap-4">
          <h2 className="font-display font-bold text-xl tracking-wider text-gray-200">
            RAY<span className="neon-text-cyan">.BOT</span>
          </h2>
          
          <div className="h-8 w-px bg-white/10" />
          
          {/* Master Control Buttons */}
          <div className="flex items-center gap-2">
            <MasterButton
              variant="start"
              isActive={masterControl.isRunning && !masterControl.isKilled}
              onClick={handleStart}
              disabled={masterControl.isRunning && !masterControl.isKilled}
            />
            <MasterButton
              variant="kill"
              isActive={masterControl.isKilled}
              onClick={handleKill}
              disabled={masterControl.isKilled}
            />
          </div>
        </div>

        {/* Center Section - Global PnL & Market Status */}
        <div className="flex items-center gap-6">
          {/* Global PnL Ticker */}
          <div className="flex flex-col items-center">
            <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">
              Global PnL
            </span>
            <motion.span
              key={globalPnl}
              initial={{ scale: 1.1 }}
              animate={{ scale: 1 }}
              className={`text-lg font-mono font-bold ${pnlColor}`}
              style={{
                textShadow: `0 0 10px ${
                  globalPnl >= 0 ? 'rgba(0, 255, 136, 0.5)' : 'rgba(255, 51, 102, 0.5)'
                }`,
              }}
            >
              {pnlSign}${Math.abs(globalPnl).toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </motion.span>
          </div>

          {/* Active Strategies Count */}
          <div className="flex flex-col items-center">
            <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">
              Strategies
            </span>
            <span className="text-lg font-mono font-bold text-neon-cyan">
              {useGlobalStore((state) => state.strategies.filter(s => s.status === 'active').length)}
            </span>
          </div>
        </div>

        {/* Right Section - System Metrics */}
        <div className="flex items-center gap-4">
          {/* Network Latency */}
          <LatencyIndicator latency={systemHealth.networkLatency} />

          {/* GPU Utilization (AMD DirectML/ROCm) */}
          <GpuMetrics gpuUsage={systemHealth.gpuUsage} gpuMemory={systemHealth.gpuMemory} />

          {/* Uptime */}
          <UptimeDisplay uptime={systemHealth.uptime} />
        </div>
      </div>

      {/* Decorative bottom glow line */}
      <div
        className="absolute bottom-0 left-0 right-0 h-px"
        style={{
          background: 'linear-gradient(90deg, transparent 0%, rgba(0, 245, 255, 0.5) 50%, transparent 100%)',
        }}
      />
    </motion.header>
  );
});

/**
 * Master Control Button Component
 */
interface MasterButtonProps {
  variant: 'start' | 'kill';
  isActive: boolean;
  onClick: () => void;
  disabled: boolean;
}

const MasterButton = memo(function MasterButton({
  variant,
  isActive,
  onClick,
  disabled,
}: MasterButtonProps) {
  const isStart = variant === 'start';
  
  const baseStyles = `
    relative px-6 py-2 rounded-lg font-mono font-bold text-sm uppercase tracking-wider
    transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-obsidian-50
    disabled:opacity-50 disabled:cursor-not-allowed
  `;
  
  const startStyles = isActive
    ? 'bg-neon-green/20 text-neon-green border border-neon-green shadow-neon-green'
    : 'bg-white/5 text-gray-300 border border-white/10 hover:border-neon-green hover:text-neon-green hover:shadow-neon-green';
  
  const killStyles = isActive
    ? 'bg-neon-red/20 text-neon-red border border-neon-red shadow-neon-red'
    : 'bg-white/5 text-gray-300 border border-white/10 hover:border-neon-red hover:text-neon-red hover:shadow-neon-red';

  return (
    <motion.button
      className={`${baseStyles} ${isStart ? startStyles : killStyles}`}
      onClick={onClick}
      disabled={disabled}
      whileHover={!disabled ? { scale: 1.05 } : {}}
      whileTap={!disabled ? { scale: 0.95 } : {}}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
    >
      {/* Glow effect when active */}
      {isActive && (
        <motion.div
          className="absolute inset-0 rounded-lg"
          style={{
            background: `radial-gradient(circle, ${
              isStart ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 51, 102, 0.2)'
            } 0%, transparent 70%)`,
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        />
      )}
      
      <span className="relative z-10 flex items-center gap-2">
        {isStart ? '▶' : '⏹'} {isStart ? '/START' : '/KILL'}
      </span>
    </motion.button>
  );
});

/**
 * Network Latency Indicator with Ping Animation
 */
const LatencyIndicator = memo(function LatencyIndicator({ latency }: { latency: number }) {
  const latencyColor = getLatencyColor(latency);

  return (
    <div className="flex items-center gap-2">
      {/* Animated ping dots */}
      <div className="relative flex items-center gap-1">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className={`w-1.5 h-1.5 rounded-full ${latencyColor}`}
            initial={{ opacity: 0.3 }}
            animate={{
              opacity: [0.3, 1, 0.3],
              scale: [0.8, 1.2, 0.8],
            }}
            transition={{
              duration: 1,
              delay: i * 0.2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        ))}
      </div>
      
      <div className="flex flex-col">
        <span className="text-[10px] font-mono text-gray-500 uppercase">Latency</span>
        <span className={`text-sm font-mono font-bold ${latencyColor}`}>
          {latency.toFixed(0)}ms
        </span>
      </div>
    </div>
  );
});

/**
 * GPU Metrics Display (AMD DirectML/ROCm Context)
 */
const GpuMetrics = memo(function GpuMetrics({
  gpuUsage,
  gpuMemory,
}: {
  gpuUsage: number;
  gpuMemory: number;
}) {
  const getGpuColor = (usage: number): string => {
    if (usage < 50) return 'text-neon-magenta';
    if (usage < 75) return 'text-neon-amber';
    return 'text-neon-red';
  };

  const gpuColor = getGpuColor(gpuUsage);

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10">
      {/* GPU Icon */}
      <div className="relative">
        <span className="text-lg">🖥️</span>
        {gpuUsage > 80 && (
          <motion.div
            className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-neon-red"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 0.5, repeat: Infinity }}
          />
        )}
      </div>
      
      <div className="flex flex-col min-w-[60px]">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-gray-500 uppercase">GPU</span>
          <span className={`text-xs font-mono font-bold ${gpuColor}`}>
            {gpuUsage.toFixed(0)}%
          </span>
        </div>
        <span className="text-[10px] font-mono text-gray-400">
          {formatGpuMemory(gpuMemory)} VRAM
        </span>
      </div>
      
      {/* Mini GPU usage bar */}
      <div className="w-12 h-1 bg-white/10 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${
            gpuUsage < 50 ? 'bg-neon-magenta' :
            gpuUsage < 75 ? 'bg-neon-amber' :
            'bg-neon-red'
          }`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(gpuUsage, 100)}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </div>
  );
});

/**
 * Uptime Display Component
 */
const UptimeDisplay = memo(function UptimeDisplay({ uptime }: { uptime: number }) {
  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
  };

  return (
    <div className="flex flex-col items-end">
      <span className="text-[10px] font-mono text-gray-500 uppercase">Uptime</span>
      <span className="text-sm font-mono font-bold text-gray-300">
        {formatUptime(uptime)}
      </span>
    </div>
  );
});

export default TopBar;
