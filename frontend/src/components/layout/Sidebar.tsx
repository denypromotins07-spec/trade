'use client';

import React, { memo, useState, useCallback } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { useGlobalStore, selectSystemHealth } from '@/store/global';

/**
 * Navigation Items Configuration
 */
const NAVIGATION_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊', href: '/' },
  { id: 'orderbook', label: 'Order Book', icon: '📖', href: '/orderbook' },
  { id: 'trades', label: 'Trades', icon: '⚡', href: '/trades' },
  { id: 'strategies', label: 'Strategies', icon: '🧠', href: '/strategies' },
  { id: 'positions', label: 'Positions', icon: '📍', href: '/positions' },
  { id: 'analytics', label: 'Analytics', icon: '📈', href: '/analytics' },
  { id: 'settings', label: 'Settings', icon: '⚙️', href: '/settings' },
];

/**
 * Sidebar Component
 * 
 * Features:
 * - Animated, collapsible navigation
 * - Real-time system health indicators
 * - RAM usage gauges
 * - Quick-access links to core trading modules
 * - GPU-accelerated animations via Framer Motion
 */
export const Sidebar = memo(function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  // Subscribe only to system health slice for minimal re-renders
  const systemHealth = useGlobalStore(selectSystemHealth);
  
  /**
   * Toggle sidebar collapse state
   */
  const handleToggle = useCallback(() => {
    setIsCollapsed((prev) => !prev);
  }, []);

  /**
   * Format bytes to human-readable string
   */
  const formatMemory = (mb: number): string => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  };

  /**
   * Get color based on usage percentage
   */
  const getUsageColor = (percentage: number): string => {
    if (percentage < 50) return 'text-neon-green';
    if (percentage < 75) return 'text-neon-amber';
    if (percentage < 90) return 'text-neon-red';
    return 'text-neon-magenta animate-pulse';
  };

  /**
   * Animation variants for sidebar
   */
  const sidebarVariants = {
    expanded: {
      width: 280,
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 30,
      },
    },
    collapsed: {
      width: 72,
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 30,
      },
    },
  };

  return (
    <motion.aside
      className="relative flex flex-col glass-panel-strong border-r border-white/10 z-[var(--z-sidebar)]"
      initial="expanded"
      animate={isCollapsed ? 'collapsed' : 'expanded'}
      variants={sidebarVariants}
      style={{
        willChange: 'width',
        transform: 'translateZ(0)',
      }}
    >
      {/* Logo / Brand */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="flex items-center gap-2"
            >
              <span className="text-2xl">🌀</span>
              <h1 className="font-display font-bold text-lg neon-text-cyan tracking-wider">
                NAUTILUS
              </h1>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Collapse Toggle Button */}
        <button
          onClick={handleToggle}
          className="p-2 rounded-lg hover:bg-white/5 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-neon-cyan/50"
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={isCollapsed ? 'Expand' : 'Collapse'}
        >
          <span className="text-xl">{isCollapsed ? '→' : '←'}</span>
        </button>
      </div>

      {/* System Health Panel */}
      <div className="p-4 border-b border-white/10">
        <AnimatePresence mode="wait">
          {!isCollapsed ? (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="space-y-3"
            >
              <h2 className="text-xs font-mono text-gray-400 uppercase tracking-wider">
                System Health
              </h2>
              
              {/* CPU Usage */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-gray-400">CPU</span>
                  <span className={getUsageColor(systemHealth.cpuUsage)}>
                    {systemHealth.cpuUsage.toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      systemHealth.cpuUsage < 50 ? 'bg-neon-green' :
                      systemHealth.cpuUsage < 75 ? 'bg-neon-amber' :
                      'bg-neon-red'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(systemHealth.cpuUsage, 100)}%` }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                  />
                </div>
              </div>

              {/* RAM Usage */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-gray-400">RAM</span>
                  <span className={getUsageColor((systemHealth.ramUsage / 8192) * 100)}>
                    {formatMemory(systemHealth.ramUsage)}
                  </span>
                </div>
                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-neon-cyan"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min((systemHealth.ramUsage / 8192) * 100, 100)}%` }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                  />
                </div>
              </div>

              {/* GPU Usage (AMD DirectML/ROCm) */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-gray-400">GPU</span>
                  <span className={getUsageColor(systemHealth.gpuUsage)}>
                    {systemHealth.gpuUsage.toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      systemHealth.gpuUsage < 50 ? 'bg-neon-magenta' :
                      systemHealth.gpuUsage < 75 ? 'bg-neon-amber' :
                      'bg-neon-red'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(systemHealth.gpuUsage, 100)}%` }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                  />
                </div>
              </div>

              {/* Backend Status */}
              <div className="flex items-center gap-2 pt-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    systemHealth.backendStatus === 'online' ? 'bg-neon-green animate-pulse' :
                    systemHealth.backendStatus === 'restarting' ? 'bg-neon-amber animate-pulse' :
                    'bg-neon-red'
                  }`}
                />
                <span className="text-xs font-mono text-gray-400 capitalize">
                  {systemHealth.backendStatus}
                </span>
              </div>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              {/* Mini status indicators for collapsed state */}
              <div className="w-2 h-2 rounded-full bg-neon-green animate-pulse" title="Backend Online" />
              <div className="w-2 h-2 rounded-full bg-neon-cyan" title="RAM OK" />
              <div className="w-2 h-2 rounded-full bg-neon-magenta" title="GPU Active" />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 overflow-y-auto p-2 space-y-1">
        {NAVIGATION_ITEMS.map((item) => (
          <Link
            key={item.id}
            href={item.href}
            className="group flex items-center gap-3 px-3 py-2.5 rounded-lg
                       text-gray-300 hover:text-white hover:bg-white/5
                       transition-all duration-200
                       focus:outline-none focus:ring-2 focus:ring-neon-cyan/50"
            title={isCollapsed ? item.label : undefined}
          >
            <span className="text-xl flex-shrink-0">{item.icon}</span>
            
            <AnimatePresence mode="wait">
              {!isCollapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.1 }}
                  className="font-medium text-sm whitespace-nowrap"
                >
                  {item.label}
                </motion.span>
              )}
            </AnimatePresence>

            {/* Hover glow effect */}
            <div
              className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 
                         transition-opacity duration-300 pointer-events-none"
              style={{
                background: 'linear-gradient(90deg, rgba(0, 245, 255, 0.1) 0%, transparent 100%)',
              }}
            />
          </Link>
        ))}
      </nav>

      {/* Footer - Connection Status */}
      <div className="p-4 border-t border-white/10">
        <ConnectionIndicator collapsed={isCollapsed} />
      </div>
    </motion.aside>
  );
});

/**
 * WebSocket Connection Indicator Component
 */
const ConnectionIndicator = memo(function ConnectionIndicator({ collapsed }: { collapsed: boolean }) {
  const isConnected = useGlobalStore((state) => state.wsConnected);
  const attempts = useGlobalStore((state) => state.wsReconnectAttempts);

  return (
    <div className={`flex items-center ${collapsed ? 'justify-center' : 'gap-3'}`}>
      <div
        className={`w-2 h-2 rounded-full ${
          isConnected ? 'bg-neon-green' : 'bg-neon-red animate-pulse'
        }`}
      />
      
      {!collapsed && (
        <div className="flex-1 min-w-0">
          <p className="text-xs font-mono text-gray-400 truncate">
            {isConnected ? 'Connected' : `Reconnecting (${attempts})`}
          </p>
          <p className="text-[10px] text-gray-500 font-mono">
            ws://localhost:8080
          </p>
        </div>
      )}
    </div>
  );
});

export default Sidebar;
