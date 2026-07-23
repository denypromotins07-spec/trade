'use client';

import React, { memo, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useGlobalStore, selectSystemHealth, selectSystemStatus, shallowSystemStatus } from '@/store/global';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

// ============================================================================
// COLLAPSIBLE SIDEBAR NAVIGATION
// Animated sidebar with real-time system health indicators and RAM gauges
// GPU-accelerated animations for smooth transitions
// ============================================================================

interface NavItem {
  id: string;
  label: string;
  href: string;
  icon: React.ReactNode;
  badge?: 'new' | 'hot' | number;
}

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

// ==========================================================================
// MEMORY GAUGE COMPONENT
// Visual indicator for RAM usage with cyberpunk styling
// ==========================================================================

const MemoryGauge = memo<{ used: number; total: number }>(({ used, total }) => {
  const percentage = useMemo(() => Math.min((used / total) * 100, 100), [used, total]);
  
  const getColor = (pct: number): string => {
    if (pct < 60) return 'text-status-success';
    if (pct < 80) return 'text-status-warning';
    return 'text-status-error animate-pulse-fast';
  };
  
  const segments = 10;
  const activeSegments = Math.floor(percentage / 10);
  
  return (
    <div className="flex flex-col gap-2" aria-label={`Memory usage: ${used.toFixed(0)}MB of ${total}MB`}>
      <div className="flex justify-between text-xs font-mono">
        <span className="text-gray-400">RAM</span>
        <span className={`font-bold ${getColor(percentage)}`}>
          {used.toFixed(0)}MB / {total}MB
        </span>
      </div>
      
      {/* Segmented gauge bar */}
      <div className="flex gap-1 h-2">
        {Array.from({ length: segments }).map((_, i) => (
          <div
            key={i}
            className={`
              flex-1 rounded-sm transition-all duration-300
              ${i < activeSegments 
                ? i < 6 ? 'bg-status-success' 
                  : i < 8 ? 'bg-status-warning' 
                  : 'bg-status-error animate-pulse-fast'
                : 'bg-obsidian-300'
              }
            `}
            style={{ willChange: 'background-color' }}
          />
        ))}
      </div>
      
      {/* Percentage indicator */}
      <div className="text-right text-xs font-mono text-gray-500">
        {percentage.toFixed(1)}% utilized
      </div>
    </div>
  );
});

MemoryGauge.displayName = 'MemoryGauge';

// ==========================================================================
// SYSTEM HEALTH INDICATOR
// Real-time CPU and uptime display
// ==========================================================================

const SystemHealthIndicator = memo<{}>(() => {
  const systemHealth = useGlobalStore(selectSystemHealth);
  
  const cpuColor = useMemo(() => {
    if (systemHealth.cpuUsage < 50) return 'text-status-success';
    if (systemHealth.cpuUsage < 80) return 'text-status-warning';
    return 'text-status-error';
  }, [systemHealth.cpuUsage]);
  
  // Format uptime
  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    }
    if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
  };
  
  return (
    <div className="space-y-3 py-4 border-t border-glass-border">
      {/* CPU Usage */}
      <div className="flex justify-between items-center">
        <span className="text-xs text-gray-400">CPU</span>
        <span className={`text-xs font-mono font-bold ${cpuColor}`}>
          {systemHealth.cpuUsage.toFixed(1)}%
        </span>
      </div>
      
      {/* Progress bar */}
      <div className="relative h-1.5 bg-obsidian-300 rounded-full overflow-hidden">
        <motion.div
          className={`absolute inset-y-0 left-0 rounded-full ${cpuColor.replace('text-', 'bg-')}`}
          initial={{ width: 0 }}
          animate={{ width: `${systemHealth.cpuUsage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          style={{ willChange: 'width' }}
        />
      </div>
      
      {/* Uptime */}
      <div className="flex justify-between items-center">
        <span className="text-xs text-gray-400">Uptime</span>
        <span className="text-xs font-mono text-neon-cyan">
          {formatUptime(systemHealth.uptimeSeconds)}
        </span>
      </div>
    </div>
  );
});

SystemHealthIndicator.displayName = 'SystemHealthIndicator';

// ==========================================================================
// NAVIGATION ITEMS
// Core trading module links
// ==========================================================================

const navItems: NavItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    href: '/',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
    ),
  },
  {
    id: 'orderbook',
    label: 'Order Book',
    href: '/orderbook',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    id: 'strategies',
    label: 'Strategies',
    href: '/strategies',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    badge: 'new',
  },
  {
    id: 'positions',
    label: 'Positions',
    href: '/positions',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
      </svg>
    ),
  },
  {
    id: 'analytics',
    label: 'Analytics',
    href: '/analytics',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    id: 'settings',
    label: 'Settings',
    href: '/settings',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
  },
];

// ==========================================================================
// MAIN SIDEBAR COMPONENT
// ==========================================================================

export const Sidebar = memo<SidebarProps>(function Sidebar({ 
  collapsed = false, 
  onToggle 
}) {
  const pathname = usePathname();
  const systemStatus = useGlobalStore(selectSystemStatus, shallowSystemStatus);
  const systemHealth = useGlobalStore(selectSystemHealth);
  
  // Status indicator color based on system state
  const statusConfig = useMemo(() => {
    switch (systemStatus) {
      case 'RUNNING':
        return { color: 'bg-status-success', glow: 'shadow-[0_0_10px_#00ff9d]', label: 'SYSTEM ACTIVE' };
      case 'STOPPING':
        return { color: 'bg-status-warning', glow: '', label: 'STOPPING...' };
      case 'ERROR':
        return { color: 'bg-status-error', glow: 'animate-pulse-fast', label: 'ERROR' };
      case 'KILLED':
        return { color: 'bg-gray-600', glow: '', label: 'KILLED' };
      default:
        return { color: 'bg-gray-500', glow: '', label: 'IDLE' };
    }
  }, [systemStatus]);
  
  // Animation variants for sidebar
  const sidebarVariants = useMemo(() => ({
    expanded: { 
      width: 280,
      transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] }
    },
    collapsed: { 
      width: 80,
      transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] }
    },
  }), []);
  
  return (
    <motion.aside
      className="fixed left-0 top-0 h-full bg-obsidian-100/90 backdrop-blur-xl border-r border-glass-border z-70"
      variants={sidebarVariants}
      animate={collapsed ? 'collapsed' : 'expanded'}
      initial={false}
      style={{ willChange: 'width' }}
    >
      <div className="flex flex-col h-full p-4">
        {/* Logo / Brand */}
        <div className="flex items-center justify-between mb-8">
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3"
            >
              <div className={`w-3 h-3 rounded-full ${statusConfig.color} ${statusConfig.glow}`} />
              <span className="font-display font-bold text-lg text-gradient">
                NAUTILUS/RAY
              </span>
            </motion.div>
          )}
          
          {/* Toggle button */}
          <button
            onClick={onToggle}
            className="p-2 rounded-lg hover:bg-glass-medium transition-colors group"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <svg 
              className={`w-5 h-5 text-gray-400 group-hover:text-neon-cyan transition-colors ${collapsed ? 'rotate-180' : ''}`}
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>
        </div>
        
        {/* System status badge */}
        {!collapsed && (
          <div className="mb-6 px-3 py-2 rounded bg-obsidian-200 border border-glass-border">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${statusConfig.color} ${statusConfig.glow}`} />
              <span className="text-xs font-mono text-gray-400">{statusConfig.label}</span>
            </div>
          </div>
        )}
        
        {/* Navigation items */}
        <nav className="flex-1 space-y-1">
          <AnimatePresence mode="popLayout">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              
              return (
                <Link key={item.id} href={item.href}>
                  <motion.div
                    className={`
                      relative flex items-center gap-3 px-3 py-3 rounded-lg
                      transition-all duration-200 cursor-pointer group
                      ${isActive 
                        ? 'bg-glass-heavy border border-glass-border' 
                        : 'hover:bg-glass-medium'
                      }
                    `}
                    whileHover={{ x: 4 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {/* Active indicator */}
                    {isActive && (
                      <motion.div
                        className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-neon-cyan rounded-r"
                        layoutId="activeNav"
                        transition={{ duration: 0.2 }}
                      />
                    )}
                    
                    {/* Icon */}
                    <span className={`${isActive ? 'text-neon-cyan' : 'text-gray-400 group-hover:text-neon-cyan'} transition-colors`}>
                      {item.icon}
                    </span>
                    
                    {/* Label */}
                    {!collapsed && (
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className={`flex-1 text-sm font-medium ${isActive ? 'text-white' : 'text-gray-400 group-hover:text-white'}`}
                      >
                        {item.label}
                      </motion.span>
                    )}
                    
                    {/* Badge */}
                    {!collapsed && item.badge && (
                      <span className={`
                        px-2 py-0.5 text-xs font-bold rounded
                        ${item.badge === 'new' ? 'bg-neon-magenta text-white' : ''}
                        ${item.badge === 'hot' ? 'bg-status-warning text-black' : ''}
                        ${typeof item.badge === 'number' ? 'bg-obsidian-300 text-gray-300' : ''}
                      `}>
                        {item.badge}
                      </span>
                    )}
                  </motion.div>
                </Link>
              );
            })}
          </AnimatePresence>
        </nav>
        
        {/* System health section */}
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="pt-4"
          >
            <MemoryGauge used={systemHealth.ramUsageMb} total={systemHealth.ramTotalMb} />
            <SystemHealthIndicator />
          </motion.div>
        )}
      </div>
    </motion.aside>
  );
});

Sidebar.displayName = 'Sidebar';

export default Sidebar;
