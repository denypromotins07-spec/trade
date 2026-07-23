/**
 * CircuitBreakerPanel.tsx - Hard and soft circuit breaker toggle matrix
 * 
 * Features:
 * - Visual toggle matrix for circuit breakers
 * - Real-time proximity alerts when PnL approaches auto-deleveraging triggers
 * - Cyberpunk aesthetic with neon status indicators
 * - AMD GPU context during high-frequency monitoring
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, AlertTriangle, Shield, Activity, TrendingDown } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface CircuitBreaker {
  id: string;
  name: string;
  type: 'hard' | 'soft';
  enabled: boolean;
  threshold: number; // Percentage or USD
  currentValue: number;
  unit: string;
  description: string;
}

interface CircuitBreakerPanelProps {
  currentPnL: number;
  dailyPnL: number;
  maxDrawdown: number;
  onBreakerTriggered: (breakerId: string) => void;
}

const INITIAL_BREAKERS: CircuitBreaker[] = [
  {
    id: 'daily_loss',
    name: 'DAILY LOSS LIMIT',
    type: 'hard',
    enabled: true,
    threshold: -5000,
    currentValue: 0,
    unit: 'USD',
    description: 'Stops trading if daily PnL falls below threshold',
  },
  {
    id: 'max_drawdown',
    name: 'MAX DRAWDOWN',
    type: 'hard',
    enabled: true,
    threshold: -15,
    currentValue: 0,
    unit: '%',
    description: 'Halts all positions if drawdown exceeds limit',
  },
  {
    id: 'volatility_spike',
    name: 'VOLATILITY SPIKE',
    type: 'soft',
    enabled: true,
    threshold: 5,
    currentValue: 0,
    unit: '%',
    description: 'Reduces position size during extreme volatility',
  },
  {
    id: 'consecutive_losses',
    name: 'CONSECUTIVE LOSSES',
    type: 'soft',
    enabled: false,
    threshold: 5,
    currentValue: 0,
    unit: 'trades',
    description: 'Pauses after N consecutive losing trades',
  },
  {
    id: 'liquidity_dry',
    name: 'LIQUIDITY DRY-UP',
    type: 'hard',
    enabled: true,
    threshold: 50,
    currentValue: 0,
    unit: '%',
    description: 'Stops if orderbook depth drops below threshold',
  },
  {
    id: 'latency_threshold',
    name: 'LATENCY CIRCUIT',
    type: 'soft',
    enabled: true,
    threshold: 100,
    currentValue: 0,
    unit: 'ms',
    description: 'Disables HFT mode if latency exceeds limit',
  },
];

export const CircuitBreakerPanel: React.FC<CircuitBreakerPanelProps> = ({
  currentPnL,
  dailyPnL,
  maxDrawdown,
  onBreakerTriggered,
}) => {
  const [breakers, setBreakers] = useState<CircuitBreaker[]>(INITIAL_BREAKERS);
  const [triggeredBreakers, setTriggeredBreakers] = useState<Set<string>>(new Set());
  const [proximityAlerts, setProximityAlerts] = useState<Map<string, number>>(new Map());
  
  const { sendMessage, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'open';

  // Update breaker values based on live data
  useEffect(() => {
    setBreakers(prev => prev.map(breaker => {
      switch (breaker.id) {
        case 'daily_loss':
          return { ...breaker, currentValue: dailyPnL };
        case 'max_drawdown':
          return { ...breaker, currentValue: maxDrawdown };
        default:
          return breaker;
      }
    }));
  }, [dailyPnL, maxDrawdown]);

  // Check for triggered breakers and proximity alerts
  useEffect(() => {
    const newTriggered = new Set<string>();
    const newProximity = new Map<string, number>();

    breakers.forEach(breaker => {
      if (!breaker.enabled) return;

      let isTriggered = false;
      let proximity = 100; // Percentage to threshold

      switch (breaker.id) {
        case 'daily_loss':
          if (breaker.currentValue <= breaker.threshold) {
            isTriggered = true;
          } else if (breaker.threshold < 0) {
            proximity = Math.min(100, Math.abs((breaker.currentValue / breaker.threshold) * 100));
          }
          break;
        case 'max_drawdown':
          if (breaker.currentValue <= breaker.threshold) {
            isTriggered = true;
          } else {
            proximity = Math.min(100, Math.abs((breaker.currentValue / breaker.threshold) * 100));
          }
          break;
        default:
          break;
      }

      if (isTriggered && !triggeredBreakers.has(breaker.id)) {
        newTriggered.add(breaker.id);
        onBreakerTriggered(breaker.id);
        
        // Send notification to backend
        const payload = {
          type: 'CIRCUIT_BREAKER_TRIGGERED',
          timestamp: Date.now(),
          data: {
            breakerId: breaker.id,
            breakerType: breaker.type,
            threshold: breaker.threshold,
            currentValue: breaker.currentValue,
          },
        };
        sendMessage(JSON.stringify(payload));
      }

      // Proximity alert at 80%
      if (proximity >= 80 && proximity < 100) {
        newProximity.set(breaker.id, proximity);
      }
    });

    setTriggeredBreakers(newTriggered);
    setProximityAlerts(newProximity);
  }, [breakers, triggeredBreakers, onBreakerTriggered, sendMessage]);

  const toggleBreaker = useCallback((breakerId: string) => {
    setBreakers(prev => prev.map(b => 
      b.id === breakerId ? { ...b, enabled: !b.enabled } : b
    ));
  }, []);

  const getBreakerStatus = (breaker: CircuitBreaker): 'normal' | 'warning' | 'danger' | 'triggered' => {
    if (!breaker.enabled) return 'normal';
    if (triggeredBreakers.has(breaker.id)) return 'triggered';
    
    const proximity = proximityAlerts.get(breaker.id);
    if (proximity && proximity >= 90) return 'danger';
    if (proximity && proximity >= 80) return 'warning';
    
    return 'normal';
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'triggered': return '#ef4444';
      case 'danger': return '#f97316';
      case 'warning': return '#eab308';
      default: return '#22d3ee';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'triggered': return <Zap className="w-5 h-5 animate-pulse" />;
      case 'danger': return <AlertTriangle className="w-5 h-5" />;
      case 'warning': return <Activity className="w-5 h-5" />;
      default: return <Shield className="w-5 h-5" />;
    }
  };

  return (
    <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <Shield className="w-5 h-5" />
          CIRCUIT BREAKER MATRIX
        </h3>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-emerald-400">● ACTIVE: {breakers.filter(b => b.enabled).length}</span>
          <span className="text-red-400">● TRIGGERED: {triggeredBreakers.size}</span>
        </div>
      </div>

      {/* Breaker Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {breakers.map((breaker) => {
          const status = getBreakerStatus(breaker);
          const color = getStatusColor(status);
          const proximity = proximityAlerts.get(breaker.id);

          return (
            <motion.div
              key={breaker.id}
              layout
              className={`relative p-4 rounded-lg border-2 overflow-hidden ${
                status === 'triggered'
                  ? 'bg-red-500/20 border-red-500'
                  : status === 'danger'
                  ? 'bg-orange-500/20 border-orange-500'
                  : status === 'warning'
                  ? 'bg-amber-500/20 border-amber-500'
                  : 'bg-slate-800/50 border-slate-700'
              }`}
              style={{
                boxShadow: status !== 'normal' ? `0 0 20px ${color}44` : undefined,
              }}
            >
              {/* Type Badge */}
              <div className="absolute top-2 right-2">
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  breaker.type === 'hard'
                    ? 'bg-red-500/30 text-red-400'
                    : 'bg-amber-500/30 text-amber-400'
                }`}>
                  {breaker.type.toUpperCase()}
                </span>
              </div>

              {/* Status Icon */}
              <div className="mb-3" style={{ color }}>
                {getStatusIcon(status)}
              </div>

              {/* Name */}
              <div className="font-bold text-sm text-slate-200 mb-1">
                {breaker.name}
              </div>

              {/* Description */}
              <div className="text-xs text-slate-400 mb-3">
                {breaker.description}
              </div>

              {/* Value Display */}
              <div className="flex items-end justify-between mb-3">
                <div>
                  <div className="text-[10px] text-slate-500">CURRENT</div>
                  <div 
                    className="text-lg font-mono font-bold"
                    style={{ color }}
                  >
                    {breaker.currentValue.toFixed(2)}{breaker.unit}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">THRESHOLD</div>
                  <div className="text-sm font-mono text-slate-300">
                    {breaker.threshold}{breaker.unit}
                  </div>
                </div>
              </div>

              {/* Proximity Bar */}
              {proximity !== undefined && (
                <div className="mb-3">
                  <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                    <span>PROXIMITY TO TRIGGER</span>
                    <span>{proximity.toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${proximity}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                </div>
              )}

              {/* Toggle Switch */}
              <button
                onClick={() => toggleBreaker(breaker.id)}
                disabled={!isConnected}
                className={`w-full py-2 rounded font-bold text-xs transition-all ${
                  breaker.enabled
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500 hover:bg-emerald-500/30'
                    : 'bg-slate-700 text-slate-400 border border-slate-600 hover:bg-slate-600'
                }`}
              >
                {breaker.enabled ? 'ENABLED' : 'DISABLED'}
              </button>

              {/* Triggered Overlay */}
              <AnimatePresence>
                {status === 'triggered' && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-red-500/10 pointer-events-none"
                  >
                    <div className="absolute inset-0 flex items-center justify-center">
                      <motion.div
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                        className="text-red-400 font-bold text-lg drop-shadow-lg"
                      >
                        TRIGGERED
                      </motion.div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Global Status Footer */}
      <div className="mt-6 pt-4 border-t border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-emerald-400" />
              <span className="text-slate-400">NORMAL</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-amber-400" />
              <span className="text-slate-400">WARNING (&gt;80%)</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-orange-400" />
              <span className="text-slate-400">DANGER (&gt;90%)</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-red-400 animate-pulse" />
              <span className="text-slate-400">TRIGGERED</span>
            </div>
          </div>
          
          <div className={`text-xs font-mono ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
            {isConnected ? 'WS CONNECTED' : 'WS DISCONNECTED'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CircuitBreakerPanel;
