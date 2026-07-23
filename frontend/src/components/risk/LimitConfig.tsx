/**
 * LimitConfig.tsx - Cyberpunk-styled risk limit configuration sliders
 * 
 * Features:
 * - Haptic-feedback style sliders for Max Drawdown, Fractional Kelly, VaR thresholds
 * - Real-time WebSocket updates to Rust core
 * - Visual threshold warnings with neon indicators
 * - AMD GPU load context during risk recalculation
 */

import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, TrendingDown, Shield, AlertTriangle } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface RiskLimits {
  maxDrawdown: number; // Percentage (0-100)
  fractionalKelly: number; // Multiplier (0-2)
  varThreshold: number; // Percentage (0-100)
  positionLimit: number; // BTC amount
  dailyLossLimit: number; // USD amount
}

interface LimitConfigProps {
  currentLimits: RiskLimits;
  onLimitsUpdated: (limits: RiskLimits) => void;
}

interface SliderConfig {
  key: keyof RiskLimits;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  icon: React.ReactNode;
  warningThreshold?: number;
  dangerThreshold?: number;
}

const SLIDER_CONFIGS: SliderConfig[] = [
  {
    key: 'maxDrawdown',
    label: 'MAX DRAWDOWN',
    min: 1,
    max: 50,
    step: 0.5,
    unit: '%',
    icon: <TrendingDown className="w-4 h-4" />,
    warningThreshold: 20,
    dangerThreshold: 30,
  },
  {
    key: 'fractionalKelly',
    label: 'FRACTIONAL KELLY',
    min: 0.1,
    max: 2,
    step: 0.1,
    unit: 'x',
    icon: <Activity className="w-4 h-4" />,
    warningThreshold: 1.5,
    dangerThreshold: 1.8,
  },
  {
    key: 'varThreshold',
    label: 'VaR THRESHOLD',
    min: 1,
    max: 20,
    step: 0.5,
    unit: '%',
    icon: <Shield className="w-4 h-4" />,
    warningThreshold: 10,
    dangerThreshold: 15,
  },
];

export const LimitConfig: React.FC<LimitConfigProps> = ({
  currentLimits,
  onLimitsUpdated,
}) => {
  const [localLimits, setLocalLimits] = useState<RiskLimits>(currentLimits);
  const [pendingChanges, setPendingChanges] = useState<Partial<RiskLimits>>({});
  const [isSynced, setIsSynced] = useState(true);
  
  const { sendMessage, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'open';

  // Sync local state when props change
  useEffect(() => {
    setLocalLimits(currentLimits);
    setPendingChanges({});
    setIsSynced(true);
  }, [currentLimits]);

  const handleSliderChange = useCallback((
    key: keyof RiskLimits,
    value: number
  ) => {
    setLocalLimits(prev => ({ ...prev, [key]: value }));
    setPendingChanges(prev => ({ ...prev, [key]: value }));
    setIsSynced(false);
  }, []);

  // Debounced sync to backend
  useEffect(() => {
    if (Object.keys(pendingChanges).length === 0 || !isConnected) {
      return;
    }

    const timer = setTimeout(() => {
      const payload = {
        type: 'UPDATE_RISK_LIMITS',
        timestamp: Date.now(),
        data: pendingChanges,
        source: 'UI_LIMIT_CONFIG',
      };

      sendMessage(JSON.stringify(payload));
      onLimitsUpdated({ ...localLimits, ...pendingChanges });
      setPendingChanges({});
      setIsSynced(true);
      
      console.log('[LimitConfig] Risk limits updated:', pendingChanges);
    }, 500); // 500ms debounce

    return () => clearTimeout(timer);
  }, [pendingChanges, isConnected, sendMessage, localLimits, onLimitsUpdated]);

  const getSliderColor = (config: SliderConfig, value: number): string => {
    if (config.dangerThreshold && value >= config.dangerThreshold) {
      return '#ef4444'; // Red
    }
    if (config.warningThreshold && value >= config.warningThreshold) {
      return '#f59e0b'; // Amber
    }
    return '#22d3ee'; // Cyan
  };

  const getSliderGlow = (config: SliderConfig, value: number): string => {
    const color = getSliderColor(config, value);
    return `0 0 15px ${color}66`;
  };

  const getValueStatus = (config: SliderConfig, value: number): string => {
    if (config.dangerThreshold && value >= config.dangerThreshold) {
      return 'DANGER';
    }
    if (config.warningThreshold && value >= config.warningThreshold) {
      return 'WARNING';
    }
    return 'NORMAL';
  };

  return (
    <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <Shield className="w-5 h-5" />
          RISK LIMIT CONFIGURATION
        </h3>
        <div className={`px-3 py-1 rounded-full text-xs font-mono font-bold ${
          isSynced 
            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500' 
            : 'bg-amber-500/20 text-amber-400 border border-amber-500 animate-pulse'
        }`}>
          {isSynced ? 'SYNCED' : 'PENDING...'}
        </div>
      </div>

      {/* Sliders Grid */}
      <div className="space-y-6">
        {SLIDER_CONFIGS.map((config) => {
          const value = localLimits[config.key] as number;
          const color = getSliderColor(config, value);
          const status = getValueStatus(config, value);
          
          return (
            <div key={config.key} className="relative">
              {/* Label Row */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-sm font-bold text-slate-300">
                  <span style={{ color }}>{config.icon}</span>
                  <span>{config.label}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span 
                    className="text-xs font-mono px-2 py-0.5 rounded"
                    style={{ 
                      backgroundColor: `${color}20`,
                      color 
                    }}
                  >
                    {status}
                  </span>
                  <span 
                    className="text-lg font-mono font-bold"
                    style={{ color }}
                  >
                    {value.toFixed(config.step < 1 ? 2 : 1)}{config.unit}
                  </span>
                </div>
              </div>

              {/* Slider Container */}
              <div className="relative h-8 flex items-center">
                {/* Track Background */}
                <div className="absolute inset-x-0 h-2 bg-slate-700 rounded-full overflow-hidden">
                  {/* Filled portion */}
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      width: `${((value - config.min) / (config.max - config.min)) * 100}%`,
                      backgroundColor: color,
                      boxShadow: getSliderGlow(config, value),
                    }}
                    layoutId={`fill-${config.key}`}
                    transition={{ duration: 0.1 }}
                  />
                </div>

                {/* Threshold Markers */}
                {config.warningThreshold && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-amber-500 z-10"
                    style={{
                      left: `${((config.warningThreshold - config.min) / (config.max - config.min)) * 100}%`,
                    }}
                  >
                    <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] text-amber-500 whitespace-nowrap">
                      WARN
                    </div>
                  </div>
                )}
                
                {config.dangerThreshold && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10"
                    style={{
                      left: `${((config.dangerThreshold - config.min) / (config.max - config.min)) * 100}%`,
                    }}
                  >
                    <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] text-red-500 whitespace-nowrap">
                      DANGER
                    </div>
                  </div>
                )}

                {/* Hidden Range Input for Interaction */}
                <input
                  type="range"
                  min={config.min}
                  max={config.max}
                  step={config.step}
                  value={value}
                  onChange={(e) => handleSliderChange(config.key, parseFloat(e.target.value))}
                  disabled={!isConnected}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                />

                {/* Custom Thumb (visual only) */}
                <motion.div
                  className="absolute w-6 h-6 rounded-full border-2 bg-slate-900 z-10 pointer-events-none"
                  style={{
                    borderColor: color,
                    boxShadow: getSliderGlow(config, value),
                    left: `calc(${((value - config.min) / (config.max - config.min)) * 100}% - 12px)`,
                  }}
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 0.2 }}
                >
                  <div 
                    className="absolute inset-2 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                </motion.div>
              </div>

              {/* Min/Max Labels */}
              <div className="flex justify-between mt-1 text-[10px] text-slate-500 font-mono">
                <span>{config.min}{config.unit}</span>
                <span>{config.max}{config.unit}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Additional Limits (Numeric Inputs) */}
      <div className="mt-6 pt-6 border-t border-slate-700">
        <h4 className="text-sm font-bold text-slate-400 mb-4">ADDITIONAL LIMITS</h4>
        
        <div className="grid grid-cols-2 gap-4">
          {/* Position Limit */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              POSITION LIMIT (BTC)
            </label>
            <input
              type="number"
              value={localLimits.positionLimit}
              onChange={(e) => handleSliderChange('positionLimit', parseFloat(e.target.value) || 0)}
              className="w-full p-2 bg-slate-800 border border-slate-700 rounded-lg text-white font-mono text-sm focus:border-cyan-400 focus:outline-none"
              step={0.1}
              min={0}
            />
          </div>

          {/* Daily Loss Limit */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              DAILY LOSS LIMIT (USD)
            </label>
            <input
              type="number"
              value={localLimits.dailyLossLimit}
              onChange={(e) => handleSliderChange('dailyLossLimit', parseFloat(e.target.value) || 0)}
              className="w-full p-2 bg-slate-800 border border-slate-700 rounded-lg text-white font-mono text-sm focus:border-cyan-400 focus:outline-none"
              step={100}
              min={0}
            />
          </div>
        </div>
      </div>

      {/* Warning Banner */}
      {(localLimits.maxDrawdown > 30 || localLimits.fractionalKelly > 1.5) && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-amber-500/20 border border-amber-500 rounded-lg flex items-start gap-3"
        >
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-amber-400 font-bold text-sm">HIGH RISK CONFIGURATION DETECTED</div>
            <div className="text-amber-400/70 text-xs mt-1">
              Current settings may expose the portfolio to significant drawdowns. 
              Consider reducing Kelly multiplier or tightening drawdown limits.
            </div>
          </div>
        </motion.div>
      )}

      {/* Connection Status */}
      {!isConnected && (
        <div className="mt-4 p-3 bg-red-500/20 border border-red-500 rounded-lg text-red-400 text-sm text-center">
          ⚠️ DISCONNECTED - Changes will not sync until connection is restored
        </div>
      )}
    </div>
  );
};

export default LimitConfig;
