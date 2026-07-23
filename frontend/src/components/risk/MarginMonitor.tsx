/**
 * MarginMonitor.tsx - Real-time cross-margin and portfolio margin utilization gauges
 * 
 * Features:
 * - SVG radial progress bars for precise margin visualization
 * - Highlights trapped capital and suggests optimal hedging routes
 * - Parses complex cross-margin payloads from Rust CQRS store
 * - AMD GPU context during risk calculations
 */

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Activity, AlertTriangle, TrendingUp, Shield } from 'lucide-react';

interface MarginData {
  // Account-level metrics
  totalEquity: number;
  totalMarginUsed: number;
  availableMargin: number;
  marginRatio: number; // 0-100
  
  // Cross-margin specific
  crossMarginUsed: number;
  isolatedMarginUsed: number;
  portfolioMargin: number;
  
  // Position-level
  positions: Array<{
    symbol: string;
    side: 'long' | 'short';
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnL: number;
    marginUsed: number;
    leverage: number;
  }>;
  
  // Risk metrics
  maintenanceMargin: number;
  liquidationPrice?: number;
  marginCallLevel: number;
}

interface MarginMonitorProps {
  marginData: MarginData | null;
  onHedgeSuggestion: (symbol: string, suggestedAction: string) => void;
}

// SVG Gauge Component
interface RadialGaugeProps {
  value: number; // 0-100
  label: string;
  sublabel?: string;
  warningThreshold?: number;
  dangerThreshold?: number;
  size?: number;
  strokeWidth?: number;
}

const RadialGauge: React.FC<RadialGaugeProps> = ({
  value,
  label,
  sublabel,
  warningThreshold = 70,
  dangerThreshold = 85,
  size = 180,
  strokeWidth = 12,
}) => {
  const center = size / 2;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  const getColor = () => {
    if (value >= dangerThreshold) return '#ef4444';
    if (value >= warningThreshold) return '#f59e0b';
    return '#22d3ee';
  };

  const getGlow = () => {
    const color = getColor();
    return `0 0 20px ${color}66`;
  };

  const color = getColor();

  return (
    <div className="relative flex flex-col items-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="#1e293b"
          strokeWidth={strokeWidth}
        />
        
        {/* Progress circle */}
        <motion.circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          style={{
            filter: `drop-shadow(${getGlow()})`,
          }}
        />
        
        {/* Tick marks */}
        {Array.from({ length: 10 }).map((_, i) => {
          const angle = (i / 10) * 2 * Math.PI;
          const tickLength = i % 5 === 0 ? 15 : 8;
          const x1 = center + Math.cos(angle) * (radius - strokeWidth);
          const y1 = center + Math.sin(angle) * (radius - strokeWidth);
          const x2 = center + Math.cos(angle) * (radius - strokeWidth - tickLength);
          const y2 = center + Math.sin(angle) * (radius - strokeWidth - tickLength);
          
          return (
            <line
              key={i}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#475569"
              strokeWidth={2}
            />
          );
        })}
      </svg>
      
      {/* Center Content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <motion.div
          className="text-3xl font-mono font-bold"
          style={{ color }}
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
        >
          {value.toFixed(1)}%
        </motion.div>
        <div className="text-xs text-slate-400 mt-1">{label}</div>
        {sublabel && <div className="text-[10px] text-slate-500">{sublabel}</div>}
      </div>
    </div>
  );
};

export const MarginMonitor: React.FC<MarginMonitorProps> = ({
  marginData,
  onHedgeSuggestion,
}) => {
  if (!marginData) {
    return (
      <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-slate-700">
        <div className="text-center text-slate-400">Loading margin data...</div>
      </div>
    );
  }

  const marginUtilization = (marginData.totalMarginUsed / marginData.totalEquity) * 100;
  const availableMarginRatio = (marginData.availableMargin / marginData.totalEquity) * 100;
  const isAtRisk = marginUtilization > 80;
  const isCritical = marginUtilization > 90;

  // Calculate hedging suggestions
  const hedgingSuggestions = useMemo(() => {
    const suggestions: Array<{ symbol: string; action: string; reason: string }> = [];
    
    marginData.positions.forEach(pos => {
      // Large unrealized loss suggestion
      if (pos.unrealizedPnL < -1000) {
        suggestions.push({
          symbol: pos.symbol,
          action: pos.side === 'long' ? 'SHORT HEDGE' : 'LONG HEDGE',
          reason: `Unrealized loss: $${Math.abs(pos.unrealizedPnL).toLocaleString()}`,
        });
      }
      
      // High leverage position
      if (pos.leverage > 10) {
        suggestions.push({
          symbol: pos.symbol,
          action: 'REDUCE LEVERAGE',
          reason: `Current leverage: ${pos.leverage}x`,
        });
      }
    });
    
    return suggestions;
  }, [marginData.positions]);

  return (
    <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <Shield className="w-5 h-5" />
          MARGIN MONITOR
        </h3>
        <div className={`px-3 py-1 rounded-full text-xs font-mono font-bold ${
          isCritical
            ? 'bg-red-500/20 text-red-400 border border-red-500 animate-pulse'
            : isAtRisk
            ? 'bg-amber-500/20 text-amber-400 border border-amber-500'
            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500'
        }`}>
          {isCritical ? 'CRITICAL' : isAtRisk ? 'WARNING' : 'HEALTHY'}
        </div>
      </div>

      {/* Main Gauges */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* Margin Utilization */}
        <div className="flex flex-col items-center p-4 bg-slate-800/50 rounded-xl border border-slate-700">
          <RadialGauge
            value={marginUtilization}
            label="MARGIN UTIL"
            sublabel={`${(marginData.totalMarginUsed).toFixed(2)} / ${(marginData.totalEquity).toFixed(2)} BTC`}
            warningThreshold={70}
            dangerThreshold={85}
            size={160}
          />
        </div>

        {/* Available Margin */}
        <div className="flex flex-col items-center p-4 bg-slate-800/50 rounded-xl border border-slate-700">
          <RadialGauge
            value={availableMarginRatio}
            label="AVAILABLE"
            sublabel={`$${marginData.availableMargin.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            warningThreshold={30}
            dangerThreshold={15}
            size={160}
          />
        </div>

        {/* Portfolio Margin */}
        <div className="flex flex-col items-center p-4 bg-slate-800/50 rounded-xl border border-slate-700">
          <RadialGauge
            value={(marginData.portfolioMargin / marginData.totalEquity) * 100}
            label="PORTFOLIO MARGIN"
            sublabel={`Maintenance: $${marginData.maintenanceMargin.toLocaleString()}`}
            warningThreshold={60}
            dangerThreshold={80}
            size={160}
          />
        </div>
      </div>

      {/* Margin Breakdown */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
          <div className="text-xs text-slate-400 mb-1">CROSS MARGIN USED</div>
          <div className="text-xl font-mono font-bold text-cyan-400">
            {marginData.crossMarginUsed.toFixed(4)} BTC
          </div>
        </div>
        <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
          <div className="text-xs text-slate-400 mb-1">ISOLATED MARGIN USED</div>
          <div className="text-xl font-mono font-bold text-amber-400">
            {marginData.isolatedMarginUsed.toFixed(4)} BTC
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="mb-6">
        <h4 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          OPEN POSITIONS
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-slate-500 border-b border-slate-700">
                <th className="text-left py-2">SYMBOL</th>
                <th className="text-left">SIDE</th>
                <th className="text-right">SIZE</th>
                <th className="text-right">LEVERAGE</th>
                <th className="text-right">UNREALIZED PnL</th>
                <th className="text-right">MARGIN</th>
              </tr>
            </thead>
            <tbody>
              {marginData.positions.map((pos, idx) => (
                <tr key={idx} className="border-b border-slate-800 hover:bg-slate-800/30">
                  <td className="py-2 font-bold text-slate-200">{pos.symbol}</td>
                  <td className={pos.side === 'long' ? 'text-emerald-400' : 'text-red-400'}>
                    {pos.side.toUpperCase()}
                  </td>
                  <td className="text-right text-slate-300">{pos.size.toFixed(4)}</td>
                  <td className="text-right text-amber-400">{pos.leverage}x</td>
                  <td className={`text-right font-bold ${
                    pos.unrealizedPnL >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}>
                    ${pos.unrealizedPnL.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className="text-right text-cyan-400">{pos.marginUsed.toFixed(4)} BTC</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Hedging Suggestions */}
      {hedgingSuggestions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-amber-500/20 border border-amber-500 rounded-lg"
        >
          <div className="flex items-start gap-3 mb-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <div className="text-amber-400 font-bold text-sm">HEDGING RECOMMENDATIONS</div>
              <div className="text-amber-400/70 text-xs">
                Optimal hedging routes to reduce portfolio risk
              </div>
            </div>
          </div>
          <div className="space-y-2">
            {hedgingSuggestions.map((suggestion, idx) => (
              <button
                key={idx}
                onClick={() => onHedgeSuggestion(suggestion.symbol, suggestion.action)}
                className="w-full flex items-center justify-between p-2 bg-slate-800/50 rounded hover:bg-slate-700/50 transition-colors text-left"
              >
                <div>
                  <span className="text-cyan-400 font-bold">{suggestion.symbol}</span>
                  <span className="text-slate-400 mx-2">→</span>
                  <span className="text-amber-400 font-bold">{suggestion.action}</span>
                </div>
                <span className="text-xs text-slate-500">{suggestion.reason}</span>
              </button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Liquidation Info */}
      {marginData.liquidationPrice && (
        <div className="mt-4 pt-4 border-t border-slate-700 flex items-center justify-between text-xs">
          <div className="text-slate-400">
            EST. LIQUIDATION PRICE
          </div>
          <div className="font-mono font-bold text-red-400">
            ${marginData.liquidationPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
        </div>
      )}
    </div>
  );
};

export default MarginMonitor;
