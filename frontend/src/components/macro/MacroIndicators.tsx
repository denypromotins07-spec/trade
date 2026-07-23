/**
 * MacroIndicators.tsx - Macro Analytics: Real-time Economic Z-Score Gauges
 * 
 * Displays real-time Z-score gauges for CPI, DXY, and Fed Rates using lightweight
 * SVG paths with CSS variables for instant color transitions without repaints.
 * 
 * Features:
 * - SVG-based gauge rendering (minimal DOM, GPU-accelerated)
 * - CSS variable-driven color transitions for 60FPS updates
 * - Z-score normalization for economic indicators
 * - Cyberpunk-styled radial gauges with neon accents
 * - Memory-efficient animation via CSS transforms
 */

'use client';

import React, { useMemo, useEffect, useState, useCallback } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

interface MacroIndicator {
  name: string;
  value: number;
  zScore: number;
  unit: string;
  description: string;
}

interface MacroIndicatorsProps {
  indicators?: MacroIndicator[];
  updateInterval?: number;
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const DEFAULT_INDICATORS: MacroIndicator[] = [
  {
    name: 'CPI',
    value: 3.2,
    zScore: 1.5,
    unit: '%',
    description: 'Consumer Price Index YoY',
  },
  {
    name: 'DXY',
    value: 103.5,
    zScore: -0.8,
    unit: '',
    description: 'US Dollar Index',
  },
  {
    name: 'Fed Rate',
    value: 5.25,
    zScore: 2.1,
    unit: '%',
    description: 'Federal Funds Rate',
  },
];

// Z-score to color mapping (CSS custom properties)
const ZSCORE_COLORS = {
  extreme_negative: '#ff0088',  // -3 or below
  negative: '#ff6600',          // -2 to -3
  neutral_low: '#ffcc00',       // -1 to -2
  neutral: '#00ff88',           // -1 to 1
  neutral_high: '#ffcc00',      // 1 to 2
  positive: '#ff6600',          // 2 to 3
  extreme_positive: '#ff0088',  // 3 or above
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Converts Z-score to color based on deviation thresholds
 */
const getZScoreColor = (zScore: number): string => {
  if (zScore <= -3) return ZSCORE_COLORS.extreme_negative;
  if (zScore <= -2) return ZSCORE_COLORS.negative;
  if (zScore <= -1) return ZSCORE_COLORS.neutral_low;
  if (zScore <= 1) return ZSCORE_COLORS.neutral;
  if (zScore <= 2) return ZSCORE_COLORS.neutral_high;
  if (zScore <= 3) return ZSCORE_COLORS.positive;
  return ZSCORE_COLORS.extreme_positive;
};

/**
 * Generates realistic mock macro data with random walk
 */
const generateMockIndicators = (prev?: MacroIndicator[]): MacroIndicator[] => {
  const baseData = prev || DEFAULT_INDICATORS;
  
  return baseData.map((indicator) => {
    // Random walk with mean reversion
    const meanReversion = indicator.name === 'DXY' ? 103.5 : 
                          indicator.name === 'CPI' ? 3.0 : 5.25;
    const volatility = indicator.name === 'DXY' ? 0.5 : 
                       indicator.name === 'CPI' ? 0.1 : 0.05;
    
    const change = (Math.random() - 0.5) * volatility;
    const newValue = Math.max(0, indicator.value + change + (meanReversion - indicator.value) * 0.1);
    
    // Calculate Z-score (simplified - in production use rolling statistics)
    const stdDev = volatility * 10;
    const zScore = (newValue - meanReversion) / stdDev;
    
    return {
      ...indicator,
      value: parseFloat(newValue.toFixed(2)),
      zScore: parseFloat(zScore.toFixed(2)),
    };
  });
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Individual Gauge Component - SVG-based radial gauge
 */
interface GaugeProps {
  indicator: MacroIndicator;
  size?: number;
}

const Gauge: React.FC<GaugeProps> = ({ indicator, size = 200 }) => {
  const [displayValue, setDisplayValue] = useState(indicator.value);
  const [displayZScore, setDisplayZScore] = useState(indicator.zScore);
  
  const color = useMemo(() => getZScoreColor(indicator.zScore), [indicator.zScore]);
  
  // Animate value changes smoothly
  useEffect(() => {
    const duration = 500;
    const startTime = Date.now();
    const startValue = displayValue;
    const startZScore = displayZScore;
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      
      setDisplayValue(startValue + (indicator.value - startValue) * eased);
      setDisplayZScore(startZScore + (indicator.zScore - startZScore) * eased);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    requestAnimationFrame(animate);
  }, [indicator.value, indicator.zScore]);
  
  // SVG geometry
  const centerX = size / 2;
  const centerY = size / 2;
  const radius = (size - 40) / 2;
  const strokeWidth = 12;
  
  // Calculate arc path
  const startAngle = -90; // Top position
  const endAngle = startAngle + (displayZScore + 3) * 30; // Map -3 to +3 onto 180 degrees
  const clampedEndAngle = Math.max(-90, Math.min(90, endAngle));
  
  // Convert polar to cartesian coordinates
  const polarToCartesian = (angle: number): { x: number; y: number } => {
    const rad = (angle * Math.PI) / 180;
    return {
      x: centerX + radius * Math.cos(rad),
      y: centerY + radius * Math.sin(rad),
    };
  };
  
  const startPoint = polarToCartesian(startAngle);
  const endPoint = polarToCartesian(clampedEndAngle);
  
  // Create arc path
  const largeArcFlag = clampedEndAngle - startAngle > 180 ? 1 : 0;
  const arcPath = `M ${startPoint.x} ${startPoint.y} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endPoint.x} ${endPoint.y}`;
  
  // Background circle (full arc)
  const backgroundPath = `M ${centerX - radius} ${centerY} A ${radius} ${radius} 0 1 1 ${centerX + radius} ${centerY}`;

  return (
    <div 
      className="relative flex flex-col items-center"
      style={{ 
        '--gauge-color': color,
        transition: '--gauge-color 0.3s ease',
      } as React.CSSProperties}
    >
      {/* Glassmorphism container */}
      <div className="relative p-4 rounded-2xl bg-white/5 backdrop-blur-md border border-white/10 shadow-lg">
        <svg 
          width={size} 
          height={size} 
          className="overflow-visible"
          aria-label={`${indicator.name} gauge showing Z-score of ${displayZScore.toFixed(2)}`}
        >
          {/* Definitions for gradients and filters */}
          <defs>
            <linearGradient id={`glow-${indicator.name}`} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="var(--gauge-color)" stopOpacity="0.8" />
              <stop offset="100%" stopColor="var(--gauge-color)" stopOpacity="0.2" />
            </linearGradient>
            
            <filter id={`glow-filter-${indicator.name}`}>
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          
          {/* Background track */}
          <path
            d={backgroundPath}
            fill="none"
            stroke="#1a1a2e"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          
          {/* Value arc with glow */}
          <path
            d={arcPath}
            fill="none"
            stroke="url(#glow-${indicator.name})"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            filter={`url(#glow-filter-${indicator.name})`}
            style={{
              transition: 'd 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          />
          
          {/* Center label */}
          <text
            x={centerX}
            y={centerY - 10}
            textAnchor="middle"
            className="fill-white font-mono text-xl font-bold"
          >
            {indicator.name}
          </text>
          
          <text
            x={centerX}
            y={centerY + 15}
            textAnchor="middle"
            className="fill-cyan-400 font-mono text-sm"
          >
            {displayValue.toFixed(2)}{indicator.unit}
          </text>
          
          {/* Z-score indicator */}
          <text
            x={centerX}
            y={centerY + 35}
            textAnchor="middle"
            className="font-mono text-xs"
            style={{ fill: 'var(--gauge-color)' }}
          >
            Z: {displayZScore > 0 ? '+' : ''}{displayZScore.toFixed(2)}
          </text>
        </svg>
      </div>
      
      {/* Description tooltip */}
      <div className="mt-2 text-xs text-gray-400 font-mono text-center max-w-[200px]">
        {indicator.description}
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const MacroIndicators: React.FC<MacroIndicatorsProps> = ({
  indicators,
  updateInterval = 3000,
}) => {
  const [data, setData] = useState<MacroIndicator[]>(indicators || DEFAULT_INDICATORS);
  
  // Auto-update with mock data (in production, subscribe to WebSocket)
  useEffect(() => {
    const interval = setInterval(() => {
      setData((prev) => generateMockIndicators(prev));
    }, updateInterval);
    
    return () => clearInterval(interval);
  }, [updateInterval]);

  // Update when external indicators change
  useEffect(() => {
    if (indicators) {
      setData(indicators);
    }
  }, [indicators]);

  return (
    <div className="w-full p-4 rounded-xl bg-gradient-to-br from-[#0a0a12]/90 to-[#12121f]/90 backdrop-blur-sm border border-cyan-900/30">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-cyan-400 font-mono text-sm tracking-wider uppercase">
          📊 Macro Indicators <span className="text-xs opacity-70">| Z-Score Analysis</span>
        </h3>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-xs text-gray-400 font-mono">LIVE</span>
        </div>
      </div>
      
      {/* Gauges Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 justify-items-center">
        {data.map((indicator) => (
          <Gauge 
            key={indicator.name} 
            indicator={indicator}
            size={180}
          />
        ))}
      </div>
      
      {/* Footer info */}
      <div className="mt-6 pt-4 border-t border-white/5">
        <div className="flex justify-between text-xs font-mono text-gray-500">
          <span>Z-Score Range: ±3σ</span>
          <span>Update: {updateInterval}ms</span>
        </div>
      </div>
    </div>
  );
};

export default MacroIndicators;
