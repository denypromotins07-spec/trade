/**
 * File 2: RateLimitGauge.tsx
 * Chapter 1: Network Topology & API Rate Limit UI
 * 
 * Binance UID and IP weight visualizers using fluid SVG radial progress bars
 * that turn neon red when approaching the 1200/6000 strict limits.
 * 
 * Features:
 * - Atomic weight counter parsing from Rust backend
 * - Dynamic color interpolation (Green -> Yellow -> Red)
 * - SVG-based rendering for crisp scaling
 * - Cyberpunk aesthetic with glow effects
 */

import React, { useEffect, useRef, useMemo } from 'react';

// --- Types ---

interface RateLimitData {
  uidWeight: number;      // Current UID weight (max 1200 per sec)
  ipWeight: number;       // Current IP weight (max 6000 per sec)
  uidMax: number;         // Usually 1200
  ipMax: number;          // Usually 6000
  resetTimeMs: number;    // Time until reset
  isThrottled: boolean;   // Backend flag if currently throttled
}

interface Props {
  data: RateLimitData;
  size?: number;
  strokeWidth?: number;
  showDetails?: boolean;
}

// --- Constants ---

const COLORS = {
  safe: '#00ff9d',      // Green
  warning: '#ffaa00',   // Orange/Yellow
  danger: '#ff0055',    // Neon Red
  bg: '#1a1a1a',
  text: '#ffffff',
  grid: '#333333',
};

/**
 * Interpolates color based on ratio (0-1)
 * 0.0-0.5: Green -> Yellow
 * 0.5-1.0: Yellow -> Red
 */
const getColorForRatio = (ratio: number): string => {
  if (ratio < 0.5) {
    // Green to Yellow
    const t = ratio * 2; // 0 to 1
    const r = Math.floor(0 + t * 255);
    const g = Math.floor(255 - t * 100); // 255 -> 155
    return `rgb(${r}, ${g}, 0)`;
  } else {
    // Yellow to Red
    const t = (ratio - 0.5) * 2; // 0 to 1
    const r = 255;
    const g = Math.floor(170 - t * 170); // 170 -> 0
    return `rgb(${r}, ${g}, 0)`;
  }
};

/**
 * Draws a radial progress gauge using SVG
 */
const RadialGauge: React.FC<{
  value: number;
  max: number;
  label: string;
  subLabel?: string;
  size: number;
  strokeWidth: number;
  colorOverride?: string;
}> = ({ value, max, label, subLabel, size, strokeWidth, colorOverride }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const ratio = Math.min(value / max, 1);
  const dashOffset = circumference * (1 - ratio);
  
  const color = colorOverride || getColorForRatio(ratio);
  
  // Generate tick marks
  const ticks = useMemo(() => {
    const tickElements = [];
    const totalTicks = 12;
    for (let i = 0; i < totalTicks; i++) {
      const angle = (i / totalTicks) * 2 * Math.PI - Math.PI / 2;
      const x1 = size / 2 + Math.cos(angle) * (radius - strokeWidth);
      const y1 = size / 2 + Math.sin(angle) * (radius - strokeWidth);
      const x2 = size / 2 + Math.cos(angle) * radius;
      const y2 = size / 2 + Math.sin(angle) * radius;
      const isWarning = i >= totalTicks * ratio;
      
      tickElements.push(
        <line
          key={i}
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke={isWarning ? '#444' : color}
          strokeWidth="2"
          opacity={isWarning ? 0.3 : 1}
        />
      );
    }
    return tickElements;
  }, [ratio, radius, size, strokeWidth, color]);

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg
        width={size}
        height={size}
        className="transform -rotate-90 drop-shadow-[0_0_8px_currentColor]"
        style={{ color }}
      >
        {/* Background Circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={COLORS.bg}
          strokeWidth={strokeWidth}
        />
        
        {/* Tick Marks */}
        {ticks}
        
        {/* Progress Arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          className="transition-all duration-300 ease-out"
          style={{
            filter: `drop-shadow(0 0 6px ${color})`,
          }}
        />
        
        {/* Center Text (rotated back) */}
        <g className="transform rotate-90" style={{ transformOrigin: `${size/2}px ${size/2}px` }}>
          <text
            x="50%"
            y="45%"
            textAnchor="middle"
            className="fill-white font-mono font-bold"
            style={{ fontSize: `${size * 0.12}px` }}
          >
            {Math.round(ratio * 100)}%
          </text>
          <text
            x="50%"
            y="65%"
            textAnchor="middle"
            className="fill-gray-400 font-mono"
            style={{ fontSize: `${size * 0.06}px` }}
          >
            {label}
          </text>
        </g>
      </svg>
      
      {subLabel && (
        <div className="absolute bottom-[-20px] text-[10px] font-mono text-cyan-400 whitespace-nowrap">
          {subLabel}
        </div>
      )}
    </div>
  );
};

/**
 * RateLimitGauge Component
 * Displays dual gauges for UID and IP rate limits with cyberpunk styling.
 */
export const RateLimitGauge: React.FC<Props> = ({
  data,
  size = 200,
  strokeWidth = 12,
  showDetails = true,
}) => {
  const gaugeSize = size < 300 ? size / 2.5 : 120;
  
  // Calculate time until reset
  const resetSeconds = Math.max(0, (data.resetTimeMs / 1000)).toFixed(1);
  
  // Determine overall status color
  const uidRatio = data.uidWeight / data.uidMax;
  const ipRatio = data.ipWeight / data.ipMax;
  const maxRatio = Math.max(uidRatio, ipRatio);
  const statusColor = getColorForRatio(maxRatio);

  return (
    <div 
      className="relative p-4 bg-black/80 backdrop-blur-md border rounded-xl overflow-hidden"
      style={{ 
        borderColor: statusColor,
        boxShadow: `0 0 20px ${statusColor}20`, // 20 in hex is ~12% opacity
      }}
    >
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-mono font-bold text-white tracking-wider">
          API_RATE_LIMITS
        </h3>
        <div 
          className="px-2 py-0.5 rounded text-[10px] font-mono font-bold"
          style={{ 
            backgroundColor: data.isThrottled ? COLORS.danger : 'transparent',
            color: data.isThrottled ? '#fff' : statusColor,
            border: `1px solid ${statusColor}`,
          }}
        >
          {data.isThrottled ? 'THROTTLED' : 'ACTIVE'}
        </div>
      </div>

      {/* Gauges Container */}
      <div className="flex flex-row items-center justify-around gap-4">
        {/* UID Gauge */}
        <RadialGauge
          value={data.uidWeight}
          max={data.uidMax}
          label="UID WT"
          subLabel={`${data.uidWeight}/${data.uidMax}`}
          size={gaugeSize}
          strokeWidth={strokeWidth / 1.5}
        />
        
        {/* IP Gauge */}
        <RadialGauge
          value={data.ipWeight}
          max={data.ipMax}
          label="IP WT"
          subLabel={`${data.ipWeight}/${data.ipMax}`}
          size={gaugeSize}
          strokeWidth={strokeWidth / 1.5}
        />
      </div>

      {/* Details Panel */}
      {showDetails && (
        <div className="mt-6 grid grid-cols-2 gap-2 text-[10px] font-mono">
          <div className="flex justify-between text-gray-400">
            <span>RESET_IN:</span>
            <span style={{ color: statusColor }}>{resetSeconds}s</span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>STATUS:</span>
            <span style={{ color: data.isThrottled ? COLORS.danger : COLORS.safe }}>
              {data.isThrottled ? 'BLOCKED' : 'OK'}
            </span>
          </div>
          
          {/* Atomic Counter Visualization */}
          <div className="col-span-2 mt-2">
            <div className="text-[9px] text-gray-500 mb-1">ATOMIC_WEIGHT_BUFFER</div>
            <div className="h-1.5 w-full bg-gray-900 rounded-full overflow-hidden">
              <div 
                className="h-full transition-all duration-200"
                style={{ 
                  width: `${maxRatio * 100}%`,
                  backgroundColor: statusColor,
                  boxShadow: `0 0 8px ${statusColor}`,
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Decorative Corners (Cyberpunk) */}
      <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2" style={{ borderColor: statusColor }} />
      <div className="absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2" style={{ borderColor: statusColor }} />
      <div className="absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2" style={{ borderColor: statusColor }} />
      <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2" style={{ borderColor: statusColor }} />
    </div>
  );
};

export default RateLimitGauge;
