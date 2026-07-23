/**
 * FearGreedGauge.tsx - Sentiment Analysis: Market Psychology Radial Gauge
 * 
 * Custom SVG radial gauge for market fear & greed index featuring fluid
 * Framer Motion path animations and glassmorphism backdrop filters.
 * 
 * Features:
 * - Custom SVG radial gauge with animated needle
 * - Framer Motion path animations for smooth transitions
 * - Glassmorphism backdrop blur effects
 * - Color gradient zones (Extreme Fear to Extreme Greed)
 * - Cyberpunk aesthetic with neon accents
 */

'use client';

import React, { useMemo, useEffect, useState } from 'react';
import { motion, useSpring, useTransform, animate } from 'framer-motion';

// ============================================================================
// Type Definitions
// ============================================================================

interface FearGreedGaugeProps {
  value?: number; // 0-100
  history?: number[];
  size?: number;
  showHistory?: boolean;
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const GAUGE_SIZE_DEFAULT = 300;

const FEAR_GREED_ZONES = [
  { min: 0, max: 25, label: 'Extreme Fear', color: '#ff0044' },
  { min: 25, max: 45, label: 'Fear', color: '#ff6600' },
  { min: 45, max: 55, label: 'Neutral', color: '#ffcc00' },
  { min: 55, max: 75, label: 'Greed', color: '#00ff88' },
  { min: 75, max: 100, label: 'Extreme Greed', color: '#00ccff' },
];

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generates mock fear & greed history data
 */
const generateMockHistory = (count: number): number[] => {
  return Array.from({ length: count }, (_, i) => {
    // Random walk with mean reversion to 50
    const prev = i > 0 ? history[i - 1] : 50;
    const change = (Math.random() - 0.5) * 10;
    const meanReversion = (50 - prev) * 0.1;
    return Math.max(0, Math.min(100, prev + change + meanReversion));
  });
};

/**
 * Gets zone label for a given value
 */
const getZoneLabel = (value: number): string => {
  const zone = FEAR_GREED_ZONES.find((z) => value >= z.min && value < z.max);
  return zone?.label || 'Unknown';
};

/**
 * Gets zone color for a given value
 */
const getZoneColor = (value: number): string => {
  const zone = FEAR_GREED_ZONES.find((z) => value >= z.min && value < z.max);
  return zone?.color || '#666666';
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Animated Needle Component using Framer Motion
 */
interface NeedleProps {
  value: number;
  radius: number;
  centerX: number;
  centerY: number;
}

const Needle: React.FC<NeedleProps> = ({ value, radius, centerX, centerY }) => {
  // Convert 0-100 value to angle (-90 to 90 degrees)
  const angle = useMemo(() => {
    return -90 + (value / 100) * 180;
  }, [value]);

  // Spring animation for smooth needle movement
  const springConfig = { damping: 15, stiffness: 100, mass: 0.5 };
  const animatedAngle = useSpring(angle, springConfig);

  useEffect(() => {
    animatedAngle.set(angle);
  }, [angle, animatedAngle]);

  // Calculate needle endpoint
  const needleLength = radius * 0.7;
  const rad = (animatedAngle.get() * Math.PI) / 180;
  const endX = centerX + needleLength * Math.cos(rad);
  const endY = centerY + needleLength * Math.sin(rad);

  return (
    <motion.line
      x1={centerX}
      y1={centerY}
      x2={endX}
      y2={endY}
      stroke="#ffffff"
      strokeWidth="3"
      strokeLinecap="round"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      style={{
        filter: 'drop-shadow(0 0 8px rgba(255, 255, 255, 0.5))',
      }}
    />
  );
};

/**
 * Center Hub Component
 */
interface HubProps {
  centerX: number;
  centerY: number;
  value: number;
  label: string;
}

const Hub: React.FC<HubProps> = ({ centerX, centerY, value, label }) => {
  const [displayValue, setDisplayValue] = useState(Math.round(value));

  // Animate value display
  useEffect(() => {
    const controls = animate(displayValue, Math.round(value), {
      duration: 0.5,
      ease: 'easeOut',
      onUpdate: (latest) => setDisplayValue(Math.round(latest)),
    });
    return () => controls.stop();
  }, [value]);

  const color = getZoneColor(value);

  return (
    <g>
      {/* Outer ring */}
      <circle
        cx={centerX}
        cy={centerY}
        r="35"
        fill="#0a0a12"
        stroke={color}
        strokeWidth="2"
        style={{
          filter: `drop-shadow(0 0 10px ${color}66)`,
        }}
      />
      
      {/* Inner glow */}
      <circle
        cx={centerX}
        cy={centerY}
        r="25"
        fill={`${color}22`}
      />
      
      {/* Value text */}
      <text
        x={centerX}
        y={centerY - 5}
        textAnchor="middle"
        className="fill-white font-mono text-xl font-bold"
      >
        {displayValue}
      </text>
      
      {/* Label text */}
      <text
        x={centerX}
        y={centerY + 15}
        textAnchor="middle"
        className="font-mono text-xs"
        fill={color}
      >
        {label}
      </text>
    </g>
  );
};

/**
 * Gradient Arc Component for gauge background
 */
interface GradientArcProps {
  centerX: number;
  centerY: number;
  radius: number;
  startAngle: number;
  endAngle: number;
  color: string;
}

const GradientArc: React.FC<GradientArcProps> = ({
  centerX,
  centerY,
  radius,
  startAngle,
  endAngle,
  color,
}) => {
  // Convert angles to radians
  const startRad = (startAngle * Math.PI) / 180;
  const endRad = (endAngle * Math.PI) / 180;

  // Calculate arc path
  const x1 = centerX + radius * Math.cos(startRad);
  const y1 = centerY + radius * Math.sin(startRad);
  const x2 = centerX + radius * Math.cos(endRad);
  const y2 = centerY + radius * Math.sin(endRad);

  const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;

  const pathData = `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`;

  return (
    <motion.path
      d={pathData}
      fill="none"
      stroke={color}
      strokeWidth="20"
      strokeLinecap="round"
      initial={{ opacity: 0, pathLength: 0 }}
      animate={{ opacity: 0.3, pathLength: 1 }}
      transition={{ duration: 1, ease: 'easeInOut' }}
    />
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const FearGreedGauge: React.FC<FearGreedGaugeProps> = ({
  value = 50,
  history,
  size = GAUGE_SIZE_DEFAULT,
  showHistory = true,
}) => {
  const [currentValue, setCurrentValue] = useState(value);
  
  // Update current value when prop changes
  useEffect(() => {
    setCurrentValue(value);
  }, [value]);

  // Generate mock history if not provided
  const historyData = useMemo(() => {
    return history || generateMockHistory(30);
  }, [history]);

  // Gauge geometry
  const centerX = size / 2;
  const centerY = size / 2 - 20;
  const radius = (size - 60) / 2;
  const strokeWidth = 20;

  // Zone arcs
  const zoneAngles = FEAR_GREED_ZONES.map((zone) => ({
    startAngle: -90 + (zone.min / 100) * 180,
    endAngle: -90 + (zone.max / 100) * 180,
    ...zone,
  }));

  return (
    <div className="relative flex flex-col items-center p-6 rounded-2xl bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl">
      {/* Header */}
      <div className="absolute top-4 left-0 right-0 text-center">
        <h3 className="text-cyan-400 font-mono text-sm tracking-wider uppercase">
          😨 Fear & Greed <span className="text-xs opacity-70">| Market Psychology</span>
        </h3>
      </div>

      {/* Main Gauge SVG */}
      <svg
        width={size}
        height={size}
        className="overflow-visible"
        aria-label={`Fear and Greed Index: ${currentValue}`}
      >
        {/* Definitions for gradients and filters */}
        <defs>
          <radialGradient id="glow-gradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </radialGradient>
          
          <filter id="outer-glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background glow circle */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius + 30}
          fill="url(#glow-gradient)"
          opacity="0.1"
        />

        {/* Zone arcs */}
        {zoneAngles.map((zone, index) => (
          <GradientArc
            key={index}
            centerX={centerX}
            centerY={centerY}
            radius={radius}
            startAngle={zone.startAngle}
            endAngle={zone.endAngle}
            color={zone.color}
          />
        ))}

        {/* Tick marks */}
        {[0, 25, 50, 75, 100].map((tick) => {
          const angle = -90 + (tick / 100) * 180;
          const rad = (angle * Math.PI) / 180;
          const innerR = radius - 35;
          const outerR = radius - 25;
          
          const x1 = centerX + innerR * Math.cos(rad);
          const y1 = centerY + innerR * Math.sin(rad);
          const x2 = centerX + outerR * Math.cos(rad);
          const y2 = centerY + outerR * Math.sin(rad);

          return (
            <g key={tick}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#ffffff"
                strokeWidth="2"
                opacity="0.5"
              />
              <text
                x={centerX + (radius - 50) * Math.cos(rad)}
                y={centerY + (radius - 50) * Math.sin(rad)}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-gray-400 font-mono text-xs"
              >
                {tick}
              </text>
            </g>
          );
        })}

        {/* Animated needle */}
        <Needle
          value={currentValue}
          radius={radius}
          centerX={centerX}
          centerY={centerY}
        />

        {/* Center hub with value */}
        <Hub
          centerX={centerX}
          centerY={centerY}
          value={currentValue}
          label={getZoneLabel(currentValue)}
        />
      </svg>

      {/* History mini-chart (if enabled) */}
      {showHistory && historyData.length > 0 && (
        <div className="w-full mt-4 pt-4 border-t border-white/10">
          <div className="text-xs font-mono text-gray-400 mb-2">30-Day Trend</div>
          <div className="h-12 relative">
            <svg viewBox="0 0 100 40" className="w-full h-full" preserveAspectRatio="none">
              {/* Grid lines */}
              <line x1="0" y1="10" x2="100" y2="10" stroke="#ffffff" strokeWidth="0.5" opacity="0.1" />
              <line x1="0" y1="20" x2="100" y2="20" stroke="#ffffff" strokeWidth="0.5" opacity="0.1" />
              <line x1="0" y1="30" x2="100" y2="30" stroke="#ffffff" strokeWidth="0.5" opacity="0.1" />
              
              {/* Area fill */}
              <motion.path
                d={`M 0,40 L ${historyData.map((v, i) => `${(i / (historyData.length - 1)) * 100},${40 - (v / 100) * 35}`).join(' L ')} L 100,40 Z`}
                fill="url(#glow-gradient)"
                opacity="0.3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.3 }}
                transition={{ duration: 0.5 }}
              />
              
              {/* Line */}
              <motion.polyline
                points={historyData.map((v, i) => `${(i / (historyData.length - 1)) * 100},${40 - (v / 100) * 35}`).join(' ')}
                fill="none"
                stroke={getZoneColor(currentValue)}
                strokeWidth="1.5"
                vectorEffect="non-scaling-stroke"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 1, ease: 'easeInOut' }}
              />
            </svg>
          </div>
          <div className="flex justify-between text-xs font-mono text-gray-500 mt-1">
            <span>30d ago</span>
            <span>Min: {Math.min(...historyData).toFixed(0)}</span>
            <span>Max: {Math.max(...historyData).toFixed(0)}</span>
            <span>Now</span>
          </div>
        </div>
      )}

      {/* Live indicator */}
      <div className="absolute top-4 right-4 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        <span className="text-xs text-gray-400 font-mono">LIVE</span>
      </div>
    </div>
  );
};

export default FearGreedGauge;
