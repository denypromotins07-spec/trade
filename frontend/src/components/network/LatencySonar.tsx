/**
 * File 3: LatencySonar.tsx
 * Chapter 1: Network Topology & API Rate Limit UI
 * 
 * Real-time ping/pong radar chart mapping WebSocket round-trip times,
 * highlighting microsecond jitter and connection degradation instantly.
 * 
 * Features:
 * - Radar/Sonar visualization style
 * - Microsecond precision display
 * - Jitter variance shading
 * - Connection health scoring
 */

import React, { useEffect, useRef, useCallback } from 'react';

// --- Types ---

interface LatencySample {
  timestamp: number;
  rttMs: number;        // Round-trip time in milliseconds
  jitterMs: number;     // Variance from previous sample
  packetLoss: boolean;  // Did this ping fail?
}

interface Props {
  samples: LatencySample[];
  maxSamples?: number;
  width?: number;
  height?: number;
  targetRttMs?: number; // Ideal RTT (e.g., 10ms)
}

// --- Constants ---

const COLORS = {
  bg: '#050505',
  grid: 'rgba(0, 243, 255, 0.15)',
  trace: '#00f3ff',
  traceBad: '#ff0055',
  traceWarn: '#ffaa00',
  fill: 'rgba(0, 243, 255, 0.1)',
  text: '#a0a0a0',
  radarLine: 'rgba(0, 243, 255, 0.3)',
};

const MAX_RTT_DISPLAY = 200; // Max RTT to display on scale (ms)

/**
 * LatencySonar Component
 * Renders a radar-style chart for WebSocket latency monitoring.
 */
export const LatencySonar: React.FC<Props> = ({
  samples,
  maxSamples = 60,
  width = 400,
  height = 400,
  targetRttMs = 10,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  // Normalize samples to fit display
  const normalizedSamples = useCallback(() => {
    const recent = samples.slice(-maxSamples);
    return recent.map((s) => ({
      ...s,
      normalizedRtt: Math.min(s.rttMs / MAX_RTT_DISPLAY, 1),
      angle: (recent.indexOf(s) / recent.length) * 2 * Math.PI,
    }));
  }, [samples, maxSamples]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) / 2 - 40; // Padding for labels

    // Clear
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    // Draw Concentric Grid Circles (RTT levels)
    const levels = [20, 50, 100, 150, 200];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '10px "JetBrains Mono", monospace';
    
    levels.forEach((level) => {
      const r = (level / MAX_RTT_DISPLAY) * radius;
      
      // Circle
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      ctx.fillStyle = COLORS.text;
      ctx.fillText(`${level}ms`, cx + r * Math.cos(Math.PI / 4), cy - r * Math.sin(Math.PI / 4));
    });

    // Draw Radial Lines (Time sectors)
    const sectorCount = 12;
    for (let i = 0; i < sectorCount; i++) {
      const angle = (i / sectorCount) * 2 * Math.PI - Math.PI / 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
      ctx.strokeStyle = COLORS.radarLine;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Draw Target Ring (Ideal RTT)
    const targetR = (targetRttMs / MAX_RTT_DISPLAY) * radius;
    ctx.beginPath();
    ctx.arc(cx, cy, targetR, 0, Math.PI * 2);
    ctx.strokeStyle = COLORS.trace;
    ctx.lineWidth = 2;
    ctx.setLineDash([2, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Data Points (Spiral outward based on time)
    const data = normalizedSamples();
    if (data.length > 1) {
      ctx.beginPath();
      
      data.forEach((sample, index) => {
        // Spiral effect: older points are closer to center, newer at edge?
        // Actually, let's do a circular buffer where angle = time, radius = RTT
        const angle = (index / data.length) * 2 * Math.PI - Math.PI / 2;
        const r = sample.normalizedRtt * radius;
        
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        // Draw point
        ctx.beginPath();
        ctx.arc(x, y, sample.packetLoss ? 4 : 2, 0, Math.PI * 2);
        ctx.fillStyle = sample.packetLoss 
          ? COLORS.traceBad 
          : sample.rttMs > 100 
            ? COLORS.traceWarn 
            : COLORS.trace;
        ctx.fill();
      });

      // Stroke the path
      ctx.beginPath();
      data.forEach((sample, index) => {
        const angle = (index / data.length) * 2 * Math.PI - Math.PI / 2;
        const r = sample.normalizedRtt * radius;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      
      ctx.strokeStyle = COLORS.trace;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Fill area under curve (spiral fill)
      ctx.lineTo(cx, cy);
      ctx.closePath();
      ctx.fillStyle = COLORS.fill;
      ctx.fill();
    }

    // Center Stats Display
    const latest = samples[samples.length - 1];
    if (latest) {
      // Glow effect for latest point
      const latestAngle = ((data.length - 1) / data.length) * 2 * Math.PI - Math.PI / 2;
      const latestR = latest.normalizedRtt * radius;
      const latestX = cx + Math.cos(latestAngle) * latestR;
      const latestY = cy + Math.sin(latestAngle) * latestR;

      ctx.beginPath();
      ctx.arc(latestX, latestY, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 243, 255, 0.3)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(latestX, latestY, 4, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.trace;
      ctx.fill();

      // Text Stats in Center
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${latest.rttMs.toFixed(2)}ms`, cx, cy - 10);
      
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`JITTER: ${latest.jitterMs.toFixed(2)}ms`, cx, cy + 10);
      
      // Health Score
      const health = Math.max(0, 100 - (latest.rttMs / 2));
      ctx.fillStyle = health > 80 ? COLORS.safe : health > 50 ? COLORS.warn : COLORS.danger;
      ctx.fillText(`HEALTH: ${health.toFixed(0)}%`, cx, cy + 25);
    }

    // Rotating Radar Sweep Line (Animation)
    const time = Date.now() / 1000;
    const sweepAngle = (time % 5) / 5 * 2 * Math.PI - Math.PI / 2;
    
    const gradient = ctx.createConicGradient(sweepAngle + Math.PI / 2, cx, cy);
    gradient.addColorStop(0, 'rgba(0, 243, 255, 0)');
    gradient.addColorStop(0.8, 'rgba(0, 243, 255, 0)');
    gradient.addColorStop(1, 'rgba(0, 243, 255, 0.2)');
    
    ctx.beginPath();
    ctx.arc(cx, cy, radius, sweepAngle, sweepAngle + 0.5);
    ctx.lineTo(cx, cy);
    ctx.fillStyle = gradient;
    ctx.fill();

  }, [samples, maxSamples, width, height, targetRttMs, normalizedSamples]);

  // Animation Loop
  useEffect(() => {
    const animate = () => {
      render();
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [render]);

  // Color constants for rendering
  const COLORS_local = COLORS; // Closure fix

  return (
    <div className="relative flex flex-col items-center justify-center p-4 bg-black/80 backdrop-blur-sm border border-cyan-900/50 rounded-lg">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="max-w-full h-auto"
        style={{ maxWidth: '100%', height: 'auto' }}
      />
      
      {/* Overlay Legend */}
      <div className="absolute top-2 right-2 flex flex-col gap-1 text-[9px] font-mono">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#00f3ff]" />
          <span className="text-gray-400">NORMAL</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#ffaa00]" />
          <span className="text-gray-400">HIGH</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#ff0055]" />
          <span className="text-gray-400">LOSS</span>
        </div>
      </div>

      {/* Title */}
      <div className="absolute top-2 left-2 text-xs font-mono text-cyan-400 font-bold tracking-wider">
        LATENCY_SONAR
      </div>
    </div>
  );
};

// Add missing color references
interface ColorSet {
  safe: string;
  warn: string;
  danger: string;
}
const extraColors: ColorSet = {
  safe: '#00ff9d',
  warn: '#ffaa00',
  danger: '#ff0055',
};
// Merge for template usage
Object.assign(COLORS, extraColors);

export default LatencySonar;
