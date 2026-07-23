/**
 * File 5: CacheMissChart.tsx
 * Chapter 2: System Diagnostics & ETW Telemetry
 * 
 * LLC (Last Level Cache) miss rate visualizer plotting hardware-level telemetry
 * from Rust ETW hooks using double-buffered Canvas rendering.
 * 
 * Features:
 * - Double-buffered Canvas for tear-free rendering
 * - Real-time cache miss percentage
 * - Per-CCD breakdown
 * - Threshold alerting visualization
 */

import React, { useEffect, useRef, useCallback } from 'react';

// --- Types ---

interface CacheSample {
  timestamp: number;
  missRate: number;       // Percentage 0-100
  hits: number;           // Absolute count
  misses: number;         // Absolute count
  ccdBreakdown?: number[]; // Miss rate per CCD
}

interface Props {
  samples: CacheSample[];
  maxSamples?: number;
  width?: number;
  height?: number;
  threshold?: number;     // Alert threshold (%)
}

// --- Constants ---

const COLORS = {
  bg: '#0a0a0a',
  grid: 'rgba(0, 243, 255, 0.1)',
  trace: '#00f3ff',
  traceHigh: '#ffaa00',
  traceCrit: '#ff0055',
  fill: 'rgba(0, 243, 255, 0.15)',
  text: '#a0a0a0',
  threshold: 'rgba(255, 0, 85, 0.5)',
};

const MAX_MISS_RATE = 100;

/**
 * CacheMissChart Component
 * Double-buffered canvas visualization for LLC miss rates.
 */
export const CacheMissChart: React.FC<Props> = ({
  samples,
  maxSamples = 120,
  width = 600,
  height = 300,
  threshold = 30,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const backBufferRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number>();

  // Initialize back buffer
  useEffect(() => {
    const backBuffer = document.createElement('canvas');
    backBuffer.width = width;
    backBuffer.height = height;
    backBufferRef.current = backBuffer;

    return () => {
      backBufferRef.current = null;
    };
  }, [width, height]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const backBuffer = backBufferRef.current;
    if (!canvas || !backBuffer) return;

    const ctx = backBuffer.getContext('2d');
    if (!ctx) return;

    // Clear back buffer
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    const recentSamples = samples.slice(-maxSamples);
    if (recentSamples.length < 2) {
      // Not enough data, just clear and return
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('WAITING FOR DATA...', width / 2, height / 2);
      
      // Copy to front buffer
      const frontCtx = canvas.getContext('2d');
      if (frontCtx) frontCtx.drawImage(backBuffer, 0, 0);
      return;
    }

    const padding = { top: 30, right: 20, bottom: 30, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Draw Grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    
    // Horizontal grid lines (0%, 25%, 50%, 75%, 100%)
    const gridLevels = [0, 25, 50, 75, 100];
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = COLORS.text;

    gridLevels.forEach((level) => {
      const y = padding.top + chartHeight - (level / MAX_MISS_RATE) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(`${level}%`, padding.left - 5, y);
    });

    // Draw Threshold Line
    const thresholdY = padding.top + chartHeight - (threshold / MAX_MISS_RATE) * chartHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, thresholdY);
    ctx.lineTo(width - padding.right, thresholdY);
    ctx.strokeStyle = COLORS.threshold;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
    
    ctx.fillStyle = COLORS.threshold;
    ctx.textAlign = 'left';
    ctx.fillText(`THRESHOLD: ${threshold}%`, width - padding.right - 100, thresholdY - 8);

    // Draw Area Fill
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    
    recentSamples.forEach((sample, i) => {
      const x = padding.left + (i / (recentSamples.length - 1)) * chartWidth;
      const y = padding.top + chartHeight - (sample.missRate / MAX_MISS_RATE) * chartHeight;
      
      if (i === 0) ctx.lineTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.closePath();
    ctx.fillStyle = COLORS.fill;
    ctx.fill();

    // Draw Main Trace
    ctx.beginPath();
    recentSamples.forEach((sample, i) => {
      const x = padding.left + (i / (recentSamples.length - 1)) * chartWidth;
      const y = padding.top + chartHeight - (sample.missRate / MAX_MISS_RATE) * chartHeight;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    // Dynamic color based on value
    const latestMissRate = recentSamples[recentSamples.length - 1]?.missRate || 0;
    ctx.strokeStyle = latestMissRate > threshold 
      ? COLORS.traceCrit 
      : latestMissRate > threshold / 2 
        ? COLORS.traceHigh 
        : COLORS.trace;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Draw CCD Breakdown (if available)
    const latestSample = recentSamples[recentSamples.length - 1];
    if (latestSample?.ccdBreakdown && latestSample.ccdBreakdown.length > 0) {
      const ccdColors = ['#00ff9d', '#00f3ff', '#bd00ff', '#ff0055'];
      const barWidth = chartWidth / latestSample.ccdBreakdown.length;
      
      latestSample.ccdBreakdown.forEach((rate, i) => {
        const x = padding.left + i * barWidth + 2;
        const barHeight = (rate / MAX_MISS_RATE) * chartHeight * 0.3; // Smaller bars
        const y = padding.top + chartHeight - barHeight;
        
        ctx.fillStyle = ccdColors[i % ccdColors.length];
        ctx.fillRect(x, y, barWidth - 4, barHeight);
      });
    }

    // Draw Latest Value Label
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${latestMissRate.toFixed(2)}%`, width / 2, padding.top + 20);
    
    // X-axis labels (time)
    ctx.fillStyle = COLORS.text;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    
    const timeLabels = [
      { idx: 0, label: '-60s' },
      { idx: Math.floor(recentSamples.length / 2), label: '-30s' },
      { idx: recentSamples.length - 1, label: 'NOW' },
    ];
    
    timeLabels.forEach(({ idx, label }) => {
      const x = padding.left + (idx / (recentSamples.length - 1)) * chartWidth;
      ctx.fillText(label, x, height - 10);
    });

    // Copy back buffer to front buffer (double buffering)
    const frontCtx = canvas.getContext('2d');
    if (frontCtx) {
      frontCtx.clearRect(0, 0, width, height);
      frontCtx.drawImage(backBuffer, 0, 0);
    }
  }, [samples, maxSamples, width, height, threshold]);

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

  // Calculate stats
  const recentSamples = samples.slice(-maxSamples);
  const avgMissRate = recentSamples.reduce((acc, s) => acc + s.missRate, 0) / recentSamples.length || 0;
  const maxMissRate = Math.max(...recentSamples.map((s) => s.missRate), 0);
  const totalMisses = recentSamples.reduce((acc, s) => acc + s.misses, 0);
  const totalHits = recentSamples.reduce((acc, s) => acc + s.hits, 0);

  return (
    <div className="p-4 bg-black/80 backdrop-blur-md border border-cyan-900/50 rounded-xl">
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-mono font-bold text-white tracking-wider">
          LLC_CACHE_MISS_RATE
        </h3>
        <div className="flex gap-4 text-[10px] font-mono">
          <div className="text-right">
            <div className="text-gray-500">AVG</div>
            <div className={avgMissRate > threshold ? 'text-red-500' : 'text-cyan-400'}>
              {avgMissRate.toFixed(2)}%
            </div>
          </div>
          <div className="text-right">
            <div className="text-gray-500">PEAK</div>
            <div className={maxMissRate > threshold ? 'text-red-500' : 'text-yellow-400'}>
              {maxMissRate.toFixed(2)}%
            </div>
          </div>
          <div className="text-right">
            <div className="text-gray-500">HIT_RATIO</div>
            <div className="text-green-400">
              {totalHits > 0 ? ((totalHits / (totalHits + totalMisses)) * 100).toFixed(2) : 0}%
            </div>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full"
        style={{ maxHeight: `${height}px` }}
      />

      {/* Legend */}
      <div className="mt-2 flex justify-center gap-4 text-[9px] font-mono">
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-[#00f3ff]" />
          <span className="text-gray-400">MISS_RATE</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-[#ff0055]" style={{ borderStyle: 'dashed' }} />
          <span className="text-gray-400">THRESHOLD</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-gradient-to-r from-green-400 via-cyan-400 to-purple-500" />
          <span className="text-gray-400">CCD_BREAKDOWN</span>
        </div>
      </div>
    </div>
  );
};

export default CacheMissChart;
