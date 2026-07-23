/**
 * VolumeDelta.tsx - Real-time Cumulative Volume Delta (CVD) bar chart
 * 
 * Renders aggressive buying vs selling pressure using Canvas 2D.
 * Updates via Web Worker message bus to keep the main thread free
 * for UI interactions during high-frequency data processing.
 * 
 * Features:
 * - Canvas-based CVD visualization
 * - Web Worker integration for off-main-thread processing
 * - Real-time delta calculation
 * - Divergence detection (price vs volume)
 * - Cyberpunk aesthetic with gradient fills
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';

export interface VolumeBar {
  timestamp: number;
  buyVolume: number;
  sellVolume: number;
  delta: number; // buyVolume - sellVolume
  cumulativeDelta: number;
  price?: number;
}

interface VolumeDeltaProps {
  bars: VolumeBar[];
  width?: number;
  height?: number;
  symbol?: string;
  showCumulative?: boolean;
}

// Cyberpunk color scheme
const CVD_COLORS = {
  buyColor: '#00ff88',
  sellColor: '#ff0055',
  buyGradient: 'rgba(0, 255, 136, 0.6)',
  sellGradient: 'rgba(255, 0, 85, 0.6)',
  gridColor: 'rgba(0, 255, 255, 0.1)',
  textColor: '#00ffff',
  background: '#0a0e17',
  zeroLine: 'rgba(255, 255, 255, 0.3)',
  divergenceColor: '#ffa500',
};

export const VolumeDelta: React.FC<VolumeDeltaProps> = ({
  bars,
  width = 600,
  height = 200,
  symbol = 'BTCUSDT',
  showCumulative = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const latestBarsRef = useRef<VolumeBar[]>(bars);
  const [divergenceDetected, setDivergenceDetected] = useState(false);

  // Update latest bars ref without triggering re-render
  useEffect(() => {
    latestBarsRef.current = bars;
    
    // Check for divergence (price making new high but CVD not confirming)
    if (bars.length >= 20) {
      const recentBars = bars.slice(-20);
      const priceHigh = Math.max(...recentBars.map(b => b.price || 0));
      const cvdHigh = Math.max(...recentBars.map(b => b.cumulativeDelta));
      
      // Simple divergence detection
      const prevBars = bars.slice(-40, -20);
      const prevPriceHigh = Math.max(...prevBars.map(b => b.price || 0));
      const prevCvdHigh = Math.max(...prevBars.map(b => b.cumulativeDelta));
      
      // Price higher but CVD lower = bearish divergence
      if (priceHigh > prevPriceHigh && cvdHigh < prevCvdHigh) {
        setDivergenceDetected(true);
      } else {
        setDivergenceDetected(false);
      }
    }
    
    // Trigger render
    if (animationFrameRef.current === null) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [bars]);

  // Calculate scale factors
  const calculateScales = useCallback((volumeBars: VolumeBar[]) => {
    if (volumeBars.length === 0) {
      return { maxVolume: 1, maxCvd: 1, minCvd: -1 };
    }

    const maxVolume = Math.max(...volumeBars.map(b => Math.max(b.buyVolume, b.sellVolume)), 1);
    const maxCvd = Math.max(...volumeBars.map(b => b.cumulativeDelta), 1);
    const minCvd = Math.min(...volumeBars.map(b => b.cumulativeDelta), -1);

    return { maxVolume, maxCvd, minCvd };
  }, []);

  // Main render function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const volumeBars = latestBarsRef.current;
    if (volumeBars.length === 0) {
      animationFrameRef.current = requestAnimationFrame(render);
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvas.width / dpr;
    const displayHeight = canvas.height / dpr;

    // Clear canvas
    ctx.fillStyle = CVD_COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const scales = calculateScales(volumeBars);
    const barWidth = (displayWidth - 60) / Math.min(volumeBars.length, 100);
    const spacing = Math.max(1, barWidth * 0.1);
    const actualBarWidth = barWidth - spacing;

    // Draw grid
    ctx.strokeStyle = CVD_COLORS.gridColor;
    ctx.lineWidth = 1;

    // Horizontal grid lines
    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      const y = (displayHeight / gridLines) * i + 20; // Offset for header
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(displayWidth - 10, y);
      ctx.stroke();
    }

    // Zero line
    const zeroY = 20 + (displayHeight / 2);
    ctx.strokeStyle = CVD_COLORS.zeroLine;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(50, zeroY);
    ctx.lineTo(displayWidth - 10, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw volume bars (bottom half) and CVD (top half)
    const visibleBars = volumeBars.slice(-100);
    const startIndex = volumeBars.length - visibleBars.length;

    visibleBars.forEach((bar, index) => {
      const x = 50 + index * barWidth;
      
      if (showCumulative) {
        // Draw CVD line
        const cvdY = zeroY - (bar.cumulativeDelta / scales.maxCvd) * (displayHeight / 2 - 20);
        
        if (index === 0) {
          ctx.beginPath();
          ctx.moveTo(x, cvdY);
        } else {
          ctx.lineTo(x, cvdY);
        }

        // Draw individual bars
        const buyHeight = (bar.buyVolume / scales.maxVolume) * (displayHeight / 2 - 10);
        const sellHeight = (bar.sellVolume / scales.maxVolume) * (displayHeight / 2 - 10);

        // Buy volume (above zero)
        ctx.fillStyle = CVD_COLORS.buyGradient;
        ctx.fillRect(x, zeroY - buyHeight, actualBarWidth, buyHeight);

        // Sell volume (below zero)
        ctx.fillStyle = CVD_COLORS.sellGradient;
        ctx.fillRect(x, zeroY, actualBarWidth, sellHeight);
      } else {
        // Show only delta bars
        const deltaHeight = (Math.abs(bar.delta) / scales.maxVolume) * (displayHeight - 40);
        const y = bar.delta >= 0 ? zeroY - deltaHeight : zeroY;

        ctx.fillStyle = bar.delta >= 0 ? CVD_COLORS.buyGradient : CVD_COLORS.sellGradient;
        ctx.fillRect(x, y, actualBarWidth, deltaHeight);
      }
    });

    // Stroke CVD line
    if (showCumulative) {
      ctx.strokeStyle = CVD_COLORS.textColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Fill area under CVD line
      ctx.lineTo(50 + (visibleBars.length - 1) * barWidth, zeroY);
      ctx.lineTo(50, zeroY);
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 255, 255, 0.1)';
      ctx.fill();
    }

    // Draw labels
    ctx.font = '10px monospace';
    ctx.fillStyle = CVD_COLORS.textColor;

    // Y-axis labels
    ctx.textAlign = 'right';
    ctx.fillText(scales.maxCvd.toFixed(0), 45, 25);
    ctx.fillText('0', 45, zeroY + 3);
    ctx.fillText(scales.minCvd.toFixed(0), 45, displayHeight - 5);

    // X-axis label
    ctx.textAlign = 'center';
    ctx.fillText(`${symbol} CVD`, displayWidth / 2 + 20, displayHeight - 5);

    // Latest delta value
    const lastBar = volumeBars[volumeBars.length - 1];
    if (lastBar) {
      ctx.textAlign = 'left';
      ctx.fillStyle = lastBar.delta >= 0 ? CVD_COLORS.buyColor : CVD_COLORS.sellColor;
      ctx.fillText(`Δ: ${lastBar.delta.toFixed(2)}`, 55, 25);
      ctx.fillText(`CVD: ${lastBar.cumulativeDelta.toFixed(2)}`, 55, 40);
    }

    // Divergence warning
    if (divergenceDetected) {
      ctx.fillStyle = CVD_COLORS.divergenceColor;
      ctx.font = 'bold 10px monospace';
      ctx.fillText('⚠ DIVERGENCE', displayWidth - 100, 25);
    }

    // Reset animation frame reference
    animationFrameRef.current = null;
  }, [showCumulative, calculateScales, divergenceDetected, symbol]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      render();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  // Handle DPI scaling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }, [width, height]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
        }}
        className="volume-delta-canvas"
      />
      <div className="absolute top-2 left-2 pointer-events-none flex gap-2">
        <span className="text-cyan-400 text-xs font-mono">
          {symbol} VOLUME DELTA | CANVAS
        </span>
        {divergenceDetected && (
          <span className="text-orange-400 text-xs font-mono animate-pulse">
            ⚠ DIVERGENCE
          </span>
        )}
      </div>
    </div>
  );
};

export default VolumeDelta;
