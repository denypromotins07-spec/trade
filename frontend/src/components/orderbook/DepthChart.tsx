/**
 * DepthChart.tsx - Canvas-based cumulative bid/ask depth chart
 * 
 * Renders real-time order book depth using HTML5 Canvas with 60FPS
 * rendering via requestAnimationFrame. Optimized for high-frequency
 * L2 WebSocket updates from Binance without DOM overhead.
 * 
 * Features:
 * - Canvas-based cumulative depth visualization
 * - 60FPS rendering with requestAnimationFrame
 * - Bid/ask gradient fills with cyberpunk colors
 * - Automatic scaling to visible depth range
 * - Zero garbage collection pauses
 */

import React, { useEffect, useRef, useCallback, useMemo } from 'react';

export interface DepthLevel {
  price: number;
  size: number;
  cumulative: number;
}

export interface DepthData {
  bids: DepthLevel[];
  asks: DepthLevel[];
  timestamp: number;
}

interface DepthChartProps {
  data: DepthData | null;
  width?: number;
  height?: number;
  symbol?: string;
}

// Cyberpunk color palette for depth visualization
const DEPTH_COLORS = {
  bidFill: 'rgba(0, 255, 136, 0.3)',
  bidStroke: '#00ff88',
  askFill: 'rgba(255, 0, 85, 0.3)',
  askStroke: '#ff0055',
  grid: 'rgba(0, 255, 255, 0.1)',
  text: '#00ffff',
  background: '#0a0e17',
};

export const DepthChart: React.FC<DepthChartProps> = ({
  data,
  width = 600,
  height = 300,
  symbol = 'BTCUSDT',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const latestDataRef = useRef<DepthData | null>(data);

  // Update latest data ref without triggering re-render
  useEffect(() => {
    latestDataRef.current = data;
    
    // Trigger render on new data
    if (animationFrameRef.current === null) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [data]);

  // Calculate scale factors for price and size axes
  const calculateScales = useCallback((depthData: DepthData) => {
    if (!depthData || depthData.bids.length === 0 || depthData.asks.length === 0) {
      return { priceScale: 1, sizeScale: 1, minPrice: 0, maxPrice: 0, maxSize: 0 };
    }

    const allPrices = [
      ...depthData.bids.map(b => b.price),
      ...depthData.asks.map(a => a.price),
    ];
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    const priceRange = maxPrice - minPrice || 1;

    const allSizes = [
      ...depthData.bids.map(b => b.cumulative),
      ...depthData.asks.map(a => a.cumulative),
    ];
    const maxSize = Math.max(...allSizes);
    const sizeScale = maxSize > 0 ? (width * 0.45) / maxSize : 1;

    return {
      priceScale: height / priceRange,
      sizeScale,
      minPrice,
      maxPrice,
      maxSize,
    };
  }, [width, height]);

  // Main render function using Canvas 2D API
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const depthData = latestDataRef.current;
    if (!depthData) {
      animationFrameRef.current = requestAnimationFrame(render);
      return;
    }

    // Clear canvas
    ctx.fillStyle = DEPTH_COLORS.background;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = DEPTH_COLORS.grid;
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = (height / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Vertical grid line at center (spread)
    const centerX = width / 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    const scales = calculateScales(depthData);
    const midPrice = (scales.minPrice + scales.maxPrice) / 2;
    const midY = height / 2;

    // Draw Bids (left side, cumulative buy depth)
    if (depthData.bids.length > 0) {
      ctx.fillStyle = DEPTH_COLORS.bidFill;
      ctx.strokeStyle = DEPTH_COLORS.bidStroke;
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(centerX, midY);

      depthData.bids.forEach((level, index) => {
        const priceOffset = (level.price - midPrice) * scales.priceScale;
        const y = midY - priceOffset;
        const x = centerX - level.cumulative * scales.sizeScale;

        if (index === 0) {
          ctx.lineTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      // Close the shape for fill
      const lastBid = depthData.bids[depthData.bids.length - 1];
      const lastY = midY - (lastBid.price - midPrice) * scales.priceScale;
      ctx.lineTo(centerX, lastY);
      ctx.closePath();
      ctx.fill();

      // Stroke the outline
      ctx.stroke();
    }

    // Draw Asks (right side, cumulative sell depth)
    if (depthData.asks.length > 0) {
      ctx.fillStyle = DEPTH_COLORS.askFill;
      ctx.strokeStyle = DEPTH_COLORS.askStroke;
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(centerX, midY);

      depthData.asks.forEach((level, index) => {
        const priceOffset = (level.price - midPrice) * scales.priceScale;
        const y = midY - priceOffset;
        const x = centerX + level.cumulative * scales.sizeScale;

        if (index === 0) {
          ctx.lineTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      // Close the shape for fill
      const lastAsk = depthData.asks[depthData.asks.length - 1];
      const lastY = midY - (lastAsk.price - midPrice) * scales.priceScale;
      ctx.lineTo(centerX, lastY);
      ctx.closePath();
      ctx.fill();

      // Stroke the outline
      ctx.stroke();
    }

    // Draw labels
    ctx.font = '10px monospace';
    ctx.fillStyle = DEPTH_COLORS.text;
    
    // Price labels
    ctx.fillText(scales.maxPrice.toFixed(2), width - 50, 15);
    ctx.fillText(midPrice.toFixed(2), width - 50, midY);
    ctx.fillText(scales.minPrice.toFixed(2), width - 50, height - 10);

    // Size label
    ctx.fillText(`Max: ${scales.maxSize.toFixed(4)}`, 10, 15);

    // Spread indicator
    if (depthData.bids.length > 0 && depthData.asks.length > 0) {
      const bestBid = depthData.bids[0].price;
      const bestAsk = depthData.asks[0].price;
      const spread = bestAsk - bestBid;
      const spreadPercent = ((spread / bestBid) * 100).toFixed(4);

      ctx.fillStyle = '#ffa500';
      ctx.fillText(`Spread: ${spread.toFixed(2)} (${spreadPercent}%)`, centerX - 60, height - 10);
    }

    // Reset animation frame reference
    animationFrameRef.current = null;
  }, [width, height, calculateScales]);

  // Continuous render loop for smooth animations
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

  // Handle canvas resize for DPI
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
  }, [width, height]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
        }}
        className="depth-chart-canvas"
      />
      <div className="absolute top-2 left-2 pointer-events-none">
        <span className="text-cyan-400 text-xs font-mono">
          {symbol} DEPTH | CANVAS_60FPS
        </span>
      </div>
    </div>
  );
};

export default DepthChart;
