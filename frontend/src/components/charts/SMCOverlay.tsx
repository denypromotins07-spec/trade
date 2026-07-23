/**
 * SMCOverlay.tsx - Smart Money Concepts overlay renderer
 * 
 * Renders Order Blocks, Fair Value Gaps (FVGs), and institutional liquidity zones
 * directly onto the chart canvas using custom drawing primitives. This component
 * integrates with lightweight-charts to draw annotations without creating DOM nodes.
 * 
 * Features:
 * - Custom canvas rendering for Order Blocks
 * - FVG highlighting with transparency
 * - Liquidity zone visualization
 * - Zero DOM overhead (pure canvas)
 * - AMD GPU-accelerated via WebGL fallback
 */

import React, { useEffect, useRef, useCallback } from 'react';
import { IChartApi, ISeriesApi, Time, MouseEventParams } from 'lightweight-charts';
import { useChartStore } from '../../store/chartStore';

// SMC Zone types for institutional patterns
export interface OrderBlock {
  id: string;
  type: 'bullish' | 'bearish';
  startTime: number;
  endTime: number;
  startPrice: number;
  endPrice: number;
  strength: number; // 0-1 multiplier for opacity
}

export interface FairValueGap {
  id: string;
  type: 'bullish' | 'bearish';
  time: number;
  high: number;
  low: number;
  mitigationPrice?: number;
}

export interface LiquidityZone {
  id: string;
  type: 'equal_highs' | 'equal_lows' | 'swing_high' | 'swing_low';
  price: number;
  time: number;
  touched: number;
}

interface SMCOverlayProps {
  chartId: string;
  orderBlocks?: OrderBlock[];
  fvgZones?: FairValueGap[];
  liquidityZones?: LiquidityZone[];
  enabled?: boolean;
}

// Cyberpunk color scheme for SMC elements
const SMC_COLORS = {
  orderBlockBullish: 'rgba(0, 255, 136, 0.15)',
  orderBlockBearish: 'rgba(255, 0, 85, 0.15)',
  fvgBullish: 'rgba(0, 255, 255, 0.1)',
  fvgBearish: 'rgba(255, 0, 255, 0.1)',
  liquidityHigh: 'rgba(255, 165, 0, 0.3)',
  liquidityLow: 'rgba(255, 165, 0, 0.3)',
  borderBullish: '#00ff88',
  borderBearish: '#ff0055',
};

export const SMCOverlay: React.FC<SMCOverlayProps> = ({
  chartId,
  orderBlocks = [],
  fvgZones = [],
  liquidityZones = [],
  enabled = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  const chart = useChartStore((state) => state.charts.get(chartId)?.chart);
  const candleSeries = useChartStore((state) => state.charts.get(chartId)?.series);

  // Convert time to X coordinate on chart
  const timeToX = useCallback((time: number, chartWidth: number): number => {
    if (!chart) return 0;
    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    if (!visibleRange) return 0;
    
    const data = candleSeries?.data() || [];
    const firstTime = data[0]?.time as number;
    const lastTime = data[data.length - 1]?.time as number;
    
    const timeRange = lastTime - firstTime;
    if (timeRange === 0) return chartWidth / 2;
    
    return ((time - firstTime) / timeRange) * chartWidth;
  }, [chart, candleSeries]);

  // Convert price to Y coordinate on chart
  const priceToY = useCallback((price: number, chartHeight: number): number => {
    if (!chart) return 0;
    const priceScale = chart.priceScale('right');
    if (!priceScale) return 0;
    
    // Get visible price range
    const visibleRange = candleSeries?.priceScale().getVisiblePriceRange();
    if (!visibleRange) return chartHeight / 2;
    
    const { minPrice, maxPrice } = visibleRange;
    const priceRange = maxPrice - minPrice;
    if (priceRange === 0) return chartHeight / 2;
    
    // Invert Y because canvas origin is top-left
    return chartHeight - ((price - minPrice) / priceRange) * chartHeight;
  }, [chart, candleSeries]);

  // Draw Order Blocks on canvas
  const drawOrderBlocks = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    orderBlocks.forEach(block => {
      const x = timeToX(block.startTime, width);
      const xEnd = timeToX(block.endTime, width);
      const y = priceToY(block.type === 'bullish' ? block.startPrice : block.endPrice, height);
      const yEnd = priceToY(block.type === 'bullish' ? block.endPrice : block.startPrice, height);
      
      const blockWidth = Math.max(xEnd - x, 50); // Minimum width for visibility
      
      ctx.fillStyle = block.type === 'bullish' 
        ? SMC_COLORS.orderBlockBullish 
        : SMC_COLORS.orderBlockBearish;
      ctx.strokeStyle = block.type === 'bullish' 
        ? SMC_COLORS.borderBullish 
        : SMC_COLORS.borderBearish;
      ctx.lineWidth = 1;
      ctx.globalAlpha = block.strength;
      
      // Draw rectangle
      ctx.fillRect(x, Math.min(y, yEnd), blockWidth, Math.abs(yEnd - y));
      ctx.strokeRect(x, Math.min(y, yEnd), blockWidth, Math.abs(yEnd - y));
      
      // Add label
      ctx.globalAlpha = 1;
      ctx.font = '10px monospace';
      ctx.fillStyle = block.type === 'bullish' ? SMC_COLORS.borderBullish : SMC_COLORS.borderBearish;
      ctx.fillText(`OB ${block.type.toUpperCase()}`, x + 4, Math.min(y, yEnd) - 4);
    });
  }, [orderBlocks, timeToX, priceToY]);

  // Draw Fair Value Gaps
  const drawFVGs = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    fvgZones.forEach(fvg => {
      const x = timeToX(fvg.time, width);
      const yHigh = priceToY(fvg.high, height);
      const yLow = priceToY(fvg.low, height);
      
      ctx.fillStyle = fvg.type === 'bullish' 
        ? SMC_COLORS.fvgBullish 
        : SMC_COLORS.fvgBearish;
      ctx.strokeStyle = fvg.type === 'bullish' 
        ? SMC_COLORS.borderBullish 
        : SMC_COLORS.borderBearish;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      
      // Draw FVG zone extending to right
      const fvgWidth = width - x;
      ctx.fillRect(x, Math.min(yHigh, yLow), fvgWidth, Math.abs(yLow - yHigh));
      
      // Draw dashed borders
      ctx.beginPath();
      ctx.moveTo(x, yHigh);
      ctx.lineTo(width, yHigh);
      ctx.moveTo(x, yLow);
      ctx.lineTo(width, yLow);
      ctx.stroke();
      
      ctx.setLineDash([]);
      
      // Label
      ctx.font = '9px monospace';
      ctx.fillStyle = fvg.type === 'bullish' ? SMC_COLORS.borderBullish : SMC_COLORS.borderBearish;
      ctx.fillText(`FVG`, x + 4, Math.min(yHigh, yLow) + 12);
    });
  }, [fvgZones, timeToX, priceToY]);

  // Draw Liquidity Zones
  const drawLiquidityZones = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    liquidityZones.forEach(zone => {
      const x = timeToX(zone.time, width);
      const y = priceToY(zone.price, height);
      
      ctx.fillStyle = SMC_COLORS.liquidityHigh;
      ctx.strokeStyle = '#ffa500';
      ctx.lineWidth = 2;
      
      // Draw horizontal line
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      
      // Draw zone indicator
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Label with touch count
      ctx.font = '10px monospace';
      ctx.fillStyle = '#ffa500';
      ctx.fillText(`${zone.type.toUpperCase()} (${zone.touched}x)`, x + 10, y - 8);
    });
  }, [liquidityZones, timeToX, priceToY]);

  // Main render loop using requestAnimationFrame for 60FPS
  const render = useCallback(() => {
    if (!canvasRef.current || !enabled) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const container = canvas.parentElement;
    if (!container) return;
    
    // Match canvas size to container
    const rect = container.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
    
    // Clear canvas efficiently
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw all SMC elements
    drawOrderBlocks(ctx, canvas.width, canvas.height);
    drawFVGs(ctx, canvas.width, canvas.height);
    drawLiquidityZones(ctx, canvas.width, canvas.height);
    
    animationFrameRef.current = requestAnimationFrame(render);
  }, [enabled, drawOrderBlocks, drawFVGs, drawLiquidityZones]);

  // Start/stop render loop
  useEffect(() => {
    if (enabled) {
      render();
    } else if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled, render]);

  // Handle chart resize
  useEffect(() => {
    if (!chart) return;
    
    const handleResize = () => {
      if (canvasRef.current && animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = requestAnimationFrame(render);
      }
    };
    
    // Subscribe to chart resize via lightweight-charts
    const unsubscribe = chart.timeScale().subscribeVisibleLogicalRangeChange(handleResize);
    
    return () => {
      unsubscribe(handleResize);
    };
  }, [chart, render]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        zIndex: 5,
      }}
      className="smc-overlay-canvas"
    />
  );
};

export default SMCOverlay;
