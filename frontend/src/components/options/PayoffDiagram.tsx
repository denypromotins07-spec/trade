/**
 * PayoffDiagram.tsx - Canvas-based Multi-leg Options Payoff Visualizer
 * 
 * Dynamically renders options strategy payoff diagrams with breakeven points
 * and max loss zones. Updates in real-time as underlying perpetual prices shift.
 * 
 * Features:
 * - Double-buffered Canvas rendering for 60FPS
 * - Multi-leg strategy support (spreads, straddles, iron condors, etc.)
 * - Dynamic breakeven calculation and P/L zones
 * - Cyberpunk aesthetic with neon glow effects
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

interface OptionLeg {
  type: 'call' | 'put';
  strike: number;
  quantity: number; // Positive for long, negative for short
  premium: number; // Paid/received per contract
}

interface PayoffPoint {
  price: number;
  pnl: number;
}

interface PayoffDiagramProps {
  legs: OptionLeg[];
  currentPrice: number;
  priceRange?: { min: number; max: number };
  onPriceSelect?: (price: number) => void;
  className?: string;
  showBreakeven?: boolean;
  showMaxLoss?: boolean;
}

interface RenderState {
  width: number;
  height: number;
  dpr: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Calculates P/L for a single option leg at a given price
 */
const calculateLegPnL = (leg: OptionLeg, price: number): number => {
  let intrinsicValue = 0;
  
  if (leg.type === 'call') {
    intrinsicValue = Math.max(0, price - leg.strike);
  } else {
    intrinsicValue = Math.max(0, leg.strike - price);
  }
  
  // P/L = (intrinsic value - premium paid) * quantity
  // For short positions, premium is received (negative cost)
  const netPremium = leg.premium * (leg.quantity > 0 ? 1 : -1);
  return (intrinsicValue - netPremium) * Math.abs(leg.quantity) * 100; // 100 shares per contract
};

/**
 * Calculates total strategy P/L at a given price
 */
const calculateStrategyPnL = (legs: OptionLeg[], price: number): number => {
  return legs.reduce((total, leg) => total + calculateLegPnL(leg, price), 0);
};

/**
 * Finds breakeven points where P/L = 0
 */
const findBreakevenPoints = (legs: OptionLeg[], priceRange: { min: number; max: number }): number[] => {
  const breakevens: number[] = [];
  const step = (priceRange.max - priceRange.min) / 1000;
  
  let prevPnL = calculateStrategyPnL(legs, priceRange.min);
  
  for (let price = priceRange.min + step; price <= priceRange.max; price += step) {
    const currentPnL = calculateStrategyPnL(legs, price);
    
    // Check for sign change (crossing zero)
    if ((prevPnL < 0 && currentPnL >= 0) || (prevPnL > 0 && currentPnL <= 0)) {
      // Linear interpolation to find exact breakeven
      const ratio = Math.abs(prevPnL) / (Math.abs(prevPnL) + Math.abs(currentPnL));
      const breakeven = price - step + (step * ratio);
      breakevens.push(breakeven);
    }
    
    prevPnL = currentPnL;
  }
  
  return breakevens;
};

/**
 * Finds max profit and max loss points
 */
const findExtremes = (legs: OptionLeg[], priceRange: { min: number; max: number }) => {
  let maxProfit = -Infinity;
  let maxLoss = Infinity;
  let maxProfitPrice = priceRange.min;
  let maxLossPrice = priceRange.min;
  
  const step = (priceRange.max - priceRange.min) / 500;
  
  for (let price = priceRange.min; price <= priceRange.max; price += step) {
    const pnl = calculateStrategyPnL(legs, price);
    
    if (pnl > maxProfit) {
      maxProfit = pnl;
      maxProfitPrice = price;
    }
    if (pnl < maxLoss) {
      maxLoss = pnl;
      maxLossPrice = price;
    }
  }
  
  return { maxProfit, maxLoss, maxProfitPrice, maxLossPrice };
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const PayoffDiagram: React.FC<PayoffDiagramProps> = ({
  legs,
  currentPrice,
  priceRange: externalRange,
  onPriceSelect,
  className = '',
  showBreakeven = true,
  showMaxLoss = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderState, setRenderState] = useState<RenderState>({ width: 0, height: 0, dpr: 1 });
  const [hoveredPrice, setHoveredPrice] = useState<number | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Calculate default price range based on strikes
  const defaultRange = useMemo(() => {
    if (!legs.length) return { min: 0, max: 100000 };
    
    const strikes = legs.map(l => l.strike);
    const minStrike = Math.min(...strikes);
    const maxStrike = Math.max(...strikes);
    const range = maxStrike - minStrike;
    const padding = range * 0.3;
    
    return {
      min: Math.max(0, minStrike - padding),
      max: maxStrike + padding
    };
  }, [legs]);
  
  const priceRange = externalRange || defaultRange;

  // Generate payoff curve points
  const payoffPoints = useMemo(() => {
    const points: PayoffPoint[] = [];
    const pointCount = 200;
    const step = (priceRange.max - priceRange.min) / pointCount;
    
    for (let i = 0; i <= pointCount; i++) {
      const price = priceRange.min + (i * step);
      const pnl = calculateStrategyPnL(legs, price);
      points.push({ price, pnl });
    }
    
    return points;
  }, [legs, priceRange]);

  // Find key levels
  const keyLevels = useMemo(() => {
    const breakevens = showBreakeven ? findBreakevenPoints(legs, priceRange) : [];
    const extremes = showMaxLoss ? findExtremes(legs, priceRange) : null;
    return { breakevens, extremes };
  }, [legs, priceRange, showBreakeven, showMaxLoss]);

  // Resize handler
  useEffect(() => {
    const updateSize = () => {
      const container = containerRef.current;
      if (!container) return;
      
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      
      setRenderState({
        width: rect.width,
        height: rect.height,
        dpr
      });
    };
    
    updateSize();
    
    const resizeObserver = new ResizeObserver(updateSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    
    return () => resizeObserver.disconnect();
  }, []);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || renderState.width === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height, dpr } = renderState;
    
    // Set actual canvas size (scaled by DPR)
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    let frameCount = 0;
    
    const render = () => {
      frameCount++;
      
      // Clear canvas
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);
      
      // Calculate scales
      const padding = { top: 40, right: 80, bottom: 50, left: 80 };
      const chartWidth = width - padding.left - padding.right;
      const chartHeight = height - padding.top - padding.bottom;
      
      // Find P/L range for Y-axis scaling
      let minPnL = Math.min(...payoffPoints.map(p => p.pnl));
      let maxPnL = Math.max(...payoffPoints.map(p => p.pnl));
      
      // Include current P/L
      const currentPnL = calculateStrategyPnL(legs, currentPrice);
      minPnL = Math.min(minPnL, currentPnL, 0);
      maxPnL = Math.max(maxPnL, currentPnL, 0);
      
      // Add padding to P/L range
      const pnlRange = maxPnL - minPnL || 1;
      minPnL -= pnlRange * 0.1;
      maxPnL += pnlRange * 0.1;
      
      // Scale functions
      const priceToX = (price: number) => 
        padding.left + ((price - priceRange.min) / (priceRange.max - priceRange.min)) * chartWidth;
      
      const pnlToY = (pnl: number) => 
        padding.top + ((maxPnL - pnl) / (maxPnL - minPnL)) * chartHeight;
      
      // Draw grid
      ctx.strokeStyle = '#1a1a2e';
      ctx.lineWidth = 1;
      
      // Horizontal grid lines (P/L levels)
      const pnlStep = (maxPnL - minPnL) / 5;
      for (let i = 0; i <= 5; i++) {
        const pnl = minPnL + (i * pnlStep);
        const y = pnlToY(pnl);
        
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        
        // P/L labels
        ctx.fillStyle = '#6b7280';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`$${Math.round(pnl).toLocaleString()}`, padding.left - 8, y + 3);
      }
      
      // Vertical grid lines (price levels)
      const priceStep = (priceRange.max - priceRange.min) / 6;
      for (let i = 0; i <= 6; i++) {
        const price = priceRange.min + (i * priceStep);
        const x = priceToX(price);
        
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
        
        // Price labels
        ctx.fillStyle = '#6b7280';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`$${Math.round(price).toLocaleString()}`, x, height - padding.bottom + 20);
      }
      
      // Draw zero line (breakeven horizontal)
      const zeroY = pnlToY(0);
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(padding.left, zeroY);
      ctx.lineTo(width - padding.right, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Fill profit area (green)
      ctx.fillStyle = 'rgba(34, 197, 94, 0.2)';
      ctx.beginPath();
      ctx.moveTo(priceToX(payoffPoints[0].price), zeroY);
      
      for (const point of payoffPoints) {
        if (point.pnl >= 0) {
          ctx.lineTo(priceToX(point.price), pnlToY(point.pnl));
        }
      }
      
      // Close the path
      for (let i = payoffPoints.length - 1; i >= 0; i--) {
        const point = payoffPoints[i];
        if (point.pnl >= 0) {
          ctx.lineTo(priceToX(point.price), zeroY);
          break;
        }
      }
      ctx.closePath();
      ctx.fill();
      
      // Fill loss area (red)
      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
      ctx.beginPath();
      ctx.moveTo(priceToX(payoffPoints[0].price), zeroY);
      
      for (const point of payoffPoints) {
        if (point.pnl < 0) {
          ctx.lineTo(priceToX(point.price), pnlToY(point.pnl));
        }
      }
      
      for (let i = payoffPoints.length - 1; i >= 0; i--) {
        const point = payoffPoints[i];
        if (point.pnl < 0) {
          ctx.lineTo(priceToX(point.price), zeroY);
          break;
        }
      }
      ctx.closePath();
      ctx.fill();
      
      // Draw payoff curve with neon glow
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 10;
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < payoffPoints.length; i++) {
        const point = payoffPoints[i];
        const x = priceToX(point.price);
        const y = pnlToY(point.pnl);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
      
      // Draw breakeven points
      if (showBreakeven && keyLevels.breakevens.length > 0) {
        for (const breakeven of keyLevels.breakevens) {
          const x = priceToX(breakeven);
          
          // Vertical dashed line
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(x, padding.top);
          ctx.lineTo(x, height - padding.bottom);
          ctx.stroke();
          ctx.setLineDash([]);
          
          // Breakeven marker
          ctx.fillStyle = '#fbbf24';
          ctx.beginPath();
          ctx.arc(x, zeroY, 5, 0, Math.PI * 2);
          ctx.fill();
          
          // Label
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText('BE', x, zeroY - 10);
        }
      }
      
      // Draw current price indicator
      const currentX = priceToX(currentPrice);
      
      // Vertical line
      ctx.strokeStyle = '#ec4899';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(currentX, padding.top);
      ctx.lineTo(currentX, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Current price label
      ctx.fillStyle = '#ec4899';
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`$${currentPrice.toLocaleString()}`, currentX, height - padding.bottom + 35);
      
      // Current P/L display
      const currentY = pnlToY(currentPnL);
      ctx.fillStyle = currentPnL >= 0 ? '#22c55e' : '#ef4444';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(
        `P/L: $${Math.round(currentPnL).toLocaleString()}`,
        currentX + 10,
        currentY - 10
      );
      
      // Draw hovered price indicator
      if (hoveredPrice !== null) {
        const hoverX = priceToX(hoveredPrice);
        const hoverPnL = calculateStrategyPnL(legs, hoveredPrice);
        const hoverY = pnlToY(hoverPnL);
        
        // Crosshair
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(hoverX, padding.top);
        ctx.lineTo(hoverX, height - padding.bottom);
        ctx.moveTo(padding.left, hoverY);
        ctx.lineTo(width - padding.right, hoverY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Hover tooltip
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(hoverX + 10, hoverY - 35, 120, 50);
        ctx.strokeStyle = '#06b6d4';
        ctx.strokeRect(hoverX + 10, hoverY - 35, 120, 50);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Price: $${Math.round(hoveredPrice)}`, hoverX + 15, hoverY - 20);
        ctx.fillText(`P/L: $${Math.round(hoverPnL).toLocaleString()}`, hoverX + 15, hoverY - 5);
      }
      
      // Title
      ctx.fillStyle = '#06b6d4';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('PAYOFF DIAGRAM', padding.left, 20);
      
      // Strategy info
      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${legs.length} LEGS`, width - padding.right, 20);
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, [renderState, payoffPoints, currentPrice, hoveredPrice, keyLevels, legs, priceRange, showBreakeven, showMaxLoss]);

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    
    const padding = { left: 80, right: 80 };
    const chartWidth = rect.width - padding.left - padding.right;
    
    const price = priceRange.min + ((x - padding.left) / chartWidth) * (priceRange.max - priceRange.min);
    
    if (price >= priceRange.min && price <= priceRange.max) {
      setHoveredPrice(price);
    }
  }, [priceRange]);

  const handleMouseLeave = useCallback(() => {
    setHoveredPrice(null);
  }, []);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (hoveredPrice !== null && onPriceSelect) {
      onPriceSelect(hoveredPrice);
    }
  }, [hoveredPrice, onPriceSelect]);

  return (
    <div ref={containerRef} className={`relative w-full h-full ${className}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair"
        style={{ width: '100%', height: '100%' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
      
      {/* Legend Overlay */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 pointer-events-none">
        <div className="flex gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500/30 border border-green-500"></div>
            <span className="text-green-400">PROFIT ZONE</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500/30 border border-red-500"></div>
            <span className="text-red-400">LOSS ZONE</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span className="text-yellow-400">BREAKEVEN</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PayoffDiagram;
