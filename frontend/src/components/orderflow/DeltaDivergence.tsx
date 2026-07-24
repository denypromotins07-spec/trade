/**
 * DeltaDivergence.tsx - Cumulative Volume Delta (CVD) Tracker
 * 
 * Visually flags absorption and exhaustion divergences against price action
 * via double-buffered Canvas rendering. Detects smart money footprints.
 * 
 * Features:
 * - Double-buffered Canvas for 60FPS updates
 * - CVD calculation and visualization
 * - Divergence detection (absorption/exhaustion)
 * - Price overlay with divergence markers
 * - Cyberpunk aesthetic
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  bidVolume: number;
  askVolume: number;
}

interface DivergencePoint {
  index: number;
  type: 'absorption' | 'exhaustion' | 'bullish' | 'bearish';
  price: number;
  cvd: number;
  strength: number;
}

interface DeltaDivergenceProps {
  candles: CandleData[];
  visibleRange?: { start: number; end: number };
  onDivergenceClick?: (divergence: DivergencePoint) => void;
  className?: string;
  showPriceOverlay?: boolean;
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
 * Calculates Cumulative Volume Delta (CVD)
 * CVD = Sum of (askVolume - bidVolume) over time
 */
const calculateCVD = (candles: CandleData[]): number[] => {
  const cvd: number[] = [];
  let cumulative = 0;
  
  for (const candle of candles) {
    const delta = candle.askVolume - candle.bidVolume;
    cumulative += delta;
    cvd.push(cumulative);
  }
  
  return cvd;
};

/**
 * Detects divergences between price and CVD
 * - Absorption: Price makes new high but CVD doesn't confirm (smart money selling)
 * - Exhaustion: Price makes new low but CVD doesn't confirm (smart money buying)
 * - Bullish Divergence: Price lower low, CVD higher low
 * - Bearish Divergence: Price higher high, CVD lower high
 */
const detectDivergences = (candles: CandleData[], cvd: number[], lookback: number = 10): DivergencePoint[] => {
  const divergences: DivergencePoint[] = [];
  
  for (let i = lookback; i < candles.length; i++) {
    const currentPrice = candles[i].close;
    const currentCVD = cvd[i];
    
    // Find price and CVD extremes in lookback window
    let maxPrice = -Infinity;
    let minPrice = Infinity;
    let maxCVD = -Infinity;
    let minCVD = Infinity;
    
    for (let j = i - lookback; j < i; j++) {
      maxPrice = Math.max(maxPrice, candles[j].high);
      minPrice = Math.min(minPrice, candles[j].low);
      maxCVD = Math.max(maxCVD, cvd[j]);
      minCVD = Math.min(minCVD, cvd[j]);
    }
    
    // Absorption: Price new high, CVD not confirming
    if (currentPrice > maxPrice && currentCVD < maxCVD * 0.95) {
      const strength = (currentPrice / maxPrice - 1) * 100 + (maxCVD - currentCVD) / Math.abs(maxCVD) * 100;
      divergences.push({
        index: i,
        type: 'absorption',
        price: currentPrice,
        cvd: currentCVD,
        strength: Math.min(strength, 100)
      });
    }
    
    // Exhaustion: Price new low, CVD not confirming
    if (currentPrice < minPrice && currentCVD > minCVD * 0.95) {
      const strength = (minPrice / currentPrice - 1) * 100 + (currentCVD - minCVD) / Math.abs(minCVD) * 100;
      divergences.push({
        index: i,
        type: 'exhaustion',
        price: currentPrice,
        cvd: currentCVD,
        strength: Math.min(strength, 100)
      });
    }
    
    // Bullish Divergence: Price lower low, CVD higher low
    if (currentPrice < minPrice && currentCVD > minCVD * 1.05) {
      divergences.push({
        index: i,
        type: 'bullish',
        price: currentPrice,
        cvd: currentCVD,
        strength: 75
      });
    }
    
    // Bearish Divergence: Price higher high, CVD lower high
    if (currentPrice > maxPrice && currentCVD < maxCVD * 0.95) {
      divergences.push({
        index: i,
        type: 'bearish',
        price: currentPrice,
        cvd: currentCVD,
        strength: 75
      });
    }
  }
  
  return divergences;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const DeltaDivergence: React.FC<DeltaDivergenceProps> = ({
  candles,
  visibleRange,
  onDivergenceClick,
  className = '',
  showPriceOverlay = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderState, setRenderState] = useState<RenderState>({ width: 0, height: 0, dpr: 1 });
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const animationFrameRef = useRef<number>(0);

  // Calculate CVD
  const cvdData = useMemo(() => {
    return calculateCVD(candles);
  }, [candles]);

  // Detect divergences
  const divergences = useMemo(() => {
    return detectDivergences(candles, cvdData, 15);
  }, [candles, cvdData]);

  // Visible range
  const visibleCandles = useMemo(() => {
    if (!visibleRange) return candles;
    const start = Math.max(0, visibleRange.start);
    const end = Math.min(candles.length, visibleRange.end);
    return candles.slice(start, end);
  }, [candles, visibleRange]);

  const visibleCVD = useMemo(() => {
    if (!visibleRange) return cvdData;
    const start = Math.max(0, visibleRange.start);
    const end = Math.min(cvdData.length, visibleRange.end);
    return cvdData.slice(start, end);
  }, [cvdData, visibleRange]);

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
    
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    const padding = { top: 40, right: 80, bottom: 50, left: 10 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const cvdChartHeight = chartHeight / 2 - 10;
    
    // Calculate CVD range
    let minCVD = Math.min(...visibleCVD);
    let maxCVD = Math.max(...visibleCVD);
    const cvdRange = maxCVD - minCVD || 1;
    
    const cvdToY = (cvd: number, baseY: number) => 
      baseY - ((cvd - minCVD) / cvdRange) * cvdChartHeight;
    
    // Calculate price range for overlay
    let minPrice = Infinity;
    let maxPrice = -Infinity;
    visibleCandles.forEach(c => {
      minPrice = Math.min(minPrice, c.low);
      maxPrice = Math.max(maxPrice, c.high);
    });
    const priceRange = maxPrice - minPrice || 1;
    
    const priceToY = (price: number, baseY: number) =>
      baseY - ((price - minPrice) / priceRange) * cvdChartHeight;
    
    let lastTime = performance.now();
    
    const render = (time: number) => {
      const deltaTime = time - lastTime;
      lastTime = time;
      
      // Clear canvas
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);
      
      // Draw titles
      ctx.fillStyle = '#06b6d4';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('CUMULATIVE VOLUME DELTA', padding.left, 25);
      
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px monospace';
      ctx.fillText(`CVD: ${visibleCVD[visibleCVD.length - 1]?.toFixed(0) || '0'}`, padding.left, 38);
      
      // Draw center divider
      const dividerY = padding.top + cvdChartHeight + 5;
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, dividerY);
      ctx.lineTo(width - padding.right, dividerY);
      ctx.stroke();
      
      // CVD labels
      ctx.fillStyle = '#9ca3af';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText('CVD+', padding.left - 5, padding.top);
      ctx.fillText('CVD-', padding.left - 5, dividerY - 5);
      
      // Draw CVD zero line
      const zeroY = cvdToY(0, dividerY);
      if (zeroY >= padding.top && zeroY <= dividerY) {
        ctx.strokeStyle = '#6b7280';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, zeroY);
        ctx.lineTo(width - padding.right, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      
      // Draw CVD line
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      
      visibleCVD.forEach((cvd, idx) => {
        const x = padding.left + (idx / visibleCVD.length) * chartWidth;
        const y = cvdToY(cvd, dividerY);
        
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
      
      // Fill area under CVD
      const gradient = ctx.createLinearGradient(padding.left, padding.top, padding.left, dividerY);
      gradient.addColorStop(0, 'rgba(6, 182, 212, 0.3)');
      gradient.addColorStop(1, 'rgba(6, 182, 212, 0.05)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.moveTo(padding.left, dividerY);
      visibleCVD.forEach((cvd, idx) => {
        const x = padding.left + (idx / visibleCVD.length) * chartWidth;
        const y = cvdToY(cvd, dividerY);
        ctx.lineTo(x, y);
      });
      ctx.lineTo(padding.left + chartWidth, dividerY);
      ctx.closePath();
      ctx.fill();
      
      // Draw price overlay (optional)
      if (showPriceOverlay) {
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        visibleCandles.forEach((candle, idx) => {
          const x = padding.left + (idx / visibleCandles.length) * chartWidth;
          const y = priceToY(candle.close, dividerY);
          
          if (idx === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      }
      
      // Draw divergence markers
      const visibleDivergences = divergences.filter(d => 
        d.index >= (visibleRange?.start || 0) && 
        d.index < (visibleRange?.end || candles.length)
      );
      
      visibleDivergences.forEach(div => {
        const localIdx = div.index - (visibleRange?.start || 0);
        const x = padding.left + (localIdx / visibleCandles.length) * chartWidth;
        const cvdY = cvdToY(div.cvd, dividerY);
        
        // Marker color based on type
        let markerColor: string;
        let label: string;
        
        switch (div.type) {
          case 'absorption':
            markerColor = '#fbbf24';
            label = 'ABS';
            break;
          case 'exhaustion':
            markerColor = '#ec4899';
            label = 'EXH';
            break;
          case 'bullish':
            markerColor = '#22c55e';
            label = 'BULL';
            break;
          case 'bearish':
            markerColor = '#ef4444';
            label = 'BEAR';
            break;
          default:
            markerColor = '#ffffff';
            label = '?';
        }
        
        // Draw marker triangle
        ctx.fillStyle = markerColor;
        ctx.beginPath();
        if (div.type === 'absorption' || div.type === 'bearish') {
          // Downward triangle
          ctx.moveTo(x - 6, cvdY - 10);
          ctx.lineTo(x + 6, cvdY - 10);
          ctx.lineTo(x, cvdY - 4);
        } else {
          // Upward triangle
          ctx.moveTo(x - 6, cvdY + 10);
          ctx.lineTo(x + 6, cvdY + 10);
          ctx.lineTo(x, cvdY + 4);
        }
        ctx.closePath();
        ctx.fill();
        
        // Label
        ctx.fillStyle = '#ffffff';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(label, x, div.type === 'absorption' || div.type === 'bearish' ? cvdY - 14 : cvdY + 20);
        
        // Hover highlight
        if (hoveredIndex === div.index) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(x, cvdY, 15, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
      
      // Draw hover line
      if (hoveredIndex !== null) {
        const localIdx = hoveredIndex - (visibleRange?.start || 0);
        if (localIdx >= 0 && localIdx < visibleCandles.length) {
          const x = padding.left + (localIdx / visibleCandles.length) * chartWidth;
          
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(x, padding.top);
          ctx.lineTo(x, dividerY);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    render(performance.now());
    
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, [renderState, visibleCVD, visibleCandles, divergences, showPriceOverlay, hoveredIndex, visibleRange, candles.length]);

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    
    const padding = { left: 10, right: 80 };
    const chartWidth = rect.width - padding.left - padding.right;
    
    const idx = Math.floor(((x - padding.left) / chartWidth) * visibleCandles.length);
    const globalIdx = idx + (visibleRange?.start || 0);
    
    if (globalIdx >= 0 && globalIdx < candles.length) {
      setHoveredIndex(globalIdx);
    } else {
      setHoveredIndex(null);
    }
  }, [visibleCandles, visibleRange, candles.length]);

  const handleMouseLeave = useCallback(() => {
    setHoveredIndex(null);
  }, []);

  const handleClick = useCallback(() => {
    if (hoveredIndex !== null) {
      const div = divergences.find(d => d.index === hoveredIndex);
      if (div) {
        onDivergenceClick?.(div);
      }
    }
  }, [hoveredIndex, divergences, onDivergenceClick]);

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
      
      {/* Legend */}
      <div className="absolute top-4 right-4 pointer-events-none">
        <div className="flex gap-3 text-xs font-mono">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-yellow-500" style={{ clipPath: 'polygon(0% 0%, 100% 0%, 50% 100%)' }}></div>
            <span className="text-yellow-400">ABSORPTION</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-pink-500" style={{ clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)' }}></div>
            <span className="text-pink-400">EXHAUSTION</span>
          </div>
        </div>
      </div>
      
      {/* Hover tooltip */}
      {hoveredIndex !== null && candles[hoveredIndex] && (
        <div className="absolute bottom-4 left-4 pointer-events-none">
          <div className="bg-black/90 backdrop-blur-sm border border-cyan-500/50 rounded px-4 py-3 text-xs font-mono">
            <div className="text-cyan-400 mb-2">CANDLE #{hoveredIndex}</div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-500">Price:</span>
                <span className="text-white">{candles[hoveredIndex].close.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">CVD:</span>
                <span className={visibleCVD[hoveredIndex - (visibleRange?.start || 0)] >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {visibleCVD[hoveredIndex - (visibleRange?.start || 0)]?.toFixed(0) || '0'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Delta:</span>
                <span className={(candles[hoveredIndex].askVolume - candles[hoveredIndex].bidVolume) >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {(candles[hoveredIndex].askVolume - candles[hoveredIndex].bidVolume).toFixed(0)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeltaDivergence;
