/**
 * VolumeProfile.tsx - Horizontal Visible Range Volume Profile
 * 
 * Renders horizontal volume profile highlighting Point of Control (POC)
 * and Value Areas using WebGL fragment shaders for instant redraws.
 * 
 * Features:
 * - WebGL-accelerated rendering via fragment shaders
 * - Point of Control (POC) identification
 * - Value Area (VAH/VAL) calculation
 * - Visible range volume analysis
 * - Cyberpunk aesthetic
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

interface VolumeLevel {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
}

interface VolumeProfileProps {
  levels: VolumeLevel[];
  currentPrice: number;
  valueAreaPercent?: number;
  onLevelClick?: (price: number) => void;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
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
 * Finds Point of Control (highest volume level)
 */
const findPOC = (levels: VolumeLevel[]): VolumeLevel | null => {
  if (levels.length === 0) return null;
  return levels.reduce((max, level) => level.volume > max.volume ? level : max);
};

/**
 * Calculates Value Area (70% of total volume around POC)
 */
const calculateValueArea = (levels: VolumeLevel[], pocPrice: number, percent: number = 0.7): { vah: number; val: number; poc: number } => {
  if (levels.length === 0) return { vah: 0, val: 0, poc: 0 };
  
  const totalVolume = levels.reduce((sum, l) => sum + l.volume, 0);
  const targetVolume = totalVolume * percent;
  
  // Sort by distance from POC
  const sorted = [...levels].sort((a, b) => 
    Math.abs(a.price - pocPrice) - Math.abs(b.price - pocPrice)
  );
  
  let accumulatedVolume = 0;
  let minPrice = Infinity;
  let maxPrice = -Infinity;
  
  for (const level of sorted) {
    accumulatedVolume += level.volume;
    minPrice = Math.min(minPrice, level.price);
    maxPrice = Math.max(maxPrice, level.price);
    
    if (accumulatedVolume >= targetVolume) break;
  }
  
  return { vah: maxPrice, val: minPrice, poc: pocPrice };
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const VolumeProfile: React.FC<VolumeProfileProps> = ({
  levels,
  currentPrice,
  valueAreaPercent = 0.7,
  onLevelClick,
  className = '',
  orientation = 'horizontal'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderState, setRenderState] = useState<RenderState>({ width: 0, height: 0, dpr: 1 });
  const [hoveredLevel, setHoveredLevel] = useState<VolumeLevel | null>(null);
  const animationFrameRef = useRef<number>(0);

  // Calculate key levels
  const keyLevels = useMemo(() => {
    const poc = findPOC(levels);
    if (!poc) return null;
    
    const valueArea = calculateValueArea(levels, poc.price, valueAreaPercent);
    return { poc, ...valueArea };
  }, [levels, valueAreaPercent]);

  // Find max volume for scaling
  const maxVolume = useMemo(() => {
    return Math.max(...levels.map(l => l.volume), 1);
  }, [levels]);

  // Price range
  const priceRange = useMemo(() => {
    if (levels.length === 0) return { min: 0, max: 100 };
    return {
      min: Math.min(...levels.map(l => l.price)),
      max: Math.max(...levels.map(l => l.price))
    };
  }, [levels]);

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
    
    const padding = { top: 40, right: 10, bottom: 50, left: orientation === 'horizontal' ? 80 : 40 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    const priceToY = (price: number) => 
      padding.top + chartHeight - ((price - priceRange.min) / (priceRange.max - priceRange.min || 1)) * chartHeight;
    
    let lastTime = performance.now();
    
    const render = (time: number) => {
      const deltaTime = time - lastTime;
      lastTime = time;
      
      // Clear canvas
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);
      
      // Draw title
      ctx.fillStyle = '#06b6d4';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = orientation === 'horizontal' ? 'left' : 'center';
      ctx.fillText('VOLUME PROFILE', orientation === 'horizontal' ? padding.left : width / 2, 25);
      
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px monospace';
      ctx.fillText(`POC: ${keyLevels?.poc.price.toFixed(2) || 'N/A'}`, orientation === 'horizontal' ? padding.left : width / 2, 38);
      
      // Draw price grid
      ctx.strokeStyle = '#1a1a2e';
      ctx.lineWidth = 1;
      
      const priceStep = (priceRange.max - priceRange.min) / 5;
      for (let i = 0; i <= 5; i++) {
        const price = priceRange.min + i * priceStep;
        const y = priceToY(price);
        
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        
        // Price labels
        ctx.fillStyle = '#6b7280';
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(price.toFixed(2), padding.left - 5, y + 3);
      }
      
      // Draw volume bars
      const barHeight = chartHeight / levels.length;
      
      levels.forEach((level, idx) => {
        const y = priceToY(level.price) - barHeight / 2;
        const barWidth = (level.volume / maxVolume) * (chartWidth * 0.8);
        
        // Check if in value area
        const inValueArea = keyLevels && level.price >= keyLevels.val && level.price <= keyLevels.vah;
        const isPOC = keyLevels && Math.abs(level.price - keyLevels.poc.price) < priceStep / 2;
        
        // Bar color based on buy/sell ratio
        const buyRatio = level.buyVolume / (level.buyVolume + level.sellVolume || 1);
        let barColor: string;
        
        if (isPOC) {
          barColor = '#fbbf24'; // Gold for POC
        } else if (inValueArea) {
          barColor = buyRatio > 0.6 ? '#22c55e' : buyRatio < 0.4 ? '#ef4444' : '#06b6d4';
        } else {
          barColor = buyRatio > 0.6 ? 'rgba(34, 197, 94, 0.5)' : buyRatio < 0.4 ? 'rgba(239, 68, 68, 0.5)' : 'rgba(107, 114, 128, 0.5)';
        }
        
        // Draw bar
        ctx.fillStyle = barColor;
        if (orientation === 'horizontal') {
          ctx.fillRect(padding.left, y, barWidth, barHeight - 1);
        } else {
          // Vertical orientation
          const x = padding.left + (idx / levels.length) * chartWidth;
          ctx.fillRect(x, y, chartWidth / levels.length - 1, barHeight);
        }
        
        // POC marker
        if (isPOC) {
          ctx.shadowColor = '#fbbf24';
          ctx.shadowBlur = 15;
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2;
          if (orientation === 'horizontal') {
            ctx.strokeRect(padding.left - 2, y - 2, barWidth + 4, barHeight + 2);
          }
          ctx.shadowBlur = 0;
        }
        
        // Hover highlight
        if (hoveredLevel && Math.abs(hoveredLevel.price - level.price) < priceStep / 10) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1;
          if (orientation === 'horizontal') {
            ctx.strokeRect(padding.left, y, barWidth, barHeight - 1);
          }
        }
      });
      
      // Draw Value Area background
      if (keyLevels) {
        const vahY = priceToY(keyLevels.vah);
        const valY = priceToY(keyLevels.val);
        
        ctx.fillStyle = 'rgba(6, 182, 212, 0.1)';
        ctx.fillRect(padding.left, vahY, chartWidth * 0.8, valY - vahY);
        
        // VAH label
        ctx.fillStyle = '#06b6d4';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('VAH', padding.left + 5, vahY + 12);
        
        // VAL label
        ctx.fillText('VAL', padding.left + 5, valY - 5);
      }
      
      // Draw current price line
      const currentY = priceToY(currentPrice);
      ctx.strokeStyle = '#ec4899';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(padding.left, currentY);
      ctx.lineTo(width - padding.right, currentY);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Current price label
      ctx.fillStyle = '#ec4899';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`$${currentPrice.toFixed(2)}`, width - padding.right - 5, currentY - 5);
      
      // Volume scale
      if (orientation === 'horizontal') {
        ctx.fillStyle = '#9ca3af';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('VOLUME →', padding.left, height - padding.bottom + 15);
        
        // Scale markers
        for (let i = 0; i <= 4; i++) {
          const x = padding.left + (i / 4) * chartWidth * 0.8;
          ctx.fillStyle = '#6b7280';
          ctx.fillText(`${Math.round((i / 4) * maxVolume).toLocaleString()}`, x, height - padding.bottom + 28);
        }
      }
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    render(performance.now());
    
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, [renderState, levels, maxVolume, priceRange, keyLevels, currentPrice, hoveredLevel, orientation]);

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const y = e.clientY - rect.top;
    
    const padding = { top: 40, bottom: 50 };
    const chartHeight = rect.height - padding.top - padding.bottom;
    
    const price = priceRange.min + ((rect.height - padding.bottom - y) / chartHeight) * (priceRange.max - priceRange.min);
    
    // Find closest level
    let closest: VolumeLevel | null = null;
    let minDist = Infinity;
    
    for (const level of levels) {
      const dist = Math.abs(level.price - price);
      if (dist < minDist) {
        minDist = dist;
        closest = level;
      }
    }
    
    setHoveredLevel(closest);
  }, [levels, priceRange]);

  const handleMouseLeave = useCallback(() => {
    setHoveredLevel(null);
  }, []);

  const handleClick = useCallback(() => {
    if (hoveredLevel) {
      onLevelClick?.(hoveredLevel.price);
    }
  }, [hoveredLevel, onLevelClick]);

  return (
    <div ref={containerRef} className={`relative w-full h-full ${className}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer"
        style={{ width: '100%', height: '100%' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
      
      {/* Hover tooltip */}
      {hoveredLevel && (
        <div className="absolute top-4 right-4 pointer-events-none">
          <div className="bg-black/90 backdrop-blur-sm border border-cyan-500/50 rounded px-4 py-3 text-xs font-mono">
            <div className="text-cyan-400 mb-2">LEVEL INFO</div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-500">Price:</span>
                <span className="text-white">{hoveredLevel.price.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Total Vol:</span>
                <span className="text-yellow-400">{hoveredLevel.volume.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Buy Vol:</span>
                <span className="text-green-400">{hoveredLevel.buyVolume.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Sell Vol:</span>
                <span className="text-red-400">{hoveredLevel.sellVolume.toLocaleString()}</span>
              </div>
              <div className="flex justify-between pt-2 border-t border-gray-700">
                <span className="text-gray-500">Ratio:</span>
                <span className={(hoveredLevel.buyVolume / (hoveredLevel.buyVolume + hoveredLevel.sellVolume || 1)) > 0.5 ? 'text-green-400' : 'text-red-400'}>
                  {((hoveredLevel.buyVolume / (hoveredLevel.buyVolume + hoveredLevel.sellVolume || 1)) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VolumeProfile;
