/**
 * RiskParityTree.tsx - Interactive Squarified Treemap for HRP Allocations
 * 
 * Visualizes Hierarchical Risk Parity (HRP) portfolio allocations using a
 * squarified treemap with real-time volatility and drawdown color coding.
 * 
 * Features:
 * - Squarified treemap algorithm for optimal rectangle aspect ratios
 * - Real-time volatility-based coloring
 * - Drawdown overlay indicators
 * - Interactive drill-down into asset clusters
 * - Cyberpunk quant aesthetic
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

interface AssetNode {
  symbol: string;
  weight: number; // Allocation percentage (0-1)
  volatility: number; // Annualized vol
  drawdown: number; // Current drawdown (negative value)
  cluster?: string; // HRP cluster name
  children?: AssetNode[];
}

interface TreemapRect {
  x: number;
  y: number;
  width: number;
  height: number;
  node: AssetNode;
}

interface RiskParityTreeProps {
  data: AssetNode | AssetNode[];
  onAssetClick?: (symbol: string) => void;
  onClusterClick?: (cluster: string) => void;
  className?: string;
  showLabels?: boolean;
  minWeight?: number;
}

interface RenderState {
  width: number;
  height: number;
  dpr: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Squarified Treemap Algorithm
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Implements the squarified treemap algorithm for optimal rectangle layout
 */
const squarify = (nodes: AssetNode[], width: number, height: number): TreemapRect[] => {
  if (nodes.length === 0) return [];
  
  // Sort by weight descending
  const sorted = [...nodes].sort((a, b) => b.weight - a.weight);
  const totalWeight = sorted.reduce((sum, n) => sum + n.weight, 0);
  
  const rects: TreemapRect[] = [];
  let row: AssetNode[] = [];
  let x = 0;
  let y = 0;
  let remainingWidth = width;
  let remainingHeight = height;
  let isHorizontal = true;
  
  const worst = (row: AssetNode[], size: number) => {
    if (row.length === 0) return Infinity;
    const rowWeight = row.reduce((sum, n) => sum + n.weight, 0);
    const areas = row.map(n => (n.weight / totalWeight) * size);
    const minArea = Math.min(...areas);
    const maxArea = Math.max(...areas);
    return Math.max(maxArea / minArea, minArea / maxArea);
  };
  
  const layoutRow = (row: AssetNode[], size: number, isHoriz: boolean) => {
    const rowWeight = row.reduce((sum, n) => sum + n.weight, 0);
    let position = isHoriz ? x : y;
    
    row.forEach(node => {
      const area = (node.weight / totalWeight) * size;
      const rectSize = area / (isHoriz ? remainingHeight : remainingWidth);
      
      rects.push({
        x: isHoriz ? x : position,
        y: isHoriz ? y : position,
        width: isHoriz ? rectSize : remainingWidth,
        height: isHoriz ? remainingHeight : rectSize,
        node
      });
      
      if (isHoriz) {
        position += rectSize;
      } else {
        position += rectSize;
      }
    });
    
    if (isHoriz) {
      x += Math.max(...row.map(n => (n.weight / totalWeight) * size / remainingHeight));
      remainingWidth -= Math.max(...row.map(n => (n.weight / totalWeight) * size / remainingHeight));
    } else {
      y += Math.max(...row.map(n => (n.weight / totalWeight) * size / remainingWidth));
      remainingHeight -= Math.max(...row.map(n => (n.weight / totalWeight) * size / remainingWidth));
    }
  };
  
  let index = 0;
  const totalSize = width * height;
  
  while (index < sorted.length) {
    if (row.length === 0) {
      row.push(sorted[index++]);
    } else {
      const currentSize = isHorizontal ? remainingWidth * height : width * remainingHeight;
      const nextWorst = worst([...row, sorted[index]], currentSize);
      const currentWorst = worst(row, currentSize);
      
      if (nextWorst <= currentWorst && index < sorted.length) {
        row.push(sorted[index++]);
      } else {
        const rowSize = isHorizontal ? remainingWidth * height : width * remainingHeight;
        layoutRow(row, rowSize, isHorizontal);
        row = [];
        isHorizontal = !isHorizontal;
      }
    }
  }
  
  if (row.length > 0) {
    const rowSize = isHorizontal ? remainingWidth * height : width * remainingHeight;
    layoutRow(row, rowSize, isHorizontal);
  }
  
  return rects;
};

// ─────────────────────────────────────────────────────────────────────────────
// Color Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Maps volatility to cyberpunk color scale
 */
const getVolatilityColor = (vol: number): string => {
  // Volatility ranges: 0-20% (low), 20-40% (med), 40-60% (high), 60%+ (extreme)
  const normalized = Math.min(vol / 0.8, 1);
  
  if (normalized < 0.33) {
    // Low vol: green to cyan
    const t = normalized / 0.33;
    const r = Math.round(34 + t * (6 - 34));
    const g = Math.round(197 + t * (182 - 197));
    const b = Math.round(94 + t * (212 - 94));
    return `rgb(${r}, ${g}, ${b})`;
  } else if (normalized < 0.66) {
    // Medium vol: cyan to purple
    const t = (normalized - 0.33) / 0.33;
    const r = Math.round(6 + t * (168 - 6));
    const g = Math.round(182 + t * (85 - 182));
    const b = Math.round(212 + t * (247 - 212));
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // High vol: purple to pink/red
    const t = (normalized - 0.66) / 0.34;
    const r = Math.round(168 + t * (239 - 168));
    const g = Math.round(85 + t * (68 - 85));
    const b = Math.round(247 + t * (68 - 247));
    return `rgb(${r}, ${g}, ${b})`;
  }
};

/**
 * Gets opacity based on drawdown severity
 */
const getDrawdownOpacity = (drawdown: number): number => {
  // More negative = more opaque overlay
  return Math.min(Math.abs(drawdown) / 0.3, 0.7);
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const RiskParityTree: React.FC<RiskParityTreeProps> = ({
  data,
  onAssetClick,
  onClusterClick,
  className = '',
  showLabels = true,
  minWeight = 0.01
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderState, setRenderState] = useState<RenderState>({ width: 0, height: 0, dpr: 1 });
  const [hoveredNode, setHoveredNode] = useState<AssetNode | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Flatten data if nested
  const flatNodes = useMemo(() => {
    if (Array.isArray(data)) {
      return data.filter(n => n.weight >= minWeight);
    }
    return [data].filter(n => n.weight >= minWeight);
  }, [data, minWeight]);

  // Calculate total weight for normalization
  const totalWeight = useMemo(() => {
    return flatNodes.reduce((sum, n) => sum + n.weight, 0);
  }, [flatNodes]);

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
    
    // Generate treemap layout
    const rects = squarify(flatNodes, width - 20, height - 80);
    
    // Offset rects to add padding
    rects.forEach(rect => {
      rect.x += 10;
      rect.y += 70;
    });
    
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
      ctx.textAlign = 'left';
      ctx.fillText('HIERARCHICAL RISK PARITY', 10, 25);
      
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px monospace';
      ctx.fillText('Volatility-Weighted Allocation', 10, 40);
      
      // Draw legend
      ctx.fillStyle = '#9ca3af';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText('LOW VOL', width - 10, 25);
      
      const gradient = ctx.createLinearGradient(width - 80, 20, width - 10, 20);
      gradient.addColorStop(0, getVolatilityColor(0.1));
      gradient.addColorStop(0.5, getVolatilityColor(0.4));
      gradient.addColorStop(1, getVolatilityColor(0.8));
      ctx.fillStyle = gradient;
      ctx.fillRect(width - 75, 15, 60, 8);
      
      ctx.fillStyle = '#9ca3af';
      ctx.textAlign = 'left';
      ctx.fillText('HIGH VOL', width - 10, 25);
      
      // Draw rectangles
      rects.forEach(rect => {
        const { x, y, width: w, height: h, node } = rect;
        
        // Skip too small rectangles
        if (w < 5 || h < 5) return;
        
        const isHovered = hoveredNode?.symbol === node.symbol;
        
        // Base color from volatility
        ctx.fillStyle = getVolatilityColor(node.volatility);
        
        // Draw rectangle
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.fill();
        
        // Draw border
        ctx.strokeStyle = isHovered ? '#ffffff' : 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.stroke();
        
        // Draw drawdown overlay (red tint for negative performance)
        if (node.drawdown < -0.05) {
          const overlayAlpha = getDrawdownOpacity(node.drawdown);
          ctx.fillStyle = `rgba(239, 68, 68, ${overlayAlpha})`;
          ctx.beginPath();
          ctx.rect(x, y, w, h);
          ctx.fill();
        }
        
        // Glow effect for hovered or high-weight nodes
        if (isHovered || node.weight > 0.2) {
          ctx.shadowColor = getVolatilityColor(node.volatility);
          ctx.shadowBlur = isHovered ? 20 : 10;
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
        
        // Draw label if large enough
        if (showLabels && w > 30 && h > 20) {
          // Symbol
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 11px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(node.symbol, x + w / 2, y + h / 2 - 5);
          
          // Weight percentage
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.font = '10px monospace';
          ctx.fillText(`${(node.weight * 100).toFixed(1)}%`, x + w / 2, y + h / 2 + 10);
          
          // Volatility indicator (small dot)
          const volDotColor = node.volatility < 0.3 ? '#22c55e' : node.volatility < 0.5 ? '#fbbf24' : '#ef4444';
          ctx.fillStyle = volDotColor;
          ctx.beginPath();
          ctx.arc(x + w - 8, y + 8, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      
      // Draw cluster groupings if present
      const clusters = new Set(flatNodes.map(n => n.cluster).filter(Boolean) as string[]);
      clusters.forEach(cluster => {
        const clusterNodes = flatNodes.filter(n => n.cluster === cluster);
        if (clusterNodes.length < 2) return;
        
        // Find bounding box for cluster
        const clusterRects = rects.filter(r => r.node.cluster === cluster);
        if (clusterRects.length === 0) return;
        
        const minX = Math.min(...clusterRects.map(r => r.x));
        const minY = Math.min(...clusterRects.map(r => r.y));
        const maxX = Math.max(...clusterRects.map(r => r.x + r.width));
        const maxY = Math.max(...clusterRects.map(r => r.y + r.height));
        
        // Draw cluster outline
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.rect(minX - 5, minY - 5, maxX - minX + 10, maxY - minY + 10);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Cluster label
        ctx.fillStyle = '#a855f7';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(cluster, minX - 3, minY - 8);
      });
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    render(performance.now());
    
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, [renderState, flatNodes, hoveredNode, showLabels]);

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Generate same layout to check hit testing
    const rects = squarify(flatNodes, rect.width - 20, rect.height - 80);
    rects.forEach(r => {
      r.x += 10;
      r.y += 70;
    });
    
    // Find hovered rectangle
    let found: AssetNode | null = null;
    for (const r of rects) {
      if (x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height) {
        found = r.node;
        break;
      }
    }
    
    setHoveredNode(found);
  }, [flatNodes]);

  const handleMouseLeave = useCallback(() => {
    setHoveredNode(null);
  }, []);

  const handleClick = useCallback(() => {
    if (hoveredNode) {
      if (hoveredNode.cluster) {
        onClusterClick?.(hoveredNode.cluster);
      }
      onAssetClick?.(hoveredNode.symbol);
    }
  }, [hoveredNode, onAssetClick, onClusterClick]);

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
      {hoveredNode && (
        <div className="absolute top-4 right-4 pointer-events-none">
          <div className="bg-black/90 backdrop-blur-sm border border-cyan-500/50 rounded px-4 py-3 text-xs font-mono min-w-[180px]">
            <div className="text-cyan-400 font-bold mb-2">{hoveredNode.symbol}</div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-500">Weight:</span>
                <span className="text-white">{(hoveredNode.weight * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Volatility:</span>
                <span className={hoveredNode.volatility > 0.5 ? 'text-red-400' : 'text-green-400'}>
                  {(hoveredNode.volatility * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Drawdown:</span>
                <span className={hoveredNode.drawdown < -0.1 ? 'text-red-400' : 'text-yellow-400'}>
                  {(hoveredNode.drawdown * 100).toFixed(2)}%
                </span>
              </div>
              {hoveredNode.cluster && (
                <div className="flex justify-between mt-2 pt-2 border-t border-gray-700">
                  <span className="text-gray-500">Cluster:</span>
                  <span className="text-purple-400">{hoveredNode.cluster}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RiskParityTree;
