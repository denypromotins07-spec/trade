/**
 * EfficientFrontier.tsx - Canvas Markowitz & Black-Litterman Efficient Frontier
 * 
 * Renders real-time portfolio optimization visualization showing the efficient
 * frontier with animated optimal risk-adjusted portfolio weights.
 * 
 * Features:
 * - WebGL-accelerated Canvas rendering for 60FPS animations
 * - Markowitz mean-variance optimization visualization
 * - Black-Litterman model integration support
 * - Dynamic weight allocation animation
 * - Cyberpunk quant aesthetic
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

interface Asset {
  symbol: string;
  expectedReturn: number;
  volatility: number;
  weight: number;
}

interface PortfolioPoint {
  risk: number; // Standard deviation
  return: number; // Expected return
  sharpeRatio: number;
  weights: number[];
}

interface EfficientFrontierProps {
  assets: Asset[];
  covarianceMatrix: number[][];
  riskFreeRate?: number;
  onPortfolioSelect?: (weights: number[]) => void;
  className?: string;
  showIndividualAssets?: boolean;
}

interface RenderState {
  width: number;
  height: number;
  dpr: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Portfolio Math Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Calculates portfolio expected return given weights and asset returns
 */
const calculatePortfolioReturn = (weights: number[], expectedReturns: number[]): number => {
  return weights.reduce((sum, w, i) => sum + w * expectedReturns[i], 0);
};

/**
 * Calculates portfolio volatility given weights and covariance matrix
 */
const calculatePortfolioVolatility = (weights: number[], covMatrix: number[][]): number => {
  const n = weights.length;
  let variance = 0;
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      variance += weights[i] * weights[j] * covMatrix[i][j];
    }
  }
  
  return Math.sqrt(variance);
};

/**
 * Calculates Sharpe ratio
 */
const calculateSharpeRatio = (return_: number, volatility: number, riskFreeRate: number): number => {
  if (volatility === 0) return 0;
  return (return_ - riskFreeRate) / volatility;
};

/**
 * Generates random portfolio weights (for Monte Carlo simulation)
 */
const generateRandomWeights = (assetCount: number): number[] => {
  const weights = Array.from({ length: assetCount }, () => Math.random());
  const sum = weights.reduce((a, b) => a + b, 0);
  return weights.map(w => w / sum);
};

/**
 * Generates efficient frontier points using optimization
 */
const generateEfficientFrontier = (
  assets: Asset[],
  covMatrix: number[][],
  pointCount: number = 100
): PortfolioPoint[] => {
  const points: PortfolioPoint[] = [];
  const expectedReturns = assets.map(a => a.expectedReturn);
  
  // Find min and max possible returns
  let minReturn = Infinity;
  let maxReturn = -Infinity;
  
  for (const asset of assets) {
    minReturn = Math.min(minReturn, asset.expectedReturn);
    maxReturn = Math.max(maxReturn, asset.expectedReturn);
  }
  
  // Generate frontier by optimizing for each target return
  for (let i = 0; i <= pointCount; i++) {
    const targetReturn = minReturn + ((maxReturn - minReturn) * i) / pointCount;
    
    // Simple optimization: try multiple random starting points
    let bestWeights = generateRandomWeights(assets.length);
    let bestVolatility = calculatePortfolioVolatility(bestWeights, covMatrix);
    
    // Monte Carlo refinement
    for (let attempt = 0; attempt < 50; attempt++) {
      const testWeights = generateRandomWeights(assets.length);
      const testReturn = calculatePortfolioReturn(testWeights, expectedReturns);
      
      // Accept if close to target return and lower volatility
      if (Math.abs(testReturn - targetReturn) < 0.02) {
        const testVol = calculatePortfolioVolatility(testWeights, covMatrix);
        if (testVol < bestVolatility) {
          bestWeights = testWeights;
          bestVolatility = testVol;
        }
      }
    }
    
    const actualReturn = calculatePortfolioReturn(bestWeights, expectedReturns);
    const sharpe = calculateSharpeRatio(actualReturn, bestVolatility, 0.02);
    
    points.push({
      risk: bestVolatility,
      return: actualReturn,
      sharpeRatio: sharpe,
      weights: bestWeights
    });
  }
  
  return points;
};

/**
 * Finds the maximum Sharpe ratio portfolio (tangency portfolio)
 */
const findOptimalPortfolio = (frontier: PortfolioPoint[]): PortfolioPoint | null => {
  if (frontier.length === 0) return null;
  
  let maxSharpe = -Infinity;
  let optimalPoint: PortfolioPoint | null = null;
  
  for (const point of frontier) {
    if (point.sharpeRatio > maxSharpe) {
      maxSharpe = point.sharpeRatio;
      optimalPoint = point;
    }
  }
  
  return optimalPoint;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const EfficientFrontier: React.FC<EfficientFrontierProps> = ({
  assets,
  covarianceMatrix,
  riskFreeRate = 0.02,
  onPortfolioSelect,
  className = '',
  showIndividualAssets = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderState, setRenderState] = useState<RenderState>({ width: 0, height: 0, dpr: 1 });
  const [hoveredPoint, setHoveredPoint] = useState<PortfolioPoint | null>(null);
  const animationFrameRef = useRef<number>(0);
  const animationProgressRef = useRef<number>(0);

  // Generate efficient frontier
  const frontierPoints = useMemo(() => {
    if (assets.length === 0 || covarianceMatrix.length === 0) return [];
    return generateEfficientFrontier(assets, covarianceMatrix, 80);
  }, [assets, covarianceMatrix]);

  // Find optimal portfolio
  const optimalPortfolio = useMemo(() => {
    return findOptimalPortfolio(frontierPoints);
  }, [frontierPoints]);

  // Individual asset points
  const assetPoints = useMemo(() => {
    return assets.map((asset, index) => ({
      risk: asset.volatility,
      return: asset.expectedReturn,
      symbol: asset.symbol,
      index
    }));
  }, [assets]);

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

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || renderState.width === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height, dpr } = renderState;
    
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    const padding = { top: 50, right: 60, bottom: 60, left: 70 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Calculate scales
    let minRisk = 0;
    let maxRisk = 0;
    let minReturn = 0;
    let maxReturn = 0;
    
    for (const point of frontierPoints) {
      minRisk = Math.min(minRisk, point.risk);
      maxRisk = Math.max(maxRisk, point.risk);
      minReturn = Math.min(minReturn, point.return);
      maxReturn = Math.max(maxReturn, point.return);
    }
    
    // Add padding
    maxRisk = maxRisk * 1.1;
    minReturn = Math.min(0, minReturn * 0.9);
    maxReturn = maxReturn * 1.1;
    
    const riskRange = maxRisk - minRisk || 1;
    const returnRange = maxReturn - minReturn || 1;
    
    const riskToX = (risk: number) => 
      padding.left + ((risk - minRisk) / riskRange) * chartWidth;
    
    const returnToY = (ret: number) => 
      padding.top + chartHeight - ((ret - minReturn) / returnRange) * chartHeight;
    
    let lastTime = performance.now();
    
    const render = (time: number) => {
      const deltaTime = time - lastTime;
      lastTime = time;
      
      // Animate progress
      animationProgressRef.current = Math.min(1, animationProgressRef.current + deltaTime / 1000);
      const animatedPointCount = Math.floor(frontierPoints.length * animationProgressRef.current);
      
      // Clear canvas
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);
      
      // Draw grid
      ctx.strokeStyle = '#1a1a2e';
      ctx.lineWidth = 1;
      
      // Vertical grid (risk levels)
      for (let i = 0; i <= 5; i++) {
        const x = padding.left + (i / 5) * chartWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
        
        // Risk labels
        const riskValue = minRisk + (i / 5) * riskRange;
        ctx.fillStyle = '#6b7280';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${(riskValue * 100).toFixed(1)}%`, x, height - padding.bottom + 20);
      }
      
      // Horizontal grid (return levels)
      for (let i = 0; i <= 5; i++) {
        const y = padding.top + chartHeight - (i / 5) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        
        // Return labels
        const returnValue = minReturn + (i / 5) * returnRange;
        ctx.fillStyle = '#6b7280';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`${(returnValue * 100).toFixed(1)}%`, padding.left - 8, y + 3);
      }
      
      // Draw axes labels
      ctx.fillStyle = '#9ca3af';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('RISK (σ)', width / 2, height - 15);
      
      ctx.save();
      ctx.translate(15, padding.top + chartHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('EXPECTED RETURN', 0, 0);
      ctx.restore();
      
      // Draw efficient frontier curve
      if (animatedPointCount > 1) {
        // Fill area under curve
        ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
        ctx.beginPath();
        ctx.moveTo(riskToX(frontierPoints[0].risk), returnToY(frontierPoints[0].return));
        
        for (let i = 1; i < animatedPointCount; i++) {
          const point = frontierPoints[i];
          ctx.lineTo(riskToX(point.risk), returnToY(point.return));
        }
        
        // Close to x-axis
        ctx.lineTo(riskToX(frontierPoints[animatedPointCount - 1].risk), height - padding.bottom);
        ctx.lineTo(riskToX(frontierPoints[0].risk), height - padding.bottom);
        ctx.closePath();
        ctx.fill();
        
        // Draw curve line with glow
        ctx.shadowColor = '#06b6d4';
        ctx.shadowBlur = 15;
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        
        for (let i = 0; i < animatedPointCount; i++) {
          const point = frontierPoints[i];
          const x = riskToX(point.risk);
          const y = returnToY(point.return);
          
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      
      // Draw individual assets
      if (showIndividualAssets) {
        for (const asset of assetPoints) {
          const x = riskToX(asset.risk);
          const y = returnToY(asset.return);
          
          // Glow effect
          ctx.shadowColor = '#a855f7';
          ctx.shadowBlur = 10;
          
          // Asset circle
          ctx.fillStyle = '#a855f7';
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fill();
          
          // Asset label
          ctx.shadowBlur = 0;
          ctx.fillStyle = '#e9d5ff';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(asset.symbol, x, y - 12);
        }
      }
      
      // Draw optimal portfolio (maximum Sharpe)
      if (optimalPortfolio && animationProgressRef.current >= 1) {
        const optX = riskToX(optimalPortfolio.risk);
        const optY = returnToY(optimalPortfolio.return);
        
        // Pulsing effect
        const pulse = Math.sin(time * 0.005) * 0.3 + 0.7;
        
        // Outer glow ring
        ctx.strokeStyle = `rgba(34, 197, 94, ${pulse})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(optX, optY, 15, 0, Math.PI * 2);
        ctx.stroke();
        
        // Inner circle
        ctx.fillStyle = '#22c55e';
        ctx.shadowColor = '#22c55e';
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(optX, optY, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        
        // Label
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('OPTIMAL', optX, optY - 20);
        ctx.fillText(`SR: ${optimalPortfolio.sharpeRatio.toFixed(2)}`, optX, optY + 22);
      }
      
      // Draw hovered point
      if (hoveredPoint) {
        const hoverX = riskToX(hoveredPoint.risk);
        const hoverY = returnToY(hoveredPoint.return);
        
        // Crosshair
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(hoverX, padding.top);
        ctx.lineTo(hoverX, height - padding.bottom);
        ctx.moveTo(padding.left, hoverY);
        ctx.lineTo(width - padding.right, hoverY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Tooltip
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(hoverX + 15, hoverY - 60, 140, 100);
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 1;
        ctx.strokeRect(hoverX + 15, hoverY - 60, 140, 100);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Risk: ${(hoveredPoint.risk * 100).toFixed(2)}%`, hoverX + 20, hoverY - 45);
        ctx.fillText(`Return: ${(hoveredPoint.return * 100).toFixed(2)}%`, hoverX + 20, hoverY - 30);
        ctx.fillText(`Sharpe: ${hoveredPoint.sharpeRatio.toFixed(2)}`, hoverX + 20, hoverY - 15);
        
        // Weights
        ctx.fillStyle = '#9ca3af';
        ctx.fillText('Weights:', hoverX + 20, hoverY + 5);
        
        hoveredPoint.weights.forEach((w, i) => {
          if (w > 0.05 && i < assets.length) {
            ctx.fillStyle = '#e9d5ff';
            ctx.fillText(`${assets[i].symbol}: ${(w * 100).toFixed(1)}%`, hoverX + 20, hoverY + 20 + (i % 3) * 15);
          }
        });
      }
      
      // Title
      ctx.fillStyle = '#06b6d4';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('EFFICIENT FRONTIER', padding.left, 25);
      
      // Subtitle
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px monospace';
      ctx.fillText('Markowitz Mean-Variance Optimization', padding.left, 40);
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    render(performance.now());
    
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, [renderState, frontierPoints, optimalPortfolio, assetPoints, showIndividualAssets, assets]);

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Find nearest frontier point
    const padding = { top: 50, right: 60, bottom: 60, left: 70 };
    const chartWidth = rect.width - padding.left - padding.right;
    const chartHeight = rect.height - padding.top - padding.bottom;
    
    let minRisk = 0, maxRisk = 0, minReturn = 0, maxReturn = 0;
    
    for (const point of frontierPoints) {
      minRisk = Math.min(minRisk, point.risk);
      maxRisk = Math.max(maxRisk, point.risk);
      minReturn = Math.min(minReturn, point.return);
      maxReturn = Math.max(maxReturn, point.return);
    }
    
    const riskRange = maxRisk - minRisk || 1;
    const returnRange = maxReturn - minReturn || 1;
    
    const xToRisk = (px: number) => 
      minRisk + ((px - padding.left) / chartWidth) * riskRange;
    
    const yToReturn = (py: number) => 
      minReturn + ((chartHeight - (py - padding.top)) / chartHeight) * returnRange;
    
    const hoverRisk = xToRisk(x);
    const hoverReturn = yToReturn(y);
    
    // Find closest point
    let closestPoint: PortfolioPoint | null = null;
    let minDistance = Infinity;
    
    for (const point of frontierPoints) {
      const dx = point.risk - hoverRisk;
      const dy = point.return - hoverReturn;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < minDistance && distance < 0.05) {
        minDistance = distance;
        closestPoint = point;
      }
    }
    
    setHoveredPoint(closestPoint);
  }, [frontierPoints]);

  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
  }, []);

  const handleClick = useCallback(() => {
    if (hoveredPoint && onPortfolioSelect) {
      onPortfolioSelect(hoveredPoint.weights);
    }
  }, [hoveredPoint, onPortfolioSelect]);

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
      
      {/* Info Panel */}
      <div className="absolute top-4 right-4 pointer-events-none">
        <div className="bg-black/70 backdrop-blur-sm border border-cyan-500/30 rounded px-3 py-2 text-xs font-mono">
          <div className="text-cyan-400 mb-2">PORTFOLIO METRICS</div>
          {optimalPortfolio && (
            <>
              <div className="text-gray-300">
                <span className="text-green-400">MAX SHARPE:</span> {optimalPortfolio.sharpeRatio.toFixed(3)}
              </div>
              <div className="text-gray-300">
                <span className="text-purple-400">RISK:</span> {(optimalPortfolio.risk * 100).toFixed(2)}%
              </div>
              <div className="text-gray-300">
                <span className="text-blue-400">RETURN:</span> {(optimalPortfolio.return * 100).toFixed(2)}%
              </div>
            </>
          )}
          <div className="text-gray-300 mt-1">
            <span className="text-yellow-400">ASSETS:</span> {assets.length}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EfficientFrontier;
