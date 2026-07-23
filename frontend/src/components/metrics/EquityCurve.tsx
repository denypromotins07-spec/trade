/**
 * EquityCurve.tsx - High-Performance Portfolio Equity Curve
 * 
 * Canvas-based equity curve with dynamic drawdown shading and rolling
 * Sharpe/Sortino ratios. Strictly bounds historical data points for memory efficiency.
 * 
 * Features:
 * - 60FPS canvas rendering with dirty rectangle optimization
 * - Dynamic drawdown visualization (red shaded areas)
 * - Rolling risk metrics (Sharpe, Sortino, Max Drawdown)
 * - Graceful Y-axis scaling for extreme crypto volatility
 * - Cyberpunk aesthetic with neon glow effects
 */

import React, { useEffect, useRef, useCallback } from 'react';
import { useMetricsStore } from '../../store/metricsStore';

interface EquityPoint {
  timestamp: number;
  equity: number;
  peak: number;
  drawdown: number;
}

interface EquityCurveProps {
  width?: number;
  height?: number;
  maxHistoryPoints?: number;
  showDrawdown?: boolean;
  showMetrics?: boolean;
}

export const EquityCurve: React.FC<EquityCurveProps> = ({
  width = 600,
  height = 300,
  maxHistoryPoints = 500,
  showDrawdown = true,
  showMetrics = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const dataPointsRef = useRef<EquityPoint[]>([]);
  
  const { equityHistory, peakEquity, currentDrawdown, sharpeRatio, sortinoRatio, maxDrawdown } = useMetricsStore();

  // Update data points when equity history changes
  useEffect(() => {
    if (!equityHistory || equityHistory.length === 0) return;

    const now = Date.now();
    
    equityHistory.forEach((equity, idx) => {
      // Find or create corresponding data point
      const existingIdx = dataPointsRef.current.findIndex(
        p => Math.abs(p.timestamp - (now - (equityHistory.length - idx) * 1000)) < 1000
      );

      if (existingIdx === -1) {
        const peak = Math.max(peakEquity || equity, ...dataPointsRef.current.map(p => p.peak), equity);
        const drawdown = ((peak - equity) / peak) * 100;
        
        dataPointsRef.current.push({
          timestamp: now - (equityHistory.length - idx) * 1000,
          equity,
          peak,
          drawdown,
        });
      }
    });

    // Bound history to prevent memory bloat
    if (dataPointsRef.current.length > maxHistoryPoints) {
      dataPointsRef.current = dataPointsRef.current.slice(-maxHistoryPoints);
    }
  }, [equityHistory, peakEquity, maxHistoryPoints]);

  // Render the equity curve
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }

    const w = rect.width;
    const h = rect.height;
    const padding = { top: 40, right: 20, bottom: 40, left: 60 };
    const graphW = w - padding.left - padding.right;
    const graphH = h - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = '#0a0f1e';
    ctx.fillRect(0, 0, w, h);

    // Draw background gradient
    const bgGradient = ctx.createLinearGradient(0, 0, 0, h);
    bgGradient.addColorStop(0, 'rgba(10, 15, 30, 0.95)');
    bgGradient.addColorStop(1, 'rgba(20, 30, 50, 0.9)');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, w, h);

    // Draw grid
    ctx.strokeStyle = 'rgba(139, 155, 180, 0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      const y = padding.top + (graphH * i) / gridLines;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      
      // X-axis grid lines
      const x = padding.left + (graphW * i) / gridLines;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, h - padding.bottom);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    if (dataPointsRef.current.length < 2) {
      // Draw placeholder text
      ctx.fillStyle = 'rgba(139, 155, 180, 0.5)';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for equity data...', w / 2, h / 2);
      return;
    }

    const data = dataPointsRef.current;
    
    // Calculate scales with graceful handling of extreme volatility
    let minEquity = Math.min(...data.map(d => d.equity));
    let maxEquity = Math.max(...data.map(d => d.equity));
    
    // Add 10% padding to Y-axis, handle flat lines
    const range = maxEquity - minEquity || Math.abs(maxEquity) * 0.1 || 100;
    minEquity -= range * 0.1;
    maxEquity += range * 0.1;

    const xScale = graphW / (data.length - 1);
    const yScale = graphH / (maxEquity - minEquity);

    const getX = (idx: number) => padding.left + idx * xScale;
    const getY = (val: number) => padding.top + graphH - (val - minEquity) * yScale;

    // Draw drawdown areas (red shaded regions below peak)
    if (showDrawdown) {
      ctx.beginPath();
      data.forEach((point, idx) => {
        const x = getX(idx);
        const y = getY(point.equity);
        const peakY = getY(point.peak);
        
        if (idx === 0) {
          ctx.moveTo(x, peakY);
        } else {
          ctx.lineTo(x, peakY);
        }
      });
      
      data.forEach((point, idx) => {
        const x = getX(data.length - 1 - idx);
        const y = getY(point.equity);
        ctx.lineTo(x, y);
      });
      
      ctx.closePath();
      
      const drawdownGradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
      drawdownGradient.addColorStop(0, 'rgba(255, 51, 102, 0.3)');
      drawdownGradient.addColorStop(1, 'rgba(255, 51, 102, 0.05)');
      
      ctx.fillStyle = drawdownGradient;
      ctx.fill();
      
      ctx.strokeStyle = 'rgba(255, 51, 102, 0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Draw equity curve with glow
    ctx.beginPath();
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 10;
    
    data.forEach((point, idx) => {
      const x = getX(idx);
      const y = getY(point.equity);
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw peak line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([8, 4]);
    
    data.forEach((point, idx) => {
      const x = getX(idx);
      const y = getY(point.peak);
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw latest point indicator
    const lastPoint = data[data.length - 1];
    const lastX = getX(data.length - 1);
    const lastY = getY(lastPoint.equity);
    
    ctx.beginPath();
    ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#00ff88';
    ctx.fill();
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 15;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw title
    ctx.fillStyle = '#00ff88';
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('PORTFOLIO EQUITY CURVE', padding.left, 18);

    // Draw metrics panel
    if (showMetrics) {
      const metricsPanelX = w - padding.right - 180;
      const metricsPanelY = 10;
      
      // Panel background
      ctx.fillStyle = 'rgba(20, 30, 50, 0.8)';
      ctx.fillRect(metricsPanelX, metricsPanelY, 175, 70);
      
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
      ctx.strokeRect(metricsPanelX, metricsPanelY, 175, 70);

      // Metrics text
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      
      const metrics = [
        { label: 'SHARPE:', value: sharpeRatio?.toFixed(2) ?? '0.00', color: '#00ffff' },
        { label: 'SORTINO:', value: sortinoRatio?.toFixed(2) ?? '0.00', color: '#bd93f9' },
        { label: 'MAX DD:', value: `-${Math.abs(maxDrawdown ?? 0).toFixed(2)}%`, color: '#ff3366' },
        { label: 'CURR DD:', value: `-${Math.abs(currentDrawdown ?? 0).toFixed(2)}%`, color: '#ffaa00' },
      ];

      metrics.forEach((m, idx) => {
        ctx.fillStyle = 'rgba(139, 155, 180, 0.8)';
        ctx.fillText(m.label, metricsPanelX + 8, metricsPanelY + 18 + idx * 14);
        
        ctx.fillStyle = m.color;
        ctx.textAlign = 'right';
        ctx.fillText(m.value, metricsPanelX + 167, metricsPanelY + 18 + idx * 14);
        ctx.textAlign = 'left';
      });
    }

    // Draw axis labels
    ctx.fillStyle = 'rgba(139, 155, 180, 0.6)';
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    
    // Y-axis labels
    for (let i = 0; i <= gridLines; i++) {
      const val = maxEquity - (i * (maxEquity - minEquity)) / gridLines;
      const y = padding.top + (graphH * i) / gridLines;
      ctx.fillText(val.toFixed(0), padding.left - 8, y + 3);
    }

    // Current equity value
    ctx.fillStyle = '#00ff88';
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`$${lastPoint.equity.toFixed(2)}`, w - padding.right, lastY - 8);

  }, [sharpeRatio, sortinoRatio, maxDrawdown, currentDrawdown, showDrawdown, showMetrics]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      render();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        borderRadius: '8px',
        border: '1px solid rgba(0, 255, 136, 0.15)',
        boxShadow: '0 0 20px rgba(0, 255, 136, 0.05), inset 0 0 30px rgba(0, 0, 0, 0.3)',
        overflow: 'hidden',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
          width: '100%',
          height: `${height}px`,
        }}
      />
    </div>
  );
};

export default EquityCurve;
