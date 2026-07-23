/**
 * RewardGraph.tsx - RL Reward & Episodic Return Visualization
 * 
 * Live plot of cumulative rewards, episodic returns, and value function
 * approximations using a custom double-buffered Canvas renderer.
 * 
 * Features:
 * - Double-buffered rendering to prevent flickering
 * - Bounded historical data points for memory efficiency
 * - Real-time Critic network value tracking
 * - Cyberpunk aesthetic with neon glow effects
 */

import React, { useEffect, useRef, useCallback } from 'react';
import { useAgentStore } from '../../store/agentStore';

interface DataPoint {
  timestamp: number;
  reward: number;
  cumulativeReward: number;
  episodicReturn: number;
  valueEstimate: number;
}

interface RewardGraphProps {
  width?: number;
  height?: number;
  maxHistoryPoints?: number;
}

export const RewardGraph: React.FC<RewardGraphProps> = ({
  width = 500,
  height = 250,
  maxHistoryPoints = 200,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const dataPointsRef = useRef<DataPoint[]>([]);
  
  const { rewards, cumulativeReward, episodicReturns, valueFunction } = useAgentStore();

  // Initialize offscreen canvas for double buffering
  useEffect(() => {
    offscreenCanvasRef.current = document.createElement('canvas');
    
    return () => {
      offscreenCanvasRef.current = null;
    };
  }, []);

  // Update data points when new rewards arrive
  useEffect(() => {
    if (!rewards || rewards.length === 0) return;

    const now = Date.now();
    const latestReward = rewards[rewards.length - 1];
    
    const newPoint: DataPoint = {
      timestamp: now,
      reward: latestReward,
      cumulativeReward: cumulativeReward || 0,
      episodicReturn: episodicReturns?.[episodicReturns.length - 1] || 0,
      valueEstimate: valueFunction?.[valueFunction.length - 1] || 0,
    };

    dataPointsRef.current.push(newPoint);

    // Bound history to prevent memory bloat
    if (dataPointsRef.current.length > maxHistoryPoints) {
      dataPointsRef.current = dataPointsRef.current.slice(-maxHistoryPoints);
    }
  }, [rewards, cumulativeReward, episodicReturns, valueFunction, maxHistoryPoints]);

  // Render with double buffering
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const offscreen = offscreenCanvasRef.current;
    
    if (!canvas || !offscreen) return;

    const ctx = canvas.getContext('2d');
    const offCtx = offscreen.getContext('2d');
    
    if (!ctx || !offCtx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      offscreen.width = canvas.width;
      offscreen.height = canvas.height;
      ctx.scale(dpr, dpr);
      offCtx.scale(dpr, dpr);
    }

    const w = rect.width;
    const h = rect.height;
    const padding = { top: 30, right: 20, bottom: 30, left: 50 };
    const graphW = w - padding.left - padding.right;
    const graphH = h - padding.top - padding.bottom;

    // Clear offscreen canvas
    offCtx.fillStyle = '#0a0f1e';
    offCtx.fillRect(0, 0, w, h);

    // Draw background gradient
    const bgGradient = offCtx.createLinearGradient(0, 0, 0, h);
    bgGradient.addColorStop(0, 'rgba(10, 15, 30, 0.95)');
    bgGradient.addColorStop(1, 'rgba(20, 30, 50, 0.9)');
    offCtx.fillStyle = bgGradient;
    offCtx.fillRect(0, 0, w, h);

    // Draw grid
    offCtx.strokeStyle = 'rgba(139, 155, 180, 0.15)';
    offCtx.lineWidth = 1;
    offCtx.setLineDash([4, 4]);

    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      const y = padding.top + (graphH * i) / gridLines;
      offCtx.beginPath();
      offCtx.moveTo(padding.left, y);
      offCtx.lineTo(w - padding.right, y);
      offCtx.stroke();
    }
    offCtx.setLineDash([]);

    if (dataPointsRef.current.length < 2) {
      // Copy to main canvas
      ctx.drawImage(offscreen, 0, 0);
      return;
    }

    const data = dataPointsRef.current;
    
    // Calculate scales
    const allValues = data.flatMap(d => [d.cumulativeReward, d.episodicReturn, d.valueEstimate]);
    let minVal = Math.min(...allValues);
    let maxVal = Math.max(...allValues);
    
    // Add padding to Y-axis
    const range = maxVal - minVal || 1;
    minVal -= range * 0.1;
    maxVal += range * 0.1;

    const xScale = graphW / (data.length - 1);
    const yScale = graphH / (maxVal - minVal);

    const getX = (idx: number) => padding.left + idx * xScale;
    const getY = (val: number) => padding.top + graphH - (val - minVal) * yScale;

    // Draw Cumulative Reward line (cyan)
    offCtx.beginPath();
    offCtx.strokeStyle = '#00ffff';
    offCtx.lineWidth = 2;
    offCtx.shadowColor = '#00ffff';
    offCtx.shadowBlur = 8;
    
    data.forEach((point, idx) => {
      const x = getX(idx);
      const y = getY(point.cumulativeReward);
      if (idx === 0) {
        offCtx.moveTo(x, y);
      } else {
        offCtx.lineTo(x, y);
      }
    });
    offCtx.stroke();
    offCtx.shadowBlur = 0;

    // Fill area under cumulative reward
    const fillGradient = offCtx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    fillGradient.addColorStop(0, 'rgba(0, 255, 255, 0.3)');
    fillGradient.addColorStop(1, 'rgba(0, 255, 255, 0.05)');
    
    offCtx.beginPath();
    data.forEach((point, idx) => {
      const x = getX(idx);
      const y = getY(point.cumulativeReward);
      if (idx === 0) {
        offCtx.moveTo(x, y);
      } else {
        offCtx.lineTo(x, y);
      }
    });
    offCtx.lineTo(getX(data.length - 1), h - padding.bottom);
    offCtx.lineTo(padding.left, h - padding.bottom);
    offCtx.closePath();
    offCtx.fillStyle = fillGradient;
    offCtx.fill();

    // Draw Episodic Return line (green)
    offCtx.beginPath();
    offCtx.strokeStyle = '#00ff88';
    offCtx.lineWidth = 1.5;
    offCtx.shadowColor = '#00ff88';
    offCtx.shadowBlur = 6;
    
    data.forEach((point, idx) => {
      const x = getX(idx);
      const y = getY(point.episodicReturn);
      if (idx === 0) {
        offCtx.moveTo(x, y);
      } else {
        offCtx.lineTo(x, y);
      }
    });
    offCtx.stroke();
    offCtx.shadowBlur = 0;

    // Draw Value Function estimate (purple)
    offCtx.beginPath();
    offCtx.strokeStyle = '#bd93f9';
    offCtx.lineWidth = 1.5;
    offCtx.setLineDash([6, 4]);
    
    data.forEach((point, idx) => {
      const x = getX(idx);
      const y = getY(point.valueEstimate);
      if (idx === 0) {
        offCtx.moveTo(x, y);
      } else {
        offCtx.lineTo(x, y);
      }
    });
    offCtx.stroke();
    offCtx.setLineDash([]);

    // Draw latest point indicators
    const lastIdx = data.length - 1;
    const lastPoint = data[lastIdx];
    
    // Cumulative reward dot
    offCtx.beginPath();
    offCtx.arc(getX(lastIdx), getY(lastPoint.cumulativeReward), 4, 0, Math.PI * 2);
    offCtx.fillStyle = '#00ffff';
    offCtx.fill();
    offCtx.shadowColor = '#00ffff';
    offCtx.shadowBlur = 10;
    offCtx.stroke();
    offCtx.shadowBlur = 0;

    // Draw title
    offCtx.fillStyle = '#00ffff';
    offCtx.font = 'bold 10px "JetBrains Mono", monospace';
    offCtx.textAlign = 'left';
    offCtx.fillText('REWARD & VALUE FUNCTION', padding.left, 16);

    // Draw legend
    const legendItems = [
      { label: 'Cumulative', color: '#00ffff' },
      { label: 'Episodic', color: '#00ff88' },
      { label: 'Value Fn', color: '#bd93f9' },
    ];

    legendItems.forEach((item, idx) => {
      const lx = w - padding.right - 80;
      const ly = 14 + idx * 14;
      
      offCtx.fillStyle = item.color;
      offCtx.font = '9px "JetBrains Mono", monospace';
      offCtx.textAlign = 'right';
      offCtx.fillText(item.label, lx, ly);
      
      offCtx.beginPath();
      offCtx.arc(lx + 5, ly - 3, 3, 0, Math.PI * 2);
      offCtx.fillStyle = item.color;
      offCtx.fill();
    });

    // Draw current values
    if (lastPoint) {
      offCtx.fillStyle = 'rgba(139, 155, 180, 0.7)';
      offCtx.font = '9px "JetBrains Mono", monospace';
      offCtx.textAlign = 'left';
      offCtx.fillText(`CUM: ${lastPoint.cumulativeReward.toFixed(2)}`, padding.left, h - 10);
      offCtx.fillText(`EP: ${lastPoint.episodicReturn.toFixed(2)}`, padding.left + 100, h - 10);
      offCtx.fillText(`VAL: ${lastPoint.valueEstimate.toFixed(2)}`, padding.left + 200, h - 10);
    }

    // Copy offscreen to main canvas (double buffer swap)
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(offscreen, 0, 0);

  }, []);

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
        border: '1px solid rgba(0, 255, 255, 0.15)',
        boxShadow: '0 0 20px rgba(0, 255, 255, 0.05), inset 0 0 30px rgba(0, 0, 0, 0.3)',
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

export default RewardGraph;
