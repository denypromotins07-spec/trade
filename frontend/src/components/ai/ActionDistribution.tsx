/**
 * ActionDistribution.tsx - Real-time RL Action Probability Distribution
 * 
 * Renders the probability distribution of the agent's chosen actions
 * using HTML5 Canvas for zero React re-renders during high-frequency updates.
 * 
 * Features:
 * - 60FPS canvas rendering via requestAnimationFrame
 * - Smooth interpolation between probability states
 * - Cyberpunk aesthetic with glow effects
 * - Memory-efficient buffer management
 */

import React, { useEffect, useRef, useCallback } from 'react';
import { useAgentStore } from '../../store/agentStore';

interface ActionDistProps {
  width?: number;
  height?: number;
}

const ACTION_LABELS = ['BUY', 'SELL', 'HOLD', 'SCALE_IN', 'SCALE_OUT'];
const ACTION_COLORS = [
  '#00ff88',  // BUY - Green
  '#ff3366',  // SELL - Red
  '#8b9bb4',  // HOLD - Gray
  '#00ffff',  // SCALE_IN - Cyan
  '#ffaa00',  // SCALE_OUT - Orange
];

export const ActionDistribution: React.FC<ActionDistProps> = ({
  width = 400,
  height = 200,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const currentProbsRef = useRef<number[]>(new Array(5).fill(0.2));
  const targetProbsRef = useRef<number[]>(new Array(5).fill(0.2));
  
  const { actionProbs } = useAgentStore();

  // Update target probabilities when store changes
  useEffect(() => {
    if (actionProbs && actionProbs.length === 5) {
      targetProbsRef.current = [...actionProbs];
    }
  }, [actionProbs]);

  // Linear interpolation for smooth transitions
  const lerp = (start: number, end: number, t: number): number => {
    return start + (end - start) * t;
  };

  // Render the distribution chart
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

    // Clear canvas with fade effect for trail
    ctx.fillStyle = 'rgba(10, 15, 30, 0.15)';
    ctx.fillRect(0, 0, w, h);

    // Interpolate current probabilities toward target
    const smoothing = 0.15;
    for (let i = 0; i < 5; i++) {
      currentProbsRef.current[i] = lerp(
        currentProbsRef.current[i],
        targetProbsRef.current[i],
        smoothing
      );
    }

    const barWidth = (w - 40) / 5;
    const barGap = 4;
    const maxBarHeight = h - 60;
    const baselineY = h - 30;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(139, 155, 180, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    
    for (let i = 0; i <= 4; i++) {
      const y = baselineY - (maxBarHeight * i) / 4;
      ctx.beginPath();
      ctx.moveTo(20, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();
      
      // Y-axis labels
      ctx.fillStyle = 'rgba(139, 155, 180, 0.5)';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${(i * 25).toFixed(0)}%`, 16, y + 3);
    }
    ctx.setLineDash([]);

    // Draw bars with glow effects
    currentProbsRef.current.forEach((prob, idx) => {
      const x = 20 + idx * barWidth + barGap / 2;
      const barHeight = prob * maxBarHeight;
      const y = baselineY - barHeight;
      const color = ACTION_COLORS[idx];

      // Outer glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 15;
      
      // Bar gradient
      const gradient = ctx.createLinearGradient(x, y, x, baselineY);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, `${color}40`);
      
      ctx.fillStyle = gradient;
      
      // Rounded rectangle bar
      const radius = 4;
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + barWidth - barGap - radius, y);
      ctx.quadraticCurveTo(x + barWidth - barGap, y, x + barWidth - barGap, y + radius);
      ctx.lineTo(x + barWidth - barGap, baselineY);
      ctx.lineTo(x, baselineY);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
      ctx.fill();

      // Reset shadow
      ctx.shadowBlur = 0;

      // Draw percentage value on top
      ctx.fillStyle = color;
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${(prob * 100).toFixed(1)}%`, x + barWidth / 2 - barGap / 4, y - 8);

      // Draw action label at bottom
      ctx.fillStyle = 'rgba(139, 155, 180, 0.8)';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      
      // Wrap text if needed
      const label = ACTION_LABELS[idx];
      const words = label.split('_');
      if (words.length > 1) {
        ctx.fillText(words[0], x + barWidth / 2 - barGap / 4, baselineY + 12);
        ctx.fillText(words[1], x + barWidth / 2 - barGap / 4, baselineY + 22);
      } else {
        ctx.fillText(label, x + barWidth / 2 - barGap / 4, baselineY + 16);
      }
    });

    // Draw title
    ctx.fillStyle = '#00ffff';
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('ACTION PROBABILITY DISTRIBUTION', 20, 14);

    // Draw selected action indicator
    const maxProb = Math.max(...currentProbsRef.current);
    const maxIdx = currentProbsRef.current.indexOf(maxProb);
    
    if (maxProb > 0.4) {
      ctx.fillStyle = ACTION_COLORS[maxIdx];
      ctx.font = 'bold 10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`→ ${ACTION_LABELS[maxIdx]}`, w - 20, 14);
    }

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
        background: 'linear-gradient(135deg, rgba(10, 15, 30, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%)',
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

export default ActionDistribution;
