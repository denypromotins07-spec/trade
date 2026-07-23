/**
 * GpuMonitor.tsx - AMD Radeon VRAM and DirectML/ROCm Utilization Charts
 * 
 * Tracks GPU offload for RL inference engines and highlights thermal throttling events.
 * Maps AMD DirectML/ROCm context visually to the rendering pipeline.
 * 
 * Features:
 * - Real-time VRAM usage monitoring
 * - GPU utilization percentage
 * - Thermal throttling detection
 * - ROCm/DirectML workload visualization
 * - Cyberpunk aesthetic with AMD branding colors
 */

import React, { useEffect, useRef, useCallback } from 'react';
import { useSystemStore } from '../../store/systemStore';

interface GpuMonitorProps {
  width?: number;
  height?: number;
}

export const GpuMonitor: React.FC<GpuMonitorProps> = ({
  width = 400,
  height = 280,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  const { 
    gpuUsage, 
    vramUsed, 
    vramTotal, 
    gpuTemperature, 
    rocmActive,
    directmlActive,
    thermalThrottling,
  } = useSystemStore();

  const vramPercentage = vramTotal ? (vramUsed / vramTotal) * 100 : 0;
  const isThrottling = thermalThrottling || (gpuTemperature && gpuTemperature > 85);

  // Render GPU monitor
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

    // Clear with fade effect
    ctx.fillStyle = 'rgba(10, 15, 30, 0.2)';
    ctx.fillRect(0, 0, w, h);

    // Background gradient
    const bgGradient = ctx.createLinearGradient(0, 0, 0, h);
    bgGradient.addColorStop(0, 'rgba(10, 15, 30, 0.95)');
    bgGradient.addColorStop(1, 'rgba(20, 30, 50, 0.9)');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, w, h);

    // Draw grid
    ctx.strokeStyle = 'rgba(139, 155, 180, 0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    
    for (let i = 0; i <= 4; i++) {
      const y = 40 + (i * (h - 60)) / 4;
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Title
    ctx.fillStyle = '#ff005a'; // AMD red
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('🎮 AMD RADEON GPU MONITOR', 50, 22);

    // ROCm/DirectML indicator
    if (rocmActive || directmlActive) {
      ctx.fillStyle = rocmActive ? '#ff005a' : '#00ffff';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(rocmActive ? '▶ ROCm ACTIVE' : '▶ DirectML ACTIVE', w - 130, 22);
    }

    // Thermal throttling warning
    if (isThrottling) {
      ctx.fillStyle = '#ff3366';
      ctx.font = 'bold 9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('⚠ THERMAL THROTTLING DETECTED', w / 2, 22);
      ctx.textAlign = 'left';
    }

    // Draw utilization bars
    const barWidth = (w - 100) / 2 - 20;
    const barHeight = 120;
    const barX1 = 50;
    const barX2 = 50 + barWidth + 20;
    const barY = 40;

    // GPU Utilization Bar
    const gpuUtil = gpuUsage ?? 0;
    
    // Background
    ctx.fillStyle = 'rgba(20, 30, 48, 0.8)';
    ctx.fillRect(barX1, barY, barWidth, barHeight);
    
    // Value fill with gradient
    const gpuGradient = ctx.createLinearGradient(barX1, barY + barHeight, barX1, barY);
    gpuGradient.addColorStop(0, '#ff005a');
    gpuGradient.addColorStop(0.5, '#ff4081');
    gpuGradient.addColorStop(1, '#ff80ab');
    
    ctx.fillStyle = gpuGradient;
    const gpuFillHeight = (gpuUtil / 100) * barHeight;
    ctx.fillRect(barX1, barY + barHeight - gpuFillHeight, barWidth, gpuFillHeight);
    
    // Glow effect
    ctx.shadowColor = '#ff005a';
    ctx.shadowBlur = gpuUtil > 90 ? 20 : 10;
    ctx.strokeStyle = '#ff005a';
    ctx.lineWidth = 2;
    ctx.strokeRect(barX1, barY + barHeight - gpuFillHeight, barWidth, gpuFillHeight);
    ctx.shadowBlur = 0;

    // VRAM Bar
    const vramFillHeight = (vramPercentage / 100) * barHeight;
    
    // Background
    ctx.fillStyle = 'rgba(20, 30, 48, 0.8)';
    ctx.fillRect(barX2, barY, barWidth, barHeight);
    
    // Value fill with gradient
    const vramGradient = ctx.createLinearGradient(barX2, barY + barHeight, barX2, barY);
    vramGradient.addColorStop(0, '#00ffff');
    vramGradient.addColorStop(0.5, '#00e5ff');
    vramGradient.addColorStop(1, '#80deea');
    
    ctx.fillStyle = vramGradient;
    ctx.fillRect(barX2, barY + barHeight - vramFillHeight, barWidth, vramFillHeight);
    
    // Glow effect
    ctx.shadowColor = '#00ffff';
    ctx.shadowBlur = vramPercentage > 90 ? 20 : 10;
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(barX2, barY + barHeight - vramFillHeight, barWidth, vramFillHeight);
    ctx.shadowBlur = 0;

    // Labels
    ctx.fillStyle = '#ff005a';
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('GPU UTILIZATION', barX1 + barWidth / 2, barY + barHeight + 18);
    
    ctx.fillStyle = '#00ffff';
    ctx.fillText('VRAM USAGE', barX2 + barWidth / 2, barY + barHeight + 18);

    // Values
    ctx.fillStyle = '#ff4081';
    ctx.font = 'bold 16px "JetBrains Mono", monospace';
    ctx.fillText(`${gpuUtil.toFixed(1)}%`, barX1 + barWidth / 2, barY - 8);
    
    ctx.fillStyle = '#00e5ff';
    ctx.fillText(`${vramPercentage.toFixed(1)}%`, barX2 + barWidth / 2, barY - 8);

    // VRAM amount
    ctx.fillStyle = 'rgba(139, 155, 180, 0.7)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(
      `${(vramUsed / 1024).toFixed(2)} GB / ${(vramTotal / 1024).toFixed(2)} GB`,
      barX2 + barWidth / 2,
      barY + barHeight + 32
    );

    // Temperature gauge
    const temp = gpuTemperature ?? 0;
    const tempColor = temp > 85 ? '#ff3366' : temp > 70 ? '#ffaa00' : '#00ff88';
    
    // Temperature arc
    const cx = w / 2;
    const cy = h - 50;
    const radius = 35;
    
    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, Math.PI * 0.75, Math.PI * 2.25);
    ctx.strokeStyle = 'rgba(20, 30, 48, 0.8)';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Value arc
    const tempAngle = Math.PI * 0.75 + (Math.PI * 1.5 * (Math.min(temp, 100) / 100));
    ctx.beginPath();
    ctx.arc(cx, cy, radius, Math.PI * 0.75, tempAngle);
    ctx.strokeStyle = tempColor;
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.shadowColor = tempColor;
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Temperature text
    ctx.fillStyle = tempColor;
    ctx.font = 'bold 14px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${temp.toFixed(0)}°C`, cx, cy);
    
    // Temperature label
    ctx.fillStyle = 'rgba(139, 155, 180, 0.6)';
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillText('TEMP', cx, cy + 28);

  }, [gpuUsage, vramUsed, vramTotal, gpuTemperature, rocmActive, directmlActive, isThrottling]);

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
        border: '1px solid rgba(255, 0, 90, 0.2)',
        boxShadow: '0 0 20px rgba(255, 0, 90, 0.1), inset 0 0 30px rgba(0, 0, 0, 0.3)',
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

export default GpuMonitor;
