/**
 * RamGauge.tsx - Precision RAM Usage Gauges
 * 
 * Shows exact 8GB RAM split (4GB Rust / 4GB Python Ray)
 * with visual alarms when memory limits are approached.
 * 
 * Features:
 * - Dual gauge visualization for Rust/Python split
 * - Threshold-based color coding
 * - Real-time updates via WebSocket
 * - Cyberpunk aesthetic with glowing arcs
 */

import React, { useEffect, useRef } from 'react';
import { useSystemStore } from '../../store/systemStore';

interface GaugeProps {
  label: string;
  value: number;
  max: number;
  color: string;
  warningThreshold: number;
  criticalThreshold: number;
}

const RamGaugeComponent: React.FC<GaugeProps> = ({
  label,
  value,
  max,
  color,
  warningThreshold,
  criticalThreshold,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const currentValueRef = useRef(0);

  const percentage = Math.min((value / max) * 100, 100);
  const isWarning = percentage >= warningThreshold;
  const isCritical = percentage >= criticalThreshold;
  
  const displayColor = isCritical ? '#ff3366' : isWarning ? '#ffaa00' : color;

  // Smooth animation
  useEffect(() => {
    const animate = () => {
      const diff = percentage - currentValueRef.current;
      if (Math.abs(diff) > 0.1) {
        currentValueRef.current += diff * 0.15;
      } else {
        currentValueRef.current = percentage;
      }
      
      render();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [percentage]);

  const render = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }

    const w = rect.width;
    const h = rect.height;
    const cx = w / 2;
    const cy = h / 2 + 10;
    const radius = Math.min(w, h) / 2 - 20;
    const lineWidth = 12;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, Math.PI * 0.75, Math.PI * 2.25);
    ctx.strokeStyle = 'rgba(20, 30, 48, 0.8)';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Value arc
    const startAngle = Math.PI * 0.75;
    const endAngle = startAngle + (Math.PI * 1.5 * (currentValueRef.current / 100));
    
    const gradient = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
    gradient.addColorStop(0, `${displayColor}40`);
    gradient.addColorStop(1, displayColor);
    
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.shadowColor = displayColor;
    ctx.shadowBlur = isCritical ? 20 : 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Percentage text
    ctx.fillStyle = displayColor;
    ctx.font = 'bold 18px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${currentValueRef.current.toFixed(1)}%`, cx, cy);

    // Value text
    ctx.fillStyle = 'rgba(139, 155, 180, 0.7)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(`${(value / 1024).toFixed(2)} GB`, cx, cy + 25);
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <canvas
        ref={canvasRef}
        style={{ width: '140px', height: '120px' }}
      />
      <div style={{ marginTop: '8px' }}>
        <div
          style={{
            fontSize: '9px',
            color: 'rgba(139, 155, 180, 0.7)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          {label}
        </div>
        {isCritical && (
          <div
            style={{
              fontSize: '7px',
              color: '#ff3366',
              marginTop: '4px',
              animation: 'pulse 0.5s infinite',
            }}
          >
            ⚠ CRITICAL
          </div>
        )}
      </div>
    </div>
  );
};

export const RamGauge: React.FC = () => {
  const { rustMemory, pythonMemory, totalMemory } = useSystemStore();

  const rustUsage = ((rustMemory?.used || 0) / (rustMemory?.total || 4294967296)) * 100;
  const pythonUsage = ((pythonMemory?.used || 0) / (pythonMemory?.total || 4294967296)) * 100;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        padding: '14px',
        background: 'linear-gradient(135deg, rgba(10, 15, 30, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%)',
        borderRadius: '8px',
        border: '1px solid rgba(0, 255, 136, 0.15)',
        boxShadow: '0 0 20px rgba(0, 255, 136, 0.05), inset 0 0 30px rgba(0, 0, 0, 0.3)',
        fontFamily: '"JetBrains Mono", monospace',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingBottom: '8px',
          borderBottom: '1px solid rgba(0, 255, 136, 0.2)',
        }}
      >
        <h3
          style={{
            margin: 0,
            fontSize: '12px',
            fontWeight: 600,
            color: '#00ff88',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            textShadow: '0 0 10px rgba(0, 255, 136, 0.5)',
          }}
        >
          💾 RAM Monitor
        </h3>
        <span
          style={{
            fontSize: '8px',
            color: 'rgba(139, 155, 180, 0.6)',
          }}
        >
          TOTAL: {(totalMemory / 1024 / 1024 / 1024).toFixed(1)} GB
        </span>
      </div>

      {/* Gauges */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '16px',
          justifyContent: 'center',
        }}
      >
        <RamGaugeComponent
          label="Rust Core"
          value={rustMemory?.used || 0}
          max={rustMemory?.total || 4294967296}
          color="#00ffff"
          warningThreshold={70}
          criticalThreshold={90}
        />
        <RamGaugeComponent
          label="Python Ray"
          value={pythonMemory?.used || 0}
          max={pythonMemory?.total || 4294967296}
          color="#bd93f9"
          warningThreshold={70}
          criticalThreshold={90}
        />
      </div>

      {/* Memory Bar */}
      <div style={{ marginTop: '8px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginBottom: '6px',
          }}
        >
          <span style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.6)' }}>
            COMBINED USAGE
          </span>
          <span
            style={{
              fontSize: '9px',
              color: rustUsage + pythonUsage > 140 ? '#ff3366' : rustUsage + pythonUsage > 100 ? '#ffaa00' : '#00ff88',
            }}
          >
            {(((rustMemory?.used || 0) + (pythonMemory?.used || 0)) / 1024 / 1024 / 1024).toFixed(2)} GB / {(totalMemory / 1024 / 1024 / 1024).toFixed(1)} GB
          </span>
        </div>
        <div
          style={{
            height: '8px',
            background: 'rgba(20, 30, 48, 0.8)',
            borderRadius: '4px',
            overflow: 'hidden',
            display: 'flex',
          }}
        >
          <div
            style={{
              width: `${rustUsage / 2}%`,
              background: 'linear-gradient(90deg, #00ffff, #00ff88)',
              boxShadow: '0 0 8px rgba(0, 255, 255, 0.5)',
            }}
          />
          <div
            style={{
              width: `${pythonUsage / 2}%`,
              background: 'linear-gradient(90deg, #bd93f9, #ff79c6)',
              boxShadow: '0 0 8px rgba(189, 147, 249, 0.5)',
            }}
          />
        </div>
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '16px',
            marginTop: '8px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '10px', height: '10px', background: '#00ffff', borderRadius: '2px' }} />
            <span style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.6)' }}>RUST</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '10px', height: '10px', background: '#bd93f9', borderRadius: '2px' }} />
            <span style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.6)' }}>PYTHON</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RamGauge;
