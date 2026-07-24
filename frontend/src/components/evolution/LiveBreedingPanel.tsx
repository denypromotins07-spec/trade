/**
 * Live Breeding Panel Component - Stage 56
 * Real-Time Progress Visualization | AMD GPU Utilization | Non-Blocking Updates
 * 
 * Real-time progress rings and AMD GPU utilization charts tracking the active
 * genetic breeding sweeps and walk-forward validation pipelines.
 * 
 * Constraints:
 * - requestAnimationFrame for smooth updates
 * - Web Worker offloading for calculations
 * - Zero layout thrashing
 * - SVG-based rendering for crisp visuals
 */

import React, { useRef, useEffect, useCallback, useState, useMemo } from 'react';

// Types
interface BreedingProgress {
  generation: number;
  populationSize: number;
  evaluatedCount: number;
  bestFitness: number;
  avgFitness: number;
  mutationRate: number;
  status: 'idle' | 'breeding' | 'validating' | 'promoting';
}

interface ValidationProgress {
  strategyId: string;
  currentPeriod: number;
  totalPeriods: number;
  dsrScore: number;
  oosSharpe: number;
  status: 'running' | 'passed' | 'failed';
}

interface GPUStats {
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
  powerDraw: number;
}

interface LiveBreedingPanelProps {
  breedingProgress?: BreedingProgress;
  validationProgress?: ValidationProgress[];
  gpuStats?: GPUStats;
  onPromoteStrategy?: (strategyId: string) => void;
}

// Constants
const PANEL_WIDTH = 400;
const PANEL_HEIGHT = 300;
const RING_SIZE = 80;
const UPDATE_INTERVAL_MS = 100;

// Progress Ring Component
const ProgressRing: React.FC<{
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  label?: string;
  value?: string;
}> = ({ progress, size = RING_SIZE, strokeWidth = 6, color = '#00ff88', label, value }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (Math.min(100, Math.max(0, progress)) / 100) * circumference;
  
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        {/* Background ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#1a2f3a"
          strokeWidth={strokeWidth}
        />
        
        {/* Progress ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{
            transition: 'stroke-dashoffset 0.3s ease-out',
          }}
        />
        
        {/* Center value */}
        {value && (
          <text
            x={size / 2}
            y={size / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            fill={color}
            fontSize="12"
            fontWeight="bold"
            style={{ transform: 'rotate(90deg)', transformOrigin: 'center' }}
          >
            {value}
          </text>
        )}
      </svg>
      
      {label && (
        <span style={{
          marginTop: 4,
          fontSize: 10,
          color: '#667788',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
        }}>
          {label}
        </span>
      )}
    </div>
  );
};

// GPU Bar Component
const GPUBar: React.FC<{
  label: string;
  value: number;
  max: number;
  unit: string;
  color: string;
}> = ({ label, value, max, unit, color }) => {
  const percentage = Math.min(100, (value / max) * 100);
  
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 10,
        color: '#667788',
        marginBottom: 2,
      }}>
        <span>{label}</span>
        <span style={{ color }}>{value.toFixed(1)}{unit}</span>
      </div>
      <div style={{
        height: 4,
        background: '#1a2f3a',
        borderRadius: 2,
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${percentage}%`,
          height: '100%',
          background: color,
          transition: 'width 0.2s ease-out',
        }} />
      </div>
    </div>
  );
};

// Validation Status Badge
const ValidationBadge: React.FC<{ status: ValidationProgress['status'] }> = ({ status }) => {
  const colors = {
    running: '#0088ff',
    passed: '#00ff88',
    failed: '#ff0044',
  };
  
  const labels = {
    running: 'RUNNING',
    passed: 'PASSED',
    failed: 'FAILED',
  };
  
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: 3,
      fontSize: 9,
      fontWeight: 'bold',
      color: '#0a0f14',
      background: colors[status],
      textTransform: 'uppercase',
      letterSpacing: 0.5,
    }}>
      {labels[status]}
    </span>
  );
};

// Main component
export const LiveBreedingPanel: React.FC<LiveBreedingPanelProps> = ({
  breedingProgress,
  validationProgress = [],
  gpuStats,
  onPromoteStrategy,
}) => {
  const [smoothedProgress, setSmoothedProgress] = useState<BreedingProgress | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Smooth progress updates using RAF
  useEffect(() => {
    if (!breedingProgress) return;
    
    const animate = () => {
      setSmoothedProgress(prev => {
        if (!prev) return breedingProgress;
        
        // Lerp for smooth transitions
        return {
          ...breedingProgress,
          evaluatedCount: prev.evaluatedCount + 
            (breedingProgress.evaluatedCount - prev.evaluatedCount) * 0.3,
          bestFitness: prev.bestFitness + 
            (breedingProgress.bestFitness - prev.bestFitness) * 0.2,
        };
      });
      
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    animationFrameRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [breedingProgress]);
  
  // Calculate evaluation percentage
  const evalPercentage = useMemo(() => {
    if (!smoothedProgress || smoothedProgress.populationSize === 0) return 0;
    return (smoothedProgress.evaluatedCount / smoothedProgress.populationSize) * 100;
  }, [smoothedProgress]);
  
  return (
    <div className="live-breeding-panel" style={{
      width: PANEL_WIDTH,
      background: '#0a0f14',
      border: '1px solid rgba(0, 255, 136, 0.3)',
      borderRadius: 4,
      padding: 16,
      fontFamily: 'system-ui, sans-serif',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: 16,
        paddingBottom: 12,
        borderBottom: '1px solid rgba(0, 255, 136, 0.2)',
      }}>
        <span style={{ fontSize: 14, fontWeight: 'bold', color: '#00ff88' }}>
          🧬 LIVE BREEDING
        </span>
        
        {smoothedProgress && (
          <span style={{
            marginLeft: 'auto',
            fontSize: 10,
            padding: '2px 8px',
            borderRadius: 3,
            background: 'rgba(0, 255, 136, 0.1)',
            color: '#00ff88',
            textTransform: 'uppercase',
          }}>
            Gen {smoothedProgress.generation}
          </span>
        )}
      </div>
      
      {/* Progress Rings Row */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-around',
        marginBottom: 20,
      }}>
        <ProgressRing
          progress={evalPercentage}
          color="#00ff88"
          label="Evaluation"
          value={`${Math.round(evalPercentage)}%`}
        />
        
        <ProgressRing
          progress={smoothedProgress?.bestFitness ? Math.min(100, (smoothedProgress.bestFitness + 1) * 50) : 0}
          color="#0088ff"
          label="Best Fitness"
          value={smoothedProgress?.bestFitness.toFixed(2)}
        />
        
        <ProgressRing
          progress={gpuStats?.utilization || 0}
          color="#ff8800"
          label="GPU Load"
          value={`${Math.round(gpuStats?.utilization || 0)}%`}
        />
      </div>
      
      {/* GPU Stats */}
      {gpuStats && (
        <div style={{
          background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: 4,
          padding: 12,
          marginBottom: 16,
        }}>
          <div style={{
            fontSize: 10,
            color: '#ff8800',
            fontWeight: 'bold',
            marginBottom: 8,
            textTransform: 'uppercase',
          }}>
            AMD ROCm Statistics
          </div>
          
          <GPUBar
            label="Utilization"
            value={gpuStats.utilization}
            max={100}
            unit="%"
            color="#ff8800"
          />
          
          <GPUBar
            label="Memory"
            value={gpuStats.memoryUsed}
            max={gpuStats.memoryTotal}
            unit={` / ${gpuStats.memoryTotal}GB`}
            color="#0088ff"
          />
          
          <GPUBar
            label="Temperature"
            value={gpuStats.temperature}
            max={100}
            unit="°C"
            color={gpuStats.temperature > 80 ? '#ff0044' : '#00ff88'}
          />
          
          <GPUBar
            label="Power"
            value={gpuStats.powerDraw}
            max={300}
            unit="W"
            color="#aabbcc"
          />
        </div>
      )}
      
      {/* Validation Pipeline */}
      {validationProgress.length > 0 && (
        <div>
          <div style={{
            fontSize: 10,
            color: '#667788',
            fontWeight: 'bold',
            marginBottom: 8,
            textTransform: 'uppercase',
          }}>
            Walk-Forward Validation
          </div>
          
          <div style={{
            maxHeight: 120,
            overflowY: 'auto',
          }}>
            {validationProgress.slice(0, 5).map((v) => (
              <div
                key={v.strategyId}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '6px 8px',
                  background: 'rgba(0, 0, 0, 0.2)',
                  borderRadius: 3,
                  marginBottom: 4,
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 10,
                    color: '#aabbcc',
                    fontFamily: 'monospace',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}>
                    {v.strategyId.slice(0, 12)}...
                  </div>
                  
                  <div style={{
                    fontSize: 9,
                    color: '#667788',
                  }}>
                    Period {v.currentPeriod}/{v.totalPeriods} • DSR: {v.dsrScore.toFixed(2)}
                  </div>
                </div>
                
                <ValidationBadge status={v.status} />
                
                {v.status === 'passed' && onPromoteStrategy && (
                  <button
                    onClick={() => onPromoteStrategy(v.strategyId)}
                    style={{
                      marginLeft: 8,
                      padding: '2px 6px',
                      fontSize: 9,
                      background: 'rgba(0, 255, 136, 0.2)',
                      border: '1px solid #00ff88',
                      borderRadius: 3,
                      color: '#00ff88',
                      cursor: 'pointer',
                    }}
                  >
                    PROMOTE
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Status indicator */}
      {smoothedProgress && (
        <div style={{
          marginTop: 16,
          paddingTop: 12,
          borderTop: '1px solid rgba(0, 255, 136, 0.2)',
          display: 'flex',
          alignItems: 'center',
          fontSize: 10,
          color: '#667788',
        }}>
          <span style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: smoothedProgress.status === 'breeding' ? '#00ff88' : '#667788',
            marginRight: 8,
            animation: smoothedProgress.status === 'breeding' ? 'pulse 1s infinite' : 'none',
          }} />
          
          <span style={{ textTransform: 'uppercase' }}>
            Status: {smoothedProgress.status}
          </span>
          
          <span style={{ marginLeft: 'auto', fontFamily: 'monospace' }}>
            μ = {smoothedProgress.mutationRate.toFixed(2)}
          </span>
        </div>
      )}
      
      {/* Pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
};

export default LiveBreedingPanel;
