/**
 * AgentState.tsx - RL Agent Observation Space & Latent State Visualization
 * 
 * Visualizes the reinforcement learning agent's internal state including:
 * - Observation space vectors (market features, order book state)
 * - Latent state embeddings from the encoder network
 * - Confidence scores for action selection
 * 
 * Optimized for 60FPS using CSS grids and minimal DOM nodes.
 * Avoids heavy charting libraries for ultra-low latency telemetry.
 */

import React, { useEffect, useRef, useMemo } from 'react';
import { useAgentStore } from '../../store/agentStore';

interface VectorBarProps {
  label: string;
  value: number;
  color: string;
  index: number;
}

const VectorBar: React.FC<VectorBarProps> = React.memo(({ label, value, color, index }) => {
  const normalizedValue = Math.max(0, Math.min(1, Math.abs(value)));
  
  return (
    <div 
      className="agent-vector-row"
      style={{
        display: 'grid',
        gridTemplateColumns: '80px 1fr 50px',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '2px',
        opacity: 0.8 + (normalizedValue * 0.2),
      }}
    >
      <span 
        className="vector-label"
        style={{
          fontSize: '10px',
          fontFamily: '"JetBrains Mono", monospace',
          color: '#8b9bb4',
          textOverflow: 'ellipsis',
          overflow: 'hidden',
          whiteSpace: 'nowrap',
        }}
      >
        {label}
      </span>
      <div 
        className="vector-track"
        style={{
          position: 'relative',
          height: '6px',
          background: 'rgba(20, 30, 48, 0.8)',
          borderRadius: '2px',
          overflow: 'hidden',
        }}
      >
        <div
          className="vector-fill"
          style={{
            position: 'absolute',
            left: value < 0 ? '50%' : '50%',
            width: `${normalizedValue * 50}%`,
            height: '100%',
            background: color,
            transform: value < 0 ? 'translateX(-100%)' : 'none',
            boxShadow: `0 0 8px ${color}`,
            transition: 'width 0.05s linear',
          }}
        />
        <div
          className="vector-center-marker"
          style={{
            position: 'absolute',
            left: '50%',
            top: 0,
            bottom: 0,
            width: '1px',
            background: 'rgba(139, 155, 180, 0.3)',
          }}
        />
      </div>
      <span 
        className="vector-value"
        style={{
          fontSize: '9px',
          fontFamily: '"JetBrains Mono", monospace',
          color: color,
          textAlign: 'right',
        }}
      >
        {value.toFixed(4)}
      </span>
    </div>
  );
});

VectorBar.displayName = 'VectorBar';

export const AgentState: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { observationSpace, latentState, confidenceScores, actionProbs } = useAgentStore();
  
  // Memoize vector data to prevent unnecessary re-renders
  const observationVectors = useMemo(() => {
    if (!observationSpace) return [];
    return Object.entries(observationSpace).slice(0, 12).map(([key, value], idx) => ({
      label: key,
      value: typeof value === 'number' ? value : 0,
      color: `hsl(${200 + idx * 15}, 80%, 60%)`,
      index: idx,
    }));
  }, [observationSpace]);

  const latentVectors = useMemo(() => {
    if (!latentState) return [];
    return latentState.slice(0, 8).map((value, idx) => ({
      label: `Z_${idx}`,
      value,
      color: `hsl(${280 + idx * 20}, 70%, 65%)`,
      index: idx,
    }));
  }, [latentState]);

  const avgConfidence = useMemo(() => {
    if (!confidenceScores || confidenceScores.length === 0) return 0;
    return confidenceScores.reduce((a, b) => a + b, 0) / confidenceScores.length;
  }, [confidenceScores]);

  return (
    <div 
      ref={containerRef}
      className="agent-state-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        padding: '12px',
        background: 'linear-gradient(135deg, rgba(10, 15, 30, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%)',
        borderRadius: '8px',
        border: '1px solid rgba(0, 255, 255, 0.15)',
        boxShadow: '0 0 20px rgba(0, 255, 255, 0.05), inset 0 0 30px rgba(0, 0, 0, 0.3)',
        fontFamily: '"JetBrains Mono", monospace',
        minHeight: '280px',
      }}
    >
      {/* Header */}
      <div 
        className="panel-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingBottom: '8px',
          borderBottom: '1px solid rgba(0, 255, 255, 0.2)',
        }}
      >
        <h3 
          style={{
            margin: 0,
            fontSize: '12px',
            fontWeight: 600,
            color: '#00ffff',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            textShadow: '0 0 10px rgba(0, 255, 255, 0.5)',
          }}
        >
          🧠 RL Agent State
        </h3>
        <div 
          className="confidence-indicator"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span 
            style={{
              fontSize: '9px',
              color: '#8b9bb4',
            }}
          >
            CONFIDENCE
          </span>
          <div 
            style={{
              width: '60px',
              height: '8px',
              background: 'rgba(20, 30, 48, 0.8)',
              borderRadius: '4px',
              overflow: 'hidden',
              position: 'relative',
            }}
          >
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${avgConfidence * 100}%`,
                background: `linear-gradient(90deg, 
                  ${avgConfidence > 0.7 ? '#00ff88' : avgConfidence > 0.4 ? '#ffaa00' : '#ff3366'}, 
                  ${avgConfidence > 0.7 ? '#00ffff' : '#ffcc00'})`,
                boxShadow: `0 0 8px ${avgConfidence > 0.7 ? '#00ff88' : '#ffaa00'}`,
                transition: 'width 0.1s ease-out',
              }}
            />
          </div>
          <span 
            style={{
              fontSize: '10px',
              color: avgConfidence > 0.7 ? '#00ff88' : avgConfidence > 0.4 ? '#ffaa00' : '#ff3366',
              minWidth: '35px',
              textAlign: 'right',
            }}
          >
            {(avgConfidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Observation Space */}
      <div className="section">
        <div 
          className="section-title"
          style={{
            fontSize: '9px',
            color: '#64ffda',
            marginBottom: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          Observation Space
        </div>
        <div 
          className="observation-vectors"
          style={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {observationVectors.map((vec) => (
            <VectorBar key={`obs-${vec.index}`} {...vec} />
          ))}
        </div>
      </div>

      {/* Latent State */}
      <div className="section">
        <div 
          className="section-title"
          style={{
            fontSize: '9px',
            color: '#bd93f9',
            marginBottom: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          Latent Embedding (Encoder Output)
        </div>
        <div 
          className="latent-vectors"
          style={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {latentVectors.map((vec) => (
            <VectorBar key={`latent-${vec.index}`} {...vec} />
          ))}
        </div>
      </div>

      {/* Action Probabilities Summary */}
      {actionProbs && (
        <div 
          className="action-summary"
          style={{
            marginTop: 'auto',
            paddingTop: '8px',
            borderTop: '1px solid rgba(139, 155, 180, 0.2)',
          }}
        >
          <div 
            style={{
              fontSize: '9px',
              color: '#8b9bb4',
              marginBottom: '4px',
            }}
          >
            ACTION DISTRIBUTION
          </div>
          <div 
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(5, 1fr)',
              gap: '4px',
            }}
          >
            {['BUY', 'SELL', 'HOLD', 'SCALE_IN', 'SCALE_OUT'].map((action, idx) => (
              <div 
                key={action}
                style={{
                  textAlign: 'center',
                  padding: '4px 2px',
                  background: `rgba(${idx === 0 ? '0, 255, 136' : idx === 1 ? '255, 51, 102' : '139, 155, 180'}, 0.1)`,
                  borderRadius: '4px',
                  border: `1px solid rgba(${idx === 0 ? '0, 255, 136' : idx === 1 ? '255, 51, 102' : '139, 155, 180'}, 0.3)`,
                }}
              >
                <div 
                  style={{
                    fontSize: '8px',
                    color: '#8b9bb4',
                    marginBottom: '2px',
                  }}
                >
                  {action.replace('_', ' ')}
                </div>
                <div 
                  style={{
                    fontSize: '10px',
                    fontWeight: 600,
                    color: idx === 0 ? '#00ff88' : idx === 1 ? '#ff3366' : '#8b9bb4',
                  }}
                >
                  {((actionProbs[idx] || 0) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentState;
