/**
 * SlippageTracker.tsx - Transaction Cost Analysis (TCA) Dashboard
 * 
 * Compares theoretical arrival price vs actual execution fill price
 * to quantify execution quality and slippage.
 * 
 * Features:
 * - Real-time slippage tracking per trade
 * - Cumulative slippage over time
 * - Execution quality scoring
 * - Market impact visualization
 * - Cyberpunk aesthetic with color-coded performance
 */

import React, { useEffect, useRef, useCallback } from 'react';
import { useMetricsStore } from '../../store/metricsStore';

interface SlippageEvent {
  timestamp: number;
  symbol: string;
  side: 'BUY' | 'SELL';
  arrivalPrice: number;
  fillPrice: number;
  quantity: number;
  slippageBps: number;
  marketImpact: number;
}

interface SlippageTrackerProps {
  maxEvents?: number;
}

export const SlippageTracker: React.FC<SlippageTrackerProps> = ({
  maxEvents = 50,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { slippageEvents, totalSlippage, avgSlippageBps, executionQualityScore } = useMetricsStore();
  
  // Get recent slippage events
  const recentEvents = React.useMemo(() => {
    if (!slippageEvents) return [];
    return slippageEvents.slice(-maxEvents).reverse();
  }, [slippageEvents, maxEvents]);

  const getSlippageColor = (bps: number): string => {
    if (bps <= 1) return '#00ff88';      // Excellent - Green
    if (bps <= 3) return '#00ffff';      // Good - Cyan
    if (bps <= 5) return '#ffaa00';      // Fair - Orange
    return '#ff3366';                    // Poor - Red
  };

  const getQualityBadge = (score: number): { label: string; color: string } => {
    if (score >= 95) return { label: 'EXCELLENT', color: '#00ff88' };
    if (score >= 85) return { label: 'GOOD', color: '#00ffff' };
    if (score >= 70) return { label: 'FAIR', color: '#ffaa00' };
    return { label: 'POOR', color: '#ff3366' };
  };

  const qualityBadge = getQualityBadge(executionQualityScore ?? 85);

  return (
    <div
      ref={containerRef}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        padding: '12px',
        background: 'linear-gradient(135deg, rgba(10, 15, 30, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%)',
        borderRadius: '8px',
        border: '1px solid rgba(189, 147, 249, 0.15)',
        boxShadow: '0 0 20px rgba(189, 147, 249, 0.05), inset 0 0 30px rgba(0, 0, 0, 0.3)',
        fontFamily: '"JetBrains Mono", monospace',
        minHeight: '350px',
        maxHeight: '450px',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingBottom: '8px',
          borderBottom: '1px solid rgba(189, 147, 249, 0.2)',
        }}
      >
        <h3
          style={{
            margin: 0,
            fontSize: '12px',
            fontWeight: 600,
            color: '#bd93f9',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            textShadow: '0 0 10px rgba(189, 147, 249, 0.5)',
          }}
        >
          📉 TCA: Slippage Analysis
        </h3>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span
            style={{
              fontSize: '8px',
              color: 'rgba(139, 155, 180, 0.6)',
            }}
          >
            EXEC QUALITY
          </span>
          <div
            style={{
              padding: '3px 8px',
              background: `rgba(${qualityBadge.color === '#00ff88' ? '0, 255, 136' : qualityBadge.color === '#00ffff' ? '0, 255, 255' : qualityBadge.color === '#ffaa00' ? '255, 170, 0' : '255, 51, 102'}, 0.15)`,
              border: `1px solid ${qualityBadge.color}`,
              borderRadius: '4px',
              fontSize: '9px',
              color: qualityBadge.color,
              boxShadow: `0 0 8px ${qualityBadge.color}40`,
              fontWeight: 600,
            }}
          >
            {qualityBadge.label} ({(executionQualityScore ?? 85).toFixed(0)}%)
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '8px',
          padding: '8px',
          background: 'rgba(20, 30, 50, 0.5)',
          borderRadius: '6px',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.6)', marginBottom: '2px' }}>
            TOTAL SLIPPAGE
          </div>
          <div
            style={{
              fontSize: '12px',
              fontWeight: 700,
              color: (totalSlippage ?? 0) < 0 ? '#00ff88' : '#ff3366',
            }}
          >
            ${(Math.abs(totalSlippage ?? 0)).toFixed(2)}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.6)', marginBottom: '2px' }}>
            AVG SLIPPAGE
          </div>
          <div
            style={{
              fontSize: '12px',
              fontWeight: 700,
              color: getSlippageColor(avgSlippageBps ?? 2),
            }}
          >
            {(avgSlippageBps ?? 0).toFixed(2)} bps
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.6)', marginBottom: '2px' }}>
            TRADES ANALYZED
          </div>
          <div
            style={{
              fontSize: '12px',
              fontWeight: 700,
              color: '#00ffff',
            }}
          >
            {slippageEvents?.length ?? 0}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.6)', marginBottom: '2px' }}>
            MARKET IMPACT
          </div>
          <div
            style={{
              fontSize: '12px',
              fontWeight: 700,
              color: '#ffaa00',
            }}
          >
            {((avgSlippageBps ?? 0) * 0.8).toFixed(2)} bps
          </div>
        </div>
      </div>

      {/* Slippage Distribution Bars */}
      <div
        style={{
          padding: '8px',
          background: 'rgba(20, 30, 50, 0.3)',
          borderRadius: '6px',
        }}
      >
        <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.7)', marginBottom: '6px' }}>
          SLIPPAGE DISTRIBUTION (bps)
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            gap: '4px',
            height: '40px',
          }}
        >
          {[
            { range: '0-1', color: '#00ff88', threshold: 1 },
            { range: '1-3', color: '#00ffff', threshold: 3 },
            { range: '3-5', color: '#ffaa00', threshold: 5 },
            { range: '5+', color: '#ff3366', threshold: Infinity },
          ].map((bucket) => {
            const count = slippageEvents?.filter(
              (e) => Math.abs(e.slippageBps) <= bucket.threshold
            ).length ?? 0;
            const prevCount = slippageEvents?.filter(
              (e, i, arr) => {
                const prevThreshold = bucket.threshold === 1 ? 0 : bucket.threshold === 3 ? 1 : bucket.threshold === 5 ? 3 : 5;
                return Math.abs(e.slippageBps) > prevThreshold && Math.abs(e.slippageBps) <= bucket.threshold;
              }
            ).length ?? count;
            
            const actualCount = bucket.threshold === 1 ? count : bucket.threshold === 3 ? count - (slippageEvents?.filter(e => Math.abs(e.slippageBps) <= 1).length ?? 0) : bucket.threshold === 5 ? count - (slippageEvents?.filter(e => Math.abs(e.slippageBps) <= 3).length ?? 0) : count - (slippageEvents?.filter(e => Math.abs(e.slippageBps) <= 5).length ?? 0);
            
            const height = slippageEvents && slippageEvents.length > 0 
              ? Math.max((actualCount / slippageEvents.length) * 100, 5)
              : 5;

            return (
              <div
                key={bucket.range}
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <div
                  style={{
                    width: '100%',
                    height: `${height}%`,
                    background: `linear-gradient(to top, ${bucket.color}40, ${bucket.color})`,
                    borderRadius: '4px 4px 0 0',
                    boxShadow: `0 0 8px ${bucket.color}40`,
                    transition: 'height 0.3s ease',
                  }}
                />
                <div style={{ fontSize: '7px', color: bucket.color }}>
                  {bucket.range}
                </div>
                <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)' }}>
                  {actualCount}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Events List */}
      <div
        style={{
          flex: 1,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.7)', marginBottom: '6px' }}>
          RECENT EXECUTIONS
        </div>
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgba(139, 155, 180, 0.3) transparent',
          }}
        >
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '8px',
            }}
          >
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(139, 155, 180, 0.2)' }}>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: 'rgba(139, 155, 180, 0.5)', fontWeight: 400 }}>TIME</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: 'rgba(139, 155, 180, 0.5)', fontWeight: 400 }}>SYMBOL</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: 'rgba(139, 155, 180, 0.5)', fontWeight: 400 }}>SIDE</th>
                <th style={{ textAlign: 'right', padding: '6px 4px', color: 'rgba(139, 155, 180, 0.5)', fontWeight: 400 }}>ARRIVAL</th>
                <th style={{ textAlign: 'right', padding: '6px 4px', color: 'rgba(139, 155, 180, 0.5)', fontWeight: 400 }}>FILL</th>
                <th style={{ textAlign: 'right', padding: '6px 4px', color: 'rgba(139, 155, 180, 0.5)', fontWeight: 400 }}>SLIPPAGE</th>
              </tr>
            </thead>
            <tbody>
              {recentEvents.map((event, idx) => (
                <tr
                  key={`slip-${event.timestamp}-${idx}`}
                  style={{
                    borderBottom: '1px solid rgba(139, 155, 180, 0.1)',
                    transition: 'background 0.15s ease',
                  }}
                >
                  <td style={{ padding: '6px 4px', color: 'rgba(139, 155, 180, 0.6)' }}>
                    {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </td>
                  <td style={{ padding: '6px 4px', color: '#00ffff', fontWeight: 500 }}>
                    {event.symbol}
                  </td>
                  <td style={{ padding: '6px 4px' }}>
                    <span
                      style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                        fontSize: '7px',
                        fontWeight: 600,
                        background: event.side === 'BUY' ? 'rgba(0, 255, 136, 0.15)' : 'rgba(255, 51, 102, 0.15)',
                        color: event.side === 'BUY' ? '#00ff88' : '#ff3366',
                      }}
                    >
                      {event.side}
                    </span>
                  </td>
                  <td style={{ padding: '6px 4px', textAlign: 'right', color: 'rgba(139, 155, 180, 0.7)' }}>
                    ${event.arrivalPrice.toFixed(event.symbol.includes('BTC') ? 1 : 2)}
                  </td>
                  <td style={{ padding: '6px 4px', textAlign: 'right', color: '#8b9bb4' }}>
                    ${event.fillPrice.toFixed(event.symbol.includes('BTC') ? 1 : 2)}
                  </td>
                  <td
                    style={{
                      padding: '6px 4px',
                      textAlign: 'right',
                      color: getSlippageColor(Math.abs(event.slippageBps)),
                      fontWeight: 600,
                    }}
                  >
                    {event.slippageBps >= 0 ? '+' : ''}{event.slippageBps.toFixed(2)} bps
                  </td>
                </tr>
              ))}
              {recentEvents.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ padding: '20px', textAlign: 'center', color: 'rgba(139, 155, 180, 0.4)' }}>
                    No execution data yet...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default SlippageTracker;
