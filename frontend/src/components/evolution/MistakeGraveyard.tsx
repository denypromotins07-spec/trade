/**
 * Mistake Graveyard Component - Stage 56
 * Cyberpunk-Styled UI | Toxic Pattern Visualization | Microsecond Precision
 * 
 * Cyberpunk-styled "graveyard" UI listing banned strategies and the exact
 * microsecond market conditions that caused their catastrophic failure.
 * 
 * Constraints:
 * - Virtual scrolling for large datasets
 * - No re-renders on data updates
 * - Precise timestamp formatting
 * - Glitch effect animations using CSS only
 */

import React, { useRef, useEffect, useCallback, useState, useMemo } from 'react';

// Types
interface CatastrophicFailure {
  id: string;
  strategyHash: string;
  strategyName: string;
  lossAmount: number;
  lossPercentage: number;
  timestamp: number;
  marketConditions: {
    symbol: string;
    volatility: number;
    spread: number;
    volume: number;
  };
  failureReason: string;
  toxicPatterns: string[];
  bannedAt: number;
}

interface MistakeGraveyardProps {
  failures: CatastrophicFailure[];
  onSelect?: (failure: CatastrophicFailure) => void;
  maxVisible?: number;
}

// Constants
const ROW_HEIGHT = 48;
const GLITCH_COLORS = ['#ff0044', '#00ff88', '#0088ff'];

// Format timestamp with microsecond precision
function formatMicroTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  const micros = Math.floor((timestamp % 1000) * 1000);
  return `${date.toISOString().slice(0, 23)}.${micros.toString().padStart(6, '0')}Z`;
}

// Format loss amount
function formatLoss(amount: number): string {
  if (Math.abs(amount) >= 1000000) {
    return `-$${(Math.abs(amount) / 1000000).toFixed(2)}M`;
  }
  if (Math.abs(amount) >= 1000) {
    return `-$${(Math.abs(amount) / 1000).toFixed(2)}K`;
  }
  return `-$${Math.abs(amount).toFixed(2)}`;
}

// Generate glitch effect style
function getGlitchStyle(index: number): React.CSSProperties {
  const delay = (index % 3) * 0.3;
  return {
    animation: `glitch ${2 + (index % 3) * 0.5}s infinite`,
    animationDelay: `${delay}s`,
  };
}

// Custom hook for virtual scrolling
function useVirtualScroll<T>(
  items: T[],
  containerHeight: number,
  itemHeight: number
) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const totalHeight = items.length * itemHeight;
  const visibleCount = Math.ceil(containerHeight / itemHeight) + 2;
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - 1);
  const endIndex = Math.min(items.length, startIndex + visibleCount);
  
  const visibleItems = useMemo(() => 
    items.slice(startIndex, endIndex),
    [items, startIndex, endIndex]
  );
  
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);
  
  return {
    containerRef,
    visibleItems,
    totalHeight,
    startIndex,
    handleScroll,
    offsetY: startIndex * itemHeight,
  };
}

// Individual row component (memoized)
const GraveyardRow = React.memo<{
  failure: CatastrophicFailure;
  index: number;
  onClick: () => void;
}>(({ failure, index, onClick }) => {
  const rowStyle: React.CSSProperties = {
    position: 'absolute',
    left: 0,
    right: 0,
    height: ROW_HEIGHT - 2,
    display: 'flex',
    alignItems: 'center',
    padding: '0 16px',
    borderBottom: '1px solid rgba(255, 0, 68, 0.2)',
    cursor: 'pointer',
    transition: 'background 0.15s',
    ...getGlitchStyle(index),
  };
  
  return (
    <div
      className="graveyard-row"
      style={rowStyle}
      onClick={onClick}
    >
      {/* Strategy hash */}
      <div style={{ 
        width: 120, 
        fontFamily: 'monospace',
        fontSize: 11,
        color: '#ff0044',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>
        {failure.strategyHash.slice(0, 12)}...
      </div>
      
      {/* Loss amount */}
      <div style={{ 
        width: 100, 
        fontWeight: 'bold',
        color: '#ff0044',
        fontSize: 13,
      }}>
        {formatLoss(failure.lossAmount)}
      </div>
      
      {/* Loss percentage */}
      <div style={{ 
        width: 70, 
        color: failure.lossPercentage > 10 ? '#ff0000' : '#ff8800',
        fontSize: 12,
      }}>
        {failure.lossPercentage.toFixed(2)}%
      </div>
      
      {/* Symbol */}
      <div style={{ 
        width: 80, 
        color: '#00ff88',
        fontFamily: 'monospace',
        fontSize: 12,
      }}>
        {failure.marketConditions.symbol}
      </div>
      
      {/* Volatility */}
      <div style={{ 
        width: 70, 
        color: '#aabbcc',
        fontSize: 11,
      }}>
        {(failure.marketConditions.volatility * 100).toFixed(1)}%
      </div>
      
      {/* Timestamp */}
      <div style={{ 
        flex: 1, 
        color: '#667788',
        fontFamily: 'monospace',
        fontSize: 10,
        textAlign: 'right',
      }}>
        {formatMicroTimestamp(failure.timestamp)}
      </div>
    </div>
  );
});

GraveyardRow.displayName = 'GraveyardRow';

// Main component
export const MistakeGraveyard: React.FC<MistakeGraveyardProps> = ({
  failures,
  onSelect,
  maxVisible = 100,
}) => {
  const containerHeight = 500;
  
  // Sort by loss amount (worst first)
  const sortedFailures = useMemo(() => 
    [...failures].sort((a, b) => a.lossAmount - b.lossAmount).slice(0, maxVisible),
    [failures, maxVisible]
  );
  
  const {
    containerRef,
    visibleItems,
    totalHeight,
    offsetY,
    handleScroll,
  } = useVirtualScroll<CatastrophicFailure>(sortedFailures, containerHeight, ROW_HEIGHT);
  
  // Calculate stats
  const stats = useMemo(() => {
    const totalLoss = failures.reduce((sum, f) => sum + f.lossAmount, 0);
    const avgLoss = failures.length > 0 ? totalLoss / failures.length : 0;
    const worstLoss = Math.min(...failures.map(f => f.lossAmount), 0);
    
    return { total: failures.length, totalLoss, avgLoss, worstLoss };
  }, [failures]);
  
  return (
    <div className="mistake-graveyard" style={{
      background: '#0a0f14',
      border: '1px solid rgba(255, 0, 68, 0.3)',
      borderRadius: 4,
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div className="graveyard-header" style={{
        display: 'flex',
        alignItems: 'center',
        padding: '12px 16px',
        borderBottom: '2px solid rgba(255, 0, 68, 0.5)',
        background: 'linear-gradient(90deg, rgba(255,0,68,0.1) 0%, transparent 100%)',
      }}>
        <span style={{ fontSize: 16, fontWeight: 'bold', color: '#ff0044', marginRight: 8 }}>
          ⚰️ MISTAKE GRAVEYARD
        </span>
        <span style={{ fontSize: 11, color: '#667788', marginLeft: 'auto' }}>
          {stats.total} CATASTROPHIC FAILURES RECORDED
        </span>
      </div>
      
      {/* Column headers */}
      <div style={{
        display: 'flex',
        padding: '8px 16px',
        borderBottom: '1px solid rgba(255, 0, 68, 0.2)',
        fontSize: 10,
        color: '#667788',
        textTransform: 'uppercase',
        letterSpacing: 1,
      }}>
        <div style={{ width: 120 }}>Strategy Hash</div>
        <div style={{ width: 100 }}>Loss Amount</div>
        <div style={{ width: 70 }}>Loss %</div>
        <div style={{ width: 80 }}>Symbol</div>
        <div style={{ width: 70 }}>Volatility</div>
        <div style={{ flex: 1, textAlign: 'right' }}>Timestamp (μs)</div>
      </div>
      
      {/* Scrollable content */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{
          height: containerHeight,
          overflowY: 'auto',
          position: 'relative',
        }}
      >
        <div style={{ height: totalHeight, position: 'relative' }}>
          <div style={{
            position: 'absolute',
            top: offsetY,
            left: 0,
            right: 0,
          }}>
            {visibleItems.map((failure, idx) => (
              <GraveyardRow
                key={failure.id}
                failure={failure}
                index={idx}
                onClick={() => onSelect?.(failure)}
              />
            ))}
          </div>
        </div>
      </div>
      
      {/* Stats footer */}
      <div className="graveyard-stats" style={{
        display: 'flex',
        padding: '12px 16px',
        borderTop: '1px solid rgba(255, 0, 68, 0.3)',
        background: 'rgba(255, 0, 68, 0.05)',
        fontSize: 11,
      }}>
        <div style={{ marginRight: 24 }}>
          <span style={{ color: '#667788' }}>Total Loss: </span>
          <span style={{ color: '#ff0044', fontWeight: 'bold' }}>
            {formatLoss(stats.totalLoss)}
          </span>
        </div>
        <div style={{ marginRight: 24 }}>
          <span style={{ color: '#667788' }}>Avg Loss: </span>
          <span style={{ color: '#ff8800' }}>
            {formatLoss(stats.avgLoss)}
          </span>
        </div>
        <div>
          <span style={{ color: '#667788' }}>Worst Single: </span>
          <span style={{ color: '#ff0000', fontWeight: 'bold' }}>
            {formatLoss(stats.worstLoss)}
          </span>
        </div>
      </div>
      
      {/* CSS for glitch effects */}
      <style>{`
        @keyframes glitch {
          0%, 100% {
            transform: translate(0);
            filter: hue-rotate(0deg);
          }
          20% {
            transform: translate(-2px, 1px);
            filter: hue-rotate(10deg);
          }
          40% {
            transform: translate(1px, -1px);
            filter: hue-rotate(-10deg);
          }
          60% {
            transform: translate(-1px, 2px);
            filter: hue-rotate(5deg);
          }
          80% {
            transform: translate(2px, -2px);
            filter: hue-rotate(-5deg);
          }
        }
        
        .graveyard-row:hover {
          background: rgba(255, 0, 68, 0.1) !important;
        }
        
        ::-webkit-scrollbar {
          width: 8px;
        }
        
        ::-webkit-scrollbar-track {
          background: #0a0f14;
        }
        
        ::-webkit-scrollbar-thumb {
          background: rgba(255, 0, 68, 0.3);
          border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 0, 68, 0.5);
        }
      `}</style>
    </div>
  );
};

export default MistakeGraveyard;
