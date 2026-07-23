/**
 * TradeTape.tsx - Hyper-fast color-coded aggressive trade tape
 * 
 * Creates a real-time ticker of executed trades using CSS transforms
 * and absolute positioning to scroll thousands of ticks without
 * layout thrashing. Optimized for high-frequency Binance trade streams.
 * 
 * Features:
 * - CSS transform-based scrolling (no layout recalc)
 * - Absolute positioning for O(1) render
 * - Color-coded by aggressor side (buy/sell)
 * - Size-proportional width indicators
 * - Cyberpunk aesthetic with glow effects
 */

import React, { useEffect, useRef, useCallback, useState, useMemo } from 'react';

export interface TradeTick {
  id: string;
  price: number;
  size: number;
  timestamp: number;
  isBuyerMaker: boolean; // true = sell, false = buy
  side: 'buy' | 'sell';
}

interface TradeTapeProps {
  trades: TradeTick[];
  maxTrades?: number;
  height?: number;
  symbol?: string;
  precision?: number;
}

// Cyberpunk color scheme
const TAPE_COLORS = {
  buyColor: '#00ff88',
  sellColor: '#ff0055',
  buyBackground: 'rgba(0, 255, 136, 0.1)',
  sellBackground: 'rgba(255, 0, 85, 0.1)',
  textDim: 'rgba(0, 255, 255, 0.5)',
  gridBorder: 'rgba(0, 255, 255, 0.05)',
  highlight: '#00ffff',
};

// Row height constant for transform calculations
const ROW_HEIGHT = 18;

export const TradeTape: React.FC<TradeTapeProps> = ({
  trades,
  maxTrades = 100,
  height = 400,
  symbol = 'BTCUSDT',
  precision = 2,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [transformY, setTransformY] = useState(0);
  const animationFrameRef = useRef<number | null>(null);

  // Limit trades to maxTrades for memory efficiency
  const limitedTrades = useMemo(() => {
    return trades.slice(-maxTrades);
  }, [trades, maxTrades]);

  // Calculate total content height
  const totalHeight = limitedTrades.length * ROW_HEIGHT;

  // Auto-scroll to bottom on new trades using transform (no layout thrash)
  useEffect(() => {
    if (!containerRef.current) return;

    const containerHeight = containerRef.current.clientHeight;
    const targetY = -(totalHeight - containerHeight + ROW_HEIGHT);
    
    // Smooth scroll using requestAnimationFrame
    const smoothScroll = () => {
      setTransformY(prev => {
        const diff = targetY - prev;
        if (Math.abs(diff) < 1) return targetY;
        return prev + diff * 0.3; // Ease out
      });
      
      if (Math.abs(targetY - transformY) > 1) {
        animationFrameRef.current = requestAnimationFrame(smoothScroll);
      }
    };

    smoothScroll();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [totalHeight, transformY]);

  // Format time for display
  const formatTime = useCallback((timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
    });
  }, []);

  // Get row style with transform optimization
  const getRowStyle = useCallback((index: number, isBuy: boolean): React.CSSProperties => ({
    position: 'absolute',
    top: 0,
    transform: `translateY(${index * ROW_HEIGHT}px)`,
    width: '100%',
    height: `${ROW_HEIGHT}px`,
    display: 'flex',
    alignItems: 'center',
    fontFamily: 'monospace',
    fontSize: '10px',
    padding: '0 6px',
    boxSizing: 'border-box',
    backgroundColor: isBuy ? TAPE_COLORS.buyBackground : TAPE_COLORS.sellBackground,
    color: isBuy ? TAPE_COLORS.buyColor : TAPE_COLORS.sellColor,
    borderBottom: `1px solid ${TAPE_COLORS.gridBorder}`,
    willChange: 'transform',
    contain: 'layout paint',
  }), []);

  // Calculate size ratio for visual indicator
  const maxSize = useMemo(() => {
    return Math.max(...limitedTrades.map(t => t.size), 1);
  }, [limitedTrades]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height,
        overflow: 'hidden',
        position: 'relative',
        backgroundColor: '#0a0e17',
        border: `1px solid ${TAPE_COLORS.gridBorder}`,
      }}
      className="trade-tape-container"
    >
      {/* Header */}
      <div
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          display: 'flex',
          height: `${ROW_HEIGHT}px`,
          backgroundColor: '#0f1420',
          borderBottom: `1px solid ${TAPE_COLORS.gridBorder}`,
          fontFamily: 'monospace',
          fontSize: '9px',
          color: TAPE_COLORS.highlight,
        }}
      >
        <span style={{ width: '25%' }}>TIME</span>
        <span style={{ width: '25%', textAlign: 'center' }}>PRICE</span>
        <span style={{ width: '25%', textAlign: 'right' }}>SIZE</span>
        <span style={{ width: '25%', textAlign: 'right' }}>SIDE</span>
      </div>

      {/* Scrollable content with transform-based scrolling */}
      <div
        style={{
          width: '100%',
          height: `${height - ROW_HEIGHT}px`,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <div
          ref={contentRef}
          style={{
            position: 'relative',
            height: `${totalHeight}px`,
            width: '100%',
            transform: `translateY(${transformY}px)`,
            willChange: 'transform',
            contain: 'layout paint',
          }}
        >
          {limitedTrades.map((trade, index) => (
            <div
              key={trade.id}
              ref={el => {
                if (el) rowRefs.current.set(trade.id, el);
                else rowRefs.current.delete(trade.id);
              }}
              style={getRowStyle(index, trade.side === 'buy')}
              className={`trade-tape-row ${trade.side === 'buy' ? 'tape-buy' : 'tape-sell'}`}
            >
              {/* Time */}
              <span style={{ width: '25%', opacity: 0.7 }}>
                {formatTime(trade.timestamp)}
              </span>

              {/* Price */}
              <span style={{ width: '25%', textAlign: 'center', fontWeight: 'bold' }}>
                {trade.price.toFixed(precision)}
              </span>

              {/* Size with visual bar */}
              <span style={{ width: '25%', textAlign: 'right', position: 'relative' }}>
                <div
                  style={{
                    position: 'absolute',
                    right: 0,
                    top: '20%',
                    height: '60%',
                    background: trade.side === 'buy'
                      ? 'rgba(0, 255, 136, 0.3)'
                      : 'rgba(255, 0, 85, 0.3)',
                    width: `${(trade.size / maxSize) * 100}%`,
                    zIndex: -1,
                  }}
                />
                {trade.size.toFixed(4)}
              </span>

              {/* Side indicator */}
              <span
                style={{
                  width: '25%',
                  textAlign: 'right',
                  fontWeight: 'bold',
                  textShadow: trade.side === 'buy'
                    ? '0 0 8px rgba(0, 255, 136, 0.5)'
                    : '0 0 8px rgba(255, 0, 85, 0.5)',
                }}
              >
                {trade.side === 'buy' ? 'BUY' : 'SELL'}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer stats */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: `${ROW_HEIGHT}px`,
          backgroundColor: '#0f1420',
          borderTop: `1px solid ${TAPE_COLORS.gridBorder}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 6px',
          fontFamily: 'monospace',
          fontSize: '9px',
        }}
      >
        <span style={{ color: TAPE_COLORS.highlight }}>{symbol}</span>
        <span style={{ color: TAPE_COLORS.textDim }}>
          {limitedTrades.length} ticks
        </span>
        <span style={{ color: TAPE_COLORS.buyColor }}>
          B: {limitedTrades.filter(t => t.side === 'buy').length}
        </span>
        <span style={{ color: TAPE_COLORS.sellColor }}>
          S: {limitedTrades.filter(t => t.side === 'sell').length}
        </span>
      </div>

      {/* CSS for glow effects */}
      <style>{`
        .tape-buy {
          animation: tape-buy-glow 0.3s ease-out;
        }
        .tape-sell {
          animation: tape-sell-glow 0.3s ease-out;
        }
        @keyframes tape-buy-glow {
          0% { box-shadow: inset 0 0 10px rgba(0, 255, 136, 0.5); }
          100% { box-shadow: none; }
        }
        @keyframes tape-sell-glow {
          0% { box-shadow: inset 0 0 10px rgba(255, 0, 85, 0.5); }
          100% { box-shadow: none; }
        }
      `}</style>
    </div>
  );
};

export default TradeTape;
