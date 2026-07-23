/**
 * ExecutionLog.tsx - Live bot order submissions, fills, and cancellations
 * 
 * Displays the trading bot's execution telemetry with Framer Motion
 * entry animations. Highlights slippage, execution quality, and order
 * lifecycle events in real-time.
 * 
 * Features:
 * - Framer Motion entry/exit animations
 * - Slippage calculation and highlighting
 * - Order lifecycle tracking (NEW, PARTIAL, FILLED, CANCELLED)
 * - Execution quality metrics
 * - Cyberpunk aesthetic with animated transitions
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface ExecutionEvent {
  id: string;
  orderId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'LIMIT' | 'MARKET' | 'STOP_LOSS' | 'TAKE_PROFIT';
  status: 'NEW' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  price: number;
  executedPrice?: number;
  quantity: number;
  executedQty: number;
  timestamp: number;
  slippage?: number; // in basis points
  latency?: number; // ms
  venue?: string;
}

interface ExecutionLogProps {
  events: ExecutionEvent[];
  maxEvents?: number;
  height?: number;
  showSlippage?: boolean;
}

// Cyberpunk color scheme
const EXEC_COLORS = {
  new: '#00ffff',
  partial: '#ffa500',
  filled: '#00ff88',
  cancelled: '#ff0055',
  rejected: '#ff4444',
  background: 'rgba(10, 14, 23, 0.9)',
  border: 'rgba(0, 255, 255, 0.1)',
  text: '#00ffff',
  textDim: 'rgba(0, 255, 255, 0.5)',
  slippageGood: '#00ff88',
  slippageBad: '#ff0055',
  slippageWarn: '#ffa500',
};

// Status badge colors
const STATUS_COLORS: Record<string, string> = {
  NEW: EXEC_COLORS.new,
  PARTIALLY_FILLED: EXEC_COLORS.partial,
  FILLED: EXEC_COLORS.filled,
  CANCELLED: EXEC_COLORS.cancelled,
  REJECTED: EXEC_COLORS.rejected,
};

// Animation variants for Framer Motion
const rowVariants = {
  hidden: {
    opacity: 0,
    x: -50,
    scale: 0.9,
  },
  visible: {
    opacity: 1,
    x: 0,
    scale: 1,
    transition: {
      type: 'spring',
      stiffness: 300,
      damping: 25,
      duration: 0.3,
    },
  },
  exit: {
    opacity: 0,
    x: 50,
    scale: 0.9,
    transition: {
      duration: 0.2,
    },
  },
};

export const ExecutionLog: React.FC<ExecutionLogProps> = ({
  events,
  maxEvents = 50,
  height = 400,
  showSlippage = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [limitedEvents, setLimitedEvents] = useState<ExecutionEvent[]>(events.slice(-maxEvents));

  // Update limited events when source changes
  useEffect(() => {
    setLimitedEvents(events.slice(-maxEvents));
  }, [events, maxEvents]);

  // Format timestamp
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

  // Calculate fill percentage
  const getFillPercent = useCallback((event: ExecutionEvent): number => {
    if (event.quantity === 0) return 0;
    return Math.round((event.executedQty / event.quantity) * 100);
  }, []);

  // Get slippage color based on severity
  const getSlippageColor = useCallback((slippage?: number): string => {
    if (slippage === undefined) return EXEC_COLORS.textDim;
    if (slippage < 5) return EXEC_COLORS.slippageGood; // < 5 bps
    if (slippage < 20) return EXEC_COLORS.slippageWarn; // 5-20 bps
    return EXEC_COLORS.slippageBad; // > 20 bps
  }, []);

  // Get status label style
  const getStatusStyle = useCallback((status: string): React.CSSProperties => ({
    padding: '2px 6px',
    borderRadius: '3px',
    fontSize: '9px',
    fontWeight: 'bold',
    backgroundColor: `${STATUS_COLORS[status]}20`,
    color: STATUS_COLORS[status],
    border: `1px solid ${STATUS_COLORS[status]}40`,
    textShadow: `0 0 4px ${STATUS_COLORS[status]}80`,
  }), []);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height,
        overflow: 'hidden',
        position: 'relative',
        backgroundColor: EXEC_COLORS.background,
        border: `1px solid ${EXEC_COLORS.border}`,
        fontFamily: 'monospace',
      }}
      className="execution-log-container"
    >
      {/* Header */}
      <div
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          display: 'flex',
          height: '24px',
          backgroundColor: '#0f1420',
          borderBottom: `1px solid ${EXEC_COLORS.border}`,
          alignItems: 'center',
          padding: '0 8px',
          fontSize: '9px',
          color: EXEC_COLORS.text,
        }}
      >
        <span style={{ width: '15%' }}>TIME</span>
        <span style={{ width: '12%' }}>SIDE</span>
        <span style={{ width: '15%' }}>TYPE</span>
        <span style={{ width: '12%' }}>STATUS</span>
        <span style={{ width: '12%', textAlign: 'right' }}>PRICE</span>
        <span style={{ width: '12%', textAlign: 'right' }}>QTY</span>
        {showSlippage && <span style={{ width: '10%', textAlign: 'right' }}>SLIP</span>}
      </div>

      {/* Scrollable content */}
      <div
        style={{
          width: '100%',
          height: `${height - 24}px`,
          overflowY: 'auto',
          overflowX: 'hidden',
        }}
      >
        <AnimatePresence initial={false}>
          {limitedEvents.map((event) => (
            <motion.div
              key={event.id}
              variants={rowVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              layout
              style={{
                display: 'flex',
                alignItems: 'center',
                height: '28px',
                padding: '0 8px',
                borderBottom: `1px solid ${EXEC_COLORS.border}`,
                backgroundColor: event.status === 'FILLED' 
                  ? 'rgba(0, 255, 136, 0.05)' 
                  : event.status === 'CANCELLED'
                  ? 'rgba(255, 0, 85, 0.05)'
                  : 'transparent',
              }}
              className="execution-log-row"
            >
              {/* Time */}
              <span style={{ width: '15%', fontSize: '9px', color: EXEC_COLORS.textDim }}>
                {formatTime(event.timestamp)}
              </span>

              {/* Side */}
              <span
                style={{
                  width: '12%',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  color: event.side === 'BUY' ? EXEC_COLORS.filled : EXEC_COLORS.cancelled,
                  textShadow: event.side === 'BUY'
                    ? '0 0 6px rgba(0, 255, 136, 0.5)'
                    : '0 0 6px rgba(255, 0, 85, 0.5)',
                }}
              >
                {event.side}
              </span>

              {/* Type */}
              <span style={{ width: '15%', fontSize: '9px', color: EXEC_COLORS.text }}>
                {event.type}
              </span>

              {/* Status */}
              <span style={{ width: '12%' }}>
                <span style={getStatusStyle(event.status)}>
                  {event.status.replace('_', ' ')}
                </span>
              </span>

              {/* Price */}
              <span
                style={{
                  width: '12%',
                  textAlign: 'right',
                  fontSize: '10px',
                  color: event.executedPrice && event.executedPrice !== event.price
                    ? EXEC_COLORS.partial
                    : EXEC_COLORS.text,
                }}
              >
                {event.executedPrice?.toFixed(2) || event.price.toFixed(2)}
                {event.executedPrice && event.executedPrice !== event.price && (
                  <span style={{ fontSize: '8px', opacity: 0.6 }}>
                    ({event.price.toFixed(2)})
                  </span>
                )}
              </span>

              {/* Quantity with fill percentage */}
              <span
                style={{
                  width: '12%',
                  textAlign: 'right',
                  fontSize: '10px',
                  color: EXEC_COLORS.text,
                }}
              >
                {event.executedQty.toFixed(4)}
                {event.quantity > 0 && (
                  <span style={{ fontSize: '8px', opacity: 0.6 }}>
                    /{event.quantity.toFixed(4)}
                  </span>
                )}
                {getFillPercent(event) > 0 && (
                  <span style={{ fontSize: '8px', color: EXEC_COLORS.partial }}>
                    {' '}({getFillPercent(event)}%)
                  </span>
                )}
              </span>

              {/* Slippage */}
              {showSlippage && (
                <span
                  style={{
                    width: '10%',
                    textAlign: 'right',
                    fontSize: '9px',
                    color: getSlippageColor(event.slippage),
                  }}
                >
                  {event.slippage !== undefined ? `${event.slippage}bp` : '-'}
                </span>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Empty state */}
        {limitedEvents.length === 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: EXEC_COLORS.textDim,
              fontSize: '11px',
            }}
          >
            No executions yet...
          </div>
        )}
      </div>

      {/* Footer stats */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '20px',
          backgroundColor: '#0f1420',
          borderTop: `1px solid ${EXEC_COLORS.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 8px',
          fontSize: '9px',
        }}
      >
        <span style={{ color: EXEC_COLORS.text }}>EXECUTION LOG</span>
        <span style={{ color: EXEC_COLORS.textDim }}>
          {limitedEvents.length} events
        </span>
        <span style={{ color: EXEC_COLORS.filled }}>
          FILLED: {limitedEvents.filter(e => e.status === 'FILLED').length}
        </span>
        <span style={{ color: EXEC_COLORS.cancelled }}>
          CANCELLED: {limitedEvents.filter(e => e.status === 'CANCELLED').length}
        </span>
      </div>
    </div>
  );
};

export default ExecutionLog;
