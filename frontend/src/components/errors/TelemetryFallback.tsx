/**
 * Telemetry Fallback Component
 * 
 * Displays cached critical PnL and open positions when WebSocket connection drops.
 * Ensures traders maintain visibility of critical data during backend disruptions.
 * 
 * Cyberpunk aesthetic: "Offline mode" with cached data indicators and pulsing reconnect status.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useSyncStore, Position, EquityPoint } from '../../hooks/useBackendSync';

interface TelemetryFallbackProps {
  maxCachedItems?: number;
  onReconnectRequest?: () => void;
}

interface CachedData {
  positions: Map<string, Position>;
  equityCurve: EquityPoint[];
  lastUpdateTime: number;
  isStale: boolean;
}

/**
 * Fallback UI for displaying cached trading data during disconnection
 */
export const TelemetryFallback: React.FC<TelemetryFallbackProps> = ({
  maxCachedItems = 100,
  onReconnectRequest,
}) => {
  const [cachedData, setCachedData] = useState<CachedData | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastKnownStatus, setLastKnownStatus] = useState<'running' | 'stopped' | 'unknown'>('unknown');

  const { state } = useSyncStore();

  /**
   * Cache current state when connection drops
   */
  useEffect(() => {
    if (state.syncStatus === 'synchronized') {
      // Update cache while connected
      setCachedData({
        positions: new Map(state.positions),
        equityCurve: [...state.equityCurve].slice(-maxCachedItems),
        lastUpdateTime: Date.now(),
        isStale: false,
      });
    } else if (state.syncStatus === 'error' || state.syncStatus === 'disconnected') {
      // Mark cache as stale when disconnected
      setCachedData((prev) => prev ? { ...prev, isStale: true } : null);
    }

    setLastKnownStatus(state.isRunning ? 'running' : 'stopped');
  }, [state, maxCachedItems]);

  /**
   * Auto-reconnect timer
   */
  useEffect(() => {
    if (state.syncStatus === 'error' || state.syncStatus === 'disconnected') {
      const interval = setInterval(() => {
        setReconnectAttempts((prev) => prev + 1);
      }, 5000);

      return () => clearInterval(interval);
    } else {
      setReconnectAttempts(0);
    }
  }, [state.syncStatus]);

  /**
   * Calculate total unrealized PnL from cached positions
   */
  const calculateTotalPnL = useCallback((): number => {
    if (!cachedData?.positions) return 0;
    
    let total = 0;
    cachedData.positions.forEach((position) => {
      total += position.unrealizedPnL;
    });
    return total;
  }, [cachedData]);

  /**
   * Format currency value
   */
  const formatCurrency = (value: number): string => {
    const absValue = Math.abs(value);
    const formatted = absValue >= 1000 
      ? `$${absValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
      : `$${absValue.toFixed(2)}`;
    return value < 0 ? `-${formatted}` : formatted;
  };

  /**
   * Format timestamp
   */
  const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp).toLocaleTimeString();
  };

  /**
   * Get staleness indicator
   */
  const getStalenessIndicator = (): string => {
    if (!cachedData) return 'NO DATA';
    
    const age = Date.now() - cachedData.lastUpdateTime;
    const minutes = Math.floor(age / 60000);
    const seconds = Math.floor((age % 60000) / 1000);
    
    if (minutes > 0) {
      return `${minutes}m ${seconds}s ago`;
    }
    return `${seconds}s ago`;
  };

  /**
   * Get latest equity value
   */
  const getLatestEquity = (): number | null => {
    if (!cachedData?.equityCurve.length) return null;
    return cachedData.equityCurve[cachedData.equityCurve.length - 1].value;
  };

  /**
   * Calculate equity change
   */
  const getEquityChange = (): number | null => {
    if (!cachedData || cachedData.equityCurve.length < 2) return null;
    
    const latest = cachedData.equityCurve[cachedData.equityCurve.length - 1].value;
    const previous = cachedData.equityCurve[cachedData.equityCurve.length - 2].value;
    return latest - previous;
  };

  if (!cachedData) {
    return (
      <div className="telemetry-fallback loading">
        <div className="loading-spinner" />
        <span>INITIALIZING OFFLINE MODE...</span>
      </div>
    );
  }

  const totalPnL = calculateTotalPnL();
  const latestEquity = getLatestEquity();
  const equityChange = getEquityChange();
  const staleness = getStalenessIndicator();

  return (
    <div className={`telemetry-fallback ${cachedData.isStale ? 'stale' : ''}`}>
      {/* Header bar */}
      <div className="fallback-header">
        <div className="connection-status">
          <span className={`status-dot ${state.syncStatus}`} />
          <span className="status-text">
            {state.syncStatus === 'error' ? 'CONNECTION LOST' : 'OFFLINE MODE'}
          </span>
        </div>
        
        <div className="cache-indicator">
          <span className="label">CACHED:</span>
          <span className="value">{staleness}</span>
          {cachedData.isStale && (
            <span className="stale-warning">⚠️ STALE DATA</span>
          )}
        </div>

        <button 
          className="btn-reconnect"
          onClick={() => {
            setReconnectAttempts(0);
            onReconnectRequest?.();
          }}
        >
          ↻ RECONNECT
        </button>
      </div>

      {/* Critical metrics grid */}
      <div className="metrics-grid">
        {/* Total PnL Card */}
        <div className="metric-card pnl">
          <div className="metric-label">TOTAL UNREALIZED PnL</div>
          <div className={`metric-value ${totalPnL >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(totalPnL)}
          </div>
          <div className="metric-subtext">
            {cachedData.positions.size} POSITIONS
          </div>
        </div>

        {/* Equity Card */}
        <div className="metric-card equity">
          <div className="metric-label">LAST KNOWN EQUITY</div>
          <div className="metric-value">
            {latestEquity !== null ? formatCurrency(latestEquity) : 'N/A'}
          </div>
          {equityChange !== null && (
            <div className={`metric-subtext ${equityChange >= 0 ? 'positive' : 'negative'}`}>
              {equityChange >= 0 ? '+' : ''}{formatCurrency(equityChange)}
            </div>
          )}
        </div>

        {/* Status Card */}
        <div className="metric-card status">
          <div className="metric-label">TRADING STATUS</div>
          <div className={`metric-value ${lastKnownStatus === 'running' ? 'running' : 'stopped'}`}>
            {lastKnownStatus === 'running' ? '● RUNNING' : '○ STOPPED'}
          </div>
          <div className="metric-subtext">
            ATTEMPTS: {reconnectAttempts}
          </div>
        </div>
      </div>

      {/* Positions table */}
      {cachedData.positions.size > 0 && (
        <div className="positions-section">
          <div className="section-header">
            <h3>CACHED POSITIONS</h3>
            <span className="count">{cachedData.positions.size} ACTIVE</span>
          </div>
          
          <div className="positions-table">
            <div className="table-header">
              <span>SYMBOL</span>
              <span>QTY</span>
              <span>ENTRY</span>
              <span>PnL</span>
            </div>
            
            {Array.from(cachedData.positions.entries()).map(([symbol, position]) => (
              <div key={symbol} className="position-row">
                <span className="symbol">{symbol}</span>
                <span className="quantity">{position.quantity.toFixed(4)}</span>
                <span className="entry">${position.entryPrice.toLocaleString()}</span>
                <span className={`pnl ${position.unrealizedPnL >= 0 ? 'positive' : 'negative'}`}>
                  {formatCurrency(position.unrealizedPnL)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Equity mini-chart placeholder */}
      {cachedData.equityCurve.length > 0 && (
        <div className="equity-section">
          <div className="section-header">
            <h3>EQUITY CURVE (CACHED)</h3>
            <span className="count">{cachedData.equityCurve.length} POINTS</span>
          </div>
          
          <div className="equity-chart-placeholder">
            <div className="chart-bars">
              {cachedData.equityCurve.slice(-50).map((point, index) => {
                const normalized = ((point.value - Math.min(...cachedData.equityCurve.map(p => p.value))) / 
                  (Math.max(...cachedData.equityCurve.map(p => p.value)) - Math.min(...cachedData.equityCurve.map(p => p.value)))) || 0;
                return (
                  <div
                    key={index}
                    className="chart-bar"
                    style={{ height: `${Math.max(10, normalized * 100)}%` }}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Warning banner */}
      {cachedData.isStale && (
        <div className="warning-banner">
          <span className="warning-icon">⚠️</span>
          <span>DATA MAY BE OUTDATED. DO NOT RELY ON CACHED VALUES FOR TRADING DECISIONS.</span>
        </div>
      )}

      {/* CSS styles */}
      <style jsx>{`
        .telemetry-fallback {
          background: linear-gradient(180deg, #0a0a14 0%, #050510 100%);
          border: 1px solid #333;
          border-radius: 8px;
          padding: 1rem;
          color: #fff;
          font-family: 'Courier New', monospace;
        }

        .telemetry-fallback.stale {
          border-color: #ffaa00;
        }

        .telemetry-fallback.loading {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          padding: 3rem;
        }

        .loading-spinner {
          width: 20px;
          height: 20px;
          border: 2px solid #333;
          border-top-color: #00f3ff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .fallback-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 1rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid #333;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #666;
        }

        .status-dot.error {
          background: #ff0055;
          animation: pulse-red 1s infinite;
        }

        .status-dot.disconnected {
          background: #ffaa00;
          animation: pulse-orange 1s infinite;
        }

        @keyframes pulse-red {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }

        @keyframes pulse-orange {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .status-text {
          font-weight: bold;
          letter-spacing: 1px;
        }

        .cache-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.85rem;
          color: #888;
        }

        .cache-indicator .value {
          color: #00f3ff;
        }

        .stale-warning {
          color: #ffaa00;
          font-weight: bold;
          animation: blink 1s step-end infinite;
        }

        @keyframes blink {
          50% { opacity: 0; }
        }

        .btn-reconnect {
          background: linear-gradient(135deg, #00f3ff, #0099ff);
          color: #050510;
          border: none;
          border-radius: 4px;
          padding: 0.5rem 1rem;
          font-family: inherit;
          font-weight: bold;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .btn-reconnect:hover {
          box-shadow: 0 0 15px rgba(0, 243, 255, 0.5);
          transform: translateY(-1px);
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .metric-card {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid #333;
          border-radius: 4px;
          padding: 1rem;
        }

        .metric-label {
          font-size: 0.75rem;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-bottom: 0.5rem;
        }

        .metric-value {
          font-size: 1.5rem;
          font-weight: bold;
          color: #fff;
        }

        .metric-value.positive {
          color: #00ff88;
        }

        .metric-value.negative {
          color: #ff0055;
        }

        .metric-value.running {
          color: #00ff88;
        }

        .metric-value.stopped {
          color: #888;
        }

        .metric-subtext {
          font-size: 0.75rem;
          color: #666;
          margin-top: 0.25rem;
        }

        .metric-subtext.positive {
          color: #00ff88;
        }

        .metric-subtext.negative {
          color: #ff0055;
        }

        .positions-section,
        .equity-section {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #333;
        }

        .section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }

        .section-header h3 {
          margin: 0;
          font-size: 0.9rem;
          color: #00f3ff;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .count {
          font-size: 0.75rem;
          color: #666;
        }

        .positions-table {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 4px;
          overflow: hidden;
        }

        .table-header {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr 1fr;
          padding: 0.75rem;
          background: rgba(0, 243, 255, 0.1);
          font-size: 0.75rem;
          font-weight: bold;
          text-transform: uppercase;
          color: #00f3ff;
        }

        .position-row {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr 1fr;
          padding: 0.75rem;
          border-top: 1px solid #222;
          font-size: 0.85rem;
        }

        .position-row:nth-child(even) {
          background: rgba(255, 255, 255, 0.02);
        }

        .symbol {
          color: #fff;
          font-weight: bold;
        }

        .quantity,
        .entry {
          color: #aaa;
        }

        .pnl.positive {
          color: #00ff88;
        }

        .pnl.negative {
          color: #ff0055;
        }

        .equity-chart-placeholder {
          height: 100px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 4px;
          padding: 0.5rem;
          display: flex;
          align-items: flex-end;
        }

        .chart-bars {
          display: flex;
          gap: 2px;
          width: 100%;
          height: 100%;
          align-items: flex-end;
        }

        .chart-bar {
          flex: 1;
          background: linear-gradient(to top, #00f3ff, #0099ff);
          border-radius: 2px 2px 0 0;
          min-height: 10%;
          transition: height 0.3s ease;
        }

        .warning-banner {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-top: 1rem;
          padding: 0.75rem;
          background: rgba(255, 170, 0, 0.1);
          border: 1px solid #ffaa00;
          border-radius: 4px;
          color: #ffaa00;
          font-size: 0.85rem;
        }

        .warning-icon {
          font-size: 1rem;
        }
      `}</style>
    </div>
  );
};

export default TelemetryFallback;
