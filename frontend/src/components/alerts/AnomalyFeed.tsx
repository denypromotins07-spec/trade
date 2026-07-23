/**
 * AnomalyFeed.tsx - Advanced Alerting: Real-time Outlier Detection Feed
 * 
 * Displays real-time feed of Isolation Forest and Autoencoder outlier detections,
 * highlighting toxic order flow events in pulsing neon red.
 * 
 * Features:
 * - Real-time anomaly detection visualization
 * - Isolation Forest and Autoencoder score display
 * - Toxic order flow event highlighting with pulsing animations
 * - Severity-based color coding
 * - Cyberpunk terminal aesthetic with scanline effects
 */

'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

type AnomalyType = 'isolation_forest' | 'autoencoder' | 'statistical' | 'toxic_flow';
type SeverityLevel = 'low' | 'medium' | 'high' | 'critical';

interface AnomalyEvent {
  id: string;
  type: AnomalyType;
  severity: SeverityLevel;
  timestamp: number;
  description: string;
  score: number; // Anomaly score (0-1, higher = more anomalous)
  metadata: {
    symbol?: string;
    volume?: number;
    priceImpact?: number;
    walletAddress?: string;
  };
}

interface AnomalyFeedProps {
  data?: AnomalyEvent[];
  maxItems?: number;
  autoScroll?: boolean;
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const MAX_ITEMS_DEFAULT = 100;
const SEVERITY_CONFIG: Record<SeverityLevel, { color: string; label: string; pulse: boolean }> = {
  low: { color: '#666666', label: 'LOW', pulse: false },
  medium: { color: '#ffcc00', label: 'MEDIUM', pulse: false },
  high: { color: '#ff6600', label: 'HIGH', pulse: true },
  critical: { color: '#ff0044', label: 'CRITICAL', pulse: true },
};

const TYPE_CONFIG: Record<AnomalyType, { icon: string; label: string }> = {
  isolation_forest: { icon: '🌲', label: 'Isolation Forest' },
  autoencoder: { icon: '🧠', label: 'Autoencoder' },
  statistical: { icon: '📊', label: 'Statistical' },
  toxic_flow: { icon: '☣️', label: 'Toxic Flow' },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generates mock anomaly events for demonstration
 */
const generateMockAnomalies = (count: number): AnomalyEvent[] => {
  const types: AnomalyType[] = ['isolation_forest', 'autoencoder', 'statistical', 'toxic_flow'];
  const severities: SeverityLevel[] = ['low', 'medium', 'high', 'critical'];
  const symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP'];
  
  const descriptions = [
    'Unusual volume spike detected in order book',
    'Price deviation exceeds 3σ threshold',
    'Toxic order flow pattern identified',
    'Wash trading signature detected',
    'Spoofing behavior flagged by model',
    'Large wallet movement anomaly',
    'Cross-exchange arbitrage anomaly',
    'Liquidity drain event detected',
  ];
  
  return Array.from({ length: count }, (_, i) => {
    const severity = severities[Math.floor(Math.random() * severities.length)];
    const type = types[Math.floor(Math.random() * types.length)];
    
    return {
      id: `anomaly-${Date.now()}-${i}`,
      type,
      severity,
      timestamp: Date.now() - Math.random() * 3600000, // Last hour
      description: descriptions[Math.floor(Math.random() * descriptions.length)],
      score: Math.random() * 0.5 + 0.5, // 0.5-1.0 for anomalies
      metadata: {
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        volume: Math.random() * 1000000,
        priceImpact: Math.random() * 5,
        walletAddress: Math.random() > 0.5 ? `0x${Math.random().toString(16).slice(2, 10)}...` : undefined,
      },
    };
  });
};

/**
 * Formats timestamp to readable format
 */
const formatTimeAgo = (timestamp: number): string => {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Individual Anomaly Card Component with Pulsing Effect
 */
interface AnomalyCardProps {
  event: AnomalyEvent;
  index: number;
}

const AnomalyCard: React.FC<AnomalyCardProps> = ({ event, index }) => {
  const severityConfig = SEVERITY_CONFIG[event.severity];
  const typeConfig = TYPE_CONFIG[event.type];
  
  return (
    <div
      className={`relative p-3 rounded-lg border transition-all duration-300 ${
        severityConfig.pulse ? 'animate-pulse' : ''
      }`}
      style={{
        backgroundColor: `${severityConfig.color}11`,
        borderColor: severityConfig.color,
        boxShadow: severityConfig.pulse ? `0 0 15px ${severityConfig.color}44` : 'none',
        transform: 'translateZ(0)',
        willChange: 'transform',
      }}
      role="article"
      aria-label={`Anomaly alert: ${event.description}`}
    >
      {/* Header Row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">{typeConfig.icon}</span>
          <span className="text-xs font-mono text-gray-400">{typeConfig.label}</span>
        </div>
        
        <div className="flex items-center gap-2">
          <span
            className="px-1.5 py-0.5 rounded text-xs font-mono font-bold"
            style={{
              backgroundColor: severityConfig.color,
              color: '#000000',
            }}
          >
            {severityConfig.label}
          </span>
          <span className="text-xs text-gray-500 font-mono">
            {formatTimeAgo(event.timestamp)}
          </span>
        </div>
      </div>
      
      {/* Description */}
      <p className="text-sm text-gray-200 font-mono mb-2 leading-tight">
        {event.description}
      </p>
      
      {/* Metadata Grid */}
      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        {event.metadata.symbol && (
          <div className="flex items-center gap-1">
            <span className="text-cyan-400">Symbol:</span>
            <span className="text-white">${event.metadata.symbol}</span>
          </div>
        )}
        {event.metadata.volume && (
          <div className="flex items-center gap-1">
            <span className="text-cyan-400">Volume:</span>
            <span className="text-white">{event.metadata.volume.toLocaleString()}</span>
          </div>
        )}
        {event.metadata.priceImpact && (
          <div className="flex items-center gap-1">
            <span className="text-cyan-400">Impact:</span>
            <span className="text-white">{event.metadata.priceImpact.toFixed(2)}%</span>
          </div>
        )}
        {event.metadata.walletAddress && (
          <div className="col-span-2 flex items-center gap-1">
            <span className="text-cyan-400">Wallet:</span>
            <span className="text-yellow-400">{event.metadata.walletAddress}</span>
          </div>
        )}
      </div>
      
      {/* Anomaly Score Bar */}
      <div className="mt-2 pt-2 border-t border-white/10">
        <div className="flex items-center justify-between text-xs font-mono mb-1">
          <span className="text-gray-500">Anomaly Score</span>
          <span style={{ color: severityConfig.color }}>{(event.score * 100).toFixed(0)}%</span>
        </div>
        <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${event.score * 100}%`,
              backgroundColor: severityConfig.color,
              boxShadow: `0 0 8px ${severityConfig.color}`,
            }}
          />
        </div>
      </div>
      
      {/* Toxic Flow Special Indicator */}
      {event.type === 'toxic_flow' && (
        <div className="absolute top-0 right-0 w-8 h-8 overflow-hidden rounded-tr-lg">
          <div
            className="absolute top-0 right-0 w-12 h-12 rotate-45 flex items-end justify-center pb-1"
            style={{ backgroundColor: severityConfig.color }}
          >
            <span className="text-[8px] font-bold text-black transform -rotate-45">TOXIC</span>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const AnomalyFeed: React.FC<AnomalyFeedProps> = ({
  data,
  maxItems = MAX_ITEMS_DEFAULT,
  autoScroll = true,
}) => {
  const [events, setEvents] = useState<AnomalyEvent[]>(data || generateMockAnomalies(20));
  const containerRef = useRef<HTMLDivElement>(null);
  const wsBufferRef = useRef<AnomalyEvent[]>([]);
  
  // Simulate WebSocket updates for real-time anomalies
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.7) { // 30% chance of new anomaly per interval
        const newAnomaly = generateMockAnomalies(1)[0];
        wsBufferRef.current.push(newAnomaly);
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  /**
   * Process buffered anomalies
   */
  const processUpdates = useCallback(() => {
    if (wsBufferRef.current.length > 0) {
      setEvents((prev) => {
        const updated = [...wsBufferRef.current, ...prev];
        return updated.slice(0, maxItems);
      });
      wsBufferRef.current = [];
    }
  }, [maxItems]);
  
  // Process updates periodically
  useEffect(() => {
    const interval = setInterval(processUpdates, 500);
    return () => clearInterval(interval);
  }, [processUpdates]);
  
  // Sync with external data prop
  useEffect(() => {
    if (data) {
      setEvents(data.slice(0, maxItems));
    }
  }, [data, maxItems]);
  
  // Auto-scroll to top when new critical events arrive
  useEffect(() => {
    const criticalEvents = events.filter((e) => e.severity === 'critical');
    if (criticalEvents.length > 0 && autoScroll && containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [events, autoScroll]);
  
  // Statistics
  const stats = useMemo(() => {
    const total = events.length;
    const critical = events.filter((e) => e.severity === 'critical').length;
    const toxicFlow = events.filter((e) => e.type === 'toxic_flow').length;
    const avgScore = events.reduce((acc, e) => acc + e.score, 0) / total || 0;
    
    return { total, critical, toxicFlow, avgScore };
  }, [events]);

  return (
    <div className="w-full rounded-xl overflow-hidden bg-[#0a0a12]/90 backdrop-blur-sm border border-red-900/30">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-b from-[#0a0a12] to-transparent border-b border-red-500/20">
        <h3 className="text-red-400 font-mono text-sm tracking-wider uppercase">
          ⚠️ Anomaly Feed <span className="text-xs opacity-70">| ML Detection</span>
        </h3>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-gray-400">LIVE</span>
          </div>
          <span className="text-xs text-gray-500 font-mono">
            {stats.total} events
          </span>
        </div>
      </div>
      
      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-2 px-4 py-2 bg-red-500/5 border-b border-red-500/10">
        <div className="text-center">
          <div className="text-xs text-gray-500 font-mono">Total</div>
          <div className="text-lg font-mono text-white">{stats.total}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-500 font-mono">Critical</div>
          <div className="text-lg font-mono text-red-400">{stats.critical}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-500 font-mono">Toxic</div>
          <div className="text-lg font-mono text-orange-400">{stats.toxicFlow}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-500 font-mono">Avg Score</div>
          <div className="text-lg font-mono text-yellow-400">{(stats.avgScore * 100).toFixed(0)}%</div>
        </div>
      </div>
      
      {/* Event Feed */}
      <div
        ref={containerRef}
        className="p-4 space-y-3 max-h-[400px] overflow-y-auto"
        role="feed"
        aria-label="Anomaly detection event feed"
      >
        {events.length === 0 ? (
          <div className="text-center py-8 text-gray-500 font-mono text-sm">
            No anomalies detected. Systems operating normally.
          </div>
        ) : (
          events.map((event, index) => (
            <AnomalyCard key={event.id} event={event} index={index} />
          ))
        )}
      </div>
      
      {/* Footer Legend */}
      <div className="px-4 py-2 bg-gradient-to-t from-[#0a0a12] to-transparent border-t border-red-500/10">
        <div className="flex items-center justify-between text-xs font-mono text-gray-500">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-gray-600" />
              Low
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-yellow-500" />
              Medium
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-orange-500" />
              High
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              Critical
            </span>
          </div>
          <span>Isolation Forest + Autoencoder</span>
        </div>
      </div>
      
      {/* Scanline Effect Overlay */}
      <div
        className="pointer-events-none absolute inset-0 opacity-5"
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 1px, #000 1px, #000 2px)',
          backgroundSize: '100% 4px',
        }}
      />
    </div>
  );
};

export default AnomalyFeed;
