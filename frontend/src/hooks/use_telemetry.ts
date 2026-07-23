import { useEffect, useRef, useCallback, useState } from 'react';
import { useGlobalStore } from '@/store/global';

/**
 * Telemetry Data Types
 */
export interface OrderBookData {
  symbol: string;
  bids: [number, number][];
  asks: [number, number][];
  spread: number;
  midPrice: number;
  totalBidDepth: number;
  totalAskDepth: number;
  sequence: number;
  timestamp: number;
}

export interface TradeData {
  symbol: string;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  tradeId: string;
}

export interface TickerData {
  symbol: string;
  lastPrice: number;
  change24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
}

interface TelemetryState {
  orderBooks: Map<string, OrderBookData>;
  recentTrades: TradeData[];
  tickers: Map<string, TickerData>;
  lastUpdate: number;
}

// Maximum trades to keep in memory (memory limit enforcement)
const MAX_RECENT_TRADES = 500;

/**
 * Custom React Hook for Real-time Telemetry Subscription
 * 
 * Features:
 * - Subscribes to Web Worker message bus for off-main-thread parsing
 * - Batches state updates via requestAnimationFrame to eliminate UI jank
 * - Prevents layout thrashing by coalescing rapid updates
 * - Maintains memory limits by dropping stale data
 */
export function useTelemetry() {
  const workerRef = useRef<Worker | null>(null);
  const rafRef = useRef<number | null>(null);
  const pendingUpdatesRef = useRef<TelemetryPartialUpdate[]>([]);
  const wsClientRef = useRef<ReturnType<typeof import('@/lib/ws_client').getWebSocketClient> | null>(null);

  // Zustand store actions for efficient updates
  const updateSystemHealth = useGlobalStore((state) => state.updateSystemHealth);
  const setWsConnected = useGlobalStore((state) => state.setWsConnected);
  const incrementReconnectAttempts = useGlobalStore((state) => state.incrementReconnectAttempts);
  const resetReconnectAttempts = useGlobalStore((state) => state.resetReconnectAttempts);

  // Local telemetry state with refs to avoid re-renders on every tick
  const [telemetryState, setTelemetryState] = useState<TelemetryState>({
    orderBooks: new Map(),
    recentTrades: [],
    tickers: new Map(),
    lastUpdate: 0,
  });

  // Refs for latest data (used for rendering without triggering re-renders)
  const orderBooksRef = useRef<Map<string, OrderBookData>>(new Map());
  const recentTradesRef = useRef<TradeData[]>([]);
  const tickersRef = useRef<Map<string, TickerData>>(new Map());

  type TelemetryPartialUpdate = Partial<TelemetryState>;

  /**
   * Initialize Web Worker and WebSocket client
   */
  useEffect(() => {
    // Initialize Web Worker for off-main-thread parsing
    workerRef.current = new Worker(new URL('./worker_parser.ts', import.meta.url));

    // Initialize WebSocket client
    const { getWebSocketClient } = require('@/lib/ws_client');
    wsClientRef.current = getWebSocketClient('ws://localhost:8080/ws');

    // Subscribe to WebSocket messages
    const unsubscribeMessage = wsClientRef.current.onMessage((message) => {
      // Forward to worker for parsing (off main thread)
      if (workerRef.current) {
        workerRef.current.postMessage({
          type: message.type,
          data: message.data,
        });
      }
    });

    // Subscribe to connection state changes
    const unsubscribeState = wsClientRef.current.onStateChange((connected) => {
      setWsConnected(connected);
      if (connected) {
        resetReconnectAttempts();
      } else {
        incrementReconnectAttempts();
      }
    });

    // Handle worker messages
    const handleWorkerMessage = (event: MessageEvent) => {
      const { type, data, processedAt } = event.data;

      // Batch updates for requestAnimationFrame
      pendingUpdatesRef.current.push({
        [type === 'orderbook' ? 'orderBooks' : type === 'trade' ? 'recentTrades' : type === 'ticker' ? 'tickers' : '']: 
          type === 'orderbook' ? new Map([[data.symbol, data]]) :
          type === 'trade' ? [data] :
          type === 'ticker' ? new Map([[data.symbol, data]]) : undefined,
      } as TelemetryPartialUpdate);

      // Schedule batched update via requestAnimationFrame
      scheduleBatchedUpdate();
    };

    workerRef.current.addEventListener('message', handleWorkerMessage);

    // Connect WebSocket
    wsClientRef.current.connect();

    // Cleanup on unmount
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      unsubscribeMessage();
      unsubscribeState();
      workerRef.current?.removeEventListener('message', handleWorkerMessage);
      workerRef.current?.terminate();
      wsClientRef.current?.disconnect();
    };
  }, [setWsConnected, incrementReconnectAttempts, resetReconnectAttempts]);

  /**
   * Schedule batched state update via requestAnimationFrame
   * This prevents UI thread jank by coalescing rapid telemetry updates
   */
  const scheduleBatchedUpdate = useCallback(() => {
    if (rafRef.current) {
      return;  // Already scheduled
    }

    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;

      if (pendingUpdatesRef.current.length === 0) {
        return;
      }

      // Process all pending updates
      const updates = [...pendingUpdatesRef.current];
      pendingUpdatesRef.current = [];

      updates.forEach((update) => {
        if ('orderBooks' in update && update.orderBooks) {
          update.orderBooks.forEach((value, key) => {
            orderBooksRef.current.set(key, value);
          });
        }
        if ('recentTrades' in update && update.recentTrades) {
          recentTradesRef.current = [
            ...update.recentTrades,
            ...recentTradesRef.current,
          ].slice(0, MAX_RECENT_TRADES);  // Enforce memory limit
        }
        if ('tickers' in update && update.tickers) {
          update.tickers.forEach((value, key) => {
            tickersRef.current.set(key, value);
          });
        }
      });

      // Trigger re-render with latest state
      setTelemetryState({
        orderBooks: new Map(orderBooksRef.current),
        recentTrades: [...recentTradesRef.current],
        tickers: new Map(tickersRef.current),
        lastUpdate: Date.now(),
      });
    });
  }, []);

  /**
   * Get latest order book for a symbol (direct ref access, no re-render)
   */
  const getOrderBook = useCallback((symbol: string): OrderBookData | undefined => {
    return orderBooksRef.current.get(symbol);
  }, []);

  /**
   * Get latest ticker for a symbol (direct ref access, no re-render)
   */
  const getTicker = useCallback((symbol: string): TickerData | undefined => {
    return tickersRef.current.get(symbol);
  }, []);

  /**
   * Get recent trades (read-only view)
   */
  const getRecentTrades = useCallback((limit: number = 100): TradeData[] => {
    return recentTradesRef.current.slice(0, limit);
  }, []);

  /**
   * Force cleanup of buffers (call on low memory warning)
   */
  const cleanupBuffers = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'cleanup' });
    }
    recentTradesRef.current = [];
    setTelemetryState((prev) => ({
      ...prev,
      recentTrades: [],
    }));
  }, []);

  return {
    // State for React rendering
    orderBooks: telemetryState.orderBooks,
    recentTrades: telemetryState.recentTrades,
    tickers: telemetryState.tickers,
    lastUpdate: telemetryState.lastUpdate,

    // Direct access methods (no re-render)
    getOrderBook,
    getTicker,
    getRecentTrades,

    // Control methods
    cleanupBuffers,

    // Connection status from global store
    isConnected: useGlobalStore((state) => state.wsConnected),
    reconnectAttempts: useGlobalStore((state) => state.wsReconnectAttempts),
  };
}

/**
 * Specialized hook for system health metrics only
 * Optimized for TopBar component with minimal subscriptions
 */
export function useSystemHealth() {
  return useGlobalStore((state) => state.systemHealth);
}

/**
 * Specialized hook for master control state
 * Optimized for START/KILL button components
 */
export function useMasterControl() {
  return useGlobalStore((state) => state.masterControl);
}
