/**
 * Master Backend Synchronization Hook
 * 
 * Ensures frontend state perfectly mirrors the Rust CQRS event store.
 * Batches updates using requestAnimationFrame to prevent React reconciliation loops.
 * 
 * Cyberpunk aesthetic: "Neural sync" status with visual feedback.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { create } from 'zustand';
import { rpcClient, RpcClient } from '../lib/ipc/rpc_client';
import { deserializeBinaryFrame, L2OrderBook } from '../lib/ipc/binary_protocol';

// State interfaces
export interface TradingState {
  isRunning: boolean;
  positions: Map<string, Position>;
  orderBooks: Map<string, L2OrderBook>;
  equityCurve: EquityPoint[];
  lastSyncTime: number;
  syncStatus: 'disconnected' | 'syncing' | 'synchronized' | 'error';
}

export interface Position {
  symbol: string;
  quantity: number;
  entryPrice: number;
  unrealizedPnL: number;
  timestamp: number;
}

export interface EquityPoint {
  timestamp: number;
  value: number;
}

interface SyncStore {
  state: TradingState;
  setPositions: (positions: Map<string, Position>) => void;
  updateOrderBook: (orderBook: L2OrderBook) => void;
  addEquityPoint: (point: EquityPoint) => void;
  setSyncStatus: (status: TradingState['syncStatus']) => void;
  setIsRunning: (running: boolean) => void;
  batchUpdates: (updates: Partial<TradingState>) => void;
}

const initialState: TradingState = {
  isRunning: false,
  positions: new Map(),
  orderBooks: new Map(),
  equityCurve: [],
  lastSyncTime: 0,
  syncStatus: 'disconnected',
};

/**
 * Zustand store for synchronized state
 */
export const useSyncStore = create<SyncStore>((set, get) => ({
  state: initialState,
  
  setPositions: (positions) => {
    set((state) => ({
      state: { ...state.state, positions: new Map(positions), lastSyncTime: Date.now() },
    }));
  },
  
  updateOrderBook: (orderBook) => {
    set((state) => {
      const newOrderBooks = new Map(state.state.orderBooks);
      newOrderBooks.set(orderBook.symbol, orderBook);
      return {
        state: { ...state.state, orderBooks: newOrderBooks, lastSyncTime: Date.now() },
      };
    });
  },
  
  addEquityPoint: (point) => {
    set((state) => ({
      state: {
        ...state.state,
        equityCurve: [...state.state.equityCurve.slice(-999), point], // Keep last 1000 points
        lastSyncTime: Date.now(),
      },
    }));
  },
  
  setSyncStatus: (status) => {
    set((state) => ({
      state: { ...state.state, syncStatus: status, lastSyncTime: Date.now() },
    }));
  },
  
  setIsRunning: (running) => {
    set((state) => ({
      state: { ...state.state, isRunning: running, lastSyncTime: Date.now() },
    }));
  },
  
  batchUpdates: (updates) => {
    set((state) => ({
      state: { ...state.state, ...updates, lastSyncTime: Date.now() },
    }));
  },
}));

/**
 * Batch update queue for efficient rendering
 */
class BatchUpdateQueue<T> {
  private queue: T[] = [];
  private pendingFrame: number | null = null;
  private callback: (items: T[]) => void;
  private maxBatchSize: number;

  constructor(callback: (items: T[]) => void, maxBatchSize: number = 100) {
    this.callback = callback;
    this.maxBatchSize = maxBatchSize;
  }

  push(item: T): void {
    this.queue.push(item);
    
    if (this.queue.length >= this.maxBatchSize) {
      this.flush();
    } else if (!this.pendingFrame) {
      this.pendingFrame = requestAnimationFrame(() => this.flush());
    }
  }

  flush(): void {
    if (this.pendingFrame) {
      cancelAnimationFrame(this.pendingFrame);
      this.pendingFrame = null;
    }
    
    if (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.maxBatchSize);
      this.callback(batch);
    }
  }
}

/**
 * Master synchronization hook
 */
export function useBackendSync(wsUrl?: string) {
  const [isInitialized, setIsInitialized] = useState(false);
  const clientRef = useRef<RpcClient | null>(null);
  const orderBookQueueRef = useRef<BatchUpdateQueue<L2OrderBook> | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const { 
    setSyncStatus, 
    updateOrderBook, 
    setPositions, 
    addEquityPoint, 
    setIsRunning,
    batchUpdates 
  } = useSyncStore();

  /**
   * Handle binary order book data
   */
  const handleBinaryData = useCallback((data: ArrayBuffer) => {
    const orderBook = deserializeBinaryFrame(data);
    if (orderBook && orderBookQueueRef.current) {
      orderBookQueueRef.current.push(orderBook);
    }
  }, []);

  /**
   * Handle JSON state updates
   */
  const handleStateUpdate = useCallback((update: Partial<TradingState>) => {
    batchUpdates(update);
  }, [batchUpdates]);

  /**
   * Connect to backend and subscribe to updates
   */
  const connect = useCallback(async () => {
    try {
      setSyncStatus('syncing');
      
      clientRef.current = new RpcClient(wsUrl || 'ws://localhost:8080/rpc');
      await clientRef.current.connect();
      
      // Initialize batch queue for order books
      orderBookQueueRef.current = new BatchUpdateQueue<L2OrderBook>(
        (books) => {
          // Only update the latest book per symbol to avoid redundant renders
          const latestBooks = new Map<string, L2OrderBook>();
          books.forEach((book) => latestBooks.set(book.symbol, book));
          latestBooks.forEach((book) => updateOrderBook(book));
        },
        50 // Max 50 updates per frame
      );
      
      // Subscribe to state updates
      const unsubscribePositions = clientRef.current.subscribe<Map<string, Position>>('positions', (data) => {
        if (data instanceof Map) {
          setPositions(data);
        }
      });
      
      const unsubscribeEquity = clientRef.current.subscribe<EquityPoint>('equity', (data) => {
        addEquityPoint(data);
      });
      
      const unsubscribeStatus = clientRef.current.subscribe<{ isRunning: boolean }>('status', (data) => {
        setIsRunning(data.isRunning);
      });
      
      setSyncStatus('synchronized');
      setIsInitialized(true);
      
      console.log('[BACKEND_SYNC] Connected and synchronized');
      
      return () => {
        unsubscribePositions();
        unsubscribeEquity();
        unsubscribeStatus();
      };
    } catch (error) {
      console.error('[BACKEND_SYNC] Connection failed:', error);
      setSyncStatus('error');
      scheduleReconnect();
    }
  }, [wsUrl, setSyncStatus, updateOrderBook, setPositions, addEquityPoint, setIsRunning, batchUpdates]);

  /**
   * Schedule reconnection attempt
   */
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    reconnectTimeoutRef.current = setTimeout(() => {
      console.log('[BACKEND_SYNC] Attempting reconnection...');
      connect();
    }, 3000);
  }, [connect]);

  /**
   * Execute command via RPC
   */
  const executeCommand = useCallback(async <TParams, TResult>(
    method: string,
    params: TParams
  ): Promise<TResult> => {
    if (!clientRef.current) {
      throw new Error('RPC client not initialized');
    }
    return clientRef.current.execute<TParams, TResult>(method, params);
  }, []);

  /**
   * Execute /START command
   */
  const startTrading = useCallback(async (): Promise<void> => {
    try {
      await executeCommand('/START', { timestamp: Date.now() });
      setIsRunning(true);
    } catch (error) {
      console.error('[BACKEND_SYNC] Failed to start trading:', error);
      throw error;
    }
  }, [executeCommand, setIsRunning]);

  /**
   * Execute /KILL command
   */
  const killTrading = useCallback(async (): Promise<void> => {
    try {
      await executeCommand('/KILL', { timestamp: Date.now(), force: true });
      setIsRunning(false);
    } catch (error) {
      console.error('[BACKEND_SYNC] Failed to kill trading:', error);
      throw error;
    }
  }, [executeCommand, setIsRunning]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (clientRef.current) {
        clientRef.current.disconnect();
      }
      if (orderBookQueueRef.current) {
        orderBookQueueRef.current.flush();
      }
    };
  }, [connect]);

  return {
    isInitialized,
    connect,
    executeCommand,
    startTrading,
    killTrading,
    getState: () => useSyncStore.getState().state,
  };
}

/**
 * Selector hooks for optimized re-renders
 */
export function useTradingStatus() {
  return useSyncStore((state) => state.state.isRunning);
}

export function useSyncStatus() {
  return useSyncStore((state) => state.state.syncStatus);
}

export function usePosition(symbol: string) {
  return useSyncStore((state) => state.state.positions.get(symbol));
}

export function useOrderBook(symbol: string) {
  return useSyncStore((state) => state.state.orderBooks.get(symbol));
}

export function useEquityCurve() {
  return useSyncStore((state) => state.state.equityCurve);
}
