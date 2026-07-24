/**
 * Reconciliation WebSocket Handler: Frontend WS handler specifically for resolving
 * state mismatches instantly. Applies delta patches to React Zustand store without
 * triggering expensive full-tree re-renders or UI jank.
 * 
 * Optimized for handling massive delta patches without blocking the main JS thread.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// Maximum delta batch size before forcing incremental apply
const MAX_DELTA_BATCH_SIZE = 1000;

// Debounce interval for non-critical updates (ms)
const DEBOUNCE_INTERVAL_MS = 16; // ~60FPS

// Reconnection backoff settings
const RECONNECT_INITIAL_MS = 100;
const RECONNECT_MAX_MS = 5000;
const RECONNECT_MULTIPLIER = 2;

/**
 * State mismatch record from backend
 */
export interface StateMismatch {
  symbolIdx: number;
  expectedHash: string;
  actualHash: string;
  expectedSeq: number;
  actualSeq: number;
  driftMs: number;
  resolved: boolean;
  resolutionTimestampMs: number;
}

/**
 * Delta patch for incremental state updates
 */
export interface DeltaPatch<T> {
  path: string;
  operation: 'set' | 'delete' | 'merge';
  value?: T;
  timestamp: number;
}

/**
 * Reconciliation state slice for Zustand
 */
interface ReconciliationState {
  // Connection state
  isConnected: boolean;
  isReconnecting: boolean;
  reconnectAttempt: number;
  lastHeartbeatMs: number;
  
  // Mismatch tracking
  pendingMismatches: StateMismatch[];
  resolvedMismatches: StateMismatch[];
  totalMismatches: number;
  
  // Delta queue for batching
  deltaQueue: DeltaPatch<unknown>[];
  isApplyingDeltas: boolean;
  
  // Statistics
  stats: {
    deltasApplied: number;
    deltasDropped: number;
    avgApplyTimeMs: number;
    lastSyncMs: number;
  };
  
  // Actions
  setConnected: (connected: boolean) => void;
  addMismatch: (mismatch: StateMismatch) => void;
  markMismatchResolved: (symbolIdx: number, expectedSeq: number) => void;
  queueDelta: <T>(patch: DeltaPatch<T>) => void;
  applyPendingDeltas: () => void;
  clearResolvedMismatches: () => void;
  resetStats: () => void;
}

/**
 * Create reconciliation store with optimized delta handling
 */
export const useReconciliationStore = create<ReconciliationState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    isConnected: false,
    isReconnecting: false,
    reconnectAttempt: 0,
    lastHeartbeatMs: 0,
    
    pendingMismatches: [],
    resolvedMismatches: [],
    totalMismatches: 0,
    
    deltaQueue: [],
    isApplyingDeltas: false,
    
    stats: {
      deltasApplied: 0,
      deltasDropped: 0,
      avgApplyTimeMs: 0,
      lastSyncMs: 0,
    },
    
    setConnected: (connected: boolean) => {
      set({ 
        isConnected: connected,
        isReconnecting: !connected,
        reconnectAttempt: connected ? 0 : get().reconnectAttempt + 1,
      });
    },
    
    addMismatch: (mismatch: StateMismatch) => {
      set((state) => ({
        pendingMismatches: [...state.pendingMismatches, mismatch].slice(-100),
        totalMismatches: state.totalMismatches + 1,
      }));
      
      // Auto-trigger resync request
      requestResync(mismatch.symbolIdx, mismatch.expectedSeq);
    },
    
    markMismatchResolved: (symbolIdx: number, expectedSeq: number) => {
      set((state) => {
        const pendingIdx = state.pendingMismatches.findIndex(
          m => m.symbolIdx === symbolIdx && m.expectedSeq === expectedSeq
        );
        
        if (pendingIdx === -1) return state;
        
        const resolved = {
          ...state.pendingMismatches[pendingIdx],
          resolved: true,
          resolutionTimestampMs: Date.now(),
        };
        
        return {
          pendingMismatches: [
            ...state.pendingMismatches.slice(0, pendingIdx),
            ...state.pendingMismatches.slice(pendingIdx + 1),
          ],
          resolvedMismatches: [resolved, ...state.resolvedMismatches].slice(-100),
        };
      });
    },
    
    queueDelta: <T>(patch: DeltaPatch<T>) => {
      set((state) => ({
        deltaQueue: [...state.deltaQueue, patch as DeltaPatch<unknown>],
      }));
      
      // Trigger async apply if queue is large enough
      const queueLength = get().deltaQueue.length;
      if (queueLength >= 10 || !get().isApplyingDeltas) {
        // Use requestAnimationFrame for non-blocking apply
        requestAnimationFrame(() => {
          get().applyPendingDeltas();
        });
      }
    },
    
    applyPendingDeltas: () => {
      const { deltaQueue, isApplyingDeltas } = get();
      
      if (isApplyingDeltas || deltaQueue.length === 0) return;
      
      set({ isApplyingDeltas: true });
      
      const startTime = performance.now();
      
      // Process deltas in chunks to avoid blocking main thread
      const processChunk = (startIndex: number) => {
        const chunkSize = Math.min(100, deltaQueue.length - startIndex);
        const chunk = deltaQueue.slice(startIndex, startIndex + chunkSize);
        
        // Apply chunk using shallow merges to minimize re-renders
        chunk.forEach((patch) => {
          applyDeltaPatch(patch);
        });
        
        const nextIndex = startIndex + chunkSize;
        
        if (nextIndex < deltaQueue.length) {
          // Schedule next chunk
          requestAnimationFrame(() => processChunk(nextIndex));
        } else {
          // Done - update stats and clear queue
          const elapsed = performance.now() - startTime;
          set((state) => ({
            isApplyingDeltas: false,
            deltaQueue: [],
            stats: {
              ...state.stats,
              deltasApplied: state.stats.deltasApplied + deltaQueue.length,
              avgApplyTimeMs: (state.stats.avgApplyTimeMs * 0.9) + (elapsed * 0.1),
              lastSyncMs: Date.now(),
            },
          }));
        }
      };
      
      processChunk(0);
    },
    
    clearResolvedMismatches: () => {
      set({ resolvedMismatches: [] });
    },
    
    resetStats: () => {
      set({
        stats: {
          deltasApplied: 0,
          deltasDropped: 0,
          avgApplyTimeMs: 0,
          lastSyncMs: 0,
        },
      });
    },
  }))
);

/**
 * Apply a single delta patch to the appropriate store
 * Uses path-based targeting to minimize re-renders
 */
function applyDeltaPatch<T>(patch: DeltaPatch<T>): void {
  const { path, operation, value } = patch;
  
  // Parse path (e.g., "orderbooks.BTCUSDT.bids.0")
  const parts = path.split('.');
  
  try {
    switch (operation) {
      case 'set':
        setAtPath(parts, value);
        break;
      case 'merge':
        mergeAtPath(parts, value as Record<string, unknown>);
        break;
      case 'delete':
        deleteAtPath(parts);
        break;
    }
  } catch (error) {
    console.warn(`Failed to apply delta patch at ${path}:`, error);
    useReconciliationStore.getState().stats.deltasDropped++;
  }
}

/**
 * Set value at nested path
 */
function setAtPath<T>(parts: string[], value: T): void {
  // Implementation depends on your state structure
  // This is a simplified example
  const targetStore = getStoreForPath(parts[0]);
  if (!targetStore) return;
  
  let current: Record<string, unknown> = targetStore.getState();
  
  for (let i = 1; i < parts.length - 1; i++) {
    const key = parts[i];
    if (!(key in current)) return;
    current = current[key] as Record<string, unknown>;
  }
  
  const lastKey = parts[parts.length - 1];
  // Use immer or similar for immutable updates
  targetStore.setState((state) => ({
    ...state,
    [parts[0]]: updateNested(state[parts[0]], parts.slice(1), value),
  }));
}

/**
 * Merge value at nested path
 */
function mergeAtPath(parts: string[], value: Record<string, unknown>): void {
  const targetStore = getStoreForPath(parts[0]);
  if (!targetStore) return;
  
  targetStore.setState((state) => ({
    ...state,
    [parts[0]]: updateNested(state[parts[0]], parts.slice(1), value, true),
  }));
}

/**
 * Delete value at nested path
 */
function deleteAtPath(parts: string[]): void {
  const targetStore = getStoreForPath(parts[0]);
  if (!targetStore) return;
  
  targetStore.setState((state) => ({
    ...state,
    [parts[0]]: deleteNested(state[parts[0]], parts.slice(1)),
  }));
}

/**
 * Helper to update nested object immutably
 */
function updateNested<T>(
  obj: unknown,
  path: string[],
  value: T,
  merge = false
): unknown {
  if (path.length === 0) {
    return merge && typeof obj === 'object' && typeof value === 'object'
      ? { ...(obj as object), ...(value as object) }
      : value;
  }
  
  if (typeof obj !== 'object' || obj === null) {
    obj = {};
  }
  
  const [first, ...rest] = path;
  const key = first as keyof typeof obj;
  
  return {
    ...(obj as Record<string, unknown>),
    [key]: updateNested(obj[key], rest, value, merge),
  };
}

/**
 * Helper to delete nested key immutably
 */
function deleteNested(obj: unknown, path: string[]): unknown {
  if (path.length === 0 || typeof obj !== 'object' || obj === null) {
    return obj;
  }
  
  const [first, ...rest] = path;
  const key = first as keyof typeof obj;
  
  if (rest.length === 0) {
    const { [key]: _, ...remaining } = obj as Record<string, unknown>;
    return remaining;
  }
  
  return {
    ...(obj as Record<string, unknown>),
    [key]: deleteNested((obj as Record<string, unknown>)[key], rest),
  };
}

/**
 * Get appropriate store for path prefix
 */
function getStoreForPath(prefix: string) {
  // Import your stores dynamically
  switch (prefix) {
    case 'orderbooks':
      return (window as unknown as { orderbookStore?: unknown }).orderbookStore;
    case 'positions':
      return (window as unknown as { positionStore?: unknown }).positionStore;
    case 'portfolio':
      return (window as unknown as { portfolioStore?: unknown }).portfolioStore;
    default:
      return null;
  }
}

/**
 * Request resync from backend
 */
async function requestResync(symbolIdx: number, expectedSeq: number): Promise<void> {
  // Send resync request to backend via WebSocket
  const ws = (window as unknown as { tradingWs?: WebSocket }).tradingWs;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'RESYNC_REQUEST',
      symbolIdx,
      expectedSeq,
      timestamp: Date.now(),
    }));
  }
}

/**
 * WebSocket message handler for reconciliation events
 */
export function handleReconciliationMessage(data: unknown): void {
  const message = data as { type: string; payload?: unknown };
  
  switch (message.type) {
    case 'STATE_MISMATCH': {
      const mismatch = message.payload as StateMismatch;
      useReconciliationStore.getState().addMismatch(mismatch);
      break;
    }
    
    case 'DELTA_PATCH': {
      const patch = message.payload as DeltaPatch<unknown>;
      useReconciliationStore.getState().queueDelta(patch);
      break;
    }
    
    case 'MISMATCH_RESOLVED': {
      const { symbolIdx, expectedSeq } = message.payload as { 
        symbolIdx: number; 
        expectedSeq: number; 
      };
      useReconciliationStore.getState().markMismatchResolved(symbolIdx, expectedSeq);
      break;
    }
    
    case 'HEARTBEAT': {
      useReconciliationStore.getState().setConnected(true);
      set((state) => ({ ...state, lastHeartbeatMs: Date.now() }));
      break;
    }
  }
}

export default useReconciliationStore;
