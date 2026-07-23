'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useGlobalStore } from '@/store/global';
import { getWsClient, TelemetryMessage, WsConnectionState } from '@/lib/ws_client';
import type { WorkerRequest, WorkerResponse } from '@/lib/worker_parser';

// ============================================================================
// CUSTOM HOOK FOR REAL-TIME TELEMETRY SUBSCRIPTION
// Batches state updates via requestAnimationFrame to eliminate UI thread jank
// ============================================================================

interface UseTelemetryOptions {
  wsUrl?: string;
  autoConnect?: boolean;
  enableWorkerParsing?: boolean;
  batchIntervalMs?: number;
}

interface TelemetryState {
  isConnected: boolean;
  connectionState: WsConnectionState;
  lastMessageTime: number | null;
  messageCount: number;
  droppedMessages: number;
  workerBusy: boolean;
}

// ============================================================================
// WORKER MANAGEMENT
// Lazy initialization of Web Worker for off-main-thread parsing
// ============================================================================

let workerInstance: Worker | null = null;
let workerRequestIdCounter = 0;

function getOrCreateWorker(): Worker {
  if (!workerInstance) {
    // Dynamic import for webpack/vite compatibility
    const workerUrl = new URL('@/lib/worker_parser', import.meta.url);
    workerInstance = new Worker(workerUrl, { type: 'module' });
    
    workerInstance.onerror = (error) => {
      console.error('[useTelemetry] Worker error:', error);
    };
  }
  
  return workerInstance;
}

function terminateWorker(): void {
  if (workerInstance) {
    workerInstance.terminate();
    workerInstance = null;
  }
}

// ============================================================================
// MESSAGE BATCHING WITH REQUESTANIMATIONFRAME
// Prevents layout thrashing by syncing updates to display refresh rate
// ============================================================================

class MessageBatcher {
  private pendingMessages: TelemetryMessage[] = [];
  private rafId: number | null = null;
  private callback: ((messages: TelemetryMessage[]) => void) | null = null;
  private lastFlushTime = 0;
  private readonly intervalMs: number;

  constructor(intervalMs: number = 16) { // ~60FPS default
    this.intervalMs = intervalMs;
  }

  setCallback(callback: (messages: TelemetryMessage[]) => void): void {
    this.callback = callback;
  }

  addMessage(message: TelemetryMessage): void {
    // Drop stale messages (>500ms old) to prevent backlog
    const age = Date.now() - message.timestamp;
    if (age > 500) {
      console.debug('[MessageBatcher] Dropping stale message:', message.type);
      return;
    }

    this.pendingMessages.push(message);
    this.scheduleFlush();
  }

  private scheduleFlush(): void {
    if (this.rafId !== null) return;

    const now = performance.now();
    const timeSinceLastFlush = now - this.lastFlushTime;

    if (timeSinceLastFlush >= this.intervalMs) {
      this.flush();
    } else {
      this.rafId = requestAnimationFrame(() => {
        this.rafId = null;
        this.flush();
      });
    }
  }

  private flush(): void {
    this.lastFlushTime = performance.now();
    
    if (this.pendingMessages.length === 0) return;

    const messagesToProcess = [...this.pendingMessages];
    this.pendingMessages = [];

    // Group messages by type for efficient batch processing
    const grouped = messagesToProcess.reduce((acc, msg) => {
      if (!acc[msg.type]) acc[msg.type] = [];
      acc[msg.type].push(msg);
      return acc;
    }, {} as Record<string, TelemetryMessage[]>);

    // Process only the latest message per type (drop intermediate updates)
    const latestMessages = Object.values(grouped).map(
      group => group[group.length - 1]!
    );

    this.callback?.(latestMessages);
  }

  destroy(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.pendingMessages = [];
    this.callback = null;
  }
}

// ============================================================================
// MAIN HOOK IMPLEMENTATION
// ============================================================================

export function useTelemetry(options: UseTelemetryOptions = {}): TelemetryState & {
  sendCommand: (type: TelemetryMessage['type'], data: unknown) => void;
  parseWithWorker: (data: ArrayBuffer | string) => Promise<unknown>;
} {
  const {
    wsUrl = 'ws://localhost:8080/ws',
    autoConnect = true,
    enableWorkerParsing = true,
    batchIntervalMs = 16,
  } = options;

  // Refs for stable access without re-renders
  const wsClientRef = useRef<ReturnType<typeof getWsClient> | null>(null);
  const batcherRef = useRef<MessageBatcher | null>(null);
  const messageCountRef = useRef(0);
  const droppedMessagesRef = useRef(0);
  const lastMessageTimeRef = useRef<number | null>(null);
  const workerBusyRef = useRef(false);
  const workerCallbacksRef = useRef<Map<string, (data: unknown) => void>>(new Map());

  // Zustand store access with selectors for optimized re-renders
  const isConnected = useGlobalStore((state) => state.wsConnected);
  const updateLatencyMetrics = useGlobalStore((state) => state.updateLatencyMetrics);
  const updateGpuMetrics = useGlobalStore((state) => state.updateGpuMetrics);
  const updateSystemHealth = useGlobalStore((state) => state.updateSystemHealth);
  const setWsConnected = useGlobalStore((state) => state.setWsConnected);

  // Local state for telemetry-specific metrics
  const [connectionState, setConnectionState] = useState<WsConnectionState>('DISCONNECTED');

  // Initialize WebSocket client
  useEffect(() => {
    wsClientRef.current = getWsClient({ url: wsUrl });
    const client = wsClientRef.current;

    // Connection state changes
    client.onStateChange((state) => {
      setConnectionState(state);
      setWsConnected(state === 'CONNECTED');
      
      if (state === 'CONNECTED') {
        console.log('[useTelemetry] WebSocket connected');
      } else if (state === 'ERROR') {
        console.error('[useTelemetry] WebSocket in error state');
      }
    });

    // Handle incoming messages
    client.onMessage((message) => {
      lastMessageTimeRef.current = message.timestamp;
      messageCountRef.current++;

      // Route messages based on type
      if (enableWorkerParsing && message.type === 'ORDER_BOOK') {
        // Offload heavy parsing to worker
        const rawData = message.data as ArrayBuffer | string;
        parseWithWorker(rawData).then((parsed) => {
          // Update store with parsed data
          // (actual implementation would route to specific store updates)
        });
      } else if (message.type === 'SYSTEM_HEALTH') {
        const health = message.data as Partial<{ cpuUsage: number; ramUsageMb: number; uptimeSeconds: number }>;
        updateSystemHealth(health);
      } else if (message.type === 'GPU_METRICS') {
        const gpu = message.data as Partial<{ utilization: number; memoryUsed: number; temperature: number }>;
        updateGpuMetrics(gpu);
      } else if (message.type === 'TICKER' || message.type === 'TRADE') {
        // Batch high-frequency updates
        batcherRef.current?.addMessage(message);
      }
    });

    // Auto-connect if enabled
    if (autoConnect) {
      client.connect();
    }

    // Cleanup
    return () => {
      if (!autoConnect) {
        client.disconnect();
      }
    };
  }, [wsUrl, autoConnect, enableWorkerParsing, setWsConnected, updateSystemHealth, updateGpuMetrics]);

  // Initialize message batcher
  useEffect(() => {
    batcherRef.current = new MessageBatcher(batchIntervalMs);
    
    batcherRef.current.setCallback((messages) => {
      // Batch process ticker/trade updates
      // This is where you'd update order book state, price charts, etc.
      const latestTicker = messages.find(m => m.type === 'TICKER');
      if (latestTicker) {
        // Update latency metrics from ticker timestamp
        const latency = Date.now() - latestTicker.timestamp;
        updateLatencyMetrics({ rustCoreLatencyMs: latency });
      }
    });

    return () => {
      batcherRef.current?.destroy();
    };
  }, [batchIntervalMs, updateLatencyMetrics]);

  // Send commands to backend
  const sendCommand = useCallback((type: TelemetryMessage['type'], data: unknown) => {
    wsClientRef.current?.send({ type, data });
  }, []);

  // Parse data using Web Worker
  const parseWithWorker = useCallback(async (data: ArrayBuffer | string): Promise<unknown> => {
    if (!enableWorkerParsing) {
      throw new Error('Worker parsing is disabled');
    }

    return new Promise((resolve, reject) => {
      const worker = getOrCreateWorker();
      const requestId = `req_${++workerRequestIdCounter}`;
      const timeoutMs = 5000;

      // Set up one-time response handler
      const timeoutId = setTimeout(() => {
        workerCallbacksRef.current.delete(requestId);
        reject(new Error('Worker parsing timeout'));
        workerBusyRef.current = false;
      }, timeoutMs);

      workerCallbacksRef.current.set(requestId, (result) => {
        clearTimeout(timeoutId);
        workerCallbacksRef.current.delete(requestId);
        workerBusyRef.current = false;
        resolve(result);
      });

      // Determine request type
      const request: WorkerRequest = {
        id: requestId,
        type: data instanceof ArrayBuffer ? 'PARSE_MESSAGEPACK' : 'PARSE_JSON',
        payload: data,
      };

      workerBusyRef.current = true;
      worker.postMessage(request);

      // Handle worker responses
      const messageHandler = (event: MessageEvent<WorkerResponse>) => {
        if (event.data.id === requestId) {
          if (event.data.type === 'ERROR') {
            reject(new Error(event.data.error));
          } else {
            workerCallbacksRef.current.get(requestId)?.(event.data.data);
          }
          worker.removeEventListener('message', messageHandler);
        }
      };

      worker.addEventListener('message', messageHandler);
    });
  }, [enableWorkerParsing]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      batcherRef.current?.destroy();
      if (!autoConnect) {
        wsClientRef.current?.disconnect();
      }
    };
  }, [autoConnect]);

  return {
    isConnected,
    connectionState,
    lastMessageTime: lastMessageTimeRef.current,
    messageCount: messageCountRef.current,
    droppedMessages: droppedMessagesRef.current,
    workerBusy: workerBusyRef.current,
    sendCommand,
    parseWithWorker,
  };
}
