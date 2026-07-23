/**
 * File 6: frontend/src/lib/syncBroadcast.ts
 * 
 * Elite Implementation:
 * - BroadcastChannel API for cross-window state synchronization.
 * - Drops stale payloads to enforce memory limits.
 * - Synchronizes crosshair timings, active symbols, and execution events.
 * - Fallback to localStorage events if BroadcastChannel is unavailable.
 */

export interface SyncMessage {
  type: 'CROSSHAIR_MOVE' | 'SYMBOL_CHANGE' | 'EXECUTION' | 'HEARTBEAT' | 'WINDOW_CLOSING' | 'WINDOW_CLOSED';
  sourceId: string;
  timestamp: number;
  payload?: any;
}

class SyncBroadcast {
  private channel: BroadcastChannel | null = null;
  private subscribers: Map<string, Set<(msg: SyncMessage) => void>> = new Map();
  private messageQueue: SyncMessage[] = [];
  private readonly MAX_QUEUE_SIZE = 100;
  private readonly STALE_THRESHOLD_MS = 5000;

  constructor() {
    this.init();
  }

  private init() {
    try {
      this.channel = new BroadcastChannel('nautilus_sync_v1');
      
      this.channel.onmessage = (event) => {
        const msg = event.data as SyncMessage;
        this.processMessage(msg);
      };

      this.channel.onmessageerror = (error) => {
        console.error('[SyncBroadcast] Message error:', error);
      };

      console.log('[SyncBroadcast] Initialized with BroadcastChannel');
    } catch (error) {
      console.warn('[SyncBroadcast] BroadcastChannel unavailable, falling back to localStorage');
      this.initLocalStorageFallback();
    }
  }

  private initLocalStorageFallback() {
    window.addEventListener('storage', (event) => {
      if (event.key === 'nautilus_sync_message' && event.newValue) {
        try {
          const msg = JSON.parse(event.newValue) as SyncMessage;
          this.processMessage(msg);
        } catch (e) {
          console.error('[SyncBroadcast] Failed to parse localStorage message:', e);
        }
      }
    });
  }

  private processMessage(msg: SyncMessage) {
    // Drop stale messages
    const age = Date.now() - msg.timestamp;
    if (age > this.STALE_THRESHOLD_MS) {
      console.debug(`[SyncBroadcast] Dropping stale message: ${msg.type}`);
      return;
    }

    // Notify subscribers
    const handlers = this.subscribers.get(msg.type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(msg);
        } catch (e) {
          console.error('[SyncBroadcast] Handler error:', e);
        }
      });
    }

    // Also notify wildcard subscribers
    const wildcardHandlers = this.subscribers.get('*');
    if (wildcardHandlers) {
      wildcardHandlers.forEach(handler => handler(msg));
    }
  }

  /**
   * Post a message to all windows
   */
  public postMessage(msg: Omit<SyncMessage, 'sourceId' | 'timestamp'>): void {
    const fullMsg: SyncMessage = {
      ...msg,
      sourceId: this.generateSourceId(),
      timestamp: Date.now(),
    };

    // Queue management - drop oldest if queue is full
    if (this.messageQueue.length >= this.MAX_QUEUE_SIZE) {
      this.messageQueue.shift();
    }
    this.messageQueue.push(fullMsg);

    // Send via BroadcastChannel
    if (this.channel) {
      this.channel.postMessage(fullMsg);
    } else {
      // Fallback to localStorage
      try {
        localStorage.setItem('nautilus_sync_message', JSON.stringify(fullMsg));
        // Clear immediately to allow duplicate events
        setTimeout(() => localStorage.removeItem('nautilus_sync_message'), 10);
      } catch (e) {
        console.error('[SyncBroadcast] Failed to send via localStorage:', e);
      }
    }
  }

  /**
   * Subscribe to specific message types
   */
  public subscribe(
    type: string | '*',
    handler: (msg: SyncMessage) => void
  ): () => void {
    if (!this.subscribers.has(type)) {
      this.subscribers.set(type, new Set());
    }
    this.subscribers.get(type)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.subscribers.get(type);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.subscribers.delete(type);
        }
      }
    };
  }

  /**
   * Broadcast crosshair movement
   */
  public broadcastCrosshair(symbol: string, price: number, time: number): void {
    this.postMessage({
      type: 'CROSSHAIR_MOVE',
      payload: { symbol, price, time },
    });
  }

  /**
   * Broadcast symbol change
   */
  public broadcastSymbolChange(symbol: string): void {
    this.postMessage({
      type: 'SYMBOL_CHANGE',
      payload: { symbol },
    });
  }

  /**
   * Broadcast execution event
   */
  public broadcastExecution(executionData: {
    symbol: string;
    side: 'BUY' | 'SELL';
    quantity: number;
    price: number;
    orderId: string;
  }): void {
    this.postMessage({
      type: 'EXECUTION',
      payload: executionData,
    });
  }

  /**
   * Get queued messages (for recovery)
   */
  public getQueuedMessages(): SyncMessage[] {
    return [...this.messageQueue];
  }

  /**
   * Clear stale messages from queue
   */
  public clearStaleMessages(): void {
    const now = Date.now();
    this.messageQueue = this.messageQueue.filter(
      msg => now - msg.timestamp < this.STALE_THRESHOLD_MS
    );
  }

  /**
   * Generate unique source ID
   */
  private generateSourceId(): string {
    if (!(window as any).__nautilusSourceId) {
      (window as any).__nautilusSourceId = `src_${Math.random().toString(36).slice(2, 10)}`;
    }
    return (window as any).__nautilusSourceId;
  }

  /**
   * Close the channel
   */
  public close(): void {
    if (this.channel) {
      this.channel.close();
      this.channel = null;
    }
    this.subscribers.clear();
    this.messageQueue = [];
  }
}

// Singleton instance
export const broadcastChannel = new SyncBroadcast();

// Auto-cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    broadcastChannel.postMessage({
      type: 'WINDOW_CLOSING',
      payload: { reason: 'page_unload' },
    });
  });
}

export default broadcastChannel;
