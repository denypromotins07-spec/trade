import { pack, unpack } from 'msgpackr';

/**
 * WebSocket Message Types from Rust Backend
 */
export interface TelemetryMessage {
  type: 'orderbook' | 'trade' | 'ticker' | 'system_health' | 'strategy_update';
  timestamp: number;
  data: unknown;
}

export interface OrderBookSnapshot {
  symbol: string;
  bids: [number, number][];  // [price, size]
  asks: [number, number][];
  sequence: number;
}

export interface TradeTick {
  symbol: string;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  tradeId: string;
}

export interface SystemHealthUpdate {
  cpuUsage: number;
  ramUsage: number;
  gpuUsage: number;
  gpuMemory: number;
  networkLatency: number;
  uptime: number;
}

/**
 * Hyper-resilient WebSocket Client with Exponential Backoff
 * Optimized for ultra-low latency telemetry ingestion from Rust backend.
 * Supports binary MessagePack decoding to minimize parsing overhead.
 */
export class ResilientWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private baseDelay = 1000;  // 1 second
  private maxDelay = 30000;  // 30 seconds
  private messageHandlers: Set<(data: TelemetryMessage) => void> = new Set();
  private stateHandlers: Set<(connected: boolean) => void> = new Set();
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private isManualClose = false;

  constructor(url: string) {
    this.url = url;
  }

  /**
   * Connect to WebSocket server with automatic reconnection logic
   */
  public connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      this.ws = new WebSocket(this.url);
      this.ws.binaryType = 'arraybuffer';  // Binary MessagePack support

      this.ws.onopen = () => {
        console.log('[WS] Connected to backend');
        this.reconnectAttempts = 0;
        this.isManualClose = false;
        this.notifyStateChange(true);
        this.startPingInterval();
      };

      this.ws.onclose = (event) => {
        console.log(`[WS] Disconnected: ${event.code} ${event.reason}`);
        this.stopPingInterval();
        this.notifyStateChange(false);

        if (!this.isManualClose) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event);
      };
    } catch (error) {
      console.error('[WS] Connection failed:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Handle incoming messages with binary MessagePack decoding
   */
  private handleMessage(event: MessageEvent): void {
    try {
      let data: TelemetryMessage;

      if (event.data instanceof ArrayBuffer) {
        // Binary MessagePack - fastest path for high-frequency data
        const uint8Array = new Uint8Array(event.data);
        data = unpack(uint8Array) as TelemetryMessage;
      } else if (typeof event.data === 'string') {
        // JSON fallback
        data = JSON.parse(event.data) as TelemetryMessage;
      } else {
        console.warn('[WS] Unknown message format');
        return;
      }

      // Notify all registered handlers
      this.messageHandlers.forEach((handler) => {
        try {
          handler(data);
        } catch (err) {
          console.error('[WS] Handler error:', err);
        }
      });
    } catch (error) {
      console.error('[WS] Message decode error:', error);
    }
  }

  /**
   * Exponential backoff reconnection strategy
   * Gracefully handles backend restarts without manual refresh
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WS] Max reconnect attempts reached');
      return;
    }

    const delay = Math.min(
      this.baseDelay * Math.pow(2, this.reconnectAttempts),
      this.maxDelay
    );

    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 1000;
    const totalDelay = delay + jitter;

    console.log(`[WS] Reconnecting in ${Math.round(totalDelay)}ms (attempt ${this.reconnectAttempts + 1})`);

    setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, totalDelay);
  }

  /**
   * Subscribe to telemetry messages
   */
  public onMessage(handler: (data: TelemetryMessage) => void): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  /**
   * Subscribe to connection state changes
   */
  public onStateChange(handler: (connected: boolean) => void): () => void {
    this.stateHandlers.add(handler);
    return () => this.stateHandlers.delete(handler);
  }

  /**
   * Send message to backend (supports MessagePack encoding)
   */
  public send(data: unknown): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn('[WS] Cannot send: not connected');
      return;
    }

    try {
      const packed = pack(data);
      this.ws.send(packed);
    } catch (error) {
      console.error('[WS] Send error:', error);
    }
  }

  /**
   * Graceful disconnect
   */
  public disconnect(): void {
    this.isManualClose = true;
    this.stopPingInterval();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Check connection status
   */
  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get current reconnect attempt count
   */
  public getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }

  /**
   * Reset reconnect counter (called on successful connection)
   */
  public resetReconnectAttempts(): void {
    this.reconnectAttempts = 0;
  }

  private notifyStateChange(connected: boolean): void {
    this.stateHandlers.forEach((handler) => {
      try {
        handler(connected);
      } catch (err) {
        console.error('[WS] State handler error:', err);
      }
    });
  }

  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, 30000);  // Ping every 30 seconds
  }

  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
}

// Singleton instance for application-wide use
let wsClientInstance: ResilientWebSocketClient | null = null;

export function getWebSocketClient(url: string = 'ws://localhost:8080/ws'): ResilientWebSocketClient {
  if (!wsClientInstance) {
    wsClientInstance = new ResilientWebSocketClient(url);
  }
  return wsClientInstance;
}
