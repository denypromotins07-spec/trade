import { pack, unpack } from 'msgpackr';

// ============================================================================
// WEBSOCKET CLIENT CONFIGURATION
// Hyper-resilient connection with exponential backoff for Rust backend
// ============================================================================

export interface WsClientConfig {
  url: string;
  maxReconnectAttempts: number;
  initialReconnectDelayMs: number;
  maxReconnectDelayMs: number;
  heartbeatIntervalMs: number;
  messageBufferSize: number;
}

export interface TelemetryMessage {
  type: 'ORDER_BOOK' | 'TRADE' | 'TICKER' | 'SYSTEM_HEALTH' | 'GPU_METRICS' | 'STRATEGY_UPDATE';
  timestamp: number;
  data: unknown;
}

export type WsConnectionState = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED' | 'RECONNECTING' | 'ERROR';

// ============================================================================
// RESILIENT WEBSOCKET CLIENT CLASS
// Handles binary MessagePack decoding and automatic reconnection
// ============================================================================

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private config: WsClientConfig;
  private reconnectAttempts = 0;
  private reconnectDelay: number;
  private reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
  private heartbeatIntervalId: ReturnType<typeof setInterval> | null = null;
  private state: WsConnectionState = 'DISCONNECTED';
  private messageQueue: TelemetryMessage[] = [];
  
  // Callbacks
  private onConnectCallback: (() => void) | null = null;
  private onDisconnectCallback: (() => void) | null = null;
  private onMessageCallback: ((msg: TelemetryMessage) => void) | null = null;
  private onErrorCallback: ((error: Error) => void) | null = null;
  private onStateChangeCallback: ((state: WsConnectionState) => void) | null = null;

  constructor(config: Partial<WsClientConfig> = {}) {
    this.config = {
      url: config.url || 'ws://localhost:8080/ws',
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      initialReconnectDelayMs: config.initialReconnectDelayMs || 100,
      maxReconnectDelayMs: config.maxReconnectDelayMs || 30000,
      heartbeatIntervalMs: config.heartbeatIntervalMs || 5000,
      messageBufferSize: config.messageBufferSize || 1000,
    };
    this.reconnectDelay = this.config.initialReconnectDelayMs;
  }

  // ==========================================================================
  // CONNECTION MANAGEMENT
  // ==========================================================================

  public connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('[WS] Already connected');
      return;
    }

    this.setState('CONNECTING');
    
    try {
      // Binary type set to arraybuffer for MessagePack support
      this.ws = new WebSocket(this.config.url);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
    } catch (error) {
      this.handleError(new Error(`Failed to create WebSocket: ${error}`));
    }
  }

  public disconnect(): void {
    this.clearReconnectTimeout();
    this.clearHeartbeat();
    
    if (this.ws) {
      this.ws.close(1000, 'Client initiated disconnect');
      this.ws = null;
    }
    
    this.setState('DISCONNECTED');
  }

  // ==========================================================================
  // EVENT HANDLERS
  // ==========================================================================

  private handleOpen(): void {
    console.log('[WS] Connected to Rust backend');
    this.reconnectAttempts = 0;
    this.reconnectDelay = this.config.initialReconnectDelayMs;
    this.setState('CONNECTED');
    this.startHeartbeat();
    this.onConnectCallback?.();
    
    // Flush any queued messages
    this.flushMessageQueue();
  }

  private handleClose(event: CloseEvent): void {
    console.log(`[WS] Disconnected: code=${event.code}, reason=${event.reason}`);
    this.clearHeartbeat();
    this.setState('DISCONNECTED');
    this.onDisconnectCallback?.();
    
    // Attempt reconnection if not manually closed
    if (event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  private handleError(event: Event & { error?: Error }): void {
    const error = event.error || new Error('WebSocket error occurred');
    console.error('[WS] Error:', error);
    this.onErrorCallback?.(error);
  }

  private handleMessage(event: MessageEvent): void {
    try {
      let message: TelemetryMessage;

      // Handle binary MessagePack data
      if (event.data instanceof ArrayBuffer) {
        const decoded = unpack(new Uint8Array(event.data)) as TelemetryMessage;
        message = decoded;
      } else if (typeof event.data === 'string') {
        // Fallback to JSON parsing
        message = JSON.parse(event.data) as TelemetryMessage;
      } else {
        console.warn('[WS] Unknown message format');
        return;
      }

      // Validate message structure
      if (!message.type || !message.timestamp) {
        console.warn('[WS] Invalid message structure');
        return;
      }

      // Enforce message buffer limit to prevent memory bloat
      if (this.messageQueue.length >= this.config.messageBufferSize) {
        // Drop oldest messages (stale order book snapshots)
        this.messageQueue.shift();
        console.warn('[WS] Message buffer full, dropping stale snapshot');
      }

      this.messageQueue.push(message);
      this.onMessageCallback?.(message);
    } catch (error) {
      console.error('[WS] Failed to parse message:', error);
      this.onErrorCallback?.(error as Error);
    }
  }

  // ==========================================================================
  // RECONNECTION LOGIC WITH EXPONENTIAL BACKOFF
  // ==========================================================================

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('[WS] Max reconnection attempts reached');
      this.setState('ERROR');
      return;
    }

    this.reconnectAttempts++;
    this.setState('RECONNECTING');

    console.log(
      `[WS] Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts})`
    );

    this.reconnectTimeoutId = setTimeout(() => {
      this.connect();
      // Exponential backoff with cap
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        this.config.maxReconnectDelayMs
      );
    }, this.reconnectDelay);
  }

  private clearReconnectTimeout(): void {
    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }
  }

  // ==========================================================================
  // HEARTBEAT MECHANISM
  // ==========================================================================

  private startHeartbeat(): void {
    this.clearHeartbeat();
    
    this.heartbeatIntervalId = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'SYSTEM_HEALTH', timestamp: Date.now(), data: { ping: true } });
      }
    }, this.config.heartbeatIntervalMs);
  }

  private clearHeartbeat(): void {
    if (this.heartbeatIntervalId) {
      clearInterval(this.heartbeatIntervalId);
      this.heartbeatIntervalId = null;
    }
  }

  // ==========================================================================
  // MESSAGE QUEUE MANAGEMENT
  // ==========================================================================

  private flushMessageQueue(): void {
    // Messages are processed in real-time via onMessageCallback
    // Queue is maintained for potential batch processing
    this.messageQueue = [];
  }

  public send(message: Omit<TelemetryMessage, 'timestamp'>): void {
    const payload = { ...message, timestamp: Date.now() };
    
    if (this.ws?.readyState === WebSocket.OPEN) {
      // Encode as MessagePack for efficient binary transmission
      const encoded = pack(payload);
      this.ws.send(encoded);
    } else {
      console.warn('[WS] Cannot send message: not connected');
    }
  }

  // ==========================================================================
  // STATE MANAGEMENT
  // ==========================================================================

  private setState(state: WsConnectionState): void {
    this.state = state;
    this.onStateChangeCallback?.(state);
  }

  public getState(): WsConnectionState {
    return this.state;
  }

  // ==========================================================================
  // CALLBACK REGISTRATION
  // ==========================================================================

  public onConnect(callback: () => void): void {
    this.onConnectCallback = callback;
  }

  public onDisconnect(callback: () => void): void {
    this.onDisconnectCallback = callback;
  }

  public onMessage(callback: (msg: TelemetryMessage) => void): void {
    this.onMessageCallback = callback;
  }

  public onError(callback: (error: Error) => void): void {
    this.onErrorCallback = callback;
  }

  public onStateChange(callback: (state: WsConnectionState) => void): void {
    this.onStateChangeCallback = callback;
  }
}

// ============================================================================
// SINGLETON INSTANCE FOR GLOBAL ACCESS
// ============================================================================

let wsClientInstance: WebSocketClient | null = null;

export function getWsClient(config?: Partial<WsClientConfig>): WebSocketClient {
  if (!wsClientInstance) {
    wsClientInstance = new WebSocketClient(config);
  }
  return wsClientInstance;
}

export function resetWsClient(): void {
  wsClientInstance?.disconnect();
  wsClientInstance = null;
}
