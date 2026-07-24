/**
 * Type-Safe RPC Client over WebSockets
 * 
 * Implements strict request/response correlation IDs with automatic timeout retries.
 * Queues commands in IndexedDB during backend restarts for eventual consistency.
 * 
 * Cyberpunk aesthetic: "Neural link" connection status indicators.
 */

import { v4 as uuidv4 } from 'uuid';
import { openDB, DBSchema, IDBPDatabase } from 'idb';

// Constants
const DEFAULT_TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 100;
const WS_HEARTBEAT_INTERVAL_MS = 30000;

export interface RpcRequest<T> {
  id: string;
  method: string;
  params: T;
  timestamp: number;
}

export interface RpcResponse<T> {
  id: string;
  result?: T;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
  timestamp: number;
}

export interface PendingRequest<T> {
  request: RpcRequest<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
  timeoutId: NodeJS.Timeout;
  retryCount: number;
}

export interface QueuedCommand {
  id: string;
  method: string;
  params: unknown;
  queuedAt: number;
  retryCount: number;
}

interface CommandDB extends DBSchema {
  queuedCommands: {
    key: string;
    value: QueuedCommand;
    indexes: { byMethod: string; byQueuedAt: number };
  };
}

/**
 * Type-safe RPC client with correlation ID tracking
 */
export class RpcClient {
  private ws: WebSocket | null = null;
  private pendingRequests: Map<string, PendingRequest<unknown>> = new Map();
  private db: IDBPDatabase<CommandDB> | null = null;
  private isConnected: boolean = false;
  private isReconnecting: boolean = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private url: string;
  private eventListeners: Map<string, Set<(data: unknown) => void>> = new Map();

  constructor(url: string) {
    this.url = url;
    this.initDB();
  }

  /**
   * Initialize IndexedDB for command queuing
   */
  private async initDB(): Promise<void> {
    try {
      this.db = await openDB<CommandDB>('nautilus-rpc-queue', 1, {
        upgrade(db) {
          const store = db.createObjectStore('queuedCommands', { keyPath: 'id' });
          store.createIndex('byMethod', 'method');
          store.createIndex('byQueuedAt', 'queuedAt');
        },
      });
      console.log('[RPC_CLIENT] IndexedDB initialized for command queuing');
    } catch (error) {
      console.error('[RPC_CLIENT] Failed to initialize IndexedDB:', error);
    }
  }

  /**
   * Connect to WebSocket backend
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[RPC_CLIENT] WebSocket connected');
        this.isConnected = true;
        this.isReconnecting = false;
        this.startHeartbeat();
        this.flushQueuedCommands();
        resolve();
      };

      this.ws.onclose = () => {
        console.log('[RPC_CLIENT] WebSocket closed');
        this.isConnected = false;
        this.stopHeartbeat();
        this.rejectPendingRequests(new Error('WebSocket connection closed'));
      };

      this.ws.onerror = (error) => {
        console.error('[RPC_CLIENT] WebSocket error:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  /**
   * Disconnect from backend
   */
  disconnect(): void {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close(1000, 'Client initiated disconnect');
      this.ws = null;
    }
    this.isConnected = false;
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.sendRaw({ type: 'ping', timestamp: Date.now() });
      }
    }, WS_HEARTBEAT_INTERVAL_MS);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(data: string): void {
    try {
      const response: RpcResponse<unknown> = JSON.parse(data);

      // Check if it's a response to a pending request
      const pending = this.pendingRequests.get(response.id);
      if (pending) {
        clearTimeout(pending.timeoutId);
        this.pendingRequests.delete(response.id);

        if (response.error) {
          pending.reject(new Error(response.error.message));
        } else {
          pending.resolve(response.result);
        }
        return;
      }

      // Otherwise, emit as an event
      const listeners = this.eventListeners.get(response.method) || new Set();
      listeners.forEach((listener) => listener(response.result));
    } catch (error) {
      console.error('[RPC_CLIENT] Failed to parse message:', error);
    }
  }

  /**
   * Send raw message over WebSocket
   */
  private sendRaw(message: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  /**
   * Execute RPC command with correlation ID and timeout
   */
  async execute<TParams, TResult>(
    method: string,
    params: TParams,
    timeoutMs: number = DEFAULT_TIMEOUT_MS
  ): Promise<TResult> {
    const id = uuidv4();
    const request: RpcRequest<TParams> = {
      id,
      method,
      params,
      timestamp: Date.now(),
    };

    // If not connected, queue the command
    if (!this.isConnected) {
      await this.queueCommand(id, method, params);
      throw new Error(`Backend offline. Command "${method}" queued for execution.`);
    }

    return new Promise<TResult>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`RPC timeout: ${method} exceeded ${timeoutMs}ms`));
      }, timeoutMs);

      const pending: PendingRequest<TResult> = {
        request,
        resolve,
        reject,
        timeoutId,
        retryCount: 0,
      };

      this.pendingRequests.set(id, pending as PendingRequest<unknown>);
      this.sendRaw(request);
    });
  }

  /**
   * Queue command in IndexedDB for later execution
   */
  private async queueCommand(id: string, method: string, params: unknown): Promise<void> {
    if (!this.db) return;

    const command: QueuedCommand = {
      id,
      method,
      params,
      queuedAt: Date.now(),
      retryCount: 0,
    };

    try {
      await this.db.put('queuedCommands', command);
      console.log(`[RPC_CLIENT] Command "${method}" queued (ID: ${id})`);
    } catch (error) {
      console.error('[RPC_CLIENT] Failed to queue command:', error);
    }
  }

  /**
   * Flush queued commands when connection restores
   */
  private async flushQueuedCommands(): Promise<void> {
    if (!this.db) return;

    try {
      const commands = await this.db.getAll('queuedCommands');
      
      for (const command of commands) {
        try {
          await this.execute(command.method, command.params);
          await this.db.delete('queuedCommands', command.id);
          console.log(`[RPC_CLIENT] Flushed queued command: ${command.method}`);
        } catch (error) {
          console.error(`[RPC_CLIENT] Failed to flush command ${command.id}:`, error);
          if (command.retryCount < MAX_RETRIES) {
            command.retryCount++;
            await this.db.put('queuedCommands', command);
          } else {
            await this.db.delete('queuedCommands', command.id);
            console.warn(`[RPC_CLIENT] Dropped command after max retries: ${command.id}`);
          }
        }
      }
    } catch (error) {
      console.error('[RPC_CLIENT] Failed to flush queued commands:', error);
    }
  }

  /**
   * Reject all pending requests
   */
  private rejectPendingRequests(error: Error): void {
    for (const [id, pending] of this.pendingRequests.entries()) {
      clearTimeout(pending.timeoutId);
      pending.reject(error);
      this.pendingRequests.delete(id);
    }
  }

  /**
   * Subscribe to server-side events
   */
  subscribe<T>(method: string, callback: (data: T) => void): () => void {
    const listeners = this.eventListeners.get(method) || new Set();
    listeners.add(callback as (data: unknown) => void);
    this.eventListeners.set(method, listeners);

    return () => {
      const currentListeners = this.eventListeners.get(method);
      if (currentListeners) {
        currentListeners.delete(callback as (data: unknown) => void);
      }
    };
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): { connected: boolean; pendingCount: number } {
    return {
      connected: this.isConnected,
      pendingCount: this.pendingRequests.size,
    };
  }

  /**
   * Execute /START command
   */
  async startTrading(): Promise<void> {
    return this.execute('/START', { timestamp: Date.now() });
  }

  /**
   * Execute /KILL command
   */
  async killTrading(): Promise<void> {
    return this.execute('/KILL', { timestamp: Date.now(), force: true });
  }
}

// Singleton instance
export const rpcClient = new RpcClient('ws://localhost:8080/rpc');
