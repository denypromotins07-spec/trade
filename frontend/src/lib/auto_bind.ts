/**
 * auto_bind.ts - Frontend-Backend Auto-Discovery & Binding Hook
 * Stage 54: Nautilus/Ray Crypto Trading Bot
 * Polls Rust gateway for dynamically allocated WebSocket port
 * Binds instantly without manual environment configs
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// Configuration constants
const PORT_FILE_POLL_INTERVAL_MS = 100;
const GATEWAY_HEALTH_CHECK_INTERVAL_MS = 500;
const MAX_RECONNECT_ATTEMPTS = 10;
const RECONNECT_BACKOFF_MS = 1000;
const DEFAULT_GATEWAY_PORT = 8080;
const SHARED_MEMORY_PATH = '/shared/gateway_port.txt';

/**
 * WebSocket connection state enumeration
 */
export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error',
}

/**
 * Auto-bind hook return type
 */
export interface AutoBindResult {
  /** Current connection state */
  connectionState: ConnectionState;
  /** Discovered WebSocket port */
  discoveredPort: number | null;
  /** WebSocket URL */
  wsUrl: string | null;
  /** Whether auto-discovery is in progress */
  isDiscovering: boolean;
  /** Last error message */
  errorMessage: string | null;
  /** Connection latency in milliseconds */
  connectionLatencyMs: number | null;
  /** Number of reconnection attempts */
  reconnectAttempts: number;
  /** Manual reconnect trigger */
  reconnect: () => void;
  /** Manual disconnect */
  disconnect: () => void;
  /** Reset discovery and start fresh */
  reset: () => void;
}

/**
 * Port discovery result from shared memory
 */
interface PortDiscovery {
  port: number;
  timestamp: number;
  source: 'file' | 'api' | 'default';
}

/**
 * Custom hook for automatic frontend-backend binding
 * 
 * Features:
 * - Auto-discovers Rust gateway WebSocket port from shared memory
 * - Handles reconnection with exponential backoff
 * - Provides real-time connection state updates
 * - Prevents phantom clicks during reconnection
 */
export function useAutoBind(
  options: {
    /** Initial gateway host (default: localhost) */
    host?: string;
    /** Fallback port if discovery fails */
    fallbackPort?: number;
    /** Enable debug logging */
    debug?: boolean;
    /** Callback when port is discovered */
    onPortDiscovered?: (port: number) => void;
    /** Callback when connection state changes */
    onStateChange?: (state: ConnectionState) => void;
  } = {}
): AutoBindResult {
  const {
    host = 'localhost',
    fallbackPort = DEFAULT_GATEWAY_PORT,
    debug = false,
    onPortDiscovered,
    onStateChange,
  } = options;

  // State management
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    ConnectionState.DISCONNECTED
  );
  const [discoveredPort, setDiscoveredPort] = useState<number | null>(null);
  const [isDiscovering, setIsDiscovering] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [connectionLatencyMs, setConnectionLatencyMs] = useState<number | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);

  // Refs for WebSocket and timers
  const wsRef = useRef<WebSocket | null>(null);
  const discoveryTimerRef = useRef<NodeJS.Timeout | null>(null);
  const healthCheckTimerRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const connectionStartTimeRef = useRef<number | null>(null);

  // Debug logging helper
  const log = useCallback((...args: unknown[]) => {
    if (debug) {
      console.log('[AutoBind]', ...args);
    }
  }, [debug]);

  /**
   * Discover gateway port from shared memory file
   */
  const discoverPort = useCallback(async (): Promise<PortDiscovery | null> => {
    try {
      // Try to fetch port from gateway API first (preferred method)
      const apiUrl = `http://${host}:${fallbackPort}/api/v1/gateway/port`;
      
      try {
        const response = await fetch(apiUrl, {
          method: 'GET',
          signal: AbortSignal.timeout(2000),
        });
        
        if (response.ok) {
          const data = await response.json();
          const port = data.port || data.websocketPort;
          
          if (typeof port === 'number' && port > 0) {
            log('Port discovered via API:', port);
            return { port, timestamp: Date.now(), source: 'api' };
          }
        }
      } catch (apiError) {
        log('API discovery failed, trying shared memory:', apiError);
      }

      // Fallback: Try to read shared memory file
      // Note: This requires the file to be served by the gateway
      try {
        const fileResponse = await fetch(`/shared/gateway_port.txt`, {
          cache: 'no-cache',
        });
        
        if (fileResponse.ok) {
          const text = await fileResponse.text();
          const port = parseInt(text.trim(), 10);
          
          if (!isNaN(port) && port > 0) {
            log('Port discovered via shared memory file:', port);
            return { port, timestamp: Date.now(), source: 'file' };
          }
        }
      } catch (fileError) {
        log('Shared memory file discovery failed:', fileError);
      }

      return null;
    } catch (error) {
      log('Port discovery error:', error);
      return null;
    }
  }, [host, fallbackPort, log]);

  /**
   * Establish WebSocket connection to discovered port
   */
  const connectWebSocket = useCallback((port: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      log('WebSocket already connected');
      return;
    }

    const wsUrl = `ws://${host}:${port}/ws`;
    log('Connecting to WebSocket:', wsUrl);
    
    setConnectionState(ConnectionState.CONNECTING);
    connectionStartTimeRef.current = performance.now();

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        const latency = connectionStartTimeRef.current
          ? performance.now() - connectionStartTimeRef.current
          : 0;
        
        setConnectionLatencyMs(Math.round(latency));
        setConnectionState(ConnectionState.CONNECTED);
        setErrorMessage(null);
        setReconnectAttempts(0);
        
        log(`WebSocket connected (latency: ${latency.toFixed(2)}ms)`);
      };

      ws.onclose = (event) => {
        log('WebSocket closed:', event.code, event.reason);
        
        if (connectionState !== ConnectionState.DISCONNECTED) {
          handleReconnect();
        }
      };

      ws.onerror = (error) => {
        log('WebSocket error:', error);
        setErrorMessage('WebSocket connection error');
        setConnectionState(ConnectionState.ERROR);
      };

      ws.onmessage = (event) => {
        // Handle incoming messages
        log('Message received:', event.data);
      };

      wsRef.current = ws;
    } catch (error) {
      log('Failed to create WebSocket:', error);
      setErrorMessage('Failed to create WebSocket connection');
      setConnectionState(ConnectionState.ERROR);
    }
  }, [host, connectionState, log]);

  /**
   * Handle reconnection with exponential backoff
   */
  const handleReconnect = useCallback(() => {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      log('Max reconnection attempts reached');
      setConnectionState(ConnectionState.ERROR);
      setErrorMessage('Max reconnection attempts reached');
      return;
    }

    const backoffTime = Math.min(
      RECONNECT_BACKOFF_MS * Math.pow(2, reconnectAttempts),
      30000
    );

    log(`Reconnecting in ${backoffTime}ms (attempt ${reconnectAttempts + 1})`);
    setConnectionState(ConnectionState.RECONNECTING);

    reconnectTimeoutRef.current = setTimeout(() => {
      setReconnectAttempts((prev) => prev + 1);
      
      if (discoveredPort) {
        connectWebSocket(discoveredPort);
      } else {
        // Re-discover port if lost
        setIsDiscovering(true);
      }
    }, backoffTime);
  }, [reconnectAttempts, discoveredPort, connectWebSocket, log]);

  /**
   * Start port discovery polling
   */
  const startDiscovery = useCallback(async () => {
    log('Starting port discovery');
    setIsDiscovering(true);

    const tryDiscovery = async () => {
      const discovery = await discoverPort();
      
      if (discovery) {
        setDiscoveredPort(discovery.port);
        setIsDiscovering(false);
        setErrorMessage(null);
        
        onPortDiscovered?.(discovery.port);
        connectWebSocket(discovery.port);
        
        // Stop polling once port is found
        if (discoveryTimerRef.current) {
          clearInterval(discoveryTimerRef.current);
          discoveryTimerRef.current = null;
        }
      } else {
        log('Port discovery attempt failed, retrying...');
      }
    };

    // Initial attempt
    await tryDiscovery();

    // Continue polling if not found
    discoveryTimerRef.current = setInterval(tryDiscovery, PORT_FILE_POLL_INTERVAL_MS);
  }, [discoverPort, connectWebSocket, onPortDiscovered, log]);

  /**
   * Start health check ping
   */
  const startHealthCheck = useCallback(() => {
    if (healthCheckTimerRef.current) {
      clearInterval(healthCheckTimerRef.current);
    }

    healthCheckTimerRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Send ping message
        wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, GATEWAY_HEALTH_CHECK_INTERVAL_MS);
  }, []);

  /**
   * Manual reconnect trigger
   */
  const reconnect = useCallback(() => {
    log('Manual reconnect triggered');
    setReconnectAttempts(0);
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (discoveredPort) {
      connectWebSocket(discoveredPort);
    } else {
      setIsDiscovering(true);
    }
  }, [discoveredPort, connectWebSocket, log]);

  /**
   * Manual disconnect
   */
  const disconnect = useCallback(() => {
    log('Manual disconnect');
    
    // Clear all timers
    if (discoveryTimerRef.current) {
      clearInterval(discoveryTimerRef.current);
      discoveryTimerRef.current = null;
    }
    
    if (healthCheckTimerRef.current) {
      clearInterval(healthCheckTimerRef.current);
      healthCheckTimerRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setConnectionState(ConnectionState.DISCONNECTED);
    setReconnectAttempts(0);
  }, []);

  /**
   * Reset everything and start fresh
   */
  const reset = useCallback(() => {
    log('Reset triggered');
    disconnect();
    setDiscoveredPort(null);
    setErrorMessage(null);
    setConnectionLatencyMs(null);
    setIsDiscovering(true);
    
    // Start discovery again
    setTimeout(startDiscovery, 100);
  }, [disconnect, startDiscovery, log]);

  // Initialize on mount
  useEffect(() => {
    startDiscovery();
    startHealthCheck();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [startDiscovery, startHealthCheck, disconnect]);

  // Notify on state change
  useEffect(() => {
    onStateChange?.(connectionState);
  }, [connectionState, onStateChange]);

  // Compute WebSocket URL
  const wsUrl = discoveredPort ? `ws://${host}:${discoveredPort}/ws` : null;

  return {
    connectionState,
    discoveredPort,
    wsUrl,
    isDiscovering,
    errorMessage,
    connectionLatencyMs,
    reconnectAttempts,
    reconnect,
    disconnect,
    reset,
  };
}

/**
 * React component wrapper for auto-bind status display
 */
export function AutoBindStatus({
  children,
  renderFallback,
}: {
  children: React.ReactNode;
  renderFallback?: (state: ConnectionState, error: string | null) => React.ReactNode;
}): JSX.Element {
  const { connectionState, errorMessage, isDiscovering } = useAutoBind({ debug: false });

  if (isDiscovering || connectionState === ConnectionState.CONNECTING) {
    return (
      <div className="auto-bind-status discovering">
        <span className="status-indicator spinning" />
        <span>Discovering gateway...</span>
      </div>
    );
  }

  if (connectionState === ConnectionState.ERROR || 
      connectionState === ConnectionState.RECONNECTING) {
    if (renderFallback) {
      return <>{renderFallback(connectionState, errorMessage)}</>;
    }
    
    return (
      <div className="auto-bind-status error">
        <span className="status-indicator error" />
        <span>{connectionState === ConnectionState.RECONNECTING ? 'Reconnecting...' : 'Connection Error'}</span>
        {errorMessage && <span className="error-message">{errorMessage}</span>}
      </div>
    );
  }

  if (connectionState === ConnectionState.DISCONNECTED) {
    return (
      <div className="auto-bind-status disconnected">
        <span className="status-indicator" />
        <span>Disconnected</span>
      </div>
    );
  }

  return <>{children}</>;
}

export default useAutoBind;
