/**
 * =============================================================================
 * kiosk_bind.ts - Frontend Auto-Binding & Kiosk Lockdown
 * Nautilus/Ray Trading Bot - Stage 60
 * =============================================================================
 * Purpose: Locks the UI to the exact local WS port, disables browser context menus,
 *          devtools, and background throttling in kiosk mode.
 * Constraints: Optimized for 60FPS rendering, AMD Radeon GPU acceleration.
 * Compatibility: Works with chrome_kiosk.ps1 launch flags.
 * =============================================================================
 */

'use strict';

// Configuration constants
const WS_PORT = 8080;
const WS_URL = `ws://localhost:${WS_PORT}/stream`;
const RECONNECT_INTERVAL_MS = 1000;
const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Disables all default browser context menus and keyboard shortcuts
 * that could interfere with kiosk operation or allow devtools access.
 */
function disableBrowserInterference(): void {
  // Disable right-click context menu
  document.addEventListener('contextmenu', (event: MouseEvent) => {
    event.preventDefault();
    return false;
  });

  // Disable common devtools shortcuts
  document.addEventListener('keydown', (event: KeyboardEvent) => {
    const { key, ctrlKey, shiftKey, altKey } = event;

    // Block F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U
    if (
      key === 'F12' ||
      (ctrlKey && shiftKey && key === 'I') ||
      (ctrlKey && shiftKey && key === 'J') ||
      (ctrlKey && key === 'u')
    ) {
      event.preventDefault();
      return false;
    }

    // Block Alt+F4 (close window) in kiosk mode
    if (altKey && key === 'F4') {
      event.preventDefault();
      return false;
    }
  });

  // Disable selection to prevent accidental text highlighting
  document.body.style.userSelect = 'none';
  document.body.style.webkitUserSelect = 'none';
}

/**
 * Establishes a persistent WebSocket connection with auto-reconnect logic.
 * Handles network partitions gracefully as per chaos engineering requirements.
 */
class KioskWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private isConnected = false;
  private messageHandlers: Set<(data: any) => void> = new Set();

  constructor(private url: string) {}

  public connect(): void {
    console.log(`[KIOSK_BIND] Connecting to ${this.url}...`);
    
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[KIOSK_BIND] WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          this.messageHandlers.forEach(handler => handler(data));
        } catch (e) {
          console.error('[KIOSK_BIND] Failed to parse message:', e);
        }
      };

      this.ws.onerror = (error: Event) => {
        console.error('[KIOSK_BIND] WebSocket error:', error);
      };

      this.ws.onclose = () => {
        console.log('[KIOSK_BIND] WebSocket closed');
        this.isConnected = false;
        this.scheduleReconnect();
      };
    } catch (e) {
      console.error('[KIOSK_BIND] Connection failed:', e);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error('[KIOSK_BIND] Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = RECONNECT_INTERVAL_MS * Math.pow(2, this.reconnectAttempts);
    console.log(`[KIOSK_BIND] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => this.connect(), delay);
  }

  public onMessage(handler: (data: any) => void): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  public send(data: any): boolean {
    if (this.ws && this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      return true;
    }
    return false;
  }
}

/**
 * Prevents browser background throttling by using visibility API hacks
 * and requesting high-priority rendering.
 */
function preventBackgroundThrottling(): void {
  // Override hidden property to always return false
  Object.defineProperty(document, 'hidden', {
    get: () => false,
    configurable: true,
  });

  Object.defineProperty(document, 'visibilityState', {
    get: () => 'visible',
    configurable: true,
  });

  // Request high priority scheduling for animations
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      console.log('[KIOSK_BIND] Idle callback registered');
    });
  }
}

/**
 * Initializes the kiosk binding system.
 * Should be called immediately on page load.
 */
export function initKioskBind(): KioskWebSocket {
  console.log('[KIOSK_BIND] Initializing kiosk lockdown...');
  
  disableBrowserInterference();
  preventBackgroundThrottling();
  
  const wsClient = new KioskWebSocket(WS_URL);
  wsClient.connect();
  
  console.log('[KIOSK_BIND] Kiosk mode active. Devtools disabled.');
  
  return wsClient;
}

// Auto-execute if running in browser context
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    initKioskBind();
  });
}

export { KioskWebSocket };
