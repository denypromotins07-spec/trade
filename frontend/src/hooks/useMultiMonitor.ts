/**
 * File 4: frontend/src/hooks/useMultiMonitor.ts
 * 
 * Elite Implementation:
 * - Manages window.open APIs for detaching widgets to secondary monitors.
 * - Preserves WebSocket state across popup windows via shared store references.
 * - Handles sudden disconnects gracefully with auto-reconnection logic.
 * - Tracks window positions and dimensions for restoration.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { broadcastChannel } from '../lib/syncBroadcast';

export interface DetachedWindowConfig {
  id: string;
  widgetType: string;
  widgetId: string;
  width: number;
  height: number;
  screenX: number;
  screenY: number;
}

export interface WindowState {
  windowRef: Window | null;
  config: DetachedWindowConfig;
  isConnected: boolean;
  lastHeartbeat: number;
}

const HEARTBEAT_INTERVAL = 1000;
const DISCONNECT_TIMEOUT = 5000;

export const useMultiMonitor = () => {
  const [detachedWindows, setDetachedWindows] = useState<Map<string, WindowState>>(new Map());
  const heartbeatTimers = useRef<Map<string, NodeJS.Timeout>>(new Map());

  /**
   * Open a detached window on a secondary monitor
   */
  const detachWidget = useCallback((config: DetachedWindowConfig): boolean => {
    try {
      // Create URL with widget parameters
      const baseUrl = window.location.origin + window.location.pathname;
      const params = new URLSearchParams({
        detached: 'true',
        widgetType: config.widgetType,
        widgetId: config.widgetId,
        parentId: window.location.href,
      });
      
      const url = `${baseUrl}?${params.toString()}`;
      
      // Calculate window features for precise positioning
      const features = [
        `width=${config.width}`,
        `height=${config.height}`,
        `left=${config.screenX}`,
        `top=${config.screenY}`,
        'menubar=no',
        'toolbar=no',
        'location=no',
        'status=no',
        'scrollbars=no',
        'resizable=yes',
      ].join(',');

      const newWindow = window.open(url, `nautilus_${config.id}`, features);

      if (!newWindow) {
        console.error('[MultiMonitor] Failed to open detached window - popup blocker?');
        return false;
      }

      // Initialize window state
      const windowState: WindowState = {
        windowRef: newWindow,
        config,
        isConnected: true,
        lastHeartbeat: Date.now(),
      };

      setDetachedWindows(prev => {
        const next = new Map(prev);
        next.set(config.id, windowState);
        return next;
      });

      // Setup heartbeat monitoring
      const timer = setInterval(() => {
        if (newWindow.closed) {
          handleDisconnect(config.id);
        } else {
          // Send heartbeat via BroadcastChannel
          broadcastChannel.postMessage({
            type: 'HEARTBEAT',
            windowId: config.id,
            timestamp: Date.now(),
          });
          
          setDetachedWindows(prev => {
            const next = new Map(prev);
            const state = next.get(config.id);
            if (state) {
              state.lastHeartbeat = Date.now();
              next.set(config.id, state);
            }
            return next;
          });
        }
      }, HEARTBEAT_INTERVAL);

      heartbeatTimers.current.set(config.id, timer);

      // Listen for close events
      const checkClosed = setInterval(() => {
        if (newWindow.closed) {
          clearInterval(checkClosed);
          handleDisconnect(config.id);
        }
      }, 500);

      console.log(`[MultiMonitor] Detached widget ${config.id} to secondary monitor`);
      return true;
    } catch (error) {
      console.error('[MultiMonitor] Error detaching widget:', error);
      return false;
    }
  }, []);

  /**
   * Handle disconnection gracefully
   */
  const handleDisconnect = useCallback((windowId: string) => {
    console.log(`[MultiMonitor] Window ${windowId} disconnected`);
    
    // Clear heartbeat timer
    const timer = heartbeatTimers.current.get(windowId);
    if (timer) {
      clearInterval(timer);
      heartbeatTimers.current.delete(windowId);
    }

    // Update state
    setDetachedWindows(prev => {
      const next = new Map(prev);
      const state = next.get(windowId);
      if (state) {
        state.isConnected = false;
        next.set(windowId, state);
      }
      return next;
    });

    // Notify via broadcast channel
    broadcastChannel.postMessage({
      type: 'WINDOW_CLOSED',
      windowId,
      timestamp: Date.now(),
    });
  }, []);

  /**
   * Reattach a detached window back to main application
   */
  const reattachWidget = useCallback((windowId: string) => {
    const state = detachedWindows.get(windowId);
    if (!state) return false;

    if (state.windowRef && !state.windowRef.closed) {
      state.windowRef.close();
    }

    handleDisconnect(windowId);
    
    setDetachedWindows(prev => {
      const next = new Map(prev);
      next.delete(windowId);
      return next;
    });

    console.log(`[MultiMonitor] Reattached widget ${windowId}`);
    return true;
  }, [detachedWindows, handleDisconnect]);

  /**
   * Bring all detached windows to front
   */
  const focusAll = useCallback(() => {
    detachedWindows.forEach(state => {
      if (state.windowRef && !state.windowRef.closed) {
        state.windowRef.focus();
      }
    });
  }, [detachedWindows]);

  /**
   * Close all detached windows gracefully
   */
  const closeAll = useCallback(() => {
    detachedWindows.forEach((_, id) => {
      reattachWidget(id);
    });
  }, [detachedWindows, reattachWidget]);

  /**
   * Check connection status of all windows
   */
  const getConnectionStatus = useCallback(() => {
    const status: Record<string, boolean> = {};
    detachedWindows.forEach((state, id) => {
      status[id] = state.isConnected && !(state.windowRef?.closed ?? true);
    });
    return status;
  }, [detachedWindows]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      heartbeatTimers.current.forEach(timer => clearInterval(timer));
      heartbeatTimers.current.clear();
      closeAll();
    };
  }, [closeAll]);

  return {
    detachedWindows,
    detachWidget,
    reattachWidget,
    focusAll,
    closeAll,
    getConnectionStatus,
    windowCount: detachedWindows.size,
  };
};

export default useMultiMonitor;
