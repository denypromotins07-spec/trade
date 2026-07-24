/**
 * `frontend/src/hooks/useCrosshairSync.ts`
 *
 * **Global Crosshair Synchronization Hook**
 * Utilizes the BroadcastChannel API to align time-axes across all 6+ charts
 * without triggering React reconciliation loops.
 *
 * **Architecture:**
 * - Uses BroadcastChannel for efficient cross-tab/cross-component communication.
 * - Direct DOM manipulation for crosshair positioning (bypasses React state).
 * - Throttled updates to prevent main thread blocking during high-frequency data.
 */

import { useEffect, useRef, useCallback } from 'react';

const CHANNEL_NAME = 'nautilus-crosshair-sync';
const THROTTLE_MS = 16; // ~60 FPS

interface CrosshairMessage {
  type: 'crosshair-move' | 'crosshair-hide';
  chartId: number;
  timestamp?: number;
  x?: number;
  y?: number;
}

type ChartRegistry = Map<number, HTMLElement>;

class CrosshairSyncManager {
  private channel: BroadcastChannel | null = null;
  private charts: ChartRegistry = new Map();
  private lastUpdate: number = 0;
  private throttleTimer: number | null = null;

  constructor() {
    if (typeof BroadcastChannel !== 'undefined') {
      this.channel = new BroadcastChannel(CHANNEL_NAME);
      this.channel.onmessage = this.handleMessage.bind(this);
    }
  }

  registerChart(chartId: number, element: HTMLElement) {
    this.charts.set(chartId, element);
  }

  unregisterChart(chartId: number) {
    this.charts.delete(chartId);
  }

  broadcastCrosshairMove(chartId: number, timestamp: number, x: number, y: number) {
    const now = Date.now();
    
    // Throttle updates
    if (now - this.lastUpdate < THROTTLE_MS) {
      if (this.throttleTimer) {
        cancelAnimationFrame(this.throttleTimer);
      }
      this.throttleTimer = requestAnimationFrame(() => {
        this.sendCrosshairMessage(chartId, timestamp, x, y);
      });
    } else {
      this.sendCrosshairMessage(chartId, timestamp, x, y);
    }
  }

  private sendCrosshairMessage(chartId: number, timestamp: number, x: number, y: number) {
    this.lastUpdate = Date.now();
    
    const message: CrosshairMessage = {
      type: 'crosshair-move',
      chartId,
      timestamp,
      x,
      y,
    };

    if (this.channel) {
      this.channel.postMessage(message);
    }

    // Update local charts immediately
    this.updateOtherCharts(chartId, timestamp, x, y);
  }

  broadcastCrosshairHide() {
    const message: CrosshairMessage = {
      type: 'crosshair-hide',
      chartId: -1,
    };

    if (this.channel) {
      this.channel.postMessage(message);
    }

    this.hideAllCrosshairs();
  }

  private handleMessage(event: MessageEvent<CrosshairMessage>) {
    const { type, chartId, timestamp, x, y } = event.data;

    if (type === 'crosshair-hide') {
      this.hideAllCrosshairs();
    } else if (type === 'crosshair-move' && timestamp !== undefined && x !== undefined && y !== undefined) {
      this.updateOtherCharts(chartId, timestamp, x, y);
    }
  }

  private updateOtherCharts(sourceChartId: number, timestamp: number, x: number, y: number) {
    this.charts.forEach((element, id) => {
      if (id !== sourceChartId) {
        // Direct DOM manipulation to avoid React re-render
        const crosshair = element.querySelector('.crosshair-line');
        if (crosshair) {
          (crosshair as HTMLElement).style.display = 'block';
          (crosshair as HTMLElement).style.left = `${x}px`;
        }
        
        // Update tooltip with synchronized data
        const tooltip = element.querySelector('.chart-tooltip');
        if (tooltip) {
          (tooltip as HTMLElement).textContent = `T: ${timestamp}`;
        }
      }
    });
  }

  private hideAllCrosshairs() {
    this.charts.forEach((element) => {
      const crosshair = element.querySelector('.crosshair-line');
      if (crosshair) {
        (crosshair as HTMLElement).style.display = 'none';
      }
    });
  }

  destroy() {
    if (this.channel) {
      this.channel.close();
      this.channel = null;
    }
    this.charts.clear();
  }
}

// Singleton instance
let syncManager: CrosshairSyncManager | null = null;

const getSyncManager = () => {
  if (!syncManager) {
    syncManager = new CrosshairSyncManager();
  }
  return syncManager;
};

/**
 * React Hook for Crosshair Synchronization
 */
export function useCrosshairSync() {
  const chartRef = useRef<HTMLElement | null>(null);
  const chartIdRef = useRef<number>(-1);
  const manager = getSyncManager();

  const registerChart = useCallback((id: number) => {
    chartIdRef.current = id;
    if (chartRef.current) {
      manager.registerChart(id, chartRef.current);
    }
  }, [manager]);

  const unregisterChart = useCallback((id: number) => {
    manager.unregisterChart(id);
    chartIdRef.current = -1;
    chartRef.current = null;
  }, [manager]);

  const setChartElement = useCallback((element: HTMLElement | null) => {
    chartRef.current = element;
    if (element && chartIdRef.current !== -1) {
      manager.registerChart(chartIdRef.current, element);
    }
  }, []);

  const broadcastMove = useCallback((timestamp: number, x: number, y: number) => {
    if (chartIdRef.current !== -1) {
      manager.broadcastCrosshairMove(chartIdRef.current, timestamp, x, y);
    }
  }, [manager]);

  const broadcastHide = useCallback(() => {
    manager.broadcastCrosshairHide();
  }, [manager]);

  useEffect(() => {
    return () => {
      if (chartIdRef.current !== -1) {
        unregisterChart(chartIdRef.current);
      }
    };
  }, [unregisterChart]);

  return {
    registerChart,
    unregisterChart,
    setChartElement,
    broadcastMove,
    broadcastHide,
  };
}

export default useCrosshairSync;
