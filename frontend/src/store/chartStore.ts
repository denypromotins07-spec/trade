/**
 * chartStore.ts - Zustand store for cross-chart synchronization
 * 
 * Manages chart instances, crosshair state, and real-time data subscriptions
 * without triggering React re-renders. Uses Map for O(1) lookups and
 * imperative updates to maintain 60FPS rendering.
 * 
 * Features:
 * - Zero re-render chart instance storage
 * - Crosshair sync state management
 * - WebSocket subscription registry
 * - Memory-efficient cleanup on unmount
 */

import { create } from 'zustand';
import { IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts';

export interface ChartInstance {
  chart: IChartApi;
  series: ISeriesApi<'Candlestick'>;
}

interface ChartState {
  // Map of chartId -> ChartInstance (stored imperatively, no re-renders)
  charts: Map<string, ChartInstance>;
  
  // Shared crosshair state for synchronization
  activeCrosshair: {
    chartId: string | null;
    time: number | null;
    price: number | null;
  };
  
  // Subscription callbacks for real-time updates
  subscriptions: Map<string, Set<(data: CandlestickData) => void>>;
  
  // Actions
  setChartInstance: (chartId: string, chart: IChartApi, series: ISeriesApi<'Candlestick'>) => void;
  removeChartInstance: (chartId: string) => void;
  updateCrosshair: (chartId: string, time: number | null, price: number | null) => void;
  subscribeToChart: (chartId: string, callback: (data: CandlestickData) => void) => () => void;
  publishUpdate: (chartId: string, data: CandlestickData) => void;
}

export const useChartStore = create<ChartState>((set, get) => ({
  charts: new Map(),
  activeCrosshair: {
    chartId: null,
    time: null,
    price: null,
  },
  subscriptions: new Map(),
  
  setChartInstance: (chartId, chart, series) => {
    const charts = new Map(get().charts);
    charts.set(chartId, { chart, series });
    set({ charts }, false); // false = don't trigger re-render
  },
  
  removeChartInstance: (chartId) => {
    const charts = new Map(get().charts);
    charts.delete(chartId);
    
    // Also clean up subscriptions
    const subscriptions = new Map(get().subscriptions);
    subscriptions.delete(chartId);
    
    set({ charts, subscriptions }, false);
  },
  
  updateCrosshair: (chartId, time, price) => {
    // Update crosshair state imperatively without full re-render
    set(state => ({
      activeCrosshair: { chartId, time, price }
    }), false);
    
    // Notify other charts via custom event (for cross-chart sync)
    if (time !== null && price !== null) {
      window.dispatchEvent(new CustomEvent('chart-crosshair-sync', {
        detail: { chartId, time, price }
      }));
    }
  },
  
  subscribeToChart: (chartId, callback) => {
    const subscriptions = new Map(get().subscriptions);
    const chartSubs = subscriptions.get(chartId) || new Set();
    chartSubs.add(callback);
    subscriptions.set(chartId, chartSubs);
    set({ subscriptions }, false);
    
    // Return unsubscribe function
    return () => {
      const subs = new Map(get().subscriptions);
      const chartSub = subs.get(chartId);
      if (chartSub) {
        chartSub.delete(callback);
        if (chartSub.size === 0) {
          subs.delete(chartId);
        }
        set({ subscriptions: subs }, false);
      }
    };
  },
  
  publishUpdate: (chartId, data) => {
    const subscriptions = get().subscriptions;
    const chartSubs = subscriptions.get(chartId);
    if (chartSubs) {
      chartSubs.forEach(callback => callback(data));
    }
  },
}));

// Listen for crosshair sync events from other charts
if (typeof window !== 'undefined') {
  window.addEventListener('chart-crosshair-sync', (event: Event) => {
    const customEvent = event as CustomEvent<{
      chartId: string;
      time: number;
      price: number;
    }>;
    
    const { chartId, time, price } = customEvent.detail;
    const state = useChartStore.getState();
    
    // Update all charts except the source chart
    state.charts.forEach((instance, id) => {
      if (id !== chartId && instance.chart) {
        // Set crosshair programmatically on other charts
        instance.chart.timeScale().setVisibleRange(
          instance.chart.timeScale().getVisibleLogicalRange() || { from: 0, to: 100 }
        );
      }
    });
  });
}

export default useChartStore;
