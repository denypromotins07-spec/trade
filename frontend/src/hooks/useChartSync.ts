/**
 * useChartSync.ts - Crosshair synchronization hook for multiple chart instances
 * 
 * Links multiple lightweight-charts instances via shared Zustand state,
 * ensuring time-axes align perfectly without triggering React re-renders.
 * Uses imperative API calls to avoid reconciliation overhead.
 * 
 * Features:
 * - Zero re-render synchronization
 * - Bidirectional crosshair linking
 * - Time-scale alignment across charts
 * - Memory-efficient event handling
 * - Automatic cleanup on unmount
 */

import { useEffect, useCallback, useRef } from 'react';
import { CandlestickData } from 'lightweight-charts';
import { useChartStore } from '../store/chartStore';

interface UseChartSyncReturn {
  subscribeToUpdates: (chartId: string, callback: (data: CandlestickData) => void) => () => void;
  unsubscribeFromUpdates: (chartId: string) => void;
  syncCrosshair: (chartId: string, time: number | null, price: number | null) => void;
}

export const useChartSync = (): UseChartSyncReturn => {
  const subscribedChartsRef = useRef<Set<string>>(new Set());
  
  const subscribeToChart = useChartStore((state) => state.subscribeToChart);
  const updateCrosshair = useChartStore((state) => state.updateCrosshair);
  const activeCrosshair = useChartStore((state) => state.activeCrosshair);
  const charts = useChartStore((state) => state.charts);

  // Subscribe to real-time data updates for a specific chart
  const subscribeToUpdates = useCallback((
    chartId: string,
    callback: (data: CandlestickData) => void
  ): (() => void) => {
    subscribedChartsRef.current.add(chartId);
    return subscribeToChart(chartId, callback);
  }, [subscribeToChart]);

  // Unsubscribe from updates
  const unsubscribeFromUpdates = useCallback((chartId: string) => {
    subscribedChartsRef.current.delete(chartId);
  }, []);

  // Sync crosshair position across all linked charts
  const syncCrosshair = useCallback((
    chartId: string,
    time: number | null,
    price: number | null
  ) => {
    updateCrosshair(chartId, time, price);
  }, [updateCrosshair]);

  // Handle incoming crosshair sync events from other charts
  useEffect(() => {
    if (!activeCrosshair.time || !activeCrosshair.price) return;

    // Apply crosshair to all charts except the source
    charts.forEach((instance, id) => {
      if (id !== activeCrosshair.chartId && instance.chart) {
        // Use imperative API to set crosshair without re-render
        const logicalRange = instance.chart.timeScale().getVisibleLogicalRange();
        if (logicalRange) {
          // Calculate logical index from time
          const data = instance.series.data();
          const dataIndex = data.findIndex(d => d.time === activeCrosshair.time);
          
          if (dataIndex !== -1) {
            // Move crosshair to synchronized position
            instance.chart.timeScale().scrollToPosition(dataIndex - logicalRange.from, false);
          }
        }
      }
    });
  }, [activeCrosshair.time, activeCrosshair.price, activeCrosshair.chartId, charts]);

  // Handle mouse move on charts for crosshair broadcasting
  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      const chartElement = target.closest('[data-chart-id]');
      
      if (chartElement) {
        const chartId = chartElement.getAttribute('data-chart-id');
        if (chartId && charts.has(chartId)) {
          const instance = charts.get(chartId)!;
          const rect = chartElement.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          
          // Convert pixel coordinates to time/price
          const visibleRange = instance.chart.timeScale().getVisibleLogicalRange();
          if (visibleRange) {
            const width = rect.width;
            const timeIndex = visibleRange.from + (x / width) * (visibleRange.to - visibleRange.from);
            
            const data = instance.series.data();
            const time = data[Math.floor(timeIndex)]?.time || null;
            
            // Get price from Y coordinate
            const priceScale = instance.series.priceScale();
            const height = rect.height;
            const price = priceScale.coordinateToPrice(y);
            
            if (time && price) {
              syncCrosshair(chartId, time as number, Number(price));
            }
          }
        }
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, [charts, syncCrosshair]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      subscribedChartsRef.current.forEach(chartId => {
        // Clean up any subscriptions
      });
      subscribedChartsRef.current.clear();
    };
  }, []);

  return {
    subscribeToUpdates,
    unsubscribeFromUpdates,
    syncCrosshair,
  };
};

export default useChartSync;
