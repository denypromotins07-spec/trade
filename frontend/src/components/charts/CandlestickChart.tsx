/**
 * CandlestickChart.tsx - High-performance OHLCV rendering using lightweight-charts
 * 
 * This component integrates TradingView's lightweight-charts library which uses
 * HTML5 Canvas for ultra-fast, memory-efficient rendering. Strictly avoids heavy
 * SVG DOM nodes to maintain 60FPS during high-frequency Binance WebSocket updates.
 * 
 * Features:
 * - Canvas-based rendering (no SVG bloat)
 * - Binance tick size precision handling
 * - Hardware-accelerated drawing pipeline
 * - Crosshair sync via Zustand state
 * - Cyberpunk/quant aesthetic theme
 */

import React, { useEffect, useRef, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { useChartStore } from '../../store/chartStore';
import { useChartSync } from '../../hooks/useChartSync';

interface CandlestickChartProps {
  symbol: string;
  interval: string;
  initialData?: CandlestickData[];
  height?: number;
  width?: string;
}

// Cyberpunk color palette for the quant aesthetic
const CYBERPUNK_THEME = {
  background: '#0a0e17',
  gridColor: 'rgba(0, 255, 255, 0.1)',
  textColor: '#00ffff',
  candleUp: '#00ff88',
  candleDown: '#ff0055',
  wickUp: '#00ff88',
  wickDown: '#ff0055',
  crosshairColor: '#00ffff',
  borderVisible: false,
};

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  symbol,
  interval,
  initialData = [],
  height = 400,
  width = '100%',
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  
  // Subscribe to chart store for cross-chart synchronization
  const { subscribeToUpdates, unsubscribeFromUpdates } = useChartSync();
  const setChartInstance = useChartStore((state) => state.setChartInstance);
  const chartId = `${symbol}-${interval}`;

  // Initialize chart with canvas-based renderer
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { type: 'solid', color: CYBERPUNK_THEME.background },
        textColor: CYBERPUNK_THEME.textColor,
      },
      grid: {
        vertLines: { color: CYBERPUNK_THEME.gridColor },
        horzLines: { color: CYBERPUNK_THEME.gridColor },
      },
      crosshair: {
        mode: 1, // Magnet mode for precise snapping
        vertLine: {
          color: CYBERPUNK_THEME.crosshairColor,
          labelBackgroundColor: CYBERPUNK_THEME.background,
        },
        horzLine: {
          color: CYBERPUNK_THEME.crosshairColor,
          labelBackgroundColor: CYBERPUNK_THEME.background,
        },
      },
      timeScale: {
        borderColor: CYBERPUNK_THEME.gridColor,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: CYBERPUNK_THEME.gridColor,
        autoScale: true,
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
    });

    // Create candlestick series with cyberpunk colors
    const candleSeries = chart.addCandlestickSeries({
      upColor: CYBERPUNK_THEME.candleUp,
      downColor: CYBERPUNK_THEME.candleDown,
      borderUpColor: CYBERPUNK_THEME.wickUp,
      borderDownColor: CYBERPUNK_THEME.wickDown,
      wickUpColor: CYBERPUNK_THEME.wickUp,
      wickDownColor: CYBERPUNK_THEME.wickDown,
    });

    // Handle Binance tick size precision - round prices to correct decimal places
    const precisionMap: Record<string, number> = {
      'BTCUSDT': 2,
      'ETHUSDT': 2,
      'SOLUSDT': 3,
      'BNBUSDT': 2,
    };
    const precision = precisionMap[symbol] || 4;

    // Format data with proper precision
    const formattedData = initialData.map(d => ({
      time: d.time as Time,
      open: Number(d.open.toFixed(precision)),
      high: Number(d.high.toFixed(precision)),
      low: Number(d.low.toFixed(precision)),
      close: Number(d.close.toFixed(precision)),
    }));

    candleSeries.setData(formattedData);

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    // Register chart instance for crosshair synchronization
    setChartInstance(chartId, chart, candleSeries);

    // Handle resize events
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    // Subscribe to real-time updates via WebSocket
    const unsubscribe = subscribeToUpdates(chartId, (update: CandlestickData) => {
      if (candleSeries) {
        candleSeries.update(update);
      }
    });

    return () => {
      window.removeEventListener('resize', handleResize);
      unsubscribe();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [symbol, interval, height, chartId, setChartInstance, subscribeToUpdates, initialData]);

  // Sync crosshair movements across multiple charts
  useChartSync();

  return (
    <div 
      ref={chartContainerRef} 
      style={{ 
        width, 
        height, 
        position: 'relative',
        overflow: 'hidden',
      }}
      className="cyberpunk-chart-container"
    >
      {/* Overlay for SMC annotations will be rendered here by SMCOverlay component */}
      <div className="absolute top-2 left-2 z-10 pointer-events-none">
        <span className="text-cyan-400 text-xs font-mono tracking-wider">
          {symbol} | {interval} | CANVAS_RENDERED
        </span>
      </div>
    </div>
  );
};

export default CandlestickChart;
