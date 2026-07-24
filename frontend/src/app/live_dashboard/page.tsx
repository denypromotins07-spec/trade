'use client';

/**
 * =============================================================================
 * page.tsx - Ultimate 60FPS Master Dashboard
 * Nautilus/Ray Trading Bot - Stage 60
 * =============================================================================
 * Purpose: Aggregates all 6 assets, SOUL.md live feed, and global risk metrics
 *          into one unified, cyberpunk-styled command center.
 * Constraints: Optimized for 60FPS, Canvas-based rendering, AMD GPU acceleration.
 * Compatibility: Works with kiosk_bind.ts lockdown and chrome_kiosk.ps1 flags.
 * =============================================================================
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { initKioskBind, KioskWebSocket } from '@/lib/kiosk_bind';

// -----------------------------------------------------------------------------
// Types & Interfaces
// -----------------------------------------------------------------------------

interface AssetTicker {
  symbol: string;
  price: number;
  change24h: number;
  volume: number;
  high24h: number;
  low24h: number;
}

interface SoulLedgerEntry {
  strategyName: string;
  version: number;
  status: 'ACTIVE' | 'PENDING' | 'DISABLED';
  pnlToday: number;
  trades: number;
}

interface GlobalRiskMetrics {
  totalEquity: number;
  marginRatio: number;
  unrealizedPnl: number;
  openPositions: number;
  circuitBreakerActive: boolean;
}

interface DashboardState {
  tickers: Record<string, AssetTicker>;
  soulEntries: SoulLedgerEntry[];
  risk: GlobalRiskMetrics;
  lastUpdate: number;
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT'];
const CYBERPUNK_COLORS = {
  bg: '#0a0a0f',
  panel: '#12121a',
  accent: '#00ff9d',
  danger: '#ff0055',
  warning: '#ffcc00',
  text: '#e0e0e0',
  grid: '#1a1a2e',
};

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------

export default function LiveDashboard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [wsClient, setWsClient] = useState<KioskWebSocket | null>(null);
  const [state, setState] = useState<DashboardState>({
    tickers: {},
    soulEntries: [],
    risk: {
      totalEquity: 0,
      marginRatio: 0,
      unrealizedPnl: 0,
      openPositions: 0,
      circuitBreakerActive: false,
    },
    lastUpdate: Date.now(),
  });
  
  const animationFrameRef = useRef<number>();

  // Initialize WebSocket connection
  useEffect(() => {
    const client = initKioskBind();
    setWsClient(client);

    client.onMessage((data) => {
      handleWebSocketMessage(data);
    });

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Handle incoming WS messages
  const handleWebSocketMessage = useCallback((data: any) => {
    setState((prev) => {
      const newState = { ...prev };

      if (data.type === 'TICKER_UPDATE') {
        const ticker = data.payload as AssetTicker;
        newState.tickers[ticker.symbol] = ticker;
      } else if (data.type === 'SOUL_LEDGER_UPDATE') {
        newState.soulEntries = data.payload as SoulLedgerEntry[];
      } else if (data.type === 'RISK_METRICS') {
        newState.risk = data.payload as GlobalRiskMetrics;
      }

      newState.lastUpdate = Date.now();
      return newState;
    });
  }, []);

  // Canvas rendering loop for 60FPS
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const render = () => {
      // Clear canvas
      ctx.fillStyle = CYBERPUNK_COLORS.bg;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw grid
      drawGrid(ctx, canvas.width, canvas.height);

      // Draw asset panels
      drawAssetPanels(ctx, state.tickers, canvas.width);

      // Draw SOUL.md feed
      drawSoulFeed(ctx, state.soulEntries, canvas.width, canvas.height);

      // Draw risk metrics
      drawRiskMetrics(ctx, state.risk, canvas.width, canvas.height);

      // Draw status bar
      drawStatusBar(ctx, state.lastUpdate, canvas.width, canvas.height);

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [state]);

  return (
    <div className="fixed inset-0 overflow-hidden bg-black">
      <canvas ref={canvasRef} className="block w-full h-full" />
      {/* Hidden overlay for accessibility / screen readers */}
      <div className="sr-only">
        <h1>Nautilus/Ray Live Trading Dashboard</h1>
        <p>Last update: {new Date(state.lastUpdate).toLocaleTimeString()}</p>
        <p>Total Equity: ${state.risk.totalEquity.toFixed(2)}</p>
        {state.risk.circuitBreakerActive && (
          <p className="text-red-500 font-bold">CIRCUIT BREAKER ACTIVE</p>
        )}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Drawing Functions (Canvas API for maximum performance)
// -----------------------------------------------------------------------------

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
) {
  ctx.strokeStyle = CYBERPUNK_COLORS.grid;
  ctx.lineWidth = 1;

  const gridSize = 50;
  for (let x = 0; x < width; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawAssetPanels(
  ctx: CanvasRenderingContext2D,
  tickers: Record<string, AssetTicker>,
  width: number
) {
  const panelHeight = 80;
  const panelWidth = width / 6 - 10;
  const padding = 5;

  ASSETS.forEach((symbol, index) => {
    const x = index * (panelWidth + padding) + padding;
    const y = padding;

    // Panel background
    ctx.fillStyle = CYBERPUNK_COLORS.panel;
    ctx.fillRect(x, y, panelWidth, panelHeight);

    // Border
    ctx.strokeStyle = CYBERPUNK_COLORS.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, panelWidth, panelHeight);

    // Text
    ctx.fillStyle = CYBERPUNK_COLORS.text;
    ctx.font = 'bold 14px monospace';
    ctx.fillText(symbol, x + 10, y + 25);

    const ticker = tickers[symbol];
    if (ticker) {
      ctx.font = '18px monospace';
      ctx.fillStyle = ticker.change24h >= 0 ? CYBERPUNK_COLORS.accent : CYBERPUNK_COLORS.danger;
      ctx.fillText(`$${ticker.price.toFixed(2)}`, x + 10, y + 50);

      ctx.font = '12px monospace';
      ctx.fillStyle = CYBERPUNK_COLORS.text;
      ctx.fillText(`${ticker.change24h.toFixed(2)}%`, x + 10, y + 70);
    } else {
      ctx.fillStyle = '#666';
      ctx.font = '12px monospace';
      ctx.fillText('WAITING...', x + 10, y + 50);
    }
  });
}

function drawSoulFeed(
  ctx: CanvasRenderingContext2D,
  entries: SoulLedgerEntry[],
  width: number,
  height: number
) {
  const startY = 120;
  const panelHeight = 200;
  const panelWidth = width / 2 - 10;
  const padding = 5;

  // Left panel: SOUL.md strategies
  ctx.fillStyle = CYBERPUNK_COLORS.panel;
  ctx.fillRect(padding, startY, panelWidth, panelHeight);
  ctx.strokeStyle = CYBERPUNK_COLORS.accent;
  ctx.lineWidth = 2;
  ctx.strokeRect(padding, startY, panelWidth, panelHeight);

  ctx.fillStyle = CYBERPUNK_COLORS.text;
  ctx.font = 'bold 16px monospace';
  ctx.fillText('SOUL.md STRATEGIES', padding + 10, startY + 25);

  entries.forEach((entry, index) => {
    const y = startY + 50 + index * 35;
    ctx.font = '12px monospace';
    ctx.fillStyle = entry.status === 'ACTIVE' ? CYBERPUNK_COLORS.accent : '#888';
    ctx.fillText(
      `${entry.strategyName} v${entry.version} | PnL: $${entry.pnlToday.toFixed(2)} | Trades: ${entry.trades}`,
      padding + 10,
      y
    );
  });
}

function drawRiskMetrics(
  ctx: CanvasRenderingContext2D,
  risk: GlobalRiskMetrics,
  width: number,
  height: number
) {
  const startY = 120;
  const panelHeight = 200;
  const panelWidth = width / 2 - 10;
  const padding = 5;
  const xOffset = width / 2 + padding;

  // Right panel: Risk Metrics
  ctx.fillStyle = CYBERPUNK_COLORS.panel;
  ctx.fillRect(xOffset, startY, panelWidth, panelHeight);
  ctx.strokeStyle = risk.circuitBreakerActive ? CYBERPUNK_COLORS.danger : CYBERPUNK_COLORS.accent;
  ctx.lineWidth = 2;
  ctx.strokeRect(xOffset, startY, panelWidth, panelHeight);

  ctx.fillStyle = CYBERPUNK_COLORS.text;
  ctx.font = 'bold 16px monospace';
  ctx.fillText('GLOBAL RISK METRICS', xOffset + 10, startY + 25);

  const metrics = [
    `Equity: $${risk.totalEquity.toFixed(2)}`,
    `Margin: ${(risk.marginRatio * 100).toFixed(1)}%`,
    `Unrealized PnL: $${risk.unrealizedPnl.toFixed(2)}`,
    `Open Positions: ${risk.openPositions}`,
  ];

  metrics.forEach((metric, index) => {
    ctx.font = '14px monospace';
    ctx.fillStyle = CYBERPUNK_COLORS.text;
    ctx.fillText(metric, xOffset + 10, startY + 55 + index * 30);
  });

  if (risk.circuitBreakerActive) {
    ctx.fillStyle = CYBERPUNK_COLORS.danger;
    ctx.font = 'bold 20px monospace';
    ctx.fillText('⚠ CIRCUIT BREAKER ACTIVE ⚠', xOffset + 10, startY + 170);
  }
}

function drawStatusBar(
  ctx: CanvasRenderingContext2D,
  lastUpdate: number,
  width: number,
  height: number
) {
  const barHeight = 30;
  const y = height - barHeight;

  ctx.fillStyle = CYBERPUNK_COLORS.panel;
  ctx.fillRect(0, y, width, barHeight);

  ctx.fillStyle = CYBERPUNK_COLORS.text;
  ctx.font = '12px monospace';
  ctx.fillText(`LAST UPDATE: ${new Date(lastUpdate).toLocaleTimeString()}`, 10, y + 20);
  ctx.fillText('SYSTEM STATUS: ONLINE', width - 150, y + 20);
}
