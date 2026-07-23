/**
 * WalkForwardMap.tsx - Isometric CSS grid for out-of-sample performance windows
 * 
 * Features:
 * - Color-codes rolling Sharpe ratios to identify overfitting and regime decay
 * - Isometric 3D visualization of walk-forward analysis results
 * - Interactive cell selection for detailed statistics
 * - Cyberpunk aesthetic with neon indicators
 */

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, AlertTriangle, Activity, Calendar } from 'lucide-react';

interface WalkForwardCell {
  trainStart: string;
  trainEnd: string;
  testStart: string;
  testEnd: string;
  sharpeRatio: number;
  returnPct: number;
  maxDrawdown: number;
  winRate: number;
  isOverfitted: boolean;
  regimeType: 'trending' | 'mean-reverting' | 'volatile' | 'crisis';
}

interface WalkForwardMapProps {
  cells: WalkForwardCell[];
  onSelectWindow: (cell: WalkForwardCell) => void;
}

const REGIME_COLORS = {
  trending: '#10b981',      // Emerald
  'mean-reverting': '#22d3ee', // Cyan
  volatile: '#f59e0b',      // Amber
  crisis: '#ef4444',        // Red
};

export const WalkForwardMap: React.FC<WalkForwardMapProps> = ({
  cells,
  onSelectWindow,
}) => {
  const [selectedCell, setSelectedCell] = useState<WalkForwardCell | null>(null);
  const [hoveredCell, setHoveredCell] = useState<WalkForwardCell | null>(null);

  // Calculate aggregate statistics
  const stats = useMemo(() => {
    if (cells.length === 0) return null;

    const avgSharpe = cells.reduce((sum, c) => sum + c.sharpeRatio, 0) / cells.length;
    const overfitCount = cells.filter(c => c.isOverfitted).length;
    const avgReturn = cells.reduce((sum, c) => sum + c.returnPct, 0) / cells.length;
    const avgDrawdown = cells.reduce((sum, c) => sum + c.maxDrawdown, 0) / cells.length;

    return { avgSharpe, overfitCount, avgReturn, avgDrawdown };
  }, [cells]);

  // Get Sharpe color
  const getSharpeColor = (sharpe: number): string => {
    if (sharpe >= 2) return '#10b981';
    if (sharpe >= 1) return '#22d3ee';
    if (sharpe >= 0.5) return '#f59e0b';
    return '#ef4444';
  };

  // Get cell height based on Sharpe (for 3D effect)
  const getCellHeight = (sharpe: number): number => {
    const normalized = Math.max(0, Math.min(sharpe / 3, 1));
    return 10 + normalized * 30; // 10-40px
  };

  return (
    <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          WALK-FORWARD ANALYSIS MAP
        </h3>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-slate-400">WINDOWS: {cells.length}</span>
          {stats && (
            <>
              <span className="text-cyan-400">AVG SHARPE: {stats.avgSharpe.toFixed(2)}</span>
              <span className={stats.overfitCount > 0 ? 'text-amber-400' : 'text-emerald-400'}>
                OVERFIT: {stats.overfitCount}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Isometric Grid */}
      <div className="relative mb-6 overflow-x-auto">
        <div className="min-w-[600px] py-8">
          {/* Grid Container with isometric transform */}
          <div 
            className="grid gap-2 px-8"
            style={{
              gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(cells.length))}, 1fr)`,
              transform: 'perspective(1000px) rotateX(20deg)',
              transformStyle: 'preserve-3d',
            }}
          >
            {cells.map((cell, index) => {
              const height = getCellHeight(cell.sharpeRatio);
              const color = getSharpeColor(cell.sharpeRatio);
              const isSelected = selectedCell === cell;
              const isHovered = hoveredCell === cell;

              return (
                <motion.div
                  key={index}
                  layout
                  onClick={() => {
                    setSelectedCell(cell);
                    onSelectWindow(cell);
                  }}
                  onMouseEnter={() => setHoveredCell(cell)}
                  onMouseLeave={() => setHoveredCell(null)}
                  className="relative cursor-pointer group"
                  style={{
                    transformStyle: 'preserve-3d',
                  }}
                  whileHover={{ scale: 1.05, zIndex: 10 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {/* 3D Block */}
                  <div
                    className="relative rounded-lg transition-all"
                    style={{
                      height: `${height}px`,
                      backgroundColor: isSelected ? '#fff' : color,
                      opacity: isSelected ? 1 : 0.8,
                      boxShadow: isSelected
                        ? `0 0 30px ${color}, 0 -10px 20px ${color}66`
                        : isHovered
                        ? `0 -5px 15px ${color}66`
                        : `0 -2px 10px ${color}33`,
                      transform: isSelected ? 'translateZ(20px)' : 'translateZ(0)',
                    }}
                  >
                    {/* Top face label */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white font-bold">
                      <span className="text-xs">{cell.sharpeRatio.toFixed(2)}</span>
                      {cell.isOverfitted && (
                        <AlertTriangle className="w-3 h-3 mt-1" />
                      )}
                    </div>

                    {/* Side faces for 3D effect */}
                    <div
                      className="absolute left-0 right-0 bottom-0 rounded-b-lg"
                      style={{
                        height: '10px',
                        backgroundColor: color,
                        filter: 'brightness(0.7)',
                        transform: 'rotateX(-90deg) translateZ(5px)',
                        transformOrigin: 'bottom',
                      }}
                    />
                  </div>

                  {/* Tooltip on hover */}
                  <AnimatePresence>
                    {isHovered && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        className="absolute -top-20 left-1/2 -translate-x-1/2 z-20 p-3 bg-slate-800 border border-cyan-500/50 rounded-lg shadow-xl whitespace-nowrap"
                      >
                        <div className="text-xs font-mono">
                          <div className="text-slate-400 mb-1">{cell.regimeType.toUpperCase()}</div>
                          <div className="text-white">Sharpe: {cell.sharpeRatio.toFixed(2)}</div>
                          <div className="text-emerald-400">Return: {cell.returnPct.toFixed(1)}%</div>
                          <div className="text-red-400">DD: {cell.maxDrawdown.toFixed(1)}%</div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Selected Cell Details */}
      <AnimatePresence>
        {selectedCell && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="p-4 bg-slate-800/50 rounded-xl border border-cyan-500/30"
          >
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-bold text-cyan-400 flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                WINDOW DETAILS
              </h4>
              <button
                onClick={() => setSelectedCell(null)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕ CLOSE
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Train Period */}
              <div>
                <div className="text-xs text-slate-500 mb-1">TRAIN PERIOD</div>
                <div className="text-sm font-mono text-slate-300">
                  {selectedCell.trainStart} → {selectedCell.trainEnd}
                </div>
              </div>

              {/* Test Period */}
              <div>
                <div className="text-xs text-slate-500 mb-1">TEST PERIOD</div>
                <div className="text-sm font-mono text-slate-300">
                  {selectedCell.testStart} → {selectedCell.testEnd}
                </div>
              </div>

              {/* Regime Type */}
              <div>
                <div className="text-xs text-slate-500 mb-1">REGIME</div>
                <div
                  className="text-sm font-bold px-2 py-1 rounded inline-block"
                  style={{
                    backgroundColor: `${REGIME_COLORS[selectedCell.regimeType]}20`,
                    color: REGIME_COLORS[selectedCell.regimeType],
                  }}
                >
                  {selectedCell.regimeType.toUpperCase()}
                </div>
              </div>

              {/* Overfit Warning */}
              <div>
                <div className="text-xs text-slate-500 mb-1">STATUS</div>
                <div className={`text-sm font-bold flex items-center gap-1 ${
                  selectedCell.isOverfitted ? 'text-amber-400' : 'text-emerald-400'
                }`}>
                  {selectedCell.isOverfitted && <AlertTriangle className="w-3 h-3" />}
                  {selectedCell.isOverfitted ? 'OVERFITTED' : 'VALID'}
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-700">
              <div className="text-center">
                <div className="text-xs text-slate-500 mb-1">SHARPE RATIO</div>
                <div
                  className="text-xl font-mono font-bold"
                  style={{ color: getSharpeColor(selectedCell.sharpeRatio) }}
                >
                  {selectedCell.sharpeRatio.toFixed(2)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-slate-500 mb-1">RETURN</div>
                <div className={`text-xl font-mono font-bold ${
                  selectedCell.returnPct >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {selectedCell.returnPct >= 0 ? '+' : ''}{selectedCell.returnPct.toFixed(1)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-slate-500 mb-1">MAX DD</div>
                <div className="text-xl font-mono font-bold text-red-400">
                  {selectedCell.maxDrawdown.toFixed(1)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-slate-500 mb-1">WIN RATE</div>
                <div className="text-xl font-mono font-bold text-purple-400">
                  {selectedCell.winRate.toFixed(0)}%
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-slate-700 flex items-center justify-between text-xs">
        <div className="flex items-center gap-4">
          <span className="text-slate-400">SHARPE:</span>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-emerald-500" />
            <span className="text-slate-500">≥2.0</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-cyan-500" />
            <span className="text-slate-500">1.0-2.0</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-amber-500" />
            <span className="text-slate-500">0.5-1.0</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-red-500" />
            <span className="text-slate-500">&lt;0.5</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <Activity className="w-3 h-3" />
          <span>OUT-OF-SAMPLE VALIDATION</span>
        </div>
      </div>
    </div>
  );
};

export default WalkForwardMap;
