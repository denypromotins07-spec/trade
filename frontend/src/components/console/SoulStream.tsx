/**
 * File 9: SoulStream.tsx
 * Chapter 3: Live Console & SOUL.md Terminal
 * 
 * Dedicated, beautifully animated scrolling feed for `SOUL.md` mutations,
 * displaying the bot's autonomous learnings and strategy pivots in real-time.
 * 
 * Features:
 * - Framer Motion-style CSS animations (without external dependency)
 * - Glassmorphism backdrop filters
 * - Entry categorization with color coding
 * - Auto-scrolling with pause on hover
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';

// --- Types ---

interface SoulEntry {
  id: string;
  timestamp: number;
  type: 'learning' | 'pivot' | 'insight' | 'warning' | 'milestone';
  category: string;
  title: string;
  content: string;
  confidence?: number; // 0-1
}

interface Props {
  entries: SoulEntry[];
  maxVisible?: number;
}

// --- Constants ---

const TYPE_CONFIG = {
  learning: { color: '#00f3ff', label: 'LEARNING', icon: '🧠' },
  pivot: { color: '#ff0055', label: 'STRATEGY_PIVOT', icon: '🔄' },
  insight: { color: '#00ff9d', label: 'INSIGHT', icon: '💡' },
  warning: { color: '#ffaa00', label: 'CAUTION', icon: '⚠️' },
  milestone: { color: '#bd00ff', label: 'MILESTONE', icon: '🎯' },
};

const COLORS = {
  bg: 'rgba(10, 10, 10, 0.6)',
  border: 'rgba(0, 243, 255, 0.2)',
  text: '#e0e0e0',
  textDim: '#808080',
};

/**
 * Format timestamp to readable format
 */
const formatTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
};

/**
 * SoulStream Component
 * Animated feed of autonomous AI learnings and strategy evolution.
 */
export const SoulStream: React.FC<Props> = ({
  entries,
  maxVisible = 10,
}) => {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleCount, setVisibleCount] = useState(maxVisible);

  // Auto-scroll animation
  useEffect(() => {
    if (!isPaused && entries.length > 0) {
      setVisibleCount((prev) => Math.min(prev + 1, maxVisible));
    }
  }, [entries, isPaused, maxVisible]);

  // Pause on container hover
  const handleMouseEnter = useCallback(() => setIsPaused(true), []);
  const handleMouseLeave = useCallback(() => setIsPaused(false), []);

  // Get latest entries
  const visibleEntries = entries.slice(-visibleCount).reverse();

  return (
    <div className="p-4 h-full flex flex-col bg-black/40 backdrop-blur-xl border border-cyan-900/30 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center mb-4 pb-3 border-b border-cyan-900/30">
        <div>
          <h3 className="text-sm font-mono font-bold text-white tracking-wider flex items-center gap-2">
            <span className="text-lg">✨</span>
            SOUL_MD_STREAM
          </h3>
          <div className="text-[10px] text-gray-500 font-mono mt-1">
            AUTONOMOUS_LEARNING_FEED
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-[10px] font-mono text-cyan-400">
            {entries.length} MUTATIONS
          </div>
          {isPaused && (
            <div className="px-2 py-0.5 bg-amber-900/50 border border-amber-700 rounded text-[9px] font-mono text-amber-400 animate-pulse">
              PAUSED
            </div>
          )}
        </div>
      </div>

      {/* Entries Feed */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-cyan-900 scrollbar-track-transparent"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {visibleEntries.map((entry, index) => {
          const config = TYPE_CONFIG[entry.type];
          const delay = index * 100; // Stagger animation

          return (
            <div
              key={entry.id}
              className={`relative p-3 rounded-lg border transition-all duration-300 cursor-pointer ${
                hoveredId === entry.id ? 'scale-[1.02]' : 'scale-100'
              }`}
              style={{
                backgroundColor: hoveredId === entry.id 
                  ? 'rgba(20, 20, 30, 0.8)' 
                  : COLORS.bg,
                borderColor: hoveredId === entry.id 
                  ? config.color 
                  : COLORS.border,
                boxShadow: hoveredId === entry.id 
                  ? `0 0 20px ${config.color}30` 
                  : `0 0 10px ${config.color}10`,
                backdropFilter: 'blur(12px)',
                animation: `slideIn 0.5s ease-out ${delay}ms both`,
              }}
              onMouseEnter={() => setHoveredId(entry.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              {/* Type Badge */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{config.icon}</span>
                  <span
                    className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded"
                    style={{
                      backgroundColor: `${config.color}20`,
                      color: config.color,
                    }}
                  >
                    {config.label}
                  </span>
                </div>
                <span className="text-[9px] font-mono text-gray-500">
                  {formatTime(entry.timestamp)}
                </span>
              </div>

              {/* Title */}
              <div className="text-xs font-mono font-bold text-white mb-1">
                {entry.title}
              </div>

              {/* Content */}
              <div className="text-[11px] font-mono text-gray-300 leading-relaxed">
                {entry.content}
              </div>

              {/* Category & Confidence */}
              <div className="flex items-center justify-between mt-3 pt-2 border-t border-gray-800">
                <span className="text-[9px] font-mono text-cyan-600">
                  #{entry.category}
                </span>
                {entry.confidence !== undefined && (
                  <div className="flex items-center gap-1">
                    <span className="text-[9px] font-mono text-gray-500">CONF:</span>
                    <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${entry.confidence * 100}%`,
                          backgroundColor: config.color,
                          boxShadow: `0 0 4px ${config.color}`,
                        }}
                      />
                    </div>
                    <span className="text-[9px] font-mono" style={{ color: config.color }}>
                      {(entry.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Decorative corner accent */}
              <div
                className="absolute top-0 right-0 w-3 h-3 rounded-bl-lg opacity-50"
                style={{
                  background: `linear-gradient(135deg, transparent 50%, ${config.color} 50%)`,
                }}
              />
            </div>
          );
        })}

        {visibleEntries.length === 0 && (
          <div className="flex flex-col items-center justify-center h-32 text-gray-500">
            <div className="text-2xl mb-2">🌱</div>
            <div className="text-[10px] font-mono">AWAITING FIRST MUTATION...</div>
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div className="mt-3 pt-3 border-t border-cyan-900/30 grid grid-cols-5 gap-2">
        {Object.entries(TYPE_CONFIG).map(([type, config]) => {
          const count = entries.filter((e) => e.type === type).length;
          return (
            <div
              key={type}
              className="text-center p-1.5 rounded bg-gray-900/50 border border-gray-800"
              style={{ borderColor: `${config.color}30` }}
            >
              <div className="text-[10px]" style={{ color: config.color }}>
                {config.icon}
              </div>
              <div className="text-[9px] font-mono text-gray-400">
                {count}
              </div>
            </div>
          );
        })}
      </div>

      {/* Inline styles for animations */}
      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
};

export default SoulStream;
