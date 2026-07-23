/**
 * File 7: LiveTerminal.tsx
 * Chapter 3: Live Console & SOUL.md Terminal
 * 
 * Custom Canvas-based hacker terminal streaming Rust and Python logs at 60FPS,
 * strictly recycling text buffers to prevent browser OOM.
 * 
 * Features:
 * - Canvas-based rendering for high-performance text output
 * - ANSI escape code sanitization (security)
 * - Circular buffer for memory management
 * - Auto-scroll with pause-on-interaction
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';

// --- Types ---

interface LogEntry {
  id: number;
  timestamp: number;
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR' | 'PANIC';
  source: 'rust' | 'python' | 'system';
  message: string;
  rawAnsi?: string;
}

interface Props {
  entries: LogEntry[];
  maxLines?: number;
  width?: number;
  height?: number;
  fontSize?: number;
  autoScroll?: boolean;
}

// --- Constants ---

const COLORS = {
  bg: '#050505',
  text: '#c0c0c0',
  debug: '#666666',
  info: '#00f3ff',
  warn: '#ffaa00',
  error: '#ff0055',
  panic: '#ff0000',
  rust: '#dea584',
  python: '#ffd343',
  system: '#00ff9d',
  cursor: '#00f3ff',
};

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: COLORS.debug,
  INFO: COLORS.info,
  WARN: COLORS.warn,
  ERROR: COLORS.error,
  PANIC: COLORS.panic,
};

const SOURCE_COLORS: Record<string, string> = {
  rust: COLORS.rust,
  python: COLORS.python,
  system: COLORS.system,
};

/**
 * Sanitize ANSI escape codes from input strings
 * Prevents terminal injection vulnerabilities
 */
const sanitizeAnsi = (input: string): string => {
  // Remove all ANSI escape sequences except for basic colors
  return input.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
};

/**
 * Parse ANSI color codes to our color map
 */
const parseAnsiColors = (text: string): { text: string; color?: string }[] => {
  const result: { text: string; color?: string }[] = [];
  let currentColor = COLORS.text;
  let currentIndex = 0;
  
  const ansiRegex = /\x1b\[([0-9;]+)m/g;
  let match: RegExpExecArray | null;
  
  while ((match = ansiRegex.exec(text)) !== null) {
    // Add text before this code
    if (match.index > currentIndex) {
      result.push({
        text: text.slice(currentIndex, match.index),
        color: currentColor,
      });
    }
    
    // Parse the color code
    const codes = match[1].split(';').map(Number);
    for (const code of codes) {
      if (code >= 30 && code <= 37) {
        // Basic foreground colors
        const colorMap = [COLORS.text, COLORS.error, COLORS.info, COLORS.warn, COLORS.rust, COLORS.python, COLORS.system, COLORS.text];
        currentColor = colorMap[code - 30] || COLORS.text;
      } else if (code === 0) {
        // Reset
        currentColor = COLORS.text;
      } else if (code === 1) {
        // Bold - we could make it brighter
        currentColor = currentColor;
      }
    }
    
    currentIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (currentIndex < text.length) {
    result.push({
      text: text.slice(currentIndex),
      color: currentColor,
    });
  }
  
  return result;
};

/**
 * LiveTerminal Component
 * High-performance Canvas terminal with circular buffer.
 */
export const LiveTerminal: React.FC<Props> = ({
  entries,
  maxLines = 500,
  width = 800,
  height = 500,
  fontSize = 12,
  autoScroll = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [scrollOffset, setScrollOffset] = useState(0);
  const requestRef = useRef<number>();
  
  // Circular buffer state
  const displayEntriesRef = useRef<LogEntry[]>([]);
  const lineHeight = fontSize + 4;
  const visibleLines = Math.floor(height / lineHeight);

  // Update circular buffer
  useEffect(() => {
    const newEntries = [...entries];
    
    // Keep only last maxLines entries
    if (newEntries.length > maxLines) {
      newEntries.splice(0, newEntries.length - maxLines);
    }
    
    displayEntriesRef.current = newEntries;
    
    // Auto-scroll if not paused
    if (autoScroll && !isPaused) {
      setScrollOffset(Math.max(0, newEntries.length - visibleLines));
    }
  }, [entries, maxLines, autoScroll, isPaused, visibleLines]);

  // Render function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    const entries = displayEntriesRef.current;
    const startIdx = Math.max(0, scrollOffset);
    const endIdx = Math.min(entries.length, startIdx + visibleLines);

    // Draw visible entries
    ctx.font = `${fontSize}px "JetBrains Mono", "Fira Code", monospace`;
    ctx.textBaseline = 'top';

    for (let i = startIdx; i < endIdx; i++) {
      const entry = entries[i];
      const y = (i - startIdx) * lineHeight;
      const x = 10;

      // Background highlight for errors/panics
      if (entry.level === 'ERROR' || entry.level === 'PANIC') {
        ctx.fillStyle = entry.level === 'PANIC' ? 'rgba(255, 0, 0, 0.1)' : 'rgba(255, 170, 0, 0.05)';
        ctx.fillRect(0, y, width, lineHeight);
      }

      // Timestamp
      const time = new Date(entry.timestamp).toISOString().substr(11, 12);
      ctx.fillStyle = '#555555';
      ctx.fillText(time, x, y);

      // Level badge
      const levelX = x + 130;
      ctx.fillStyle = LEVEL_COLORS[entry.level] || COLORS.text;
      ctx.font = `bold ${fontSize}px "JetBrains Mono", monospace`;
      ctx.fillText(`[${entry.level}]`, levelX, y);

      // Source
      const sourceX = levelX + 80;
      ctx.fillStyle = SOURCE_COLORS[entry.source] || COLORS.text;
      ctx.font = `${fontSize}px "JetBrains Mono", monospace`;
      ctx.fillText(`(${entry.source})`, sourceX, y);

      // Message (with ANSI parsing if available)
      const msgX = sourceX + 100;
      const maxWidth = width - msgX - 20;
      
      if (entry.rawAnsi) {
        const parsed = parseAnsiColors(entry.rawAnsi);
        let currentX = msgX;
        parsed.forEach(({ text, color }) => {
          ctx.fillStyle = color || COLORS.text;
          ctx.fillText(text.substring(0, 100), currentX, y); // Truncate long lines
          currentX += ctx.measureText(text.substring(0, 100)).width;
        });
      } else {
        ctx.fillStyle = COLORS.text;
        const sanitized = sanitizeAnsi(entry.message);
        ctx.fillText(sanitized.substring(0, 150), msgX, y);
      }
    }

    // Draw cursor blink
    if (!isPaused && entries.length > 0) {
      const cursorY = (Math.min(entries.length, startIdx + visibleLines) - startIdx - 1) * lineHeight + fontSize - 2;
      const cursorBlink = Math.floor(Date.now() / 500) % 2 === 0;
      
      if (cursorBlink) {
        ctx.fillStyle = COLORS.cursor;
        ctx.fillRect(width - 10, cursorY, 8, 2);
      }
    }

    // Draw pause indicator
    if (isPaused) {
      ctx.fillStyle = 'rgba(255, 170, 0, 0.8)';
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('⏸ SCROLL PAUSED - Click to resume', width / 2, 20);
      ctx.textAlign = 'left';
    }

    // Draw line count
    ctx.fillStyle = '#444444';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${entries.length}/${maxLines} LINES`, width - 10, height - 5);
    ctx.textAlign = 'left';
  }, [width, height, fontSize, visibleLines, scrollOffset, isPaused, maxLines]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      render();
      requestRef.current = requestAnimationFrame(animate);
    };
    animate();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [render]);

  // Handle wheel for manual scrolling
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!containerRef.current) return;
    
    const delta = e.deltaY > 0 ? 3 : -3;
    const maxScroll = Math.max(0, displayEntriesRef.current.length - visibleLines);
    
    setScrollOffset((prev) => {
      const newOffset = Math.max(0, Math.min(maxScroll, prev + delta));
      // If user scrolls up, pause auto-scroll
      if (newOffset < maxScroll) {
        setIsPaused(true);
      }
      return newOffset;
    });
  }, [visibleLines]);

  // Resume on click
  const handleClick = useCallback(() => {
    setIsPaused(false);
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden rounded-lg border border-cyan-900/50 bg-black"
      style={{ width, height }}
      onWheel={handleWheel}
      onClick={handleClick}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="block"
        style={{ cursor: isPaused ? 'pointer' : 'default' }}
      />
      
      {/* Overlay controls */}
      <div className="absolute top-2 right-2 flex gap-2">
        <button
          onClick={(e) => {
            e.stopPropagation();
            setScrollOffset(Math.max(0, displayEntriesRef.current.length - visibleLines));
            setIsPaused(false);
          }}
          className="px-2 py-1 text-[10px] font-mono bg-cyan-900/50 text-cyan-400 rounded hover:bg-cyan-800 transition-colors"
        >
          SCROLL_END
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setIsPaused(!isPaused);
          }}
          className="px-2 py-1 text-[10px] font-mono bg-gray-800/50 text-gray-400 rounded hover:bg-gray-700 transition-colors"
        >
          {isPaused ? 'RESUME' : 'PAUSE'}
        </button>
      </div>

      {/* Title */}
      <div className="absolute top-2 left-2 text-xs font-mono text-cyan-400 font-bold tracking-wider pointer-events-none">
        LIVE_TERMINAL_V4
      </div>
    </div>
  );
};

export default LiveTerminal;
