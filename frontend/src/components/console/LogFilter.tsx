/**
 * File 8: LogFilter.tsx
 * Chapter 3: Live Console & SOUL.md Terminal
 * 
 * Regex-based log filtering panel with GPU-accelerated syntax highlighting
 * for panic traces, IPC payloads, and execution reports.
 * 
 * Features:
 * - Real-time regex filtering
 * - GPU-accelerated syntax highlighting via CSS transforms
 * - Preset filter groups
 * - Match count statistics
 */

import React, { useState, useMemo, useCallback } from 'react';

// --- Types ---

interface LogEntry {
  id: number;
  timestamp: number;
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR' | 'PANIC';
  source: 'rust' | 'python' | 'system';
  message: string;
}

interface FilterPreset {
  id: string;
  label: string;
  pattern: string;
  color: string;
}

interface Props {
  entries: LogEntry[];
  onFiltered?: (filtered: LogEntry[]) => void;
}

// --- Constants ---

const PRESETS: FilterPreset[] = [
  { id: 'panic', label: 'PANIC_TRACES', pattern: '(panic|thread.*panicked|unwrapped)', color: '#ff0055' },
  { id: 'ipc', label: 'IPC_PAYLOADS', pattern: '(ipc|payload|serialize|deserialize|bincode)', color: '#00f3ff' },
  { id: 'exec', label: 'EXEC_REPORTS', pattern: '(fill|order|executed|latency|slippage)', color: '#00ff9d' },
  { id: 'error', label: 'ERRORS', pattern: '(error|failed|exception|timeout)', color: '#ffaa00' },
  { id: 'ws', label: 'WS_EVENTS', pattern: '(websocket|ws::|connected|disconnected|reconnect)', color: '#bd00ff' },
];

const COLORS = {
  bg: '#0a0a0a',
  border: '#333333',
  text: '#c0c0c0',
  highlight: 'rgba(255, 255, 0, 0.2)',
};

/**
 * Highlight text matching a regex pattern
 */
const highlightMatches = (text: string, pattern: RegExp, color: string): React.ReactNode[] => {
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  
  // Reset regex lastIndex
  pattern.lastIndex = 0;
  
  while ((match = pattern.exec(text)) !== null) {
    // Add non-matching text before this match
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`} className="text-gray-400">
          {text.slice(lastIndex, match.index)}
        </span>
      );
    }
    
    // Add highlighted match
    parts.push(
      <span
        key={`match-${match.index}`}
        className="px-0.5 rounded font-bold"
        style={{
          backgroundColor: `${color}40`,
          color: color,
          boxShadow: `0 0 8px ${color}60`,
          transform: 'translateZ(0)', // GPU acceleration hint
        }}
      >
        {match[0]}
      </span>
    );
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(
      <span key={`text-${lastIndex}`} className="text-gray-400">
        {text.slice(lastIndex)}
      </span>
    );
  }
  
  return parts;
};

/**
 * LogFilter Component
 * Interactive regex filtering panel with syntax highlighting.
 */
export const LogFilter: React.FC<Props> = ({ entries, onFiltered }) => {
  const [customPattern, setCustomPattern] = useState('');
  const [activePresets, setActivePresets] = useState<string[]>([]);
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [useRegex, setUseRegex] = useState(true);

  // Build combined filter pattern
  const combinedPattern = useMemo(() => {
    const patterns: string[] = [];
    
    // Add preset patterns
    activePresets.forEach((presetId) => {
      const preset = PRESETS.find((p) => p.id === presetId);
      if (preset) patterns.push(preset.pattern);
    });
    
    // Add custom pattern
    if (customPattern.trim()) {
      patterns.push(customPattern.trim());
    }
    
    if (patterns.length === 0) return null;
    
    try {
      const flags = caseSensitive ? 'g' : 'gi';
      return new RegExp(patterns.join('|'), flags);
    } catch (e) {
      return null; // Invalid regex
    }
  }, [activePresets, customPattern, caseSensitive]);

  // Filter entries
  const filteredEntries = useMemo(() => {
    if (!combinedPattern) return entries;
    
    return entries.filter((entry) => {
      const searchText = `${entry.message} ${entry.level} ${entry.source}`;
      return combinedPattern.test(searchText);
    });
  }, [entries, combinedPattern]);

  // Notify parent of filtered results
  React.useEffect(() => {
    onFiltered?.(filteredEntries);
  }, [filteredEntries, onFiltered]);

  // Toggle preset
  const togglePreset = useCallback((id: string) => {
    setActivePresets((prev) =>
      prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id]
    );
  }, []);

  // Calculate stats
  const stats = useMemo(() => {
    const byLevel: Record<string, number> = {};
    const bySource: Record<string, number> = {};
    
    filteredEntries.forEach((entry) => {
      byLevel[entry.level] = (byLevel[entry.level] || 0) + 1;
      bySource[entry.source] = (bySource[entry.source] || 0) + 1;
    });
    
    return { byLevel, bySource, total: filteredEntries.length };
  }, [filteredEntries]);

  return (
    <div className="p-4 bg-black/80 backdrop-blur-md border border-cyan-900/50 rounded-xl">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-mono font-bold text-white tracking-wider">
          LOG_FILTER_PANEL
        </h3>
        <div className="text-[10px] font-mono text-cyan-400">
          MATCHING: {stats.total}/{entries.length}
        </div>
      </div>

      {/* Preset Filters */}
      <div className="mb-4">
        <div className="text-[9px] font-mono text-gray-500 mb-2">PRESET_FILTERS</div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => togglePreset(preset.id)}
              className={`px-2 py-1 text-[10px] font-mono rounded border transition-all duration-200 ${
                activePresets.includes(preset.id)
                  ? 'border-opacity-100'
                  : 'border-opacity-30 opacity-50'
              }`}
              style={{
                borderColor: preset.color,
                backgroundColor: activePresets.includes(preset.id)
                  ? `${preset.color}20`
                  : 'transparent',
                color: preset.color,
                boxShadow: activePresets.includes(preset.id)
                  ? `0 0 8px ${preset.color}40`
                  : 'none',
              }}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Custom Pattern Input */}
      <div className="mb-4">
        <div className="text-[9px] font-mono text-gray-500 mb-2">CUSTOM_REGEX</div>
        <div className="flex gap-2">
          <input
            type="text"
            value={customPattern}
            onChange={(e) => setCustomPattern(e.target.value)}
            placeholder="Enter regex pattern..."
            className="flex-1 px-3 py-1.5 bg-gray-900 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
          />
          <label className="flex items-center gap-1 text-[10px] font-mono text-gray-400">
            <input
              type="checkbox"
              checked={caseSensitive}
              onChange={(e) => setCaseSensitive(e.target.checked)}
              className="rounded border-gray-700 bg-gray-900"
            />
            Aa
          </label>
          <label className="flex items-center gap-1 text-[10px] font-mono text-gray-400">
            <input
              type="checkbox"
              checked={useRegex}
              onChange={(e) => setUseRegex(e.target.checked)}
              className="rounded border-gray-700 bg-gray-900"
              disabled
            />
            .*
          </label>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-[9px] font-mono text-gray-500 mb-1">BY_LEVEL</div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(stats.byLevel).map(([level, count]) => (
              <div
                key={level}
                className="px-1.5 py-0.5 bg-gray-900 rounded text-[9px] font-mono"
                style={{ color: level === 'PANIC' ? '#ff0055' : level === 'ERROR' ? '#ffaa00' : '#00f3ff' }}
              >
                {level}: {count}
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-mono text-gray-500 mb-1">BY_SOURCE</div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(stats.bySource).map(([source, count]) => (
              <div
                key={source}
                className="px-1.5 py-0.5 bg-gray-900 rounded text-[9px] font-mono text-cyan-400"
              >
                {source}: {count}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sample Highlighted Output */}
      {filteredEntries.length > 0 && combinedPattern && (
        <div>
          <div className="text-[9px] font-mono text-gray-500 mb-1">PREVIEW</div>
          <div className="p-2 bg-gray-900/50 rounded border border-gray-800 max-h-32 overflow-y-auto">
            {filteredEntries.slice(0, 5).map((entry) => (
              <div key={entry.id} className="text-[10px] font-mono mb-1 last:mb-0">
                <span className="text-gray-500">[{entry.level}]</span>{' '}
                {highlightMatches(entry.message, combinedPattern, '#ffff00')}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clear Button */}
      {(activePresets.length > 0 || customPattern) && (
        <button
          onClick={() => {
            setActivePresets([]);
            setCustomPattern('');
          }}
          className="mt-3 w-full px-2 py-1 text-[10px] font-mono text-gray-400 border border-gray-700 rounded hover:bg-gray-800 transition-colors"
        >
          CLEAR_FILTERS
        </button>
      )}
    </div>
  );
};

export default LogFilter;
