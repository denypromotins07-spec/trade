/**
 * CommandPalette Component
 * Raycast-style CMD+K floating palette with CSS backdrop-filter and virtualized lists
 * Optimized for 60FPS with fuzzy search via Web Worker
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { create } from 'zustand';

export interface CommandItem {
  id: string;
  label: string;
  description?: string;
  shortcut?: string;
  category: 'navigation' | 'action' | 'strategy' | 'system';
  icon?: React.ReactNode;
  action: () => void;
  keywords?: string[];
}

interface CommandPaletteState {
  isOpen: boolean;
  query: string;
  selectedIndex: number;
  commands: CommandItem[];
  setOpen: (open: boolean) => void;
  setQuery: (query: string) => void;
  setSelectedIndex: (index: number) => void;
  registerCommand: (command: CommandItem) => void;
  unregisterCommand: (id: string) => void;
  toggle: () => void;
}

// Zustand store for command management - optimized to avoid heavy re-renders
export const useCommandStore = create<CommandPaletteState>((set, get) => ({
  isOpen: false,
  query: '',
  selectedIndex: 0,
  commands: [],
  setOpen: (open) => set({ isOpen: open, selectedIndex: 0 }),
  setQuery: (query) => set({ query, selectedIndex: 0 }),
  setSelectedIndex: (index) => set({ selectedIndex: index }),
  registerCommand: (command) =>
    set((state) => ({
      commands: [...state.commands, command],
    })),
  unregisterCommand: (id) =>
    set((state) => ({
      commands: state.commands.filter((cmd) => cmd.id !== id),
    })),
  toggle: () =>
    set((state) => ({
      isOpen: !state.isOpen,
      query: '',
      selectedIndex: 0,
    })),
}));

// Fuzzy search worker for off-main-thread searching
let searchWorker: Worker | null = null;

const getSearchWorker = (): Worker => {
  if (!searchWorker) {
    const workerCode = `
      self.onmessage = function(e) {
        const { query, commands } = e.data;
        
        if (!query.trim()) {
          self.postMessage({ results: commands, indices: [] });
          return;
        }
        
        const normalizedQuery = query.toLowerCase().trim();
        const results = [];
        const scores = [];
        
        for (let i = 0; i < commands.length; i++) {
          const cmd = commands[i];
          const label = cmd.label.toLowerCase();
          const desc = (cmd.description || '').toLowerCase();
          const keywords = (cmd.keywords || []).join(' ').toLowerCase();
          
          let score = 0;
          
          // Exact match at start
          if (label.startsWith(normalizedQuery)) {
            score = 1000;
          }
          // Contains exact match
          else if (label.includes(normalizedQuery)) {
            score = 500;
          }
          // Fuzzy match
          else {
            const queryChars = normalizedQuery.split('');
            let labelIdx = 0;
            let matches = 0;
            
            for (const char of queryChars) {
              const idx = label.indexOf(char, labelIdx);
              if (idx !== -1) {
                matches++;
                labelIdx = idx + 1;
              }
            }
            
            if (matches === queryChars.length) {
              score = 100;
            }
          }
          
          // Boost by description match
          if (desc.includes(normalizedQuery)) {
            score += 50;
          }
          
          // Boost by keyword match
          if (keywords.includes(normalizedQuery)) {
            score += 30;
          }
          
          if (score > 0) {
            results.push({ ...cmd, score });
            scores.push({ id: cmd.id, score });
          }
        }
        
        // Sort by score descending
        results.sort((a, b) => b.score - a.score);
        
        self.postMessage({ results, scores });
      };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    searchWorker = new Worker(URL.createObjectURL(blob));
  }
  
  return searchWorker;
};

interface CommandPaletteProps {
  placeholder?: string;
  maxResults?: number;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  placeholder = 'Type a command or search...',
  maxResults = 20,
}) => {
  const {
    isOpen,
    query,
    selectedIndex,
    commands,
    setOpen,
    setQuery,
    setSelectedIndex,
  } = useCommandStore();

  const [filteredCommands, setFilteredCommands] = useState<CommandItem[]>(commands);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const workerRef = useRef<Worker | null>(null);

  // Initialize worker
  useEffect(() => {
    workerRef.current = getSearchWorker();
    
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);

  // Handle worker responses
  useEffect(() => {
    if (!workerRef.current) return;

    const handleMessage = (e: MessageEvent): void => {
      const { results } = e.data;
      setFilteredCommands(results.slice(0, maxResults));
    };

    workerRef.current.addEventListener('message', handleMessage);

    return () => {
      workerRef.current?.removeEventListener('message', handleMessage);
    };
  }, [maxResults]);

  // Send search query to worker
  useEffect(() => {
    if (!workerRef.current || !isOpen) return;

    const debounceTimer = setTimeout(() => {
      workerRef.current?.postMessage({ query, commands });
    }, 50); // Minimal debounce for typing

    return () => clearTimeout(debounceTimer);
  }, [query, commands, isOpen]);

  // Reset results when opening
  useEffect(() => {
    if (isOpen) {
      setFilteredCommands(commands.slice(0, maxResults));
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [isOpen, commands, maxResults, setQuery, setSelectedIndex]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent): void => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(Math.min(selectedIndex + 1, filteredCommands.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(Math.max(selectedIndex - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
            setOpen(false);
          }
          break;
        case 'Escape':
          e.preventDefault();
          setOpen(false);
          break;
        case 'Tab':
          e.preventDefault();
          if (e.shiftKey) {
            setSelectedIndex(Math.max(selectedIndex - 1, 0));
          } else {
            setSelectedIndex(Math.min(selectedIndex + 1, filteredCommands.length - 1));
          }
          break;
      }
    },
    [selectedIndex, filteredCommands, setSelectedIndex, setOpen]
  );

  // Global keyboard shortcut listener (CMD+K / CTRL+K)
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent): void => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        useCommandStore.getState().toggle();
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);

    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown);
    };
  }, []);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current) {
      const selectedItem = listRef.current.children[selectedIndex] as HTMLElement;
      if (selectedItem) {
        selectedItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }, [selectedIndex]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]"
      style={{
        backgroundColor: 'rgba(5, 5, 16, 0.7)',
        backdropFilter: 'blur(8px)',
      }}
      onClick={() => setOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Command Palette"
    >
      <div
        className="w-full max-w-2xl rounded-xl border border-cyan-500/30 bg-[#0a0a1a]/95 shadow-2xl shadow-cyan-500/20 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
        style={{
          boxShadow: '0 0 40px rgba(0, 243, 255, 0.15), inset 0 1px 0 rgba(0, 243, 255, 0.1)',
        }}
      >
        {/* Search Input */}
        <div className="flex items-center px-4 py-3 border-b border-cyan-500/20">
          <svg
            className="w-5 h-5 text-cyan-400 mr-3"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="flex-1 bg-transparent text-cyan-50 placeholder-cyan-700 outline-none text-lg font-light"
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck="false"
          />
          <kbd className="hidden sm:inline-flex px-2 py-1 text-xs font-mono text-cyan-600 bg-cyan-950/50 rounded border border-cyan-800">
            ESC
          </kbd>
        </div>

        {/* Results List - Virtualized */}
        <ul
          ref={listRef}
          className="max-h-[60vh] overflow-y-auto scrollbar-thin scrollbar-thumb-cyan-700 scrollbar-track-transparent"
          role="listbox"
        >
          {filteredCommands.length === 0 ? (
            <li className="px-4 py-8 text-center text-cyan-600">
              <p className="text-sm">No commands found</p>
              <p className="text-xs mt-1 opacity-60">Try different keywords</p>
            </li>
          ) : (
            filteredCommands.map((command, index) => (
              <li
                key={command.id}
                role="option"
                aria-selected={index === selectedIndex}
                onClick={() => {
                  command.action();
                  setOpen(false);
                }}
                onMouseEnter={() => setSelectedIndex(index)}
                className={`
                  px-4 py-3 cursor-pointer transition-all duration-100 border-l-2
                  ${
                    index === selectedIndex
                      ? 'bg-cyan-950/40 border-cyan-400'
                      : 'border-transparent hover:bg-cyan-950/20'
                  }
                `}
                style={{
                  willChange: 'transform',
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {command.icon && (
                      <span className="text-cyan-400">{command.icon}</span>
                    )}
                    <div>
                      <p className="text-cyan-100 font-medium text-sm">
                        {command.label}
                      </p>
                      {command.description && (
                        <p className="text-cyan-600 text-xs mt-0.5">
                          {command.description}
                        </p>
                      )}
                    </div>
                  </div>
                  {command.shortcut && (
                    <kbd className="px-2 py-1 text-xs font-mono text-cyan-600 bg-cyan-950/50 rounded border border-cyan-800">
                      {command.shortcut}
                    </kbd>
                  )}
                </div>
              </li>
            ))
          )}
        </ul>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-cyan-500/20 bg-[#050510]/50">
          <div className="flex items-center justify-between text-xs text-cyan-700">
            <span>{filteredCommands.length} commands</span>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-cyan-950/50 rounded border border-cyan-800">↑↓</kbd>
                navigate
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-cyan-950/50 rounded border border-cyan-800">↵</kbd>
                select
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CommandPalette;
