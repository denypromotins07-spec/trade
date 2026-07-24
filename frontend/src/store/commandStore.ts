/**
 * commandStore - Zustand Store for Command Management
 * Manages registered commands with dynamic fuzzy search indices
 * Optimized to avoid heavy React re-renders on main UI thread
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

export interface CommandDefinition {
  id: string;
  label: string;
  description?: string;
  category: 'navigation' | 'action' | 'strategy' | 'system' | 'trading';
  shortcut?: string;
  keywords?: string[];
  action: () => void | Promise<void>;
  icon?: string;
  enabled: boolean;
  priority: number; // Higher = more important in search results
}

interface SearchIndex {
  term: string;
  commandIds: Set<string>;
}

interface CommandStoreState {
  commands: Map<string, CommandDefinition>;
  searchIndices: Map<string, SearchIndex>;
  recentCommands: string[]; // Command IDs ordered by recency
  favoriteCommands: Set<string>; // Command IDs marked as favorites
  isInitialized: boolean;
}

interface CommandStoreActions {
  registerCommand: (command: CommandDefinition) => void;
  unregisterCommand: (id: string) => void;
  updateCommand: (id: string, updates: Partial<CommandDefinition>) => void;
  enableCommand: (id: string) => void;
  disableCommand: (id: string) => void;
  toggleCommand: (id: string) => void;
  getCommand: (id: string) => CommandDefinition | undefined;
  getAllCommands: () => CommandDefinition[];
  getCommandsByCategory: (category: string) => CommandDefinition[];
  searchCommands: (query: string, limit?: number) => CommandDefinition[];
  markRecent: (id: string) => void;
  toggleFavorite: (id: string) => void;
  rebuildSearchIndex: () => void;
  reset: () => void;
}

type CommandStore = CommandStoreState & CommandStoreActions;

// Default system commands
const DEFAULT_COMMANDS: CommandDefinition[] = [
  {
    id: 'start-bot',
    label: '/START',
    description: 'Initialize all Ray workers and start trading strategies',
    category: 'system',
    shortcut: 'Ctrl+Shift+S',
    keywords: ['launch', 'init', 'begin', 'run', 'activate'],
    action: () => console.log('[COMMAND] /START executed'),
    icon: '▶',
    enabled: true,
    priority: 100,
  },
  {
    id: 'kill-bot',
    label: '/KILL',
    description: 'Emergency stop: flatten all positions and shutdown workers',
    category: 'system',
    shortcut: 'Ctrl+Shift+K',
    keywords: ['stop', 'halt', 'emergency', 'abort', 'terminate'],
    action: () => console.log('[COMMAND] /KILL executed'),
    icon: '⏹',
    enabled: true,
    priority: 100,
  },
  {
    id: 'panic-flatten',
    label: 'PANIC FLATTEN',
    description: 'Immediately close all open positions',
    category: 'trading',
    shortcut: 'Ctrl+Shift+P',
    keywords: ['emergency', 'close', 'exit', 'sell', 'liquidate'],
    action: () => console.log('[COMMAND] PANIC FLATTEN executed'),
    icon: '⚠',
    enabled: true,
    priority: 95,
  },
  {
    id: 'toggle-strategy',
    label: 'SWAP STRATEGY',
    description: 'Hot-swap the active trading strategy',
    category: 'strategy',
    shortcut: 'Ctrl+Shift+T',
    keywords: ['switch', 'change', 'rotate', 'algorithm'],
    action: () => console.log('[COMMAND] SWAP STRATEGY executed'),
    icon: '🔄',
    enabled: true,
    priority: 80,
  },
  {
    id: 'open-command-palette',
    label: 'Command Palette',
    description: 'Open the command palette for quick navigation',
    category: 'navigation',
    shortcut: 'Cmd+K',
    keywords: ['search', 'navigate', 'menu', 'quick'],
    action: () => console.log('[COMMAND] Command Palette opened'),
    icon: '🔍',
    enabled: true,
    priority: 70,
  },
  {
    id: 'toggle-pip',
    label: 'Toggle PiP',
    description: 'Toggle Picture-in-Picture display mode',
    category: 'navigation',
    shortcut: 'Ctrl+Shift+D',
    keywords: ['picture', 'overlay', 'float', 'window'],
    action: () => console.log('[COMMAND] Toggle PiP executed'),
    icon: '🖼',
    enabled: true,
    priority: 60,
  },
  {
    id: 'generate-report',
    label: 'Generate Report',
    description: 'Generate daily PDF performance report',
    category: 'system',
    shortcut: 'Ctrl+Shift+R',
    keywords: ['pdf', 'daily', 'pnl', 'performance', 'export'],
    action: () => console.log('[COMMAND] Generate Report executed'),
    icon: '📊',
    enabled: true,
    priority: 50,
  },
  {
    id: 'refresh-data',
    label: 'Refresh Data',
    description: 'Force refresh all market data feeds',
    category: 'system',
    shortcut: 'F5',
    keywords: ['reload', 'update', 'sync', 'fetch'],
    action: () => console.log('[COMMAND] Refresh Data executed'),
    icon: '⟳',
    enabled: true,
    priority: 40,
  },
];

/**
 * Build search index terms from a command
 */
function buildSearchTerms(command: CommandDefinition): string[] {
  const terms = new Set<string>();
  
  // Add label words
  command.label.toLowerCase().split(/\s+/).forEach((word) => {
    if (word.length > 1) terms.add(word);
  });
  
  // Add description words
  if (command.description) {
    command.description.toLowerCase().split(/\s+/).forEach((word) => {
      if (word.length > 2) terms.add(word);
    });
  }
  
  // Add keywords
  command.keywords?.forEach((keyword) => {
    terms.add(keyword.toLowerCase());
  });
  
  return Array.from(terms);
}

/**
 * Create the command store with optimized indexing
 */
export const useCommandStore = create<CommandStore>()(
  subscribeWithSelector((set, get) => ({
    // State
    commands: new Map(),
    searchIndices: new Map(),
    recentCommands: [],
    favoriteCommands: new Set(),
    isInitialized: false,

    // Actions
    registerCommand: (command: CommandDefinition) => {
      set((state) => {
        const newCommands = new Map(state.commands);
        newCommands.set(command.id, command);
        
        // Update search indices asynchronously to avoid blocking
        setTimeout(() => {
          get().rebuildSearchIndex();
        }, 0);
        
        return { commands: newCommands };
      });
    },

    unregisterCommand: (id: string) => {
      set((state) => {
        const newCommands = new Map(state.commands);
        newCommands.delete(id);
        
        const newFavorites = new Set(state.favoriteCommands);
        newFavorites.delete(id);
        
        const newRecents = state.recentCommands.filter((cmdId) => cmdId !== id);
        
        setTimeout(() => {
          get().rebuildSearchIndex();
        }, 0);
        
        return {
          commands: newCommands,
          favoriteCommands: newFavorites,
          recentCommands: newRecents,
        };
      });
    },

    updateCommand: (id: string, updates: Partial<CommandDefinition>) => {
      set((state) => {
        const command = state.commands.get(id);
        if (!command) return state;
        
        const updatedCommand = { ...command, ...updates };
        const newCommands = new Map(state.commands);
        newCommands.set(id, updatedCommand);
        
        setTimeout(() => {
          get().rebuildSearchIndex();
        }, 0);
        
        return { commands: newCommands };
      });
    },

    enableCommand: (id: string) => {
      get().updateCommand(id, { enabled: true });
    },

    disableCommand: (id: string) => {
      get().updateCommand(id, { enabled: false });
    },

    toggleCommand: (id: string) => {
      const command = get().getCommand(id);
      if (command) {
        get().updateCommand(id, { enabled: !command.enabled });
      }
    },

    getCommand: (id: string) => {
      return get().commands.get(id);
    },

    getAllCommands: () => {
      return Array.from(get().commands.values());
    },

    getCommandsByCategory: (category: string) => {
      return Array.from(get().commands.values()).filter(
        (cmd) => cmd.category === category && cmd.enabled
      );
    },

    searchCommands: (query: string, limit = 20) => {
      const normalizedQuery = query.toLowerCase().trim();
      
      if (!normalizedQuery) {
        // Return all enabled commands sorted by priority
        return get()
          .getAllCommands()
          .filter((cmd) => cmd.enabled)
          .sort((a, b) => b.priority - a.priority)
          .slice(0, limit);
      }

      const state = get();
      const results: Array<{ command: CommandDefinition; score: number }> = [];

      // Use pre-built search indices for fast lookup
      const queryTerms = normalizedQuery.split(/\s+/);
      const matchingIds = new Set<string>();

      for (const term of queryTerms) {
        const index = state.searchIndices.get(term);
        if (index) {
          index.commandIds.forEach((id) => matchingIds.add(id));
        }
      }

      // Score matching commands
      matchingIds.forEach((id) => {
        const command = state.commands.get(id);
        if (!command || !command.enabled) return;

        let score = command.priority;

        // Exact label match boost
        if (command.label.toLowerCase().includes(normalizedQuery)) {
          score += 1000;
        }

        // Starts with match boost
        if (command.label.toLowerCase().startsWith(normalizedQuery)) {
          score += 500;
        }

        // Description match
        if (command.description?.toLowerCase().includes(normalizedQuery)) {
          score += 50;
        }

        // Recent command boost
        const recentIndex = state.recentCommands.indexOf(id);
        if (recentIndex !== -1) {
          score += Math.max(0, 30 - recentIndex * 3);
        }

        // Favorite boost
        if (state.favoriteCommands.has(id)) {
          score += 100;
        }

        results.push({ command, score });
      });

      // Sort by score and return limited results
      return results
        .sort((a, b) => b.score - a.score)
        .map((r) => r.command)
        .slice(0, limit);
    },

    markRecent: (id: string) => {
      set((state) => {
        const newRecents = [
          id,
          ...state.recentCommands.filter((cmdId) => cmdId !== id),
        ].slice(0, 10); // Keep last 10
        
        return { recentCommands: newRecents };
      });
    },

    toggleFavorite: (id: string) => {
      set((state) => {
        const newFavorites = new Set(state.favoriteCommands);
        if (newFavorites.has(id)) {
          newFavorites.delete(id);
        } else {
          newFavorites.add(id);
        }
        return { favoriteCommands: newFavorites };
      });
    },

    rebuildSearchIndex: () => {
      const state = get();
      const newIndex = new Map<string, SearchIndex>();

      state.commands.forEach((command) => {
        if (!command.enabled) return;

        const terms = buildSearchTerms(command);
        
        terms.forEach((term) => {
          if (!newIndex.has(term)) {
            newIndex.set(term, { term, commandIds: new Set() });
          }
          newIndex.get(term)!.commandIds.add(command.id);
        });
      });

      set({ searchIndices: newIndex });
    },

    reset: () => {
      set({
        commands: new Map(),
        searchIndices: new Map(),
        recentCommands: [],
        favoriteCommands: new Set(),
        isInitialized: false,
      });
    },
  }))
);

// Initialize with default commands
export function initializeCommandStore(): void {
  const store = useCommandStore.getState();
  
  if (!store.isInitialized) {
    DEFAULT_COMMANDS.forEach((command) => {
      store.registerCommand(command);
    });
    
    store.rebuildSearchIndex();
    
    useCommandStore.setState({ isInitialized: true });
    
    console.log('[CommandStore] Initialized with', DEFAULT_COMMANDS.length, 'default commands');
  }
}

// Auto-initialize when module loads
if (typeof window !== 'undefined') {
  initializeCommandStore();
}

export default useCommandStore;
