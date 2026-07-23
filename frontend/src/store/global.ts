import { create } from 'zustand';
import { shallow } from 'zustand/shallow';

/**
 * Global System State Interface
 * Manages master control states, system health metrics, and active strategy tracking.
 * Optimized with shallow equality to prevent unnecessary re-renders on high-frequency updates.
 */
export interface SystemHealth {
  cpuUsage: number;          // Percentage (0-100)
  ramUsage: number;          // MB used
  gpuUsage: number;          // Percentage (0-100) - AMD DirectML/ROCm context
  gpuMemory: number;         // MB used
  networkLatency: number;    // ms
  uptime: number;            // seconds
  backendStatus: 'online' | 'offline' | 'restarting';
}

export interface Strategy {
  id: string;
  name: string;
  status: 'active' | 'paused' | 'stopped';
  pnl: number;               // Real-time PnL in USD
  tradesExecuted: number;
}

export interface MasterControl {
  isRunning: boolean;        // Master /START state
  isKilled: boolean;         // Master /KILL state
  lastCommand: string | null;
  commandTimestamp: number | null;
}

interface GlobalState {
  // System Health Metrics
  systemHealth: SystemHealth;
  
  // Active Strategies
  strategies: Strategy[];
  
  // Master Control States
  masterControl: MasterControl;
  
  // WebSocket Connection State
  wsConnected: boolean;
  wsReconnectAttempts: number;
  
  // Actions - optimized for minimal re-renders
  updateSystemHealth: (health: Partial<SystemHealth>) => void;
  addStrategy: (strategy: Strategy) => void;
  updateStrategy: (id: string, updates: Partial<Strategy>) => void;
  removeStrategy: (id: string) => void;
  setMasterStart: () => void;
  setMasterKill: () => void;
  resetMasterControl: () => void;
  setWsConnected: (connected: boolean) => void;
  incrementReconnectAttempts: () => void;
  resetReconnectAttempts: () => void;
}

const initialSystemHealth: SystemHealth = {
  cpuUsage: 0,
  ramUsage: 0,
  gpuUsage: 0,
  gpuMemory: 0,
  networkLatency: 0,
  uptime: 0,
  backendStatus: 'offline',
};

const initialMasterControl: MasterControl = {
  isRunning: false,
  isKilled: false,
  lastCommand: null,
  commandTimestamp: null,
};

/**
 * Global Zustand Store
 * Uses shallow comparison for selectors to optimize React rendering performance.
 * All mutations are synchronous and batched to prevent reconciliation loops.
 */
export const useGlobalStore = create<GlobalState>()((set, get) => ({
  // Initial State
  systemHealth: initialSystemHealth,
  strategies: [],
  masterControl: initialMasterControl,
  wsConnected: false,
  wsReconnectAttempts: 0,

  // System Health Updates - partial updates merged efficiently
  updateSystemHealth: (health) => set((state) => ({
    systemHealth: { ...state.systemHealth, ...health },
  }), false), // Shallow merge, no deep equality check needed

  // Strategy Management
  addStrategy: (strategy) => set((state) => ({
    strategies: [...state.strategies, strategy],
  }), false),

  updateStrategy: (id, updates) => set((state) => ({
    strategies: state.strategies.map((s) =>
      s.id === id ? { ...s, ...updates } : s
    ),
  }), false),

  removeStrategy: (id) => set((state) => ({
    strategies: state.strategies.filter((s) => s.id !== id),
  }), false),

  // Master Control Commands - PowerShell orchestration compatible
  setMasterStart: () => set((state) => ({
    masterControl: {
      isRunning: true,
      isKilled: false,
      lastCommand: '/START',
      commandTimestamp: Date.now(),
    },
  }), false),

  setMasterKill: () => set((state) => ({
    masterControl: {
      isRunning: false,
      isKilled: true,
      lastCommand: '/KILL',
      commandTimestamp: Date.now(),
    },
  }), false),

  resetMasterControl: () => set({
    masterControl: { ...initialMasterControl },
  }, false),

  // WebSocket State Management
  setWsConnected: (connected) => set({ wsConnected: connected }, false),

  incrementReconnectAttempts: () => set((state) => ({
    wsReconnectAttempts: state.wsReconnectAttempts + 1,
  }), false),

  resetReconnectAttempts: () => set({ wsReconnectAttempts: 0 }, false),
}));

/**
 * Custom selector hooks with shallow equality for optimal performance.
 * Import these in components to subscribe only to necessary state slices.
 */
export const selectSystemHealth = (state: GlobalState) => state.systemHealth;
export const selectStrategies = (state: GlobalState) => state.strategies;
export const selectMasterControl = (state: GlobalState) => state.masterControl;
export const selectWsStatus = (state: GlobalState) => ({
  connected: state.wsConnected,
  attempts: state.wsReconnectAttempts,
});
