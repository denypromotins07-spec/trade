import { create } from 'zustand';
import { shallow } from 'zustand/shallow';

// ============================================================================
// SYSTEM HEALTH & STATE TYPES
// ============================================================================

export type SystemStatus = 'IDLE' | 'RUNNING' | 'STOPPING' | 'ERROR' | 'KILLED';

export interface GpuMetrics {
  utilization: number; // 0-100
  memoryUsed: number;  // MB
  memoryTotal: number; // MB
  temperature: number; // Celsius
  directmlEnabled: boolean;
  rocmEnabled: boolean;
}

export interface LatencyMetrics {
  wsPingMs: number;
  rustCoreLatencyMs: number;
  lastHeartbeat: number;
}

export interface ActiveStrategy {
  id: string;
  name: string;
  status: 'ACTIVE' | 'PAUSED' | 'STOPPED';
  pnl: number;
  trades: number;
}

export interface GlobalState {
  // Master control states
  systemStatus: SystemStatus;
  isTradingEnabled: boolean;
  
  // System health
  systemHealth: {
    cpuUsage: number;
    ramUsageMb: number;
    ramTotalMb: number;
    uptimeSeconds: number;
  };
  
  // GPU metrics (AMD DirectML/ROCm context)
  gpuMetrics: GpuMetrics | null;
  
  // Latency tracking
  latencyMetrics: LatencyMetrics;
  
  // Active strategies
  activeStrategies: ActiveStrategy[];
  
  // Connection state
  wsConnected: boolean;
  wsReconnectAttempts: number;
  
  // Actions
  setSystemStatus: (status: SystemStatus) => void;
  toggleTrading: () => void;
  updateSystemHealth: (health: Partial<GlobalState['systemHealth']>) => void;
  updateGpuMetrics: (metrics: Partial<GpuMetrics>) => void;
  updateLatencyMetrics: (metrics: Partial<LatencyMetrics>) => void;
  addStrategy: (strategy: ActiveStrategy) => void;
  removeStrategy: (strategyId: string) => void;
  updateStrategy: (strategyId: string, updates: Partial<ActiveStrategy>) => void;
  setWsConnected: (connected: boolean) => void;
  incrementReconnectAttempts: () => void;
  resetReconnectAttempts: () => void;
}

// ============================================================================
// INITIAL STATE
// ============================================================================

const initialState: Omit<GlobalState, keyof Pick<GlobalState, 
  'setSystemStatus' | 'toggleTrading' | 'updateSystemHealth' | 'updateGpuMetrics' | 
  'updateLatencyMetrics' | 'addStrategy' | 'removeStrategy' | 'updateStrategy' | 
  'setWsConnected' | 'incrementReconnectAttempts' | 'resetReconnectAttempts'
>> = {
  systemStatus: 'IDLE',
  isTradingEnabled: false,
  systemHealth: {
    cpuUsage: 0,
    ramUsageMb: 0,
    ramTotalMb: 8192, // 8GB limit enforced
    uptimeSeconds: 0,
  },
  gpuMetrics: null,
  latencyMetrics: {
    wsPingMs: 0,
    rustCoreLatencyMs: 0,
    lastHeartbeat: Date.now(),
  },
  activeStrategies: [],
  wsConnected: false,
  wsReconnectAttempts: 0,
};

// ============================================================================
// ZUSTAND STORE CREATION
// Utilizes shallow equality checks to prevent unnecessary re-renders
// ============================================================================

export const useGlobalStore = create<GlobalState>()((set, get) => ({
  ...initialState,
  
  setSystemStatus: (status) => set({ systemStatus: status }, false, 'setSystemStatus'),
  
  toggleTrading: () => set(
    (state) => ({ isTradingEnabled: !state.isTradingEnabled }),
    false,
    'toggleTrading'
  ),
  
  updateSystemHealth: (health) => set(
    (state) => ({ 
      systemHealth: { ...state.systemHealth, ...health } 
    }),
    false,
    'updateSystemHealth'
  ),
  
  updateGpuMetrics: (metrics) => set(
    (state) => ({ 
      gpuMetrics: state.gpuMetrics 
        ? { ...state.gpuMetrics, ...metrics } 
        : { ...metrics, directmlEnabled: false, rocmEnabled: false } as GpuMetrics 
    }),
    false,
    'updateGpuMetrics'
  ),
  
  updateLatencyMetrics: (metrics) => set(
    (state) => ({ 
      latencyMetrics: { ...state.latencyMetrics, ...metrics } 
    }),
    false,
    'updateLatencyMetrics'
  ),
  
  addStrategy: (strategy) => set(
    (state) => ({ activeStrategies: [...state.activeStrategies, strategy] }),
    false,
    'addStrategy'
  ),
  
  removeStrategy: (strategyId) => set(
    (state) => ({ 
      activeStrategies: state.activeStrategies.filter(s => s.id !== strategyId) 
    }),
    false,
    'removeStrategy'
  ),
  
  updateStrategy: (strategyId, updates) => set(
    (state) => ({
      activeStrategies: state.activeStrategies.map(s => 
        s.id === strategyId ? { ...s, ...updates } : s
      ),
    }),
    false,
    'updateStrategy'
  ),
  
  setWsConnected: (connected) => set({ wsConnected: connected }, false, 'setWsConnected'),
  
  incrementReconnectAttempts: () => set(
    (state) => ({ wsReconnectAttempts: state.wsReconnectAttempts + 1 }),
    false,
    'incrementReconnectAttempts'
  ),
  
  resetReconnectAttempts: () => set({ wsReconnectAttempts: 0 }, false, 'resetReconnectAttempts'),
}));

// ============================================================================
// SELECTORS WITH SHALLOW EQUALITY FOR OPTIMIZED RE-RENDERS
// Prevents React reconciliation loops during high-frequency updates
// ============================================================================

export const selectSystemStatus = (state: GlobalState) => state.systemStatus;
export const selectIsTradingEnabled = (state: GlobalState) => state.isTradingEnabled;
export const selectWsConnected = (state: GlobalState) => state.wsConnected;
export const selectGpuMetrics = (state: GlobalState) => state.gpuMetrics;
export const selectLatencyMetrics = (state: GlobalState) => state.latencyMetrics;
export const selectActiveStrategies = (state: GlobalState) => state.activeStrategies;
export const selectSystemHealth = (state: GlobalState) => state.systemHealth;

// Shallow equality comparators for use with useSelector pattern
export const shallowSystemStatus = shallow;
export const shallowGpuMetrics = shallow;
export const shallowLatencyMetrics = shallow;
