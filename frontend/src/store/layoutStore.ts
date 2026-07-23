/**
 * File 3: frontend/src/store/layoutStore.ts
 * 
 * Elite Implementation:
 * - Zustand store with shallow equality to prevent unnecessary re-renders.
 * - LocalStorage persistence with corruption quarantine.
 * - Atomic updates for layout geometry changes.
 * - Supports multiple saved layouts per user.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { shallow } from 'zustand/shallow';

export interface WidgetItem {
  id: string;
  type: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface LayoutData {
  widgets: WidgetItem[];
  createdAt: number;
  updatedAt: number;
}

interface LayoutState {
  layouts: Record<string, LayoutData>;
  activeLayoutId: string | null;
  isDirty: boolean;
  
  // Actions
  loadLayout: (id: string) => LayoutData | null;
  saveLayout: (id: string, data: Omit<LayoutData, 'createdAt' | 'updatedAt'>) => void;
  resetLayout: (id: string) => void;
  setActiveLayout: (id: string | null) => void;
  addWidget: (layoutId: string, widget: WidgetItem) => void;
  removeWidget: (layoutId: string, widgetId: string) => void;
  updateWidget: (layoutId: string, widgetId: string, updates: Partial<WidgetItem>) => void;
  markDirty: () => void;
  markClean: () => void;
}

const STORAGE_KEY = 'nautilus_layouts_v1';
const QUARANTINE_KEY = 'nautilus_layouts_quarantine';

// Safe JSON parse with fallback
const safeJsonParse = <T>(str: string | null, fallback: T): T => {
  if (!str) return fallback;
  try {
    return JSON.parse(str);
  } catch (e) {
    console.warn('[LayoutStore] Corrupted JSON detected:', e);
    return fallback;
  }
};

// Quarantine corrupted data
const quarantineData = (key: string, data: any) => {
  try {
    const existing = safeJsonParse<any[]>(localStorage.getItem(QUARANTINE_KEY), []);
    const quarantined = [
      ...existing,
      { timestamp: Date.now(), key, data, error: 'Corruption detected' }
    ].slice(-50); // Keep last 50 incidents
    localStorage.setItem(QUARANTINE_KEY, JSON.stringify(quarantined));
  } catch (e) {
    console.error('[LayoutStore] Failed to quarantine data:', e);
  }
};

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set, get) => ({
      layouts: {},
      activeLayoutId: null,
      isDirty: false,

      loadLayout: (id: string) => {
        const state = get();
        const layout = state.layouts[id];
        
        if (!layout) {
          // Create default empty layout
          const defaultLayout: LayoutData = {
            widgets: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
          };
          set(state => ({
            layouts: { ...state.layouts, [id]: defaultLayout }
          }));
          return defaultLayout;
        }
        
        // Validate layout integrity
        if (!Array.isArray(layout.widgets)) {
          console.warn(`[LayoutStore] Layout ${id} has invalid widgets array. Quarantining.`);
          quarantineData(id, layout);
          get().resetLayout(id);
          return null;
        }
        
        return layout;
      },

      saveLayout: (id: string, data: Omit<LayoutData, 'createdAt' | 'updatedAt'>) => {
        set(state => {
          const existing = state.layouts[id];
          const updated: LayoutData = {
            ...data,
            createdAt: existing?.createdAt || Date.now(),
            updatedAt: Date.now(),
          };
          
          return {
            layouts: { ...state.layouts, [id]: updated },
            isDirty: false,
          };
        });
      },

      resetLayout: (id: string) => {
        set(state => {
          const existing = state.layouts[id];
          if (existing) {
            quarantineData(id, existing);
          }
          
          const defaultLayout: LayoutData = {
            widgets: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
          };
          
          return {
            layouts: { ...state.layouts, [id]: defaultLayout },
            isDirty: false,
          };
        });
      },

      setActiveLayout: (id: string | null) => {
        set({ activeLayoutId: id, isDirty: false });
      },

      addWidget: (layoutId: string, widget: WidgetItem) => {
        set(state => {
          const layout = state.layouts[layoutId];
          if (!layout) return state;
          
          return {
            layouts: {
              ...state.layouts,
              [layoutId]: {
                ...layout,
                widgets: [...layout.widgets, widget],
                updatedAt: Date.now(),
              }
            },
            isDirty: true,
          };
        });
      },

      removeWidget: (layoutId: string, widgetId: string) => {
        set(state => {
          const layout = state.layouts[layoutId];
          if (!layout) return state;
          
          return {
            layouts: {
              ...state.layouts,
              [layoutId]: {
                ...layout,
                widgets: layout.widgets.filter(w => w.id !== widgetId),
                updatedAt: Date.now(),
              }
            },
            isDirty: true,
          };
        });
      },

      updateWidget: (layoutId: string, widgetId: string, updates: Partial<WidgetItem>) => {
        set(state => {
          const layout = state.layouts[layoutId];
          if (!layout) return state;
          
          return {
            layouts: {
              ...state.layouts,
              [layoutId]: {
                ...layout,
                widgets: layout.widgets.map(w => 
                  w.id === widgetId ? { ...w, ...updates } : w
                ),
                updatedAt: Date.now(),
              }
            },
            isDirty: true,
          };
        });
      },

      markDirty: () => set({ isDirty: true }),
      markClean: () => set({ isDirty: false }),
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ layouts: state.layouts, activeLayoutId: state.activeLayoutId }),
      version: 1,
      migrate: (persistedState: any, version: number) => {
        if (version === 0) {
          // Migration from v0 to v1
          return {
            ...persistedState,
            layouts: persistedState.layouts || {},
          };
        }
        return persistedState as any;
      },
      onRehydrateStorage: () => (state, error) => {
        if (error) {
          console.error('[LayoutStore] Rehydration failed:', error);
          // Clear corrupted storage
          localStorage.removeItem(STORAGE_KEY);
        } else {
          console.log('[LayoutStore] Rehydration successful');
        }
      },
    }
  )
);

// Custom hook for optimized selectors
export const useLayoutSelector = <T,>(selector: (state: LayoutState) => T) => {
  return useLayoutStore(selector, shallow);
};

export default useLayoutStore;
