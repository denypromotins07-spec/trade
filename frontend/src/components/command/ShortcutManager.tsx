/**
 * ShortcutManager Component
 * Global hotkey listener for complex chorded shortcuts
 * Triggers panic flattens, strategy hot-swaps, and master /START toggles
 * Optimized for 60FPS with minimal main thread interference
 */

import React, { useEffect, useCallback, useRef } from 'react';
import { useCommandStore } from './CommandPalette';

export interface ShortcutConfig {
  id: string;
  keys: string[];
  description: string;
  action: () => void;
  category: 'trading' | 'navigation' | 'system' | 'strategy';
  enabled?: boolean;
}

interface ShortcutManagerProps {
  shortcuts?: ShortcutConfig[];
  onShortcutTriggered?: (shortcut: ShortcutConfig) => void;
  debugMode?: boolean;
}

// Default trading bot shortcuts
const DEFAULT_SHORTCUTS: ShortcutConfig[] = [
  {
    id: 'panic-flatten',
    keys: ['Ctrl', 'Shift', 'P'],
    description: 'PANIC: Flatten all positions immediately',
    action: () => console.log('[SHORTCUT] PANIC FLATTEN TRIGGERED'),
    category: 'trading',
    enabled: true,
  },
  {
    id: 'start-bot',
    keys: ['Ctrl', 'Shift', 'S'],
    description: '/START: Initialize all Ray workers',
    action: () => console.log('[SHORTCUT] /START TRIGGERED'),
    category: 'system',
    enabled: true,
  },
  {
    id: 'kill-bot',
    keys: ['Ctrl', 'Shift', 'K'],
    description: '/KILL: Emergency stop all operations',
    action: () => console.log('[SHORTCUT] /KILL TRIGGERED'),
    category: 'system',
    enabled: true,
  },
  {
    id: 'toggle-strategy',
    keys: ['Ctrl', 'Shift', 'T'],
    description: 'Hot-swap active strategy',
    action: () => console.log('[SHORTCUT] STRATEGY SWAP TRIGGERED'),
    category: 'strategy',
    enabled: true,
  },
  {
    id: 'command-palette',
    keys: ['Meta', 'k'],
    description: 'Open command palette',
    action: () => useCommandStore.getState().toggle(),
    category: 'navigation',
    enabled: true,
  },
  {
    id: 'quick-buy',
    keys: ['Alt', 'B'],
    description: 'Quick buy order panel',
    action: () => console.log('[SHORTCUT] QUICK BUY TRIGGERED'),
    category: 'trading',
    enabled: true,
  },
  {
    id: 'quick-sell',
    keys: ['Alt', 'S'],
    description: 'Quick sell order panel',
    action: () => console.log('[SHORTCUT] QUICK SELL TRIGGERED'),
    category: 'trading',
    enabled: true,
  },
  {
    id: 'toggle-pip',
    keys: ['Ctrl', 'Shift', 'D'],
    description: 'Toggle Picture-in-Picture display',
    action: () => console.log('[SHORTCUT] PiP TOGGLE TRIGGERED'),
    category: 'navigation',
    enabled: true,
  },
  {
    id: 'generate-report',
    keys: ['Ctrl', 'Shift', 'R'],
    description: 'Generate daily PDF report',
    action: () => console.log('[SHORTCUT] REPORT GENERATION TRIGGERED'),
    category: 'system',
    enabled: true,
  },
  {
    id: 'refresh-data',
    keys: ['F5'],
    description: 'Force refresh market data',
    action: () => console.log('[SHORTCUT] DATA REFRESH TRIGGERED'),
    category: 'system',
    enabled: true,
  },
];

/**
 * Normalize key for comparison across different keyboard layouts
 */
function normalizeKey(key: string): string {
  const keyMap: Record<string, string> = {
    ' ': 'Space',
    'Control': 'Ctrl',
    'Meta': 'Cmd',
    'Escape': 'Esc',
    'ArrowUp': '↑',
    'ArrowDown': '↓',
    'ArrowLeft': '←',
    'ArrowRight': '→',
  };
  
  return keyMap[key] || key;
}

/**
 * Parse shortcut string into normalized key array
 */
function parseShortcut(shortcut: string): string[] {
  return shortcut.split('+').map((k) => normalizeKey(k.trim()));
}

/**
 * Check if current pressed keys match the shortcut
 */
function matchesShortcut(
  event: KeyboardEvent,
  shortcutKeys: string[]
): boolean {
  const pressedKeys = new Set<string>();
  
  if (event.ctrlKey) pressedKeys.add('Ctrl');
  if (event.metaKey) pressedKeys.add('Cmd');
  if (event.altKey) pressedKeys.add('Alt');
  if (event.shiftKey) pressedKeys.add('Shift');
  
  const mainKey = normalizeKey(event.key);
  pressedKeys.add(mainKey);
  
  // Check if all shortcut keys are pressed
  const shortcutSet = new Set(shortcutKeys.map((k) => k.toLowerCase()));
  
  for (const key of pressedKeys) {
    if (!shortcutSet.has(key.toLowerCase())) {
      return false;
    }
  }
  
  // Ensure same number of keys
  return pressedKeys.size === shortcutKeys.length;
}

/**
 * ShortcutManager - Global hotkey system for trading bot
 */
export const ShortcutManager: React.FC<ShortcutManagerProps> = ({
  shortcuts = DEFAULT_SHORTCUTS,
  onShortcutTriggered,
  debugMode = false,
}) => {
  const activeKeys = useRef<Set<string>>(new Set());
  const shortcutMap = useRef<Map<string, ShortcutConfig>>(new Map());
  const lastTriggerTime = useRef<Map<string, number>>(new Map());
  const cooldownMs = 300; // Prevent rapid re-triggering

  // Build shortcut map
  useEffect(() => {
    shortcutMap.current.clear();
    
    shortcuts.forEach((shortcut) => {
      if (shortcut.enabled !== false) {
        const key = shortcut.keys.join('+').toLowerCase();
        shortcutMap.current.set(key, shortcut);
      }
    });

    if (debugMode) {
      console.log('[ShortcutManager] Registered shortcuts:', shortcutMap.current.size);
    }
  }, [shortcuts, debugMode]);

  // Handle key down events
  const handleKeyDown = useCallback(
    (event: KeyboardEvent): void => {
      // Ignore when typing in input fields
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      activeKeys.current.add(event.key);

      // Check all registered shortcuts
      shortcutMap.current.forEach((shortcut, shortcutKey) => {
        if (matchesShortcut(event, shortcut.keys)) {
          const now = Date.now();
          const lastTime = lastTriggerTime.current.get(shortcut.id) || 0;

          // Cooldown check
          if (now - lastTime < cooldownMs) {
            return;
          }

          if (debugMode) {
            console.log(
              `[ShortcutManager] Triggered: ${shortcut.description}`
            );
          }

          lastTriggerTime.current.set(shortcut.id, now);
          
          try {
            shortcut.action();
            onShortcutTriggered?.(shortcut);
            
            // Prevent default browser behavior for certain shortcuts
            if (
              shortcut.category === 'trading' ||
              shortcut.category === 'system'
            ) {
              event.preventDefault();
              event.stopPropagation();
            }
          } catch (error) {
            console.error(
              `[ShortcutManager] Error executing ${shortcut.id}:`,
              error
            );
          }
        }
      });
    },
    [onShortcutTriggered, debugMode]
  );

  // Handle key up events
  const handleKeyUp = useCallback((event: KeyboardEvent): void => {
    activeKeys.current.delete(event.key);
  }, []);

  // Handle window blur (clear active keys)
  const handleBlur = useCallback((): void => {
    activeKeys.current.clear();
  }, []);

  // Register global event listeners
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown, { capture: true });
    window.addEventListener('keyup', handleKeyUp, { capture: true });
    window.addEventListener('blur', handleBlur);

    return () => {
      window.removeEventListener('keydown', handleKeyDown, { capture: true });
      window.removeEventListener('keyup', handleKeyUp, { capture: true });
      window.removeEventListener('blur', handleBlur);
    };
  }, [handleKeyDown, handleKeyUp, handleBlur]);

  // Expose API for dynamic shortcut registration
  useEffect(() => {
    const api = {
      register: (shortcut: ShortcutConfig): void => {
        if (shortcut.enabled !== false) {
          const key = shortcut.keys.join('+').toLowerCase();
          shortcutMap.current.set(key, shortcut);
        }
      },
      unregister: (id: string): void => {
        shortcutMap.current.forEach((shortcut, key) => {
          if (shortcut.id === id) {
            shortcutMap.current.delete(key);
          }
        });
      },
      enable: (id: string): void => {
        shortcutMap.current.forEach((shortcut, key) => {
          if (shortcut.id === id) {
            shortcutMap.current.set(key, { ...shortcut, enabled: true });
          }
        });
      },
      disable: (id: string): void => {
        shortcutMap.current.forEach((shortcut, key) => {
          if (shortcut.id === id) {
            shortcutMap.current.set(key, { ...shortcut, enabled: false });
          }
        });
      },
      trigger: (id: string): void => {
        shortcutMap.current.forEach((shortcut) => {
          if (shortcut.id === id) {
            shortcut.action();
          }
        });
      },
    };

    // Store API reference globally for external access
    (window as unknown as Record<string, unknown>).shortcutManagerApi = api;

    return () => {
      delete (window as unknown as Record<string, unknown>).shortcutManagerApi;
    };
  }, []);

  // Debug overlay (optional)
  if (debugMode) {
    return (
      <div
        className="fixed bottom-4 right-4 p-3 bg-[#0a0a1a]/90 border border-cyan-800 rounded-lg text-xs font-mono text-cyan-400 z-[100]"
        style={{
          boxShadow: '0 0 20px rgba(0, 243, 255, 0.2)',
        }}
      >
        <p className="font-bold mb-2">Shortcut Debugger</p>
        <p>Active Keys: {Array.from(activeKeys.current).join(' + ') || 'None'}</p>
        <p>Registered: {shortcutMap.current.size}</p>
      </div>
    );
  }

  return null; // Hidden component - only handles logic
};

/**
 * Hook to use shortcut manager in other components
 */
export function useShortcutManager() {
  const apiRef = React.useRef<Record<string, unknown> | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      apiRef.current = (window as unknown as Record<string, unknown>)
        .shortcutManagerApi as Record<string, unknown> | null;
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return {
    register: (shortcut: ShortcutConfig): void => {
      apiRef.current?.register?.(shortcut);
    },
    unregister: (id: string): void => {
      apiRef.current?.unregister?.(id);
    },
    enable: (id: string): void => {
      apiRef.current?.enable?.(id);
    },
    disable: (id: string): void => {
      apiRef.current?.disable?.(id);
    },
    trigger: (id: string): void => {
      apiRef.current?.trigger?.(id);
    },
  };
}

export default ShortcutManager;
