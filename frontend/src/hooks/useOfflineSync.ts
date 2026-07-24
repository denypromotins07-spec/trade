/**
 * useOfflineSync Hook
 * Manages IndexedDB storage for queued execution intents
 * Automatically flushes commands to WebSocket gateway on connectivity restore
 * Optimized for 60FPS with minimal main thread blocking
 */

import { useState, useEffect, useCallback, useRef } from 'react';

export interface QueuedCommand {
  id?: number;
  type: string;
  payload: Record<string, unknown>;
  timestamp: number;
  retryCount: number;
  status: 'pending' | 'executing' | 'completed' | 'failed';
}

export interface OfflineSyncState {
  isOnline: boolean;
  queueCount: number;
  isFlushing: boolean;
  lastSyncTime: number | null;
  error: string | null;
}

const DB_NAME = 'NautilusOfflineQueue';
const DB_VERSION = 1;
const STORE_NAME = 'execution-intents';
const MAX_QUEUE_SIZE = 100;

/**
 * Custom hook for managing offline command synchronization
 */
export function useOfflineSync(
  wsGatewayUrl: string,
  options: {
    autoFlush?: boolean;
    maxRetries?: number;
    onFlushComplete?: () => void;
    onCommandQueued?: (command: QueuedCommand) => void;
  } = {}
): OfflineSyncState & {
  queueCommand: (type: string, payload: Record<string, unknown>) => Promise<void>;
  flushQueue: () => Promise<void>;
  clearQueue: () => Promise<void>;
  getQueueStatus: () => Promise<{ count: number; oldestTimestamp: number | null }>;
} {
  const [state, setState] = useState<OfflineSyncState>({
    isOnline: navigator.onLine,
    queueCount: 0,
    isFlushing: false,
    lastSyncTime: null,
    error: null,
  });

  const dbRef = useRef<IDBDatabase | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const flushInProgress = useRef(false);
  const { autoFlush = true, maxRetries = 3, onFlushComplete, onCommandQueued } = options;

  // Initialize IndexedDB connection
  useEffect(() => {
    let mounted = true;

    const initDB = async (): Promise<void> => {
      try {
        const db = await openDB();
        if (mounted) {
          dbRef.current = db;
          await updateQueueCount(db);
        }
      } catch (error) {
        console.error('[useOfflineSync] Failed to initialize IndexedDB:', error);
        if (mounted) {
          setState((prev) => ({ ...prev, error: 'IndexedDB initialization failed' }));
        }
      }
    };

    initDB();

    return () => {
      mounted = false;
      if (dbRef.current) {
        dbRef.current.close();
      }
    };
  }, []);

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = async (): Promise<void> => {
      setState((prev) => ({ ...prev, isOnline: true, error: null }));
      
      if (autoFlush && !flushInProgress.current) {
        await flushQueueInternal();
      }
    };

    const handleOffline = (): void => {
      setState((prev) => ({ ...prev, isOnline: false }));
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [autoFlush]);

  // Listen for service worker messages about executed commands
  useEffect(() => {
    const handleMessage = (event: MessageEvent): void => {
      if (event.data?.type === 'COMMAND_EXECUTED') {
        setState((prev) => ({
          ...prev,
          lastSyncTime: Date.now(),
        }));
        updateQueueCount(dbRef.current);
      }
    };

    navigator.serviceWorker?.addEventListener('message', handleMessage);

    return () => {
      navigator.serviceWorker?.removeEventListener('message', handleMessage);
    };
  }, []);

  /**
   * Queue a command for later execution
   */
  const queueCommand = useCallback(
    async (type: string, payload: Record<string, unknown>): Promise<void> => {
      const db = dbRef.current;
      if (!db) {
        throw new Error('IndexedDB not initialized');
      }

      const command: Omit<QueuedCommand, 'id'> = {
        type,
        payload,
        timestamp: Date.now(),
        retryCount: 0,
        status: 'pending',
      };

      try {
        const storedCommand = await addCommandToStore(db, command);
        
        // Enforce queue size limit
        await enforceQueueLimit(db);
        
        // Update UI state
        await updateQueueCount(db);
        
        // Callback notification
        onCommandQueued?.(storedCommand as QueuedCommand);

        // Auto-flush if online
        if (state.isOnline && autoFlush && !flushInProgress.current) {
          await flushQueueInternal();
        }
      } catch (error) {
        console.error('[useOfflineSync] Failed to queue command:', error);
        setState((prev) => ({ ...prev, error: 'Failed to queue command' }));
        throw error;
      }
    },
    [state.isOnline, autoFlush, onCommandQueued]
  );

  /**
   * Flush all queued commands to the WebSocket gateway
   */
  const flushQueue = useCallback(async (): Promise<void> => {
    await flushQueueInternal();
  }, []);

  /**
   * Clear all queued commands
   */
  const clearQueue = useCallback(async (): Promise<void> => {
    const db = dbRef.current;
    if (!db) return;

    try {
      await clearAllCommands(db);
      await updateQueueCount(db);
    } catch (error) {
      console.error('[useOfflineSync] Failed to clear queue:', error);
      throw error;
    }
  }, []);

  /**
   * Get current queue status
   */
  const getQueueStatus = useCallback(async (): Promise<{
    count: number;
    oldestTimestamp: number | null;
  }> => {
    const db = dbRef.current;
    if (!db) return { count: 0, oldestTimestamp: null };

    return getQueueMetrics(db);
  }, []);

  /**
   * Internal flush implementation
   */
  const flushQueueInternal = async (): Promise<void> => {
    const db = dbRef.current;
    if (!db || flushInProgress.current || !state.isOnline) return;

    flushInProgress.current = true;
    setState((prev) => ({ ...prev, isFlushing: true, error: null }));

    try {
      const commands = await getPendingCommands(db);
      let successCount = 0;

      for (const command of commands) {
        try {
          // Update status to executing
          await updateCommandStatus(db, command.id!, 'executing');

          // Send to WebSocket gateway
          const success = await sendToWebSocket(command);

          if (success) {
            await updateCommandStatus(db, command.id!, 'completed');
            successCount++;
          } else {
            // Increment retry count
            if (command.retryCount >= maxRetries) {
              await updateCommandStatus(db, command.id!, 'failed');
            } else {
              await incrementRetryCount(db, command.id!);
            }
          }
        } catch (error) {
          console.error('[useOfflineSync] Command execution failed:', command.id, error);
          if (command.retryCount >= maxRetries) {
            await updateCommandStatus(db, command.id!, 'failed');
          } else {
            await incrementRetryCount(db, command.id!);
          }
        }
      }

      setState((prev) => ({
        ...prev,
        lastSyncTime: Date.now(),
        isFlushing: false,
      }));

      await updateQueueCount(db);

      if (successCount > 0 && onFlushComplete) {
        onFlushComplete();
      }
    } catch (error) {
      console.error('[useOfflineSync] Flush failed:', error);
      setState((prev) => ({
        ...prev,
        isFlushing: false,
        error: 'Flush operation failed',
      }));
    } finally {
      flushInProgress.current = false;
    }
  };

  /**
   * Send command to WebSocket gateway
   */
  const sendToWebSocket = async (command: QueuedCommand): Promise<boolean> => {
    return new Promise((resolve) => {
      try {
        // Create temporary WebSocket for command execution
        const ws = new WebSocket(wsGatewayUrl);
        let resolved = false;

        const timeout = setTimeout(() => {
          if (!resolved) {
            ws.close();
            resolved = true;
            resolve(false);
          }
        }, 5000); // 5 second timeout

        ws.onopen = () => {
          const message = JSON.stringify({
            type: command.type,
            payload: command.payload,
            timestamp: command.timestamp,
          });
          ws.send(message);
        };

        ws.onmessage = (event) => {
          clearTimeout(timeout);
          const response = JSON.parse(event.data);
          if (response.status === 'ok' || response.acknowledged) {
            ws.close();
            resolved = true;
            resolve(true);
          } else {
            ws.close();
            resolved = true;
            resolve(false);
          }
        };

        ws.onerror = () => {
          clearTimeout(timeout);
          ws.close();
          if (!resolved) {
            resolved = true;
            resolve(false);
          }
        };

        ws.onclose = () => {
          if (!resolved) {
            resolved = true;
            resolve(false);
          }
        };
      } catch (error) {
        console.error('[useOfflineSync] WebSocket send failed:', error);
        resolve(false);
      }
    });
  };

  return {
    ...state,
    queueCommand,
    flushQueue,
    clearQueue,
    getQueueStatus,
  };
}

// IndexedDB helper functions

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, {
          keyPath: 'id',
          autoIncrement: true,
        });
        store.createIndex('status', 'status', { unique: false });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('retryCount', 'retryCount', { unique: false });
      }
    };
  });
}

async function addCommandToStore(
  db: IDBDatabase,
  command: Omit<QueuedCommand, 'id'>
): Promise<QueuedCommand> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.add(command);

    request.onsuccess = () => {
      resolve({ ...command, id: request.result as number });
    };
    request.onerror = () => reject(request.error);
  });
}

async function getPendingCommands(db: IDBDatabase): Promise<QueuedCommand[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('timestamp');
    const request = index.getAll();

    request.onsuccess = () => {
      const commands = request.result as QueuedCommand[];
      resolve(commands.filter((c) => c.status === 'pending'));
    };
    request.onerror = () => reject(request.error);
  });
}

async function updateCommandStatus(
  db: IDBDatabase,
  id: number,
  status: QueuedCommand['status']
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const getRequest = store.get(id);

    getRequest.onsuccess = () => {
      const command = getRequest.result as QueuedCommand;
      if (command) {
        command.status = status;
        store.put(command);
      }
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    };

    getRequest.onerror = () => reject(getRequest.error);
  });
}

async function incrementRetryCount(db: IDBDatabase, id: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const getRequest = store.get(id);

    getRequest.onsuccess = () => {
      const command = getRequest.result as QueuedCommand;
      if (command) {
        command.retryCount++;
        store.put(command);
      }
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    };

    getRequest.onerror = () => reject(getRequest.error);
  });
}

async function updateQueueCount(db: IDBDatabase | null): Promise<void> {
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.count();

    request.onsuccess = () => {
      // Use functional update to avoid stale closure
      const countElement = document.querySelector('[data-queue-count]');
      if (countElement) {
        countElement.textContent = request.result.toString();
      }
      resolve();
    };
    request.onerror = () => resolve(); // Don't fail on count error
  });
}

async function clearAllCommands(db: IDBDatabase): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.clear();

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

async function getQueueMetrics(
  db: IDBDatabase
): Promise<{ count: number; oldestTimestamp: number | null }> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const countRequest = store.count();
    const index = store.index('timestamp');
    const oldestRequest = index.openCursor();

    let oldestTimestamp: number | null = null;

    oldestRequest.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor) {
        oldestTimestamp = cursor.value.timestamp;
      }
    };

    countRequest.onsuccess = () => {
      resolve({ count: countRequest.result, oldestTimestamp });
    };

    countRequest.onerror = () => reject(countRequest.error);
  });
}

async function enforceQueueLimit(db: IDBDatabase): Promise<void> {
  const metrics = await getQueueMetrics(db);
  
  if (metrics.count > MAX_QUEUE_SIZE) {
    const toDelete = metrics.count - MAX_QUEUE_SIZE;
    await deleteOldestCommands(db, toDelete);
  }
}

async function deleteOldestCommands(db: IDBDatabase, count: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('timestamp');
    const request = index.openCursor();

    let deleted = 0;

    request.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor && deleted < count) {
        store.delete(cursor.primaryKey);
        deleted++;
        cursor.continue();
      } else {
        resolve();
      }
    };

    request.onerror = () => reject(request.error);
  });
}

export default useOfflineSync;
