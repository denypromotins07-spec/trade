/**
 * Service Worker for Nautilus Ray Trading Bot
 * Implements offline caching strategy with fallback queue for manual override commands
 * Optimized for cyberpunk aesthetic PWA with strict storage quota enforcement
 */

const CACHE_NAME = 'nautilus-ray-v1';
const OFFLINE_QUEUE_KEY = 'offline-commands-queue';
const MAX_QUEUE_SIZE = 100;
const STORAGE_QUOTA_BYTES = 50 * 1024 * 1024; // 50MB limit for safety

// Static assets to precache for offline functionality
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
];

// Install event - precache static assets
self.addEventListener('install', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate event - clean old caches and enforce storage quotas
self.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(
    Promise.all([
      caches.keys().then((keys) => {
        return Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        );
      }),
      enforceStorageQuota(),
    ])
  );
  self.clients.claim();
});

/**
 * Enforce browser storage quota by清理 oldest entries if exceeding limit
 */
async function enforceStorageQuota(): Promise<void> {
  try {
    const db = await openOfflineQueueDB();
    const count = await countQueuedCommands(db);
    
    if (count > MAX_QUEUE_SIZE) {
      const toDelete = count - MAX_QUEUE_SIZE;
      await deleteOldestCommands(db, toDelete);
    }
  } catch (error) {
    console.error('[SW] Storage quota enforcement failed:', error);
  }
}

// Fetch event - network first with cache fallback
self.addEventListener('fetch', (event: FetchEvent) => {
  const { request } = event;
  
  // Handle WebSocket upgrade requests
  if (request.url.startsWith('ws://') || request.url.startsWith('wss://')) {
    return; // Let WebSocket connections pass through
  }
  
  // Handle API requests with offline queue fallback
  if (request.url.includes('/api/')) {
    event.respondWith(handleApiRequest(request));
    return;
  }
  
  // Standard cache-first for static assets
  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(request).then((networkResponse) => {
        // Cache successful responses
        if (networkResponse.ok) {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(request, responseClone);
          });
        }
        return networkResponse;
      });
    })
  );
});

/**
 * Handle API requests with offline fallback queue
 * Queues commands when backend is unavailable
 */
async function handleApiRequest(request: Request): Promise<Response> {
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    // Network error - queue the request if it's a mutation
    if (request.method !== 'GET') {
      await queueOfflineCommand(request);
    }
    
    // Return offline fallback response
    return new Response(
      JSON.stringify({
        status: 'queued',
        message: 'Backend unavailable. Command queued for execution.',
        timestamp: Date.now(),
      }),
      {
        status: 202,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

/**
 * Queue offline command to IndexedDB
 */
async function queueOfflineCommand(request: Request): Promise<void> {
  const db = await openOfflineQueueDB();
  
  const commandData = {
    url: request.url,
    method: request.method,
    body: await request.clone().text(),
    headers: Object.fromEntries(request.headers.entries()),
    timestamp: Date.now(),
    retryCount: 0,
  };
  
  await addCommandToQueue(db, commandData);
}

// Message handler for flushing queued commands
self.addEventListener('message', (event: ExtendableMessageEvent) => {
  if (event.data && event.data.type === 'FLUSH_OFFLINE_QUEUE') {
    event.waitUntil(flushOfflineQueue());
  }
  
  if (event.data && event.data.type === 'GET_QUEUE_STATUS') {
    event.waitUntil(
      getQueueStatus().then((status) => {
        event.source?.postMessage({
          type: 'QUEUE_STATUS',
          payload: status,
        });
      })
    );
  }
});

/**
 * Flush all queued commands to the backend when connectivity restores
 */
async function flushOfflineQueue(): Promise<void> {
  const db = await openOfflineQueueDB();
  const commands = await getAllQueuedCommands(db);
  
  for (const command of commands) {
    try {
      const response = await fetch(command.url, {
        method: command.method,
        headers: command.headers as HeadersInit,
        body: command.body,
      });
      
      if (response.ok) {
        await removeCommandFromQueue(db, command.id);
        
        // Notify client of successful execution
        self.clients.matchAll().then((clients) => {
          clients.forEach((client) => {
            client.postMessage({
              type: 'COMMAND_EXECUTED',
              payload: { id: command.id, url: command.url },
            });
          });
        });
      }
    } catch (error) {
      console.error('[SW] Failed to flush command:', command.id, error);
      // Increment retry count
      await incrementRetryCount(db, command.id);
    }
  }
}

// IndexedDB helpers for offline queue
interface OfflineCommand {
  id?: number;
  url: string;
  method: string;
  body: string;
  headers: Record<string, string>;
  timestamp: number;
  retryCount: number;
}

function openOfflineQueueDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('NautilusOfflineQueue', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      
      if (!db.objectStoreNames.contains(OFFLINE_QUEUE_KEY)) {
        const store = db.createObjectStore(OFFLINE_QUEUE_KEY, {
          keyPath: 'id',
          autoIncrement: true,
        });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('retryCount', 'retryCount', { unique: false });
      }
    };
  });
}

async function addCommandToQueue(
  db: IDBDatabase,
  command: Omit<OfflineCommand, 'id'>
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(OFFLINE_QUEUE_KEY, 'readwrite');
    const store = tx.objectStore(OFFLINE_QUEUE_KEY);
    store.add(command);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function getAllQueuedCommands(db: IDBDatabase): Promise<OfflineCommand[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(OFFLINE_QUEUE_KEY, 'readonly');
    const store = tx.objectStore(OFFLINE_QUEUE_KEY);
    const index = store.index('timestamp');
    const request = index.getAll();
    
    request.onsuccess = () => resolve(request.result as OfflineCommand[]);
    request.onerror = () => reject(request.error);
  });
}

async function removeCommandFromQueue(
  db: IDBDatabase,
  id: number
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(OFFLINE_QUEUE_KEY, 'readwrite');
    const store = tx.objectStore(OFFLINE_QUEUE_KEY);
    store.delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function countQueuedCommands(db: IDBDatabase): Promise<number> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(OFFLINE_QUEUE_KEY, 'readonly');
    const store = tx.objectStore(OFFLINE_QUEUE_KEY);
    const request = store.count();
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function deleteOldestCommands(
  db: IDBDatabase,
  count: number
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(OFFLINE_QUEUE_KEY, 'readwrite');
    const store = tx.objectStore(OFFLINE_QUEUE_KEY);
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

async function incrementRetryCount(
  db: IDBDatabase,
  id: number
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(OFFLINE_QUEUE_KEY, 'readwrite');
    const store = tx.objectStore(OFFLINE_QUEUE_KEY);
    const getRequest = store.get(id);
    
    getRequest.onsuccess = () => {
      const command = getRequest.result as OfflineCommand;
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

async function getQueueStatus(): Promise<{
  count: number;
  maxSize: number;
  oldestTimestamp: number | null;
}> {
  const db = await openOfflineQueueDB();
  const count = await countQueuedCommands(db);
  
  let oldestTimestamp: number | null = null;
  if (count > 0) {
    const commands = await getAllQueuedCommands(db);
    if (commands.length > 0) {
      oldestTimestamp = Math.min(...commands.map((c) => c.timestamp));
    }
  }
  
  return { count, maxSize: MAX_QUEUE_SIZE, oldestTimestamp };
}

// Background sync for periodic flush attempts
self.addEventListener('sync', (event: SyncEvent) => {
  if (event.tag === 'flush-offline-queue') {
    event.waitUntil(flushOfflineQueue());
  }
});

// Push notification handler
self.addEventListener('push', (event: PushEvent) => {
  const data = event.data?.json() ?? {};
  
  event.waitUntil(
    self.registration.showNotification(data.title || 'Nautilus Ray Alert', {
      body: data.body || 'Critical system alert',
      icon: '/icons/icon-192x192.png',
      badge: '/icons/icon-96x96.png',
      tag: data.tag || 'nautilus-alert',
      requireInteraction: true,
      actions: [
        { action: 'view', title: 'View Dashboard' },
        { action: 'dismiss', title: 'Dismiss' },
      ],
      data: {
        url: data.url || '/dashboard',
        type: data.type || 'alert',
      },
    })
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event: NotificationEvent) => {
  event.notification.close();
  
  if (event.action === 'dismiss') {
    return;
  }
  
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clients) => {
      const url = event.notification.data?.url || '/dashboard';
      
      for (const client of clients) {
        if (client.url.includes(url) && 'focus' in client) {
          return client.focus();
        }
      }
      
      if (self.clients.openWindow) {
        return self.clients.openWindow(url);
      }
    })
  );
});

export {};
