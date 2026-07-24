/**
 * ServiceWorkerPush - Background Service Worker Push Handler
 * Renders custom cyberpunk-styled notification payloads with instant action buttons
 * Handles push events even when the main application is not active
 */

// This file should be imported and used within the service worker context
// It provides notification rendering utilities for push events

export interface CyberpunkNotificationPayload {
  title: string;
  body: string;
  type?: 'alert' | 'warning' | 'info' | 'circuit-breaker' | 'trade';
  severity?: 'low' | 'medium' | 'high' | 'critical';
  icon?: string;
  badge?: string;
  image?: string;
  actions?: Array<{
    action: string;
    title: string;
    icon?: string;
  }>;
  data?: {
    url?: string;
    timestamp?: number;
    pair?: string;
    side?: 'BUY' | 'SELL';
    price?: number;
    amount?: number;
    pnl?: number;
    reason?: string;
  };
  vibrate?: number[];
  silent?: boolean;
  tag?: string;
  renotify?: boolean;
}

/**
 * Color schemes for different notification types (cyberpunk aesthetic)
 */
const NOTIFICATION_COLORS: Record<string, { primary: string; secondary: string }> = {
  'circuit-breaker': { primary: '#ff0055', secondary: '#ff66aa' },
  alert: { primary: '#ff3300', secondary: '#ff8866' },
  warning: { primary: '#ffaa00', secondary: '#ffdd66' },
  trade: { primary: '#00f3ff', secondary: '#66ffff' },
  info: { primary: '#0066ff', secondary: '#66aaff' },
};

/**
 * Format a cyberpunk-styled notification body with rich text
 */
export function formatNotificationBody(payload: CyberpunkNotificationPayload): string {
  let body = payload.body;

  // Add timestamp if available
  if (payload.data?.timestamp) {
    const time = new Date(payload.data.timestamp).toLocaleTimeString();
    body += `\n⏰ ${time}`;
  }

  // Add trading details if available
  if (payload.type === 'trade' && payload.data) {
    const { pair, side, price, amount } = payload.data;
    if (pair && side && price !== undefined && amount !== undefined) {
      body += `\n━━━━━━━━━━━━━━━━━━━━\n`;
      body += `${side} ${amount} @ $${price.toLocaleString()}\n`;
      body += `━━━━━━━━━━━━━━━━━━━━`;
    }
  }

  // Add PnL details if available
  if (payload.data?.pnl !== undefined) {
    const pnl = payload.data.pnl;
    const sign = pnl >= 0 ? '+' : '';
    const emoji = pnl >= 0 ? '📈' : '📉';
    body += `\n${emoji} PnL: ${sign}$${pnl.toLocaleString()}`;
  }

  // Add circuit breaker reason
  if (payload.type === 'circuit-breaker' && payload.data?.reason) {
    body += `\n⚠️ Reason: ${payload.data.reason}`;
  }

  return body;
}

/**
 * Get vibration pattern based on severity
 */
export function getVibrationPattern(
  severity: CyberpunkNotificationPayload['severity']
): number[] {
  switch (severity) {
    case 'critical':
      return [200, 100, 200, 100, 400]; // Long urgent pattern
    case 'high':
      return [150, 80, 150, 80, 200];
    case 'medium':
      return [100, 50, 100];
    case 'low':
    default:
      return [50]; // Short subtle pulse
  }
}

/**
 * Create notification options from payload
 */
export function createNotificationOptions(
  payload: CyberpunkNotificationPayload
): NotificationOptions {
  const colors =
    NOTIFICATION_COLORS[payload.type || 'info'] || NOTIFICATION_COLORS.info;

  const defaultActions = [
    { action: 'view', title: '📊 View Dashboard', icon: '/icons/view-icon.png' },
    { action: 'dismiss', title: '✕ Dismiss', icon: '/icons/dismiss-icon.png' },
  ];

  // Add type-specific actions
  let actions = defaultActions;
  if (payload.type === 'circuit-breaker') {
    actions = [
      { action: 'resume', title: '▶ Resume Trading', icon: '/icons/resume-icon.png' },
      { action: 'details', title: '📋 Details', icon: '/icons/details-icon.png' },
      ...defaultActions,
    ];
  } else if (payload.type === 'trade') {
    actions = [
      { action: 'undo', title: '↩ Undo', icon: '/icons/undo-icon.png' },
      ...defaultActions,
    ];
  }

  return {
    body: formatNotificationBody(payload),
    icon: payload.icon || '/icons/icon-192x192.png',
    badge: payload.badge || '/icons/icon-96x96.png',
    image: payload.image,
    tag: payload.tag || `nautilus-${Date.now()}`,
    requireInteraction: true,
    silent: payload.silent ?? false,
    vibrate: payload.vibrate || getVibrationPattern(payload.severity),
    renotify: payload.renotify ?? true,
    actions: payload.actions || actions,
    data: {
      url: payload.data?.url || '/dashboard',
      type: payload.type || 'info',
      severity: payload.severity || 'medium',
      timestamp: payload.data?.timestamp || Date.now(),
      ...payload.data,
    },
    // Custom styling hints (for browsers that support it)
    // Note: Full customization requires notification builder apps
    silent: false,
  };
}

/**
 * Handle notification click events
 */
export async function handleNotificationClick(
  event: NotificationEvent
): Promise<void> {
  const notification = event.notification;
  const data = notification.data as { url?: string; type?: string } | null;

  event.waitUntil(
    (async () => {
      // Close the notification
      notification.close();

      // Handle specific actions
      switch (event.action) {
        case 'dismiss':
          // Just close, no further action
          return;

        case 'view':
        case 'details':
          await openDashboard(data?.url || '/dashboard');
          break;

        case 'resume':
          await resumeTrading();
          await openDashboard('/alerts/circuit-breaker');
          break;

        case 'undo':
          await undoLastTrade();
          await openDashboard('/trades');
          break;

        default:
          await openDashboard(data?.url || '/dashboard');
      }
    })()
  );
}

/**
 * Open dashboard URL in existing or new window
 */
async function openDashboard(url: string): Promise<void> {
  if (typeof self === 'undefined') return;

  const clients = await (self as unknown as ServiceWorkerGlobalScope).clients.matchAll({
    type: 'window',
    includeUncontrolled: true,
  });

  // Try to focus an existing client with matching URL
  for (const client of clients) {
    if (client.url.includes(url) && 'focus' in client) {
      await client.focus();
      return;
    }
  }

  // Open new window if no matching client found
  if ((self as unknown as ServiceWorkerGlobalScope).clients.openWindow) {
    await (self as unknown as ServiceWorkerGlobalScope).clients.openWindow(url);
  }
}

/**
 * Resume trading after circuit breaker
 */
async function resumeTrading(): Promise<void> {
  // Send message to all clients to resume trading
  const clients = await (self as unknown as ServiceWorkerGlobalScope).clients.matchAll({
    type: 'window',
    includeUncontrolled: true,
  });

  clients.forEach((client) => {
    client.postMessage({
      type: 'RESUME_TRADING',
      timestamp: Date.now(),
    });
  });
}

/**
 * Undo last trade
 */
async function undoLastTrade(): Promise<void> {
  // Send message to all clients to undo last trade
  const clients = await (self as unknown as ServiceWorkerGlobalScope).clients.matchAll({
    type: 'window',
    includeUncontrolled: true,
  });

  clients.forEach((client) => {
    client.postMessage({
      type: 'UNDO_LAST_TRADE',
      timestamp: Date.now(),
    });
  });
}

/**
 * Register push notification handler for service worker
 * This should be called in your service worker file
 */
export function registerPushHandler(): void {
  if (typeof self === 'undefined') return;

  const swSelf = self as unknown as ServiceWorkerGlobalScope;

  // Push event handler
  swSelf.addEventListener('push', (event: PushEvent) => {
    let payload: CyberpunkNotificationPayload;

    try {
      const data = event.data?.json();
      payload = {
        title: data?.title || 'Nautilus Ray Alert',
        body: data?.body || 'System notification',
        ...data,
      };
    } catch {
      payload = {
        title: 'Nautilus Ray Alert',
        body: 'Critical system alert',
        type: 'alert',
        severity: 'high',
      };
    }

    const options = createNotificationOptions(payload);

    event.waitUntil(
      swSelf.registration.showNotification(payload.title, options)
    );
  });

  // Notification click handler
  swSelf.addEventListener('notificationclick', handleNotificationClick);

  // Notification close handler (optional logging)
  swSelf.addEventListener('notificationclose', (event: NotificationEvent) => {
    console.log('[ServiceWorkerPush] Notification closed:', event.notification.tag);
  });
}

/**
 * Send a local notification from the service worker
 * Useful for scheduled alerts or background tasks
 */
export async function showLocalNotification(
  payload: CyberpunkNotificationPayload
): Promise<void> {
  if (typeof self === 'undefined') return;

  const swSelf = self as unknown as ServiceWorkerGlobalScope;
  const options = createNotificationOptions(payload);

  await swSelf.registration.showNotification(payload.title, options);
}

/**
 * Schedule a notification for later delivery
 * Uses the Notification API's tag feature for replacement
 */
export async function scheduleNotification(
  payload: CyberpunkNotificationPayload,
  delayMs: number
): Promise<void> {
  if (typeof self === 'undefined') return;

  const swSelf = self as unknown as ServiceWorkerGlobalScope;

  // Store notification in IndexedDB for later retrieval
  const db = await openNotificationQueueDB();
  await addScheduledNotification(db, {
    ...payload,
    scheduledTime: Date.now() + delayMs,
  });

  // Set up periodic check for due notifications
  await swSelf.registration.sync.register('check-scheduled-notifications');
}

// IndexedDB helpers for scheduled notifications
interface ScheduledNotification extends CyberpunkNotificationPayload {
  scheduledTime: number;
  id?: string;
}

function openNotificationQueueDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('NautilusNotificationQueue', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('scheduledNotifications')) {
        db.createObjectStore('scheduledNotifications', { keyPath: 'id' });
      }
    };
  });
}

async function addScheduledNotification(
  db: IDBDatabase,
  notification: ScheduledNotification
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('scheduledNotifications', 'readwrite');
    const store = tx.objectStore('scheduledNotifications');
    const request = store.put({
      ...notification,
      id: `notif-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    });

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Check and deliver any due scheduled notifications
 */
export async function checkScheduledNotifications(): Promise<void> {
  if (typeof self === 'undefined') return;

  const db = await openNotificationQueueDB();
  const now = Date.now();

  const dueNotifications = await getDueNotifications(db, now);

  for (const notification of dueNotifications) {
    await showLocalNotification(notification);
    await removeScheduledNotification(db, notification.id!);
  }
}

async function getDueNotifications(
  db: IDBDatabase,
  cutoffTime: number
): Promise<ScheduledNotification[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('scheduledNotifications', 'readonly');
    const store = tx.objectStore('scheduledNotifications');
    const request = store.getAll();

    request.onsuccess = () => {
      const notifications = request.result as ScheduledNotification[];
      resolve(notifications.filter((n) => n.scheduledTime <= cutoffTime));
    };
    request.onerror = () => reject(request.error);
  });
}

async function removeScheduledNotification(
  db: IDBDatabase,
  id: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('scheduledNotifications', 'readwrite');
    const store = tx.objectStore('scheduledNotifications');
    const request = store.delete(id);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Export for service worker usage
export default {
  registerPushHandler,
  showLocalNotification,
  scheduleNotification,
  checkScheduledNotifications,
  handleNotificationClick,
  createNotificationOptions,
};
