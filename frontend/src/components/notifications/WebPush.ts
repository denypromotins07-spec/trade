/**
 * WebPush - Web Push API Subscription Manager
 * Routes critical circuit breaker alerts to OS notification center
 * Works even when browser is minimized via service worker integration
 */

export interface PushSubscriptionOptions {
  vapidPublicKey: string;
  applicationServerKey?: string;
}

export interface PushNotificationPayload {
  title: string;
  body: string;
  icon?: string;
  badge?: string;
  tag?: string;
  requireInteraction?: boolean;
  actions?: Array<{ action: string; title: string }>;
  data?: {
    url?: string;
    type?: 'alert' | 'warning' | 'info' | 'circuit-breaker';
    timestamp?: number;
    severity?: 'low' | 'medium' | 'high' | 'critical';
  };
}

export interface WebPushState {
  isSupported: boolean;
  permission: NotificationPermission;
  subscription: PushSubscription | null;
  isSubscribing: boolean;
  error: string | null;
}

const DEFAULT_ICON = '/icons/icon-192x192.png';
const DEFAULT_BADGE = '/icons/icon-96x96.png';

/**
 * WebPush Manager Class for handling push notifications
 */
class WebPushManager {
  private state: WebPushState;
  private listeners: Set<(state: WebPushState) => void> = new Set();

  constructor() {
    this.state = {
      isSupported: this.checkSupport(),
      permission: typeof window !== 'undefined' ? Notification.permission : 'default',
      subscription: null,
      isSubscribing: false,
      error: null,
    };
  }

  /**
   * Check if Web Push is supported in current environment
   */
  private checkSupport(): boolean {
    if (typeof window === 'undefined') return false;
    return 'serviceWorker' in navigator && 'PushManager' in window;
  }

  /**
   * Get current state
   */
  getState(): WebPushState {
    return { ...this.state };
  }

  /**
   * Subscribe to state changes
   */
  subscribe(callback: (state: WebPushState) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  /**
   * Notify all listeners of state change
   */
  private notifyListeners(): void {
    this.listeners.forEach((listener) => listener({ ...this.state }));
  }

  /**
   * Request notification permission
   */
  async requestPermission(): Promise<NotificationPermission> {
    if (!this.state.isSupported) {
      this.state.error = 'Web Push not supported in this browser';
      this.notifyListeners();
      return 'denied';
    }

    try {
      const permission = await Notification.requestPermission();
      this.state.permission = permission;
      this.notifyListeners();
      return permission;
    } catch (error) {
      this.state.error = 'Failed to request permission';
      this.notifyListeners();
      return 'denied';
    }
  }

  /**
   * Subscribe to push notifications
   */
  async subscribeToPush(options: PushSubscriptionOptions): Promise<boolean> {
    if (!this.state.isSupported) {
      this.state.error = 'Web Push not supported';
      this.notifyListeners();
      return false;
    }

    if (this.state.permission !== 'granted') {
      const permission = await this.requestPermission();
      if (permission !== 'granted') {
        this.state.error = 'Notification permission denied';
        this.notifyListeners();
        return false;
      }
    }

    this.state.isSubscribing = true;
    this.notifyListeners();

    try {
      const registration = await navigator.serviceWorker.ready;

      // Convert VAPID key from base64 to Uint8Array
      const applicationServerKey = this.urlBase64ToUint8Array(options.vapidPublicKey);

      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey,
      });

      this.state.subscription = subscription;
      this.state.isSubscribing = false;
      this.state.error = null;
      this.notifyListeners();

      // Send subscription to your backend
      await this.sendSubscriptionToBackend(subscription);

      console.log('[WebPush] Successfully subscribed to push notifications');
      return true;
    } catch (error) {
      console.error('[WebPush] Failed to subscribe:', error);
      this.state.isSubscribing = false;
      this.state.error = error instanceof Error ? error.message : 'Subscription failed';
      this.notifyListeners();
      return false;
    }
  }

  /**
   * Unsubscribe from push notifications
   */
  async unsubscribeFromPush(): Promise<boolean> {
    if (!this.state.subscription) {
      return true;
    }

    try {
      const success = await this.state.subscription.unsubscribe();
      
      if (success) {
        // Remove from backend
        await this.removeSubscriptionFromBackend(this.state.subscription);
        
        this.state.subscription = null;
        this.notifyListeners();
        console.log('[WebPush] Unsubscribed from push notifications');
      }
      
      return success;
    } catch (error) {
      console.error('[WebPush] Failed to unsubscribe:', error);
      return false;
    }
  }

  /**
   * Show a local notification (fallback when push is not available)
   */
  showLocalNotification(payload: PushNotificationPayload): void {
    if (this.state.permission !== 'granted') {
      console.warn('[WebPush] Cannot show notification: permission not granted');
      return;
    }

    const notification = new Notification(payload.title, {
      body: payload.body,
      icon: payload.icon || DEFAULT_ICON,
      badge: payload.badge || DEFAULT_BADGE,
      tag: payload.tag || 'nautilus-notification',
      requireInteraction: payload.requireInteraction ?? true,
      actions: payload.actions || [
        { action: 'view', title: 'View Dashboard' },
        { action: 'dismiss', title: 'Dismiss' },
      ],
      data: {
        url: payload.data?.url || '/dashboard',
        type: payload.data?.type || 'info',
        timestamp: payload.data?.timestamp || Date.now(),
        severity: payload.data?.severity || 'medium',
      },
    });

    notification.onclick = (event) => {
      event.preventDefault();
      notification.close();

      const url = payload.data?.url || '/dashboard';
      
      // Focus existing window or open new one
      window.open(url, '_blank');
    };

    notification.onclose = () => {
      console.log('[WebPush] Notification closed');
    };

    notification.onerror = (error) => {
      console.error('[WebPush] Notification error:', error);
    };
  }

  /**
   * Send subscription to backend server
   */
  private async sendSubscriptionToBackend(
    subscription: PushSubscription
  ): Promise<void> {
    try {
      // This would typically be an API call to your backend
      // For now, we'll just log it and store locally
      console.log('[WebPush] Sending subscription to backend:', {
        endpoint: subscription.endpoint,
        expirationTime: subscription.expirationTime,
      });

      // Store locally for persistence
      localStorage.setItem('push-subscription', JSON.stringify(subscription));
    } catch (error) {
      console.error('[WebPush] Failed to send subscription to backend:', error);
    }
  }

  /**
   * Remove subscription from backend server
   */
  private async removeSubscriptionFromBackend(
    subscription: PushSubscription
  ): Promise<void> {
    try {
      console.log('[WebPush] Removing subscription from backend');
      localStorage.removeItem('push-subscription');
    } catch (error) {
      console.error('[WebPush] Failed to remove subscription from backend:', error);
    }
  }

  /**
   * Restore subscription from local storage
   */
  async restoreSubscription(): Promise<boolean> {
    try {
      const stored = localStorage.getItem('push-subscription');
      if (!stored) return false;

      const subscription = JSON.parse(stored) as PushSubscription;
      
      // Verify subscription is still valid
      const registration = await navigator.serviceWorker.ready;
      const activeSubscription = await registration.pushManager.getSubscription();

      if (activeSubscription) {
        this.state.subscription = activeSubscription;
        this.notifyListeners();
        return true;
      }

      // Subscription expired, re-subscribe
      localStorage.removeItem('push-subscription');
      return false;
    } catch (error) {
      console.error('[WebPush] Failed to restore subscription:', error);
      return false;
    }
  }

  /**
   * Helper: Convert base64 URL to Uint8Array
   */
  private urlBase64ToUint8Array(base64String: string): Uint8Array {
    const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }

    return outputArray;
  }

  /**
   * Get circuit breaker alert payload
   */
  static createCircuitBreakerAlert(params: {
    pair: string;
    reason: string;
    timestamp: number;
  }): PushNotificationPayload {
    return {
      title: '⚠️ CIRCUIT BREAKER TRIGGERED',
      body: `${params.pair}: ${params.reason}`,
      icon: '/icons/alert-icon.png',
      tag: `circuit-breaker-${params.pair}-${params.timestamp}`,
      requireInteraction: true,
      actions: [
        { action: 'view', title: 'View Details' },
        { action: 'resume', title: 'Resume Trading' },
      ],
      data: {
        type: 'circuit-breaker',
        severity: 'critical',
        timestamp: params.timestamp,
        url: `/alerts/circuit-breaker?pair=${params.pair}`,
      },
    };
  }

  /**
   * Get trade execution alert payload
   */
  static createTradeAlert(params: {
    pair: string;
    side: 'BUY' | 'SELL';
    amount: number;
    price: number;
  }): PushNotificationPayload {
    return {
      title: `✅ Trade Executed: ${params.pair}`,
      body: `${params.side} ${params.amount} @ $${params.price.toLocaleString()}`,
      icon: '/icons/trade-icon.png',
      tag: `trade-${params.pair}-${Date.now()}`,
      data: {
        type: 'info',
        severity: 'low',
        timestamp: Date.now(),
        url: `/trades/${params.pair}`,
      },
    };
  }

  /**
   * Get PnL threshold alert payload
   */
  static createPnLAlert(params: {
    dailyPnL: number;
    threshold: number;
    percentage: number;
  }): PushNotificationPayload {
    const isPositive = params.dailyPnL >= 0;
    return {
      title: isPositive ? '📈 PnL Target Reached' : '📉 PnL Threshold Breached',
      body: `Daily PnL: $${params.dailyPnL.toLocaleString()} (${params.percentage.toFixed(2)}%)`,
      icon: isPositive ? '/icons/profit-icon.png' : '/icons/loss-icon.png',
      tag: `pnl-alert-${Date.now()}`,
      requireInteraction: true,
      data: {
        type: 'warning',
        severity: params.percentage < -5 ? 'high' : 'medium',
        timestamp: Date.now(),
        url: '/dashboard/pnl',
      },
    };
  }
}

// Export singleton instance
export const webPushManager = new WebPushManager();

/**
 * React Hook for using WebPush in components
 */
export function useWebPush() {
  const [state, setState] = React.useState<WebPushState>(webPushManager.getState());

  React.useEffect(() => {
    const unsubscribe = webPushManager.subscribe(setState);
    return unsubscribe;
  }, []);

  return {
    ...state,
    requestPermission: () => webPushManager.requestPermission(),
    subscribeToPush: (options: PushSubscriptionOptions) =>
      webPushManager.subscribeToPush(options),
    unsubscribeFromPush: () => webPushManager.unsubscribeFromPush(),
    showLocalNotification: (payload: PushNotificationPayload) =>
      webPushManager.showLocalNotification(payload),
    restoreSubscription: () => webPushManager.restoreSubscription(),
  };
}

// React import for the hook
import React from 'react';

export default WebPushManager;
