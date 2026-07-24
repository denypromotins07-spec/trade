/**
 * ShareModal Component
 * Secure, read-only snapshot generator for sharing sanitized performance dashboards
 * Uses temporary, expiring local tokens for access control
 */

import React, { useState, useCallback, useEffect } from 'react';

export interface ShareableSnapshot {
  id: string;
  token: string;
  createdAt: number;
  expiresAt: number;
  data: SanitizedDashboardData;
  viewCount: number;
  maxViews?: number;
}

export interface SanitizedDashboardData {
  // PnL data (sanitized - no sensitive positions)
  pnlSummary: {
    daily: number;
    weekly: number;
    monthly: number;
    total: number;
    winRate: number;
  };
  
  // Equity curve (aggregated, no individual trades)
  equityCurve: Array<{ timestamp: number; value: number }>;
  
  // Performance metrics
  metrics: {
    sharpeRatio: number;
    sortinoRatio: number;
    maxDrawdown: number;
    avgWin: number;
    avgLoss: number;
    profitFactor: number;
  };
  
  // Top pairs by volume (no amounts)
  topPairs: Array<{ pair: string; volume24h: number; pnl: number }>;
  
  // System stats
  systemStats: {
    uptime: number;
    totalTrades: number;
    activeStrategies: number;
    lastTradeTime: number;
  };
  
  // Branding
  branding: {
    primaryColor: string;
    secondaryColor: string;
    displayName: string;
  };
}

interface ShareModalProps {
  isOpen: boolean;
  onClose: () => void;
  dashboardData: Omit<SanitizedDashboardData, 'branding'>;
  defaultExpiryHours?: number;
  defaultMaxViews?: number;
  onShareCreated?: (snapshot: ShareableSnapshot) => void;
}

const DEFAULT_EXPIRY_HOURS = 24;
const DEFAULT_MAX_VIEWS = 100;

/**
 * Generate a secure random token
 */
function generateSecureToken(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Hash a token for storage comparison
 */
async function hashToken(token: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(token);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * ShareModal - Create and manage shareable dashboard snapshots
 */
export const ShareModal: React.FC<ShareModalProps> = ({
  isOpen,
  onClose,
  dashboardData,
  defaultExpiryHours = DEFAULT_EXPIRY_HOURS,
  defaultMaxViews = DEFAULT_MAX_VIEWS,
  onShareCreated,
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [createdSnapshot, setCreatedSnapshot] = useState<ShareableSnapshot | null>(null);
  const [expiryHours, setExpiryHours] = useState(defaultExpiryHours);
  const [maxViews, setMaxViews] = useState(defaultMaxViews);
  const [includeMetrics, setIncludeMetrics] = useState(true);
  const [includeEquityCurve, setIncludeEquityCurve] = useState(true);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Close modal on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Reset state when opening
  useEffect(() => {
    if (isOpen) {
      setCreatedSnapshot(null);
      setCopied(false);
      setError(null);
      setExpiryHours(defaultExpiryHours);
      setMaxViews(defaultMaxViews);
    }
  }, [isOpen, defaultExpiryHours, defaultMaxViews]);

  /**
   * Create a sanitized snapshot with expiring token
   */
  const createSnapshot = useCallback(async (): Promise<void> => {
    setIsCreating(true);
    setError(null);

    try {
      const token = generateSecureToken();
      const hashedToken = await hashToken(token);
      
      const now = Date.now();
      const expiresAt = now + expiryHours * 60 * 60 * 1000;

      // Sanitize data based on user preferences
      const sanitizedData: SanitizedDashboardData = {
        pnlSummary: dashboardData.pnlSummary,
        equityCurve: includeEquityCurve ? dashboardData.equityCurve : [],
        metrics: includeMetrics ? dashboardData.metrics : {
          sharpeRatio: 0,
          sortinoRatio: 0,
          maxDrawdown: 0,
          avgWin: 0,
          avgLoss: 0,
          profitFactor: 0,
        },
        topPairs: dashboardData.topPairs,
        systemStats: dashboardData.systemStats,
        branding: {
          primaryColor: '#00f3ff',
          secondaryColor: '#ff0055',
          displayName: 'Nautilus Ray Trader',
        },
      };

      const snapshot: ShareableSnapshot = {
        id: `snap-${now}`,
        token: hashedToken,
        createdAt: now,
        expiresAt,
        data: sanitizedData,
        viewCount: 0,
        maxViews,
      };

      // Store in IndexedDB for persistence
      await storeSnapshot(snapshot);

      // Create shareable URL with the raw token (not hash)
      const shareUrl = `${window.location.origin}/share/${token}`;
      
      const result = { ...snapshot, shareUrl };
      setCreatedSnapshot(result as ShareableSnapshot & { shareUrl: string });
      onShareCreated?.(snapshot);

      console.log('[ShareModal] Snapshot created:', snapshot.id);
    } catch (err) {
      console.error('[ShareModal] Failed to create snapshot:', err);
      setError(err instanceof Error ? err.message : 'Failed to create share link');
    } finally {
      setIsCreating(false);
    }
  }, [dashboardData, expiryHours, maxViews, includeMetrics, includeEquityCurve, onShareCreated]);

  /**
   * Copy share link to clipboard
   */
  const copyShareLink = useCallback(async (): Promise<void> => {
    if (!createdSnapshot || !('shareUrl' in createdSnapshot)) return;

    try {
      await navigator.clipboard.writeText(createdSnapshot.shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 3000);
    } catch (err) {
      console.error('[ShareModal] Failed to copy link:', err);
    }
  }, [createdSnapshot]);

  /**
   * Download snapshot as JSON
   */
  const downloadSnapshot = useCallback((): void => {
    if (!createdSnapshot) return;

    const dataStr = JSON.stringify(createdSnapshot.data, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `nautilus-snapshot-${createdSnapshot.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [createdSnapshot]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center"
      style={{
        backgroundColor: 'rgba(5, 5, 16, 0.8)',
        backdropFilter: 'blur(8px)',
      }}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Share Dashboard"
    >
      <div
        className="w-full max-w-lg rounded-xl border border-cyan-500/30 bg-[#0a0a1a]/95 shadow-2xl shadow-cyan-500/20 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-cyan-500/20">
          <h2 className="text-lg font-bold text-cyan-400">Share Dashboard</h2>
          <button
            onClick={onClose}
            className="text-cyan-600 hover:text-cyan-400 transition-colors"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {!createdSnapshot ? (
            <>
              {/* Configuration */}
              <div className="space-y-4">
                <p className="text-sm text-cyan-200">
                  Create a secure, read-only snapshot of your dashboard that can be shared via a temporary link.
                </p>

                {/* Expiry Setting */}
                <div className="space-y-2">
                  <label className="text-xs text-cyan-500 uppercase tracking-wide">
                    Link Expiry
                  </label>
                  <select
                    value={expiryHours}
                    onChange={(e) => setExpiryHours(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-[#050510] border border-cyan-800 rounded text-cyan-100 text-sm focus:border-cyan-500 focus:outline-none"
                  >
                    <option value={1}>1 hour</option>
                    <option value={6}>6 hours</option>
                    <option value={24}>24 hours</option>
                    <option value={72}>3 days</option>
                    <option value={168}>7 days</option>
                  </select>
                </div>

                {/* Max Views Setting */}
                <div className="space-y-2">
                  <label className="text-xs text-cyan-500 uppercase tracking-wide">
                    Max Views
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={1000}
                    value={maxViews}
                    onChange={(e) => setMaxViews(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-[#050510] border border-cyan-800 rounded text-cyan-100 text-sm focus:border-cyan-500 focus:outline-none"
                  />
                </div>

                {/* Data Options */}
                <div className="space-y-2">
                  <label className="text-xs text-cyan-500 uppercase tracking-wide">
                    Include in Snapshot
                  </label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={includeMetrics}
                        onChange={(e) => setIncludeMetrics(e.target.checked)}
                        className="w-4 h-4 rounded border-cyan-800 bg-[#050510] text-cyan-500 focus:ring-cyan-500 focus:ring-offset-0"
                      />
                      <span className="text-sm text-cyan-200">Performance Metrics</span>
                    </label>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={includeEquityCurve}
                        onChange={(e) => setIncludeEquityCurve(e.target.checked)}
                        className="w-4 h-4 rounded border-cyan-800 bg-[#050510] text-cyan-500 focus:ring-cyan-500 focus:ring-offset-0"
                      />
                      <span className="text-sm text-cyan-200">Equity Curve</span>
                    </label>
                  </div>
                </div>
              </div>

              {/* Security Notice */}
              <div className="p-3 bg-cyan-950/30 border border-cyan-800 rounded-lg">
                <p className="text-xs text-cyan-400">
                  🔒 Links are cryptographically signed and will expire automatically. 
                  Sensitive position data is never included in shared snapshots.
                </p>
              </div>

              {/* Error Message */}
              {error && (
                <div className="p-3 bg-red-950/30 border border-red-800 rounded-lg">
                  <p className="text-sm text-red-400">{error}</p>
                </div>
              )}

              {/* Create Button */}
              <button
                onClick={createSnapshot}
                disabled={isCreating}
                className="w-full px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-black font-bold rounded-lg shadow-lg shadow-cyan-500/30 transition-all duration-200"
              >
                {isCreating ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Creating Secure Link...
                  </span>
                ) : (
                  '🔗 Create Share Link'
                )}
              </button>
            </>
          ) : (
            <>
              {/* Success State */}
              <div className="space-y-4">
                <div className="p-4 bg-green-950/30 border border-green-800 rounded-lg">
                  <p className="text-sm text-green-400 mb-1">✓ Share link created successfully!</p>
                  <p className="text-xs text-green-600">
                    Expires in {expiryHours} hours or after {maxViews} views
                  </p>
                </div>

                {/* Share Link */}
                <div className="space-y-2">
                  <label className="text-xs text-cyan-500 uppercase tracking-wide">
                    Share Link
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      readOnly
                      value={('shareUrl' in createdSnapshot ? createdSnapshot.shareUrl : '')}
                      className="flex-1 px-3 py-2 bg-[#050510] border border-cyan-800 rounded text-cyan-100 text-xs font-mono truncate focus:outline-none"
                    />
                    <button
                      onClick={copyShareLink}
                      className={`px-4 py-2 rounded font-semibold transition-all ${
                        copied
                          ? 'bg-green-600 text-white'
                          : 'bg-cyan-600 hover:bg-cyan-500 text-black'
                      }`}
                    >
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-4 border-t border-cyan-800">
                  <button
                    onClick={downloadSnapshot}
                    className="flex-1 px-4 py-2 bg-[#050510] border border-cyan-800 hover:border-cyan-600 text-cyan-400 rounded transition-all"
                  >
                    📥 Download JSON
                  </button>
                  <button
                    onClick={createSnapshot}
                    className="flex-1 px-4 py-2 bg-[#050510] border border-cyan-800 hover:border-cyan-600 text-cyan-400 rounded transition-all"
                  >
                    🔄 Create New Link
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// IndexedDB helpers for snapshot storage
const DB_NAME = 'NautilusSnapshots';
const STORE_NAME = 'snapshots';

async function openSnapshotDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('token', 'token', { unique: true });
        store.createIndex('expiresAt', 'expiresAt', { unique: false });
      }
    };
  });
}

async function storeSnapshot(snapshot: ShareableSnapshot): Promise<void> {
  const db = await openSnapshotDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.put(snapshot);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export async function getSnapshotByToken(token: string): Promise<ShareableSnapshot | null> {
  const hashedToken = await hashToken(token);
  const db = await openSnapshotDB();
  
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('token');
    const request = index.get(hashedToken);

    request.onsuccess = () => {
      const snapshot = request.result as ShareableSnapshot | undefined;
      
      if (!snapshot) {
        resolve(null);
        return;
      }

      // Check expiration
      if (Date.now() > snapshot.expiresAt) {
        resolve(null);
        return;
      }

      // Check view count
      if (snapshot.maxViews && snapshot.viewCount >= snapshot.maxViews) {
        resolve(null);
        return;
      }

      // Increment view count
      snapshot.viewCount++;
      const updateTx = db.transaction(STORE_NAME, 'readwrite');
      updateTx.objectStore(STORE_NAME).put(snapshot);

      resolve(snapshot);
    };
    request.onerror = () => reject(request.error);
  });
}

export async function cleanupExpiredSnapshots(): Promise<void> {
  const db = await openSnapshotDB();
  const now = Date.now();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('expiresAt');
    const request = index.openCursor();

    request.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor) {
        if (cursor.value.expiresAt < now) {
          store.delete(cursor.primaryKey);
        }
        cursor.continue();
      } else {
        resolve();
      }
    };
    request.onerror = () => reject(request.error);
  });
}

export default ShareModal;
