/**
 * Network Throttle - Client-Side Bandwidth Estimator
 * 
 * Dynamically requests lower-resolution tick streams from Rust via RPC
 * during local network congestion or Wi-Fi drops.
 * 
 * Cyberpunk aesthetic: "Neural link quality indicator" with signal degradation visuals.
 */

export interface NetworkStats {
  online: boolean;
  connectionType?: string;
  downlink?: number; // Mbps
  rtt?: number; // milliseconds
  effectiveConnectionType?: 'slow-2g' | '2g' | '3g' | '4g';
  estimatedBandwidth: number; // bits per second
  isCongested: boolean;
  packetLossEstimate: number; // 0-1
  timestamp: number;
}

export interface NetworkThrottleConfig {
  checkIntervalMs: number;
  congestionThreshold: number; // Mbps below which to consider congested
  rttThreshold: number; // ms above which to consider high latency
  onThrottle?: (stats: NetworkStats, recommendedQuality: StreamQuality) => void;
  onRecovery?: (stats: NetworkStats) => void;
}

export type StreamQuality = 'ultra' | 'high' | 'medium' | 'low' | 'minimal';

/**
 * Network throttle class for monitoring bandwidth and adjusting stream quality
 */
export class NetworkThrottle {
  private static instance: NetworkThrottle;
  private config: NetworkThrottleConfig;
  private isMonitoring: boolean = false;
  private checkInterval: NodeJS.Timeout | null = null;
  private statsHistory: NetworkStats[] = [];
  private readonly MAX_HISTORY = 60;
  private currentQuality: StreamQuality = 'ultra';
  private lastThrottledTime: number = 0;
  
  // Bandwidth estimation
  private recentTransfers: TransferRecord[] = [];
  private readonly TRANSFER_WINDOW_MS = 5000;

  private constructor(config?: Partial<NetworkThrottleConfig>) {
    this.config = {
      checkIntervalMs: 2000,
      congestionThreshold: 5, // Mbps
      rttThreshold: 200, // ms
      ...config,
    };
  }

  /**
   * Get singleton instance
   */
  static getInstance(config?: Partial<NetworkThrottleConfig>): NetworkThrottle {
    if (!NetworkThrottle.instance) {
      NetworkThrottle.instance = new NetworkThrottle(config);
    }
    return NetworkThrottle.instance;
  }

  /**
   * Start network monitoring
   */
  start(): void {
    if (this.isMonitoring) {
      return;
    }

    this.isMonitoring = true;
    console.log('[NETWORK_THROTTLE] Starting network monitoring');

    // Listen for network events
    window.addEventListener('online', this.handleOnline);
    window.addEventListener('offline', this.handleOffline);

    this.check();
    this.checkInterval = setInterval(() => {
      this.check();
    }, this.config.checkIntervalMs);
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    this.isMonitoring = false;

    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }

    window.removeEventListener('online', this.handleOnline);
    window.removeEventListener('offline', this.handleOffline);
  }

  /**
   * Handle online event
   */
  private handleOnline = (): void => {
    console.log('[NETWORK_THROTTLE] Network back online');
    this.check();
  };

  /**
   * Handle offline event
   */
  private handleOffline = (): void => {
    console.warn('[NETWORK_THROTTLE] Network went offline');
    this.setQuality('minimal');
  };

  /**
   * Perform network check
   */
  check(): void {
    const stats = this.getNetworkStats();
    
    if (!stats) {
      return;
    }

    // Add to history
    this.statsHistory.push(stats);
    if (this.statsHistory.length > this.MAX_HISTORY) {
      this.statsHistory.shift();
    }

    // Determine if network is congested
    const isCongested = this.isNetworkCongested(stats);

    // Get recommended quality based on current conditions
    const recommendedQuality = this.getRecommendedQuality(stats);

    // Apply throttling if needed
    if (isCongested && this.currentQuality !== recommendedQuality) {
      this.setQuality(recommendedQuality);
      this.lastThrottledTime = Date.now();

      if (this.config.onThrottle) {
        this.config.onThrottle(stats, recommendedQuality);
      }

      console.warn('[NETWORK_THROTTLE] Network congested - throttling to', recommendedQuality);
    } else if (!isCongested && this.currentQuality !== 'ultra') {
      // Check if we've been stable for a while before upgrading
      const timeSinceThrottle = Date.now() - this.lastThrottledTime;
      if (timeSinceThrottle > 10000) { // 10 second hysteresis
        this.setQuality('ultra');

        if (this.config.onRecovery) {
          this.config.onRecovery(stats);
        }

        console.log('[NETWORK_THROTTLE] Network recovered - quality restored to ultra');
      }
    }
  }

  /**
   * Get current network statistics
   */
  getNetworkStats(): NetworkStats | null {
    const online = navigator.onLine;
    
    let connectionType: string | undefined;
    let downlink: number | undefined;
    let rtt: number | undefined;
    let effectiveConnectionType: NetworkStats['effectiveConnectionType'];

    // Check for Network Information API
    if ('connection' in navigator) {
      const conn = (navigator as { connection?: {
        effectiveType?: string;
        downlink?: number;
        rtt?: number;
        saveData?: boolean;
      } }).connection;

      if (conn) {
        connectionType = conn.effectiveType;
        downlink = conn.downlink; // Mbps
        rtt = conn.rtt; // ms
        effectiveConnectionType = conn.effectiveType as NetworkStats['effectiveConnectionType'];
      }
    }

    // Estimate bandwidth from recent transfers
    const estimatedBandwidth = this.calculateEstimatedBandwidth();

    // Estimate packet loss (simplified - would need actual metrics in production)
    const packetLossEstimate = this.estimatePacketLoss();

    // Determine if congested
    const isCongested = this.isNetworkCongested({
      online,
      connectionType,
      downlink,
      rtt,
      effectiveConnectionType,
      estimatedBandwidth,
      packetLossEstimate,
      timestamp: Date.now(),
    });

    return {
      online,
      connectionType,
      downlink,
      rtt,
      effectiveConnectionType,
      estimatedBandwidth,
      isCongested,
      packetLossEstimate,
      timestamp: Date.now(),
    };
  }

  /**
   * Calculate estimated bandwidth from recent transfers
   */
  private calculateEstimatedBandwidth(): number {
    const now = Date.now();
    const windowStart = now - this.TRANSFER_WINDOW_MS;

    // Filter to recent transfers
    this.recentTransfers = this.recentTransfers.filter(t => t.timestamp >= windowStart);

    if (this.recentTransfers.length === 0) {
      return 0;
    }

    const totalBytes = this.recentTransfers.reduce((sum, t) => sum + t.bytes, 0);
    const durationSeconds = this.TRANSFER_WINDOW_MS / 1000;

    // Return bits per second
    return (totalBytes * 8) / durationSeconds;
  }

  /**
   * Estimate packet loss (placeholder - would use actual metrics in production)
   */
  private estimatePacketLoss(): number {
    // In production, this would analyze WebSocket message gaps,
    // TCP retransmissions, or use the WebRTC statistics API
    const stats = this.getAverageStats();
    
    if (!stats || !stats.rtt) {
      return 0;
    }

    // Simple heuristic: higher RTT correlates with potential packet loss
    if (stats.rtt > 500) return 0.1;
    if (stats.rtt > 300) return 0.05;
    if (stats.rtt > 150) return 0.02;
    return 0;
  }

  /**
   * Check if network is congested
   */
  private isNetworkCongested(stats: NetworkStats): boolean {
    // Offline is definitely congested
    if (!stats.online) {
      return true;
    }

    // Check downlink threshold
    if (stats.downlink !== undefined && stats.downlink < this.config.congestionThreshold) {
      return true;
    }

    // Check RTT threshold
    if (stats.rtt !== undefined && stats.rtt > this.config.rttThreshold) {
      return true;
    }

    // Check effective connection type
    if (stats.effectiveConnectionType === 'slow-2g' || 
        stats.effectiveConnectionType === '2g' || 
        stats.effectiveConnectionType === '3g') {
      return true;
    }

    // Check estimated bandwidth
    const estimatedMbps = stats.estimatedBandwidth / 1_000_000;
    if (estimatedMbps > 0 && estimatedMbps < this.config.congestionThreshold) {
      return true;
    }

    return false;
  }

  /**
   * Get recommended quality based on network conditions
   */
  getRecommendedQuality(stats: NetworkStats): StreamQuality {
    if (!stats.online) {
      return 'minimal';
    }

    // Use the lower of downlink and estimated bandwidth
    const effectiveBandwidth = Math.min(
      stats.downlink || Infinity,
      stats.estimatedBandwidth > 0 ? stats.estimatedBandwidth / 1_000_000 : Infinity
    );

    // Consider RTT for interactive streams
    const highLatency = (stats.rtt || 0) > this.config.rttThreshold;

    if (effectiveBandwidth < 0.5 || highLatency) {
      return 'minimal';
    }
    if (effectiveBandwidth < 1) {
      return 'low';
    }
    if (effectiveBandwidth < 3) {
      return 'medium';
    }
    if (effectiveBandwidth < 10) {
      return 'high';
    }
    return 'ultra';
  }

  /**
   * Set stream quality
   */
  setQuality(quality: StreamQuality): void {
    if (this.currentQuality === quality) {
      return;
    }

    const oldQuality = this.currentQuality;
    this.currentQuality = quality;

    console.log('[NETWORK_THROTTLE] Quality changed:', oldQuality, '->', quality);

    // Dispatch custom event for components to listen to
    window.dispatchEvent(new CustomEvent('nautilus:network-quality-change', {
      detail: { oldQuality, newQuality: quality },
    }));
  }

  /**
   * Get current quality
   */
  getQuality(): StreamQuality {
    return this.currentQuality;
  }

  /**
   * Get stream settings for current quality
   */
  getStreamSettings(): StreamSettings {
    switch (this.currentQuality) {
      case 'ultra':
        return {
          tickResolution: 'full',
          updateFrequency: 100, // ms
          orderBookDepth: 100,
          includeTrades: true,
          compressionLevel: 'none',
        };
      case 'high':
        return {
          tickResolution: 'full',
          updateFrequency: 200,
          orderBookDepth: 50,
          includeTrades: true,
          compressionLevel: 'low',
        };
      case 'medium':
        return {
          tickResolution: 'aggregated',
          updateFrequency: 500,
          orderBookDepth: 25,
          includeTrades: true,
          compressionLevel: 'medium',
        };
      case 'low':
        return {
          tickResolution: 'aggregated',
          updateFrequency: 1000,
          orderBookDepth: 10,
          includeTrades: false,
          compressionLevel: 'high',
        };
      case 'minimal':
        return {
          tickResolution: 'snapshot',
          updateFrequency: 5000,
          orderBookDepth: 5,
          includeTrades: false,
          compressionLevel: 'maximum',
        };
    }
  }

  /**
   * Record a data transfer for bandwidth estimation
   */
  recordTransfer(bytes: number): void {
    this.recentTransfers.push({
      bytes,
      timestamp: Date.now(),
    });
  }

  /**
   * Get average stats over history
   */
  getAverageStats(): NetworkStats | null {
    if (this.statsHistory.length === 0) {
      return null;
    }

    const sum = this.statsHistory.reduce((acc, stats) => ({
      online: acc.online || stats.online,
      downlink: (acc.downlink || 0) + (stats.downlink || 0),
      rtt: (acc.rtt || 0) + (stats.rtt || 0),
      estimatedBandwidth: acc.estimatedBandwidth + stats.estimatedBandwidth,
      packetLossEstimate: acc.packetLossEstimate + stats.packetLossEstimate,
      timestamp: Date.now(),
      isCongested: false,
    }), {
      online: false,
      downlink: 0,
      rtt: 0,
      estimatedBandwidth: 0,
      packetLossEstimate: 0,
      timestamp: Date.now(),
      isCongested: false,
    });

    const count = this.statsHistory.length;

    return {
      online: sum.online,
      downlink: sum.downlink / count,
      rtt: sum.rtt / count,
      estimatedBandwidth: sum.estimatedBandwidth / count,
      packetLossEstimate: sum.packetLossEstimate / count,
      isCongested: sum.isCongested,
      timestamp: Date.now(),
    };
  }

  /**
   * Get formatted network report
   */
  getReport(): string {
    const stats = this.getNetworkStats();
    if (!stats) {
      return 'Network Information API not available';
    }

    return `
=== NETWORK THROTTLE REPORT ===
Online: ${stats.online ? 'YES' : 'NO'}
Connection: ${stats.connectionType || stats.effectiveConnectionType || 'UNKNOWN'}
Downlink: ${stats.downlink?.toFixed(2) || 'N/A'} Mbps
RTT: ${stats.rtt?.toFixed(0) || 'N/A'} ms
Estimated BW: ${(stats.estimatedBandwidth / 1_000_000).toFixed(2)} Mbps
Packet Loss: ${(stats.packetLossEstimate * 100).toFixed(1)}%
Congested: ${stats.isCongested ? 'YES' : 'NO'}
Quality: ${this.currentQuality.toUpperCase()}
================================`.trim();
  }
}

/**
 * Transfer record for bandwidth tracking
 */
interface TransferRecord {
  bytes: number;
  timestamp: number;
}

/**
 * Stream settings interface for RPC requests
 */
export interface StreamSettings {
  tickResolution: 'full' | 'aggregated' | 'snapshot';
  updateFrequency: number; // ms
  orderBookDepth: number;
  includeTrades: boolean;
  compressionLevel: 'none' | 'low' | 'medium' | 'high' | 'maximum';
}

// Export singleton instance
export const networkThrottle = NetworkThrottle.getInstance();

/**
 * Hook-friendly getter for current quality
 */
export function getCurrentNetworkQuality(): StreamQuality {
  return networkThrottle.getQuality();
}

/**
 * Hook-friendly getter for stream settings
 */
export function getStreamSettings(): StreamSettings {
  return networkThrottle.getStreamSettings();
}

/**
 * Auto-start monitoring when module loads
 */
if (typeof window !== 'undefined') {
  // Delay start to allow app initialization
  setTimeout(() => {
    networkThrottle.start();
  }, 4000);
}
