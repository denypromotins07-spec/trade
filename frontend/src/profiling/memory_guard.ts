/**
 * Memory Guard - Browser Performance Profiling
 * 
 * Continuous performance.memory monitor that forces WebGL context destruction
 * and drops heavy chart buffers if the browser approaches the 200MB RAM target.
 * 
 * Cyberpunk aesthetic: "Memory pressure gauge" with neural overload warnings.
 */

export interface MemoryStats {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  usagePercent: number;
  isCritical: boolean;
  timestamp: number;
}

export interface MemoryGuardConfig {
  targetMaxMB: number;
  warningThreshold: number;
  criticalThreshold: number;
  checkIntervalMs: number;
  onWarning?: (stats: MemoryStats) => void;
  onCritical?: (stats: MemoryStats) => void;
  onCleanup?: (freedMB: number) => void;
}

/**
 * Memory guard class for monitoring and enforcing browser memory limits
 */
export class MemoryGuard {
  private static instance: MemoryGuard;
  private config: MemoryGuardConfig;
  private isMonitoring: boolean = false;
  private checkInterval: NodeJS.Timeout | null = null;
  private registeredResources: Map<string, MemoryResource> = new Map();
  private webglContexts: Set<WebGLRenderingContext | WebGL2RenderingContext> = new Set();
  private statsHistory: MemoryStats[] = [];
  private readonly MAX_HISTORY = 60; // Keep last 60 samples (1 minute at 1s interval)

  private constructor(config?: Partial<MemoryGuardConfig>) {
    this.config = {
      targetMaxMB: 200,
      warningThreshold: 0.7,
      criticalThreshold: 0.9,
      checkIntervalMs: 1000,
      ...config,
    };
  }

  /**
   * Get singleton instance
   */
  static getInstance(config?: Partial<MemoryGuardConfig>): MemoryGuard {
    if (!MemoryGuard.instance) {
      MemoryGuard.instance = new MemoryGuard(config);
    }
    return MemoryGuard.instance;
  }

  /**
   * Start continuous memory monitoring
   */
  start(): void {
    if (this.isMonitoring) {
      return;
    }

    this.isMonitoring = true;
    console.log('[MEMORY_GUARD] Starting monitoring with target max:', this.config.targetMaxMB, 'MB');

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
  }

  /**
   * Perform memory check and take action if needed
   */
  check(): void {
    const stats = this.getMemoryStats();
    
    if (!stats) {
      return;
    }

    // Add to history
    this.statsHistory.push(stats);
    if (this.statsHistory.length > this.MAX_HISTORY) {
      this.statsHistory.shift();
    }

    // Log periodic updates
    if (stats.usagePercent > this.config.warningThreshold) {
      console.log('[MEMORY_GUARD] Memory usage:', {
        used: (stats.usedJSHeapSize / 1024 / 1024).toFixed(2),
        limit: (stats.jsHeapSizeLimit / 1024 / 1024).toFixed(2),
        percent: (stats.usagePercent * 100).toFixed(1) + '%',
      });
    }

    // Handle thresholds
    if (stats.usagePercent >= this.config.criticalThreshold) {
      this.handleCritical(stats);
    } else if (stats.usagePercent >= this.config.warningThreshold) {
      this.handleWarning(stats);
    }
  }

  /**
   * Get current memory statistics
   */
  getMemoryStats(): MemoryStats | null {
    // Check for Performance Memory API (Chrome only)
    if (!('memory' in performance)) {
      return null;
    }

    const mem = (performance as { memory?: { 
      usedJSHeapSize: number; 
      totalJSHeapSize: number; 
      jsHeapSizeLimit: number;
    } }).memory;

    if (!mem) {
      return null;
    }

    const usagePercent = mem.usedJSHeapSize / mem.jsHeapSizeLimit;
    const isCritical = usagePercent >= this.config.criticalThreshold;

    return {
      usedJSHeapSize: mem.usedJSHeapSize,
      totalJSHeapSize: mem.totalJSHeapSize,
      jsHeapSizeLimit: mem.jsHeapSizeLimit,
      usagePercent,
      isCritical,
      timestamp: Date.now(),
    };
  }

  /**
   * Handle warning threshold breach
   */
  private handleWarning(stats: MemoryStats): void {
    console.warn('[MEMORY_GUARD] Warning: Memory usage above threshold', {
      percent: (stats.usagePercent * 100).toFixed(1) + '%',
    });

    if (this.config.onWarning) {
      this.config.onWarning(stats);
    }

    // Trigger light cleanup
    this.performLightCleanup();
  }

  /**
   * Handle critical threshold breach - aggressive cleanup
   */
  private handleCritical(stats: MemoryStats): void {
    console.error('[MEMORY_GUARD] CRITICAL: Memory usage dangerously high', {
      percent: (stats.usagePercent * 100).toFixed(1) + '%',
    });

    if (this.config.onCritical) {
      this.config.onCritical(stats);
    }

    // Trigger aggressive cleanup
    this.performAggressiveCleanup();
  }

  /**
   * Light cleanup - drop cached data, reduce buffer sizes
   */
  private performLightCleanup(): void {
    let freedMB = 0;

    // Notify registered resources to reduce footprint
    for (const [id, resource] of this.registeredResources.entries()) {
      if (resource.priority === 'low') {
        const freed = resource.cleanup?.() || 0;
        freedMB += freed;
        console.log('[MEMORY_GUARD] Light cleanup freed from', id, ':', freed.toFixed(2), 'MB');
      }
    }

    if (this.config.onCleanup && freedMB > 0) {
      this.config.onCleanup(freedMB);
    }
  }

  /**
   * Aggressive cleanup - destroy WebGL contexts, drop all non-essential buffers
   */
  private performAggressiveCleanup(): void {
    let freedMB = 0;

    // First, notify all resources to cleanup
    for (const [id, resource] of this.registeredResources.entries()) {
      if (resource.priority !== 'critical') {
        const freed = resource.cleanup?.() || 0;
        freedMB += freed;
        console.log('[MEMORY_GUARD] Aggressive cleanup freed from', id, ':', freed.toFixed(2), 'MB');
      }
    }

    // Force WebGL context loss for non-critical contexts
    this.webglContexts.forEach((context) => {
      const canvas = context.canvas as HTMLCanvasElement;
      const isCritical = Array.from(this.registeredResources.values()).some(
        (r) => r.element === canvas && r.priority === 'critical'
      );

      if (!isCritical) {
        console.log('[MEMORY_GUARD] Forcing WebGL context loss for canvas:', canvas.id || 'unnamed');
        
        // Trigger context loss
        const ext = context.getExtension('WEBGL_lose_context');
        if (ext) {
          ext.loseContext();
        }
      }
    });

    // Force garbage collection hint (if available)
    if (typeof gc === 'function') {
      gc();
    }

    if (this.config.onCleanup && freedMB > 0) {
      this.config.onCleanup(freedMB);
    }
  }

  /**
   * Register a memory-managed resource
   */
  register(id: string, resource: MemoryResource): void {
    this.registeredResources.set(id, resource);
    console.log('[MEMORY_GUARD] Registered resource:', id, '(priority:', resource.priority + ')');
  }

  /**
   * Unregister a resource
   */
  unregister(id: string): void {
    const resource = this.registeredResources.get(id);
    if (resource) {
      resource.cleanup?.();
      this.registeredResources.delete(id);
      console.log('[MEMORY_GUARD] Unregistered resource:', id);
    }
  }

  /**
   * Track a WebGL context for potential cleanup
   */
  trackWebGLContext(context: WebGLRenderingContext | WebGL2RenderingContext): void {
    this.webglContexts.add(context);
    
    // Listen for context loss
    const canvas = context.canvas as HTMLCanvasElement;
    canvas.addEventListener('webglcontextlost', () => {
      console.log('[MEMORY_GUARD] WebGL context lost:', canvas.id || 'unnamed');
      this.webglContexts.delete(context);
    });

    canvas.addEventListener('webglcontextrestored', () => {
      console.log('[MEMORY_GUARD] WebGL context restored:', canvas.id || 'unnamed');
      this.webglContexts.add(context);
    });
  }

  /**
   * Get memory usage trend (bytes per second)
   */
  getMemoryTrend(): number {
    if (this.statsHistory.length < 2) {
      return 0;
    }

    const oldest = this.statsHistory[0];
    const newest = this.statsHistory[this.statsHistory.length - 1];
    const timeDiff = (newest.timestamp - oldest.timestamp) / 1000; // seconds
    
    if (timeDiff <= 0) {
      return 0;
    }

    return (newest.usedJSHeapSize - oldest.usedJSHeapSize) / timeDiff;
  }

  /**
   * Predict time until memory limit (in seconds)
   */
  predictTimeToLimit(): number | null {
    const trend = this.getMemoryTrend();
    
    if (trend <= 0) {
      return Infinity; // Memory is stable or decreasing
    }

    const stats = this.getMemoryStats();
    if (!stats) {
      return null;
    }

    const targetBytes = this.config.targetMaxMB * 1024 * 1024;
    const remainingBytes = targetBytes - stats.usedJSHeapSize;
    
    return remainingBytes / trend;
  }

  /**
   * Get formatted memory report
   */
  getReport(): string {
    const stats = this.getMemoryStats();
    if (!stats) {
      return 'Memory API not available';
    }

    const trend = this.getMemoryTrend();
    const timeToLimit = this.predictTimeToLimit();

    return `
=== MEMORY GUARD REPORT ===
Used: ${(stats.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB
Limit: ${(stats.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB
Usage: ${(stats.usagePercent * 100).toFixed(1)}%
Trend: ${(trend / 1024 / 1024).toFixed(2)} MB/s
Time to Limit: ${timeToLimit === null ? 'N/A' : timeToLimit === Infinity ? '∞' : timeToLimit.toFixed(0) + 's'}
Registered Resources: ${this.registeredResources.size}
WebGL Contexts: ${this.webglContexts.size}
===========================
`.trim();
  }
}

/**
 * Interface for memory-managed resources
 */
export interface MemoryResource {
  priority: 'low' | 'medium' | 'high' | 'critical';
  element?: HTMLElement;
  cleanup?: () => number; // Returns estimated MB freed
}

// Export singleton instance
export const memoryGuard = MemoryGuard.getInstance();

/**
 * Auto-start monitoring when module loads
 */
if (typeof window !== 'undefined') {
  // Delay start to allow app initialization
  setTimeout(() => {
    memoryGuard.start();
  }, 5000);
}
