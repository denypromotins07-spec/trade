/**
 * FPS Monitor - Frame Rate Tracking with GPU Compositor Health
 * 
 * Microsecond requestAnimationFrame delta tracker that dynamically degrades
 * particle effects and chart fidelity if the UI thread FPS drops below 55.
 * Tracks GPU compositor thread health for AMD DirectML/ROCm context visualization.
 * 
 * Cyberpunk aesthetic: "Neural pulse monitor" with real-time frame timing graphs.
 */

export interface FPSStats {
  currentFPS: number;
  averageFPS: number;
  minFPS: number;
  maxFPS: number;
  frameTime: number; // milliseconds
  averageFrameTime: number;
  droppedFrames: number;
  gpuCompositorLatency: number; // Estimated GPU latency in ms
  isDegraded: boolean;
  timestamp: number;
}

export interface FPSMonitorConfig {
  targetFPS: number;
  degradationThreshold: number;
  criticalThreshold: number;
  sampleWindowSize: number;
  onDegradation?: (stats: FPSStats) => void;
  onRecovery?: (stats: FPSStats) => void;
  onCritical?: (stats: FPSStats) => void;
}

export type QualityLevel = 'ultra' | 'high' | 'medium' | 'low' | 'potato';

/**
 * FPS monitor class for tracking frame rates and triggering quality adjustments
 */
export class FPSMonitor {
  private static instance: FPSMonitor;
  private config: FPSMonitorConfig;
  private isMonitoring: boolean = false;
  private animationFrameId: number | null = null;
  private frameTimes: number[] = [];
  private statsHistory: FPSStats[] = [];
  private lastFrameTime: number = 0;
  private droppedFramesCount: number = 0;
  private currentQuality: QualityLevel = 'ultra';
  private readonly MAX_HISTORY = 120; // Keep last 120 samples (2 seconds at 60fps)
  
  // GPU compositor tracking
  private gpuTaskQueue: number[] = [];
  private lastGpuSyncPoint: number = 0;

  private constructor(config?: Partial<FPSMonitorConfig>) {
    this.config = {
      targetFPS: 60,
      degradationThreshold: 55,
      criticalThreshold: 30,
      sampleWindowSize: 60,
      ...config,
    };
  }

  /**
   * Get singleton instance
   */
  static getInstance(config?: Partial<FPSMonitorConfig>): FPSMonitor {
    if (!FPSMonitor.instance) {
      FPSMonitor.instance = new FPSMonitor(config);
    }
    return FPSMonitor.instance;
  }

  /**
   * Start FPS monitoring
   */
  start(): void {
    if (this.isMonitoring) {
      return;
    }

    this.isMonitoring = true;
    this.lastFrameTime = performance.now();
    console.log('[FPS_MONITOR] Starting monitoring with target:', this.config.targetFPS, 'FPS');

    this.tick();
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    this.isMonitoring = false;
    
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Main monitoring loop
   */
  private tick = (): void => {
    if (!this.isMonitoring) {
      return;
    }

    const now = performance.now();
    const frameTime = now - this.lastFrameTime;
    this.lastFrameTime = now;

    // Track frame times
    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > this.config.sampleWindowSize) {
      this.frameTimes.shift();
    }

    // Detect dropped frames (frame time > 2x expected)
    const expectedFrameTime = 1000 / this.config.targetFPS;
    if (frameTime > expectedFrameTime * 1.5) {
      this.droppedFramesCount++;
    }

    // Estimate GPU compositor latency using sync points
    this.updateGPUEstimate(frameTime);

    // Calculate stats
    const stats = this.calculateStats();

    // Add to history
    this.statsHistory.push(stats);
    if (this.statsHistory.length > this.MAX_HISTORY) {
      this.statsHistory.shift();
    }

    // Check thresholds and adjust quality
    this.checkThresholds(stats);

    // Continue loop
    this.animationFrameId = requestAnimationFrame(this.tick);
  };

  /**
   * Update GPU compositor latency estimate
   */
  private updateGPUEstimate(frameTime: number): void {
    // Use frame time as a proxy for GPU latency
    // In a real implementation, this would use WebGL fence sync or GPU timestamps
    this.gpuTaskQueue.push(frameTime);
    if (this.gpuTaskQueue.length > 16) {
      this.gpuTaskQueue.shift();
    }

    this.lastGpuSyncPoint = performance.now();
  }

  /**
   * Calculate current FPS statistics
   */
  private calculateStats(): FPSStats {
    if (this.frameTimes.length === 0) {
      return this.createEmptyStats();
    }

    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    const minFrameTime = Math.min(...this.frameTimes);
    const maxFrameTime = Math.max(...this.frameTimes);
    
    const currentFPS = 1000 / this.frameTimes[this.frameTimes.length - 1];
    const averageFPS = 1000 / avgFrameTime;
    const minFPS = 1000 / maxFrameTime;
    const maxFPS = 1000 / minFrameTime;

    // Estimate GPU compositor latency (average of recent frame times)
    const gpuCompositorLatency = this.gpuTaskQueue.length > 0
      ? this.gpuTaskQueue.reduce((a, b) => a + b, 0) / this.gpuTaskQueue.length
      : avgFrameTime;

    const isDegraded = averageFPS < this.config.degradationThreshold;

    return {
      currentFPS,
      averageFPS,
      minFPS,
      maxFPS,
      frameTime: this.frameTimes[this.frameTimes.length - 1],
      averageFrameTime: avgFrameTime,
      droppedFrames: this.droppedFramesCount,
      gpuCompositorLatency,
      isDegraded,
      timestamp: Date.now(),
    };
  }

  /**
   * Create empty stats object
   */
  private createEmptyStats(): FPSStats {
    return {
      currentFPS: 0,
      averageFPS: 0,
      minFPS: 0,
      maxFPS: 0,
      frameTime: 0,
      averageFrameTime: 0,
      droppedFrames: 0,
      gpuCompositorLatency: 0,
      isDegraded: false,
      timestamp: Date.now(),
    };
  }

  /**
   * Check thresholds and trigger callbacks/quality adjustments
   */
  private checkThresholds(stats: FPSStats): void {
    const wasDegraded = this.currentQuality !== 'ultra';
    
    if (stats.averageFPS < this.config.criticalThreshold) {
      // Critical - drop to lowest quality
      this.setQuality('potato');
      
      if (this.config.onCritical) {
        this.config.onCritical(stats);
      }
      
      console.error('[FPS_MONITOR] CRITICAL: FPS below', this.config.criticalThreshold, '- Quality set to POTATO');
    } else if (stats.averageFPS < this.config.degradationThreshold) {
      // Degraded - step down quality
      const newQuality = this.getDegradedQuality();
      this.setQuality(newQuality);
      
      if (this.config.onDegradation && !wasDegraded) {
        this.config.onDegradation(stats);
      }
      
      console.warn('[FPS_MONITOR] DEGRADED: FPS below', this.config.degradationThreshold, '- Quality:', newQuality);
    } else if (wasDegraded && stats.averageFPS >= this.config.degradationThreshold) {
      // Recovery - restore quality
      this.setQuality('ultra');
      
      if (this.config.onRecovery) {
        this.config.onRecovery(stats);
      }
      
      console.log('[FPS_MONITOR] RECOVERED: FPS back to normal - Quality restored to ULTRA');
    }
  }

  /**
   * Get appropriate degraded quality level
   */
  private getDegradedQuality(): QualityLevel {
    const stats = this.calculateStats();
    
    if (stats.averageFPS < 20) return 'potato';
    if (stats.averageFPS < 30) return 'low';
    if (stats.averageFPS < 45) return 'medium';
    return 'high';
  }

  /**
   * Set rendering quality level
   */
  setQuality(level: QualityLevel): void {
    if (this.currentQuality === level) {
      return;
    }

    const oldQuality = this.currentQuality;
    this.currentQuality = level;

    console.log('[FPS_MONITOR] Quality changed:', oldQuality, '->', level);

    // Dispatch custom event for components to listen to
    window.dispatchEvent(new CustomEvent('nautilus:quality-change', {
      detail: { oldQuality, newQuality: level },
    }));
  }

  /**
   * Get current quality level
   */
  getQuality(): QualityLevel {
    return this.currentQuality;
  }

  /**
   * Get quality settings for rendering components
   */
  getQualitySettings(): QualitySettings {
    switch (this.currentQuality) {
      case 'ultra':
        return {
          particles: true,
          particleCount: 10000,
          shadows: true,
          antialiasing: true,
          resolution: 1,
          chartPoints: 1000,
          updateInterval: 16,
        };
      case 'high':
        return {
          particles: true,
          particleCount: 5000,
          shadows: true,
          antialiasing: true,
          resolution: 1,
          chartPoints: 500,
          updateInterval: 16,
        };
      case 'medium':
        return {
          particles: true,
          particleCount: 2000,
          shadows: false,
          antialiasing: true,
          resolution: 0.75,
          chartPoints: 200,
          updateInterval: 32,
        };
      case 'low':
        return {
          particles: false,
          particleCount: 0,
          shadows: false,
          antialiasing: false,
          resolution: 0.5,
          chartPoints: 100,
          updateInterval: 50,
        };
      case 'potato':
        return {
          particles: false,
          particleCount: 0,
          shadows: false,
          antialiasing: false,
          resolution: 0.25,
          chartPoints: 50,
          updateInterval: 100,
        };
    }
  }

  /**
   * Get current FPS statistics
   */
  getStats(): FPSStats {
    return this.calculateStats();
  }

  /**
   * Get FPS history for graphing
   */
  getHistory(): FPSStats[] {
    return [...this.statsHistory];
  }

  /**
   * Reset dropped frames counter
   */
  resetDroppedFrames(): void {
    this.droppedFramesCount = 0;
  }

  /**
   * Get formatted FPS report
   */
  getReport(): string {
    const stats = this.getStats();
    
    return `
=== FPS MONITOR REPORT ===
Current: ${stats.currentFPS.toFixed(1)} FPS
Average: ${stats.averageFPS.toFixed(1)} FPS
Min: ${stats.minFPS.toFixed(1)} FPS
Max: ${stats.maxFPS.toFixed(1)} FPS
Frame Time: ${stats.frameTime.toFixed(2)}ms
Avg Frame Time: ${stats.averageFrameTime.toFixed(2)}ms
Dropped Frames: ${stats.droppedFrames}
GPU Latency: ${stats.gpuCompositorLatency.toFixed(2)}ms
Quality: ${this.currentQuality.toUpperCase()}
=========================`.trim();
  }
}

/**
 * Quality settings interface for rendering components
 */
export interface QualitySettings {
  particles: boolean;
  particleCount: number;
  shadows: boolean;
  antialiasing: boolean;
  resolution: number;
  chartPoints: number;
  updateInterval: number; // ms between updates
}

// Export singleton instance
export const fpsMonitor = FPSMonitor.getInstance();

/**
 * Hook-friendly getter for current quality
 */
export function getCurrentQuality(): QualityLevel {
  return fpsMonitor.getQuality();
}

/**
 * Hook-friendly getter for quality settings
 */
export function getQualitySettings(): QualitySettings {
  return fpsMonitor.getQualitySettings();
}

/**
 * Auto-start monitoring when module loads
 */
if (typeof window !== 'undefined') {
  // Delay start to allow app initialization
  setTimeout(() => {
    fpsMonitor.start();
  }, 3000);
}
