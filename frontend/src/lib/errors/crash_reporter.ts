/**
 * Automated Crash Reporter
 * 
 * Bundles React error stacks and sends them to the Rust backend for permanent logging
 * into the SOUL.md ledger. Includes system state snapshot for post-mortem analysis.
 * 
 * Cyberpunk aesthetic: "Black box recorder" metaphor with encrypted transmission visuals.
 */

import { rpcClient } from '../ipc/rpc_client';

export interface CrashReport {
  id: string;
  timestamp: number;
  error: {
    name: string;
    message: string;
    stack?: string;
  };
  componentStack?: string;
  systemState: SystemStateSnapshot;
  tradingContext: TradingContextSnapshot;
  networkInfo: NetworkInfo;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface SystemStateSnapshot {
  memoryUsage: number;
  heapSizeLimit: number;
  gpuMemory?: number;
  openWebGLContexts: number;
  activeWebSockets: number;
  pendingRequests: number;
  userAgent: string;
  screenResolution: string;
  language: string;
  timezone: string;
}

export interface TradingContextSnapshot {
  isRunning: boolean;
  openPositionsCount: number;
  lastOrderTime: number | null;
  lastTradeTime: number | null;
  sessionStartTime: number;
  commandsExecuted: number;
}

export interface NetworkInfo {
  online: boolean;
  connectionType?: string;
  downlink?: number;
  rtt?: number;
}

/**
 * Crash reporter class for bundling and transmitting error data
 */
export class CrashReporter {
  private static instance: CrashReporter;
  private reportQueue: CrashReport[] = [];
  private isTransmitting: boolean = false;
  private readonly MAX_QUEUE_SIZE = 10;
  private readonly SOUL_MD_ENDPOINT = '/api/soul/ledger';

  private constructor() {}

  /**
   * Get singleton instance
   */
  static getInstance(): CrashReporter {
    if (!CrashReporter.instance) {
      CrashReporter.instance = new CrashReporter();
    }
    return CrashReporter.instance;
  }

  /**
   * Capture and report a crash
   */
  async report(
    error: Error,
    componentStack?: string,
    additionalContext?: Record<string, unknown>
  ): Promise<void> {
    const report = await this.buildCrashReport(error, componentStack, additionalContext);
    
    // Add to queue
    this.reportQueue.push(report);
    
    // Trim queue if needed
    if (this.reportQueue.length > this.MAX_QUEUE_SIZE) {
      this.reportQueue.shift();
    }

    console.log('[CRASH_REPORTER] Crash captured:', {
      id: report.id,
      severity: report.severity,
      error: error.message,
    });

    // Attempt immediate transmission
    await this.transmitQueue();
  }

  /**
   * Build comprehensive crash report
   */
  private async buildCrashReport(
    error: Error,
    componentStack?: string,
    additionalContext?: Record<string, unknown>
  ): Promise<CrashReport> {
    const systemState = this.captureSystemState();
    const tradingContext = await this.captureTradingContext();
    const networkInfo = this.captureNetworkInfo();

    // Determine severity based on error type and system state
    const severity = this.determineSeverity(error, systemState, tradingContext);

    return {
      id: this.generateReportId(),
      timestamp: Date.now(),
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack,
      },
      componentStack,
      systemState,
      tradingContext,
      networkInfo,
      severity,
      ...additionalContext,
    };
  }

  /**
   * Capture current system state
   */
  private captureSystemState(): SystemStateSnapshot {
    let memoryUsage = 0;
    let heapSizeLimit = 0;
    let gpuMemory: number | undefined;

    // Check for Performance Memory API (Chrome only)
    if ('memory' in performance) {
      const mem = (performance as { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
      if (mem) {
        memoryUsage = mem.usedJSHeapSize;
        heapSizeLimit = mem.jsHeapSizeLimit;
      }
    }

    // Count WebGL contexts
    let openWebGLContexts = 0;
    try {
      const canvases = document.querySelectorAll('canvas');
      canvases.forEach((canvas) => {
        try {
          const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
          if (gl && !gl.isContextLost()) {
            openWebGLContexts++;
          }
        } catch {
          // Ignore
        }
      });
    } catch {
      // Ignore
    }

    // Count active WebSockets (tracked globally)
    const activeWebSockets = (window as { __activeWebSockets?: number }).__activeWebSockets || 0;

    return {
      memoryUsage,
      heapSizeLimit,
      gpuMemory,
      openWebGLContexts,
      activeWebSockets,
      pendingRequests: 0, // Would be tracked by RPC client
      userAgent: navigator.userAgent,
      screenResolution: `${screen.width}x${screen.height}`,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    };
  }

  /**
   * Capture trading context from localStorage/state
   */
  private async captureTradingContext(): Promise<TradingContextSnapshot> {
    try {
      const storedState = localStorage.getItem('nautilus_trading_state');
      const state = storedState ? JSON.parse(storedState) : {};

      return {
        isRunning: state.isRunning || false,
        openPositionsCount: state.openPositions?.length || 0,
        lastOrderTime: state.lastOrderTime || null,
        lastTradeTime: state.lastTradeTime || null,
        sessionStartTime: state.sessionStartTime || Date.now(),
        commandsExecuted: state.commandsExecuted || 0,
      };
    } catch {
      return {
        isRunning: false,
        openPositionsCount: 0,
        lastOrderTime: null,
        lastTradeTime: null,
        sessionStartTime: Date.now(),
        commandsExecuted: 0,
      };
    }
  }

  /**
   * Capture network information
   */
  private captureNetworkInfo(): NetworkInfo {
    const info: NetworkInfo = {
      online: navigator.onLine,
    };

    // Check for Network Information API
    if ('connection' in navigator) {
      const conn = (navigator as { connection?: { effectiveType?: string; downlink?: number; rtt?: number } }).connection;
      if (conn) {
        info.connectionType = conn.effectiveType;
        info.downlink = conn.downlink;
        info.rtt = conn.rtt;
      }
    }

    return info;
  }

  /**
   * Determine crash severity
   */
  private determineSeverity(
    error: Error,
    systemState: SystemStateSnapshot,
    tradingContext: TradingContextSnapshot
  ): 'low' | 'medium' | 'high' | 'critical' {
    // Critical: Trading was running with open positions
    if (tradingContext.isRunning && tradingContext.openPositionsCount > 0) {
      return 'critical';
    }

    // High: Memory pressure or WebGL context loss
    if (systemState.memoryUsage > systemState.heapSizeLimit * 0.9) {
      return 'high';
    }
    if (systemState.openWebGLContexts === 0 && document.querySelectorAll('canvas').length > 0) {
      return 'high';
    }

    // Medium: Network issues
    if (!systemState.online) {
      return 'medium';
    }

    // Low: Standard errors
    return 'low';
  }

  /**
   * Generate unique report ID
   */
  private generateReportId(): string {
    return `CRASH_${Date.now()}_${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
  }

  /**
   * Transmit queued reports to backend
   */
  async transmitQueue(): Promise<void> {
    if (this.isTransmitting || this.reportQueue.length === 0) {
      return;
    }

    this.isTransmitting = true;

    try {
      while (this.reportQueue.length > 0) {
        const report = this.reportQueue[0];
        
        try {
          await this.sendToBackend(report);
          this.reportQueue.shift();
          console.log('[CRASH_REPORTER] Report transmitted:', report.id);
        } catch (error) {
          console.error('[CRASH_REPORTER] Failed to transmit report:', error);
          
          // If offline, save to localStorage for later
          if (!navigator.onLine) {
            this.saveToLocalStorage(report);
            this.reportQueue.shift();
          } else {
            // Stop on first failure to avoid infinite loop
            break;
          }
        }
      }
    } finally {
      this.isTransmitting = false;
    }
  }

  /**
   * Send report to Rust backend
   */
  private async sendToBackend(report: CrashReport): Promise<void> {
    try {
      // Try RPC first
      await rpcClient.execute('crash_report', report);
    } catch {
      // Fallback to fetch
      const response = await fetch(this.SOUL_MD_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(report),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    }
  }

  /**
   * Save report to localStorage for later transmission
   */
  private saveToLocalStorage(report: CrashReport): void {
    try {
      const key = `pending_crash_${report.id}`;
      localStorage.setItem(key, JSON.stringify(report));
      
      // Also add to index
      const indexKey = 'pending_crash_index';
      const index = JSON.parse(localStorage.getItem(indexKey) || '[]');
      index.push(report.id);
      localStorage.setItem(indexKey, JSON.stringify(index));
      
      console.log('[CRASH_REPORTER] Report saved to localStorage:', report.id);
    } catch (error) {
      console.error('[CRASH_REPORTER] Failed to save to localStorage:', error);
    }
  }

  /**
   * Flush any pending reports from localStorage
   */
  async flushPendingReports(): Promise<void> {
    try {
      const indexKey = 'pending_crash_index';
      const index = JSON.parse(localStorage.getItem(indexKey) || '[]');
      
      for (const id of index) {
        const key = `pending_crash_${id}`;
        const reportJson = localStorage.getItem(key);
        
        if (reportJson) {
          const report = JSON.parse(reportJson) as CrashReport;
          
          try {
            await this.sendToBackend(report);
            localStorage.removeItem(key);
            console.log('[CRASH_REPORTER] Flushed pending report:', id);
          } catch (error) {
            console.error('[CRASH_REPORTER] Failed to flush report:', error);
          }
        }
      }
      
      // Clear index
      localStorage.removeItem(indexKey);
    } catch (error) {
      console.error('[CRASH_REPORTER] Failed to flush pending reports:', error);
    }
  }

  /**
   * Get all queued reports (for debugging)
   */
  getQueuedReports(): CrashReport[] {
    return [...this.reportQueue];
  }

  /**
   * Format report for SOUL.md ledger entry
   */
  formatForSoulLedger(report: CrashReport): string {
    const date = new Date(report.timestamp).toISOString();
    const severityEmoji = {
      low: 'ℹ️',
      medium: '⚠️',
      high: '🔴',
      critical: '☠️',
    }[report.severity];

    return `## ${severityEmoji} CRASH REPORT: ${report.id}

**Timestamp:** ${date}
**Severity:** ${report.severity.toUpperCase()}
**Error:** ${report.error.name}: ${report.error.message}

### Stack Trace
\`\`\`
${report.error.stack || 'No stack trace available'}
\`\`\`

### Component Stack
\`\`\`
${report.componentStack || 'No component stack available'}
\`\`\`

### System State
- Memory Usage: ${(report.systemState.memoryUsage / 1024 / 1024).toFixed(2)} MB
- Heap Limit: ${(report.systemState.heapSizeLimit / 1024 / 1024).toFixed(2)} MB
- WebGL Contexts: ${report.systemState.openWebGLContexts}
- Active WebSockets: ${report.systemState.activeWebSockets}
- Screen: ${report.systemState.screenResolution}
- Browser: ${report.systemState.userAgent.split(' ').pop()}

### Trading Context
- Running: ${report.tradingContext.isRunning ? 'YES' : 'NO'}
- Open Positions: ${report.tradingContext.openPositionsCount}
- Last Order: ${report.tradingContext.lastOrderTime ? new Date(report.tradingContext.lastOrderTime).toISOString() : 'N/A'}
- Session Start: ${new Date(report.tradingContext.sessionStartTime).toISOString()}

### Network
- Online: ${report.networkInfo.online ? 'YES' : 'NO'}
- Connection: ${report.networkInfo.connectionType || 'UNKNOWN'}
- Downlink: ${report.networkInfo.downlink || 'N/A'} Mbps
- RTT: ${report.networkInfo.rtt || 'N/A'} ms

---
*Report generated by Nautilus Crash Reporter v1.0*
`;
  }
}

// Export singleton instance
export const crashReporter = CrashReporter.getInstance();

/**
 * Hook-friendly wrapper for crash reporting
 */
export function reportCrash(
  error: Error,
  componentStack?: string,
  additionalContext?: Record<string, unknown>
): Promise<void> {
  return crashReporter.report(error, componentStack, additionalContext);
}
