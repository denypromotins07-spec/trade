/**
 * Global Error Boundary Component
 * 
 * Catches React rendering crashes and displays a cyberpunk "System Halted" UI.
 * Preserves master /KILL switch functionality even during critical failures.
 * 
 * Cyberpunk aesthetic: Glitch effects, neon red alerts, terminal-style error display.
 */

import React, { Component, ErrorInfo, ReactNode, useState, useCallback } from 'react';
import { rpcClient } from '../../lib/ipc/rpc_client';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onCrash?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  crashTime: number | null;
}

/**
 * Class-based error boundary for catching render errors
 */
export class GlobalBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      crashTime: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
      crashTime: Date.now(),
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    
    console.error('[GLOBAL_BOUNDARY] Critical error caught:', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
    });

    // Report crash to backend for SOUL.md logging
    this.reportCrash(error, errorInfo);

    // Invoke custom crash handler if provided
    if (this.props.onCrash) {
      this.props.onCrash(error, errorInfo);
    }
  }

  /**
   * Send crash report to Rust backend
   */
  private async reportCrash(error: Error, errorInfo: ErrorInfo): Promise<void> {
    try {
      const crashReport = {
        timestamp: Date.now(),
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        userAgent: navigator.userAgent,
        url: window.location.href,
      };

      // Attempt to send via RPC if available
      await rpcClient.execute('crash_report', crashReport).catch(() => {
        // Fallback: store in localStorage for later transmission
        const existingReports = JSON.parse(
          localStorage.getItem('pending_crash_reports') || '[]'
        );
        existingReports.push(crashReport);
        localStorage.setItem('pending_crash_reports', JSON.stringify(existingReports));
      });
    } catch (reportError) {
      console.error('[GLOBAL_BOUNDARY] Failed to report crash:', reportError);
    }
  }

  /**
   * Attempt recovery by reloading the component tree
   */
  handleRecovery = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      crashTime: null,
    });
  };

  /**
   * Execute emergency /KILL command
   */
  handleEmergencyKill = async (): Promise<void> => {
    try {
      await rpcClient.killTrading();
    } catch (error) {
      console.error('[GLOBAL_BOUNDARY] Emergency kill failed:', error);
    }
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <SystemHaltedUI
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          crashTime={this.state.crashTime}
          onRecovery={this.handleRecovery}
          onEmergencyKill={this.handleEmergencyKill}
        />
      );
    }

    return this.props.children;
  }
}

/**
 * Cyberpunk-styled "System Halted" error display
 */
interface SystemHaltedUIProps {
  error: Error | null;
  errorInfo: ErrorInfo | null;
  crashTime: number | null;
  onRecovery: () => void;
  onEmergencyKill: () => void;
}

const SystemHaltedUI: React.FC<SystemHaltedUIProps> = ({
  error,
  errorInfo,
  crashTime,
  onRecovery,
  onEmergencyKill,
}) => {
  const [isKilling, setIsKilling] = useState(false);

  const handleKillClick = useCallback(async () => {
    setIsKilling(true);
    await onEmergencyKill();
    setIsKilling(false);
  }, [onEmergencyKill]);

  const formatTime = (timestamp: number | null): string => {
    if (!timestamp) return 'UNKNOWN';
    return new Date(timestamp).toISOString();
  };

  return (
    <div className="system-halted-container">
      {/* Animated background grid */}
      <div className="cyber-grid" />
      
      {/* Glitch overlay */}
      <div className="glitch-overlay" data-text="SYSTEM HALTED" />
      
      {/* Main error panel */}
      <div className="error-panel">
        <div className="error-header">
          <span className="alert-icon">⚠️</span>
          <h1 className="error-title">SYSTEM HALTED</h1>
          <span className="alert-icon">⚠️</span>
        </div>

        <div className="error-timestamp">
          CRASH TIME: <span className="neon-text">{formatTime(crashTime)}</span>
        </div>

        <div className="error-details">
          <div className="error-message">
            <span className="label">ERROR:</span>
            <code>{error?.message || 'Unknown error'}</code>
          </div>

          {error?.stack && (
            <div className="error-stack">
              <span className="label">STACK TRACE:</span>
              <pre className="stack-trace">{error.stack}</pre>
            </div>
          )}

          {errorInfo?.componentStack && (
            <div className="error-component-stack">
              <span className="label">COMPONENT STACK:</span>
              <pre className="component-trace">{errorInfo.componentStack}</pre>
            </div>
          )}
        </div>

        {/* Action buttons - /KILL always available */}
        <div className="action-buttons">
          <button
            className="btn-recovery"
            onClick={onRecovery}
            disabled={isKilling}
          >
            <span className="btn-icon">↻</span>
            ATTEMPT RECOVERY
          </button>

          <button
            className="btn-kill"
            onClick={handleKillClick}
            disabled={isKilling}
          >
            <span className="btn-icon">{isKilling ? '⏳' : '☠️'}</span>
            {isKilling ? 'EXECUTING...' : 'EMERGENCY /KILL'}
          </button>
        </div>

        {/* Status indicator */}
        <div className="status-indicator halted">
          <span className="status-dot" />
          CRITICAL FAILURE - MANUAL INTERVENTION REQUIRED
        </div>
      </div>

      {/* CSS styles injected dynamically */}
      <style jsx>{`
        .system-halted-container {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: #050510;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: 'Courier New', monospace;
          overflow: hidden;
          z-index: 999999;
        }

        .cyber-grid {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-image: 
            linear-gradient(rgba(0, 243, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.1) 1px, transparent 1px);
          background-size: 50px 50px;
          animation: grid-pulse 2s ease-in-out infinite;
        }

        @keyframes grid-pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }

        .glitch-overlay {
          position: absolute;
          font-size: 8rem;
          font-weight: bold;
          color: rgba(255, 0, 85, 0.1);
          text-transform: uppercase;
          pointer-events: none;
          animation: glitch 1s infinite;
        }

        @keyframes glitch {
          0% { transform: translate(0); }
          20% { transform: translate(-2px, 2px); }
          40% { transform: translate(-2px, -2px); }
          60% { transform: translate(2px, 2px); }
          80% { transform: translate(2px, -2px); }
          100% { transform: translate(0); }
        }

        .error-panel {
          position: relative;
          z-index: 10;
          background: rgba(10, 10, 20, 0.95);
          border: 2px solid #ff0055;
          border-radius: 8px;
          padding: 2rem;
          max-width: 800px;
          width: 90%;
          box-shadow: 
            0 0 20px rgba(255, 0, 85, 0.5),
            inset 0 0 60px rgba(255, 0, 85, 0.1);
        }

        .error-header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .error-title {
          color: #ff0055;
          font-size: 2rem;
          text-transform: uppercase;
          letter-spacing: 4px;
          margin: 0;
          text-shadow: 0 0 10px rgba(255, 0, 85, 0.8);
        }

        .alert-icon {
          font-size: 2rem;
          animation: blink 0.5s step-end infinite;
        }

        @keyframes blink {
          50% { opacity: 0; }
        }

        .error-timestamp {
          text-align: center;
          color: #00f3ff;
          margin-bottom: 1.5rem;
          font-size: 0.9rem;
        }

        .neon-text {
          color: #00f3ff;
          text-shadow: 0 0 5px rgba(0, 243, 255, 0.8);
        }

        .error-details {
          background: rgba(0, 0, 0, 0.5);
          border: 1px solid #333;
          border-radius: 4px;
          padding: 1rem;
          margin-bottom: 1.5rem;
          max-height: 300px;
          overflow-y: auto;
        }

        .error-message,
        .error-stack,
        .error-component-stack {
          margin-bottom: 1rem;
        }

        .error-message:last-child,
        .error-stack:last-child,
        .error-component-stack:last-child {
          margin-bottom: 0;
        }

        .label {
          color: #ff0055;
          font-weight: bold;
          display: block;
          margin-bottom: 0.5rem;
        }

        code,
        pre {
          color: #00f3ff;
          font-family: 'Courier New', monospace;
          font-size: 0.85rem;
          white-space: pre-wrap;
          word-break: break-all;
        }

        .stack-trace,
        .component-trace {
          background: rgba(0, 243, 255, 0.05);
          padding: 0.5rem;
          border-radius: 2px;
          max-height: 150px;
          overflow-y: auto;
        }

        .action-buttons {
          display: flex;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .btn-recovery,
        .btn-kill {
          flex: 1;
          padding: 1rem;
          border: none;
          border-radius: 4px;
          font-family: 'Courier New', monospace;
          font-size: 1rem;
          font-weight: bold;
          cursor: pointer;
          text-transform: uppercase;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          transition: all 0.2s ease;
        }

        .btn-recovery {
          background: linear-gradient(135deg, #00f3ff, #0099ff);
          color: #050510;
        }

        .btn-recovery:hover:not(:disabled) {
          box-shadow: 0 0 20px rgba(0, 243, 255, 0.6);
          transform: translateY(-2px);
        }

        .btn-kill {
          background: linear-gradient(135deg, #ff0055, #ff3377);
          color: #fff;
        }

        .btn-kill:hover:not(:disabled) {
          box-shadow: 0 0 20px rgba(255, 0, 85, 0.6);
          transform: translateY(-2px);
        }

        button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-icon {
          font-size: 1.2rem;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.75rem;
          border-radius: 4px;
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 2px;
        }

        .status-indicator.halted {
          background: rgba(255, 0, 85, 0.1);
          color: #ff0055;
          border: 1px solid #ff0055;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #ff0055;
          animation: pulse-red 1s ease-in-out infinite;
        }

        @keyframes pulse-red {
          0%, 100% { box-shadow: 0 0 5px rgba(255, 0, 85, 0.8); }
          50% { box-shadow: 0 0 15px rgba(255, 0, 85, 1); }
        }
      `}</style>
    </div>
  );
};

export default GlobalBoundary;
