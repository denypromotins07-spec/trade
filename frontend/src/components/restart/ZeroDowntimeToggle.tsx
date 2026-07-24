/**
 * ZeroDowntimeToggle Component - Master UI for Binary Hot-Swap
 * 
 * Provides the master toggle to initiate zero-downtime binary hot-swap,
 * displaying live shadow process synchronization metrics.
 * 
 * Features:
 * - Real-time shadow state visualization
 * - Memory budget enforcement display (8GB limit)
 * - Synchronization progress with checksum verification
 * - Safe abort capability during any phase
 */

import React, { useState, useEffect, useCallback } from 'react';

interface ShadowProcessMetrics {
  pid: number;
  state: ShadowState;
  memoryUsageBytes: number;
  syncProgress: number;
  lastHeartbeat: Date;
  checksumMatch: boolean;
}

type ShadowState = 
  | 'IDLE'
  | 'INITIALIZING'
  | 'SYNCING'
  | 'VERIFYING'
  | 'READY'
  | 'ACTIVE'
  | 'FAILED';

interface ZeroDowntimeToggleProps {
  onHotSwapComplete?: (success: boolean) => void;
  maxMemoryBytes?: number; // Default 8GB
}

const ZeroDowntimeToggle: React.FC<ZeroDowntimeToggleProps> = ({
  onHotSwapComplete,
  maxMemoryBytes = 8 * 1024 * 1024 * 1024, // 8GB default
}) => {
  const [isToggled, setIsToggled] = useState(false);
  const [shadowMetrics, setShadowMetrics] = useState<ShadowProcessMetrics | null>(null);
  const [hotSwapInProgress, setHotSwapInProgress] = useState(false);
  const [hotSwapPhase, setHotSwapPhase] = useState<string>('');
  const [logs, setLogs] = useState<string[]>([]);

  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toISOString();
    setLogs(prev => [...prev.slice(-99), `[${timestamp}] ${message}`]);
  }, []);

  // Poll shadow process status
  useEffect(() => {
    if (!hotSwapInProgress) return;

    const pollStatus = async () => {
      try {
        const response = await fetch('/api/restart/shadow/status');
        if (response.ok) {
          const data = await response.json();
          setShadowMetrics(data);
          
          // Update phase based on state
          const phaseMap: Record<ShadowState, string> = {
            IDLE: 'Waiting...',
            INITIALIZING: 'Initializing shadow process...',
            SYNCING: `Synchronizing state (${data.syncProgress.toFixed(1)}%)...`,
            VERIFYING: 'Verifying memory checksums...',
            READY: 'Shadow ready for handoff...',
            ACTIVE: 'Handoff complete - shadow is now primary',
            FAILED: 'Hot swap failed - rolling back',
          };
          setHotSwapPhase(phaseMap[data.state] || 'Unknown');

          // Check for completion
          if (data.state === 'ACTIVE') {
            setHotSwapInProgress(false);
            setIsToggled(false);
            addLog('Hot swap completed successfully');
            onHotSwapComplete?.(true);
          } else if (data.state === 'FAILED') {
            setHotSwapInProgress(false);
            addLog('Hot swap failed - see logs for details');
            onHotSwapComplete?.(false);
          }
        }
      } catch (error) {
        console.warn('Failed to poll shadow status:', error);
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 500); // 500ms polling
    return () => clearInterval(interval);
  }, [hotSwapInProgress, onHotSwapComplete, addLog]);

  // Initiate hot swap
  const handleToggle = async () => {
    if (isToggled) {
      // Toggle off - just reset state
      setIsToggled(false);
      return;
    }

    // Toggle on - initiate hot swap
    setIsToggled(true);
    setHotSwapInProgress(true);
    addLog('Initiating zero-downtime hot swap...');

    try {
      const response = await fetch('/api/restart/shadow/spawn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          maxMemoryBytes,
          preserveConnections: true,
          verifyChecksums: true,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to spawn shadow process');
      }

      const result = await response.json();
      addLog(`Shadow process spawned with PID: ${result.pid}`);
      setShadowMetrics({
        pid: result.pid,
        state: 'INITIALIZING',
        memoryUsageBytes: 0,
        syncProgress: 0,
        lastHeartbeat: new Date(),
        checksumMatch: false,
      });
    } catch (error) {
      addLog(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setHotSwapInProgress(false);
      setIsToggled(false);
      onHotSwapComplete?.(false);
    }
  };

  // Abort hot swap
  const handleAbort = async () => {
    addLog('Aborting hot swap...');
    
    try {
      await fetch('/api/restart/shadow/abort', { method: 'POST' });
      addLog('Hot swap aborted successfully');
    } catch (error) {
      addLog(`Abort failed: ${error instanceof Error ? error.message : 'Unknown'}`);
    }

    setHotSwapInProgress(false);
    setIsToggled(false);
  };

  // Format bytes to human readable
  const formatBytes = (bytes: number): string => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(2)} GB`;
  };

  // Calculate memory percentage
  const memoryPercentage = shadowMetrics 
    ? (shadowMetrics.memoryUsageBytes / maxMemoryBytes) * 100 
    : 0;

  // State color coding
  const getStateColor = (state: ShadowState): string => {
    switch (state) {
      case 'IDLE': return '#718096';
      case 'INITIALIZING': return '#4299e1';
      case 'SYNCING': return '#ed8936';
      case 'VERIFYING': return '#ed8936';
      case 'READY': return '#48bb78';
      case 'ACTIVE': return '#48bb78';
      case 'FAILED': return '#f56565';
      default: return '#718096';
    }
  };

  return (
    <div className="zero-downtime-toggle" style={{ padding: '20px', backgroundColor: '#1a202c', borderRadius: '8px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ color: '#ffffff', margin: 0 }}>Zero-Downtime Hot Swap</h2>
        
        {/* Toggle Switch */}
        <label className="toggle-switch" style={{ position: 'relative', display: 'inline-block', width: '60px', height: '34px' }}>
          <input
            type="checkbox"
            checked={isToggled}
            onChange={handleToggle}
            disabled={hotSwapInProgress}
            style={{ opacity: 0, width: 0, height: 0 }}
          />
          <span style={{
            position: 'absolute',
            cursor: hotSwapInProgress ? 'not-allowed' : 'pointer',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: isToggled ? '#48bb78' : '#4a5568',
            borderRadius: '34px',
            transition: 'background-color 0.3s',
          }}>
            <span style={{
              position: 'absolute',
              content: '""',
              height: '26px',
              width: '26px',
              left: isToggled ? 26 : 4,
              bottom: 4,
              backgroundColor: '#ffffff',
              borderRadius: '50%',
              transition: 'left 0.3s',
            }} />
          </span>
        </label>
      </div>

      {/* Status Panel */}
      {hotSwapInProgress && (
        <div className="status-panel" style={{ 
          padding: '20px', 
          backgroundColor: '#2d3748', 
          borderRadius: '8px',
          marginBottom: '20px',
        }}>
          {/* Current Phase */}
          <div style={{ marginBottom: '15px' }}>
            <span style={{ color: '#a0aec0', fontSize: '14px' }}>Current Phase:</span>
            <div style={{ color: '#ffffff', fontWeight: 'bold', marginTop: '5px' }}>
              {hotSwapPhase}
            </div>
          </div>

          {/* Progress Bar */}
          {shadowMetrics && (
            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: '#a0aec0', fontSize: '12px' }}>Synchronization Progress</span>
                <span style={{ color: '#ffffff', fontSize: '12px' }}>{shadowMetrics.syncProgress.toFixed(1)}%</span>
              </div>
              <div style={{ 
                width: '100%', 
                height: '8px', 
                backgroundColor: '#4a5568',
                borderRadius: '4px',
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${shadowMetrics.syncProgress}%`,
                  height: '100%',
                  backgroundColor: '#4299e1',
                  transition: 'width 0.3s',
                }} />
              </div>
            </div>
          )}

          {/* Memory Usage */}
          {shadowMetrics && (
            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: '#a0aec0', fontSize: '12px' }}>Memory Usage (8GB Limit)</span>
                <span style={{ 
                  color: memoryPercentage > 90 ? '#f56565' : '#a0aec0', 
                  fontSize: '12px',
                }}>
                  {formatBytes(shadowMetrics.memoryUsageBytes)} / {formatBytes(maxMemoryBytes)}
                </span>
              </div>
              <div style={{ 
                width: '100%', 
                height: '8px', 
                backgroundColor: '#4a5568',
                borderRadius: '4px',
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${Math.min(memoryPercentage, 100)}%`,
                  height: '100%',
                  backgroundColor: memoryPercentage > 90 ? '#f56565' : '#48bb78',
                  transition: 'width 0.3s',
                }} />
              </div>
            </div>
          )}

          {/* Checksum Status */}
          {shadowMetrics && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ color: '#a0aec0', fontSize: '12px' }}>Checksum Verification:</span>
              <span style={{ 
                color: shadowMetrics.checksumMatch ? '#48bb78' : '#f56565',
                fontWeight: 'bold',
              }}>
                {shadowMetrics.checksumMatch ? '✓ MATCH' : '✗ MISMATCH'}
              </span>
            </div>
          )}

          {/* Abort Button */}
          <button
            onClick={handleAbort}
            style={{
              marginTop: '15px',
              padding: '10px 20px',
              backgroundColor: '#f56565',
              color: '#ffffff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
          >
            ABORT HOT SWAP
          </button>
        </div>
      )}

      {/* Shadow Process Info */}
      {shadowMetrics && !hotSwapInProgress && shadowMetrics.state === 'ACTIVE' && (
        <div className="shadow-info" style={{ 
          padding: '15px', 
          backgroundColor: 'rgba(72, 187, 120, 0.1)',
          border: '1px solid #48bb78',
          borderRadius: '8px',
        }}>
          <div style={{ color: '#48bb78', fontWeight: 'bold' }}>
            ✓ Shadow process active (PID: {shadowMetrics.pid})
          </div>
        </div>
      )}

      {/* Logs Panel */}
      {logs.length > 0 && (
        <div className="logs-panel" style={{ 
          marginTop: '20px',
          padding: '15px',
          backgroundColor: '#000000',
          borderRadius: '8px',
          maxHeight: '200px',
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: '12px',
        }}>
          {logs.map((log, index) => (
            <div key={index} style={{ color: '#a0aec0', marginBottom: '4px' }}>
              {log}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ZeroDowntimeToggle;
