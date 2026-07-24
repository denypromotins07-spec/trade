/**
 * RayCluster.tsx - Python Ray Worker Topology Manager
 * 
 * Manages Python Ray worker topology with manual memory quota adjustments
 * and forced garbage collection triggers for the 4GB AI ecosystem.
 * Provides real-time visualization of cluster health and resource allocation.
 * 
 * Features:
 * - Real-time worker topology visualization
 * - Memory quota management per worker
 * - Forced GC trigger for Python workers
 * - Resource utilization monitoring
 * - Cyberpunk terminal aesthetic
 */

import React, { useState, useCallback, useEffect } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

interface RayWorker {
  id: string;
  name: string;
  status: 'running' | 'idle' | 'error' | 'starting';
  cpuUsage: number; // 0-100
  memoryUsed: number; // MB
  memoryQuota: number; // MB
  gpuUsage?: number; // 0-100 (if GPU available)
  tasksCompleted: number;
  lastHeartbeat: number;
}

interface RayClusterConfig {
  maxWorkers: number;
  defaultMemoryQuota: number; // MB per worker
  enableGPU: boolean;
  autoScale: boolean;
  gcThreshold: number; // Trigger GC when memory > this %
}

interface RayClusterProps {
  workers: RayWorker[];
  config: RayClusterConfig;
  onConfigChange?: (config: RayClusterConfig) => void;
  onWorkerAction?: (workerId: string, action: 'restart' | 'stop' | 'gc') => Promise<void>;
  onAddWorker?: () => void;
  onRemoveWorker?: (workerId: string) => void;
  className?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const RayCluster: React.FC<RayClusterProps> = ({
  workers,
  config,
  onConfigChange,
  onWorkerAction,
  onAddWorker,
  onRemoveWorker,
  className = ''
}) => {
  const [selectedWorker, setSelectedWorker] = useState<string | null>(null);
  const [gcInProgress, setGcInProgress] = useState<Set<string>>(new Set());
  const [terminalOutput, setTerminalOutput] = useState<string[]>([]);

  // Add terminal log entry
  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setTerminalOutput(prev => [...prev.slice(-50), `[${timestamp}] ${message}`]);
  }, []);

  // Handle worker action
  const handleWorkerAction = async (workerId: string, action: 'restart' | 'stop' | 'gc') => {
    if (!onWorkerAction) return;
    
    addLog(`Executing ${action.toUpperCase()} on worker ${workerId}...`);
    
    if (action === 'gc') {
      setGcInProgress(prev => new Set(prev).add(workerId));
    }
    
    try {
      await onWorkerAction(workerId, action);
      addLog(`${action.toUpperCase()} completed on ${workerId}`);
    } catch (error) {
      addLog(`ERROR: ${action.toUpperCase()} failed on ${workerId}: ${error}`);
    } finally {
      if (action === 'gc') {
        setGcInProgress(prev => {
          const next = new Set(prev);
          next.delete(workerId);
          return next;
        });
      }
    }
  };

  // Handle memory quota change
  const handleMemoryQuotaChange = (workerId: string, newQuota: number) => {
    addLog(`Adjusting memory quota for ${workerId}: ${newQuota}MB`);
    // In production, this would send IPC to Rust backend
  };

  // Calculate cluster stats
  const clusterStats = {
    totalWorkers: workers.length,
    runningWorkers: workers.filter(w => w.status === 'running').length,
    totalMemoryUsed: workers.reduce((sum, w) => sum + w.memoryUsed, 0),
    totalMemoryQuota: workers.reduce((sum, w) => sum + w.memoryQuota, 0),
    avgCpuUsage: workers.reduce((sum, w) => sum + w.cpuUsage, 0) / (workers.length || 1),
    totalTasks: workers.reduce((sum, w) => sum + w.tasksCompleted, 0)
  };

  return (
    <div className={`p-6 ${className}`}>
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-cyan-400 font-mono">RAY CLUSTER MANAGER</h2>
          <p className="text-gray-500 text-sm mt-1">Python worker topology & memory management</p>
        </div>
        
        {/* Quick Stats */}
        <div className="flex gap-4 text-xs font-mono">
          <div className="text-center">
            <div className="text-gray-500">WORKERS</div>
            <div className="text-cyan-400">{clusterStats.runningWorkers}/{clusterStats.totalWorkers}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">MEMORY</div>
            <div className="text-purple-400">{(clusterStats.totalMemoryUsed / 1024).toFixed(1)}GB</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">CPU AVG</div>
            <div className={clusterStats.avgCpuUsage > 80 ? 'text-red-400' : 'text-green-400'}>
              {clusterStats.avgCpuUsage.toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
      
      {/* Cluster Configuration */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 mb-6">
        <h3 className="text-sm font-mono text-gray-400 mb-4">CLUSTER CONFIGURATION</h3>
        
        <div className="grid grid-cols-4 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Max Workers</label>
            <input
              type="number"
              value={config.maxWorkers}
              onChange={e => onConfigChange?.({ ...config, maxWorkers: parseInt(e.target.value) })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none"
              min={1}
              max={64}
            />
          </div>
          
          <div>
            <label className="block text-xs text-gray-500 mb-1">Default Quota (MB)</label>
            <input
              type="number"
              value={config.defaultMemoryQuota}
              onChange={e => onConfigChange?.({ ...config, defaultMemoryQuota: parseInt(e.target.value) })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none"
              min={256}
              max={8192}
              step={256}
            />
          </div>
          
          <div>
            <label className="block text-xs text-gray-500 mb-1">GC Threshold (%)</label>
            <input
              type="number"
              value={config.gcThreshold}
              onChange={e => onConfigChange?.({ ...config, gcThreshold: parseInt(e.target.value) })}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none"
              min={50}
              max={95}
            />
          </div>
          
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={config.autoScale}
                onChange={e => onConfigChange?.({ ...config, autoScale: e.target.checked })}
                className="w-4 h-4 rounded bg-gray-800 border-gray-700 text-cyan-500 focus:ring-cyan-500"
              />
              Auto-Scale
            </label>
            
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={config.enableGPU}
                onChange={e => onConfigChange?.({ ...config, enableGPU: e.target.checked })}
                className="w-4 h-4 rounded bg-gray-800 border-gray-700 text-cyan-500 focus:ring-cyan-500"
              />
              GPU Enabled
            </label>
          </div>
        </div>
      </div>
      
      {/* Worker Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-6">
        {workers.map(worker => (
          <div
            key={worker.id}
            className={`bg-gray-900/50 border rounded-lg p-4 cursor-pointer transition-all ${
              selectedWorker === worker.id 
                ? 'border-cyan-500 bg-gray-800/50' 
                : 'border-gray-800 hover:border-gray-700'
            }`}
            onClick={() => setSelectedWorker(worker.id)}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  worker.status === 'running' ? 'bg-green-500 animate-pulse' :
                  worker.status === 'idle' ? 'bg-yellow-500' :
                  worker.status === 'error' ? 'bg-red-500' : 'bg-gray-500'
                }`} />
                <span className="font-mono text-sm text-white">{worker.name}</span>
              </div>
              
              <span className={`text-xs px-2 py-0.5 rounded ${
                worker.status === 'running' ? 'bg-green-500/20 text-green-400' :
                worker.status === 'idle' ? 'bg-yellow-500/20 text-yellow-400' :
                worker.status === 'error' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'
              }`}>
                {worker.status.toUpperCase()}
              </span>
            </div>
            
            {/* Metrics */}
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-gray-500">CPU:</span>
                <span className={worker.cpuUsage > 80 ? 'text-red-400' : 'text-cyan-400'}>
                  {worker.cpuUsage.toFixed(0)}%
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-500">Memory:</span>
                <span className={worker.memoryUsed / worker.memoryQuota > 0.9 ? 'text-red-400' : 'text-purple-400'}>
                  {(worker.memoryUsed / 1024).toFixed(1)}GB / {(worker.memoryQuota / 1024).toFixed(1)}GB
                </span>
              </div>
              
              {worker.gpuUsage !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-500">GPU:</span>
                  <span className="text-orange-400">{worker.gpuUsage.toFixed(0)}%</span>
                </div>
              )}
              
              <div className="flex justify-between">
                <span className="text-gray-500">Tasks:</span>
                <span className="text-gray-300">{worker.tasksCompleted.toLocaleString()}</span>
              </div>
            </div>
            
            {/* CPU Progress Bar */}
            <div className="mt-3 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all ${
                  worker.cpuUsage > 80 ? 'bg-red-500' : worker.cpuUsage > 50 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${worker.cpuUsage}%` }}
              />
            </div>
            
            {/* Memory Progress Bar */}
            <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all ${
                  worker.memoryUsed / worker.memoryQuota > 0.9 ? 'bg-red-500' : 'bg-purple-500'
                }`}
                style={{ width: `${(worker.memoryUsed / worker.memoryQuota) * 100}%` }}
              />
            </div>
          </div>
        ))}
        
        {/* Add Worker Button */}
        <button
          onClick={onAddWorker}
          disabled={workers.length >= config.maxWorkers}
          className={`border-2 border-dashed rounded-lg p-4 flex flex-col items-center justify-center gap-2 transition-all ${
            workers.length >= config.maxWorkers
              ? 'border-gray-800 text-gray-700 cursor-not-allowed'
              : 'border-gray-700 text-gray-500 hover:border-cyan-500 hover:text-cyan-400'
          }`}
        >
          <span className="text-2xl">+</span>
          <span className="text-xs font-mono">ADD WORKER</span>
        </button>
      </div>
      
      {/* Selected Worker Actions */}
      {selectedWorker && (
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-mono text-gray-400">
              WORKER ACTIONS: <span className="text-cyan-400">{workers.find(w => w.id === selectedWorker)?.name}</span>
            </h3>
            <button
              onClick={() => setSelectedWorker(null)}
              className="text-gray-500 hover:text-white"
            >
              ✕
            </button>
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={() => handleWorkerAction(selectedWorker, 'restart')}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-xs font-mono text-white transition-colors"
            >
              RESTART WORKER
            </button>
            
            <button
              onClick={() => handleWorkerAction(selectedWorker, 'gc')}
              disabled={gcInProgress.has(selectedWorker)}
              className={`px-4 py-2 rounded text-xs font-mono transition-colors ${
                gcInProgress.has(selectedWorker)
                  ? 'bg-yellow-500/20 text-yellow-500 animate-pulse'
                  : 'bg-cyan-600 hover:bg-cyan-500 text-white'
              }`}
            >
              {gcInProgress.has(selectedWorker) ? 'RUNNING GC...' : 'FORCE GARBAGE COLLECTION'}
            </button>
            
            <button
              onClick={() => handleWorkerAction(selectedWorker, 'stop')}
              className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded text-xs font-mono text-red-400 transition-colors"
            >
              STOP WORKER
            </button>
            
            <div className="ml-auto flex items-center gap-2">
              <label className="text-xs text-gray-500">Memory Quota:</label>
              <input
                type="range"
                min={256}
                max={4096}
                step={256}
                value={workers.find(w => w.id === selectedWorker)?.memoryQuota || 1024}
                onChange={e => handleMemoryQuotaChange(selectedWorker, parseInt(e.target.value))}
                className="w-32 accent-cyan-500"
              />
              <span className="text-xs font-mono text-cyan-400 w-16 text-right">
                {(workers.find(w => w.id === selectedWorker)?.memoryQuota || 1024) / 1024}GB
              </span>
            </div>
          </div>
        </div>
      )}
      
      {/* Terminal Output */}
      <div className="bg-black border border-gray-800 rounded-lg p-4 font-mono text-xs h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700">
        <div className="text-gray-500 mb-2">=== CLUSTER TERMINAL ===</div>
        {terminalOutput.length === 0 ? (
          <div className="text-gray-600">No recent activity...</div>
        ) : (
          terminalOutput.map((line, i) => (
            <div key={i} className={`${line.includes('ERROR') ? 'text-red-400' : line.includes('GC') ? 'text-yellow-400' : 'text-gray-400'}`}>
              {line}
            </div>
          ))
        )}
      </div>
      
      {/* Footer Info */}
      <div className="mt-4 flex items-center justify-between text-xs text-gray-500 font-mono">
        <span>Ray v2.9.0 • Python 3.11 • 4GB AI Ecosystem</span>
        <span>Total Tasks Completed: {clusterStats.totalTasks.toLocaleString()}</span>
      </div>
    </div>
  );
};

export default RayCluster;
