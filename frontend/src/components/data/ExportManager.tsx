/**
 * ExportManager.tsx - Data Exploration: Parquet/CSV Export Manager
 * 
 * Manages Parquet and CSV exports with a cyberpunk-styled progress ring
 * that tracks the Rust backend's asynchronous NVMe disk writes.
 * 
 * Features:
 * - Multi-format export (Parquet, CSV, JSON)
 * - Real-time progress tracking via Rust IPC
 * - Cyberpunk progress ring animation
 * - Batch export queue management
 * - NVMe write speed visualization
 */

'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

type ExportFormat = 'parquet' | 'csv' | 'json';
type ExportStatus = 'pending' | 'writing' | 'complete' | 'error';

interface ExportJob {
  id: string;
  name: string;
  format: ExportFormat;
  status: ExportStatus;
  progress: number; // 0-100
  totalRows: number;
  writtenRows: number;
  writeSpeedMBps: number;
  error?: string;
  createdAt: number;
  completedAt?: number;
}

interface ExportManagerProps {
  maxConcurrent?: number;
  onExportStart?: (job: ExportJob) => void;
  onExportComplete?: (job: ExportJob) => void;
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const MAX_CONCURRENT_DEFAULT = 3;
const RING_SIZE = 120;
const STROKE_WIDTH = 8;

const FORMAT_CONFIG: Record<ExportFormat, { icon: string; color: string; label: string; extension: string }> = {
  parquet: { icon: '📦', color: '#ff6600', label: 'Parquet', extension: '.parquet' },
  csv: { icon: '📄', color: '#00ff88', label: 'CSV', extension: '.csv' },
  json: { icon: '{}', color: '#00ccff', label: 'JSON', extension: '.json' },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generates a unique ID for export jobs
 */
const generateId = (): string => {
  return `export-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
};

/**
 * Formats file size in human-readable format
 */
const formatSize = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let unitIndex = 0;
  let size = bytes;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
};

/**
 * Formats duration in human-readable format
 */
const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Cyberpunk Progress Ring Component
 */
interface ProgressRingProps {
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  showLabel?: boolean;
  label?: string;
}

const ProgressRing: React.FC<ProgressRingProps> = ({
  progress,
  size = RING_SIZE,
  strokeWidth = STROKE_WIDTH,
  color = '#00ff88',
  showLabel = true,
  label,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;
  
  const center = size / 2;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
        aria-label={`Progress: ${progress}%`}
      >
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="#1a1a2e"
          strokeWidth={strokeWidth}
        />
        
        {/* Progress circle with gradient */}
        <defs>
          <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={color} />
            <stop offset="100%" stopColor={`${color}88`} />
          </linearGradient>
          
          <filter id={`glow-${color.replace('#', '')}`}>
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={`url(#gradient-${color})`}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          filter={`url(#glow-${color.replace('#', '')})`}
          className="transition-all duration-300 ease-out"
        />
        
        {/* Animated segments for cyberpunk effect */}
        {progress < 100 && (
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${circumference * 0.1}, ${circumference}`}
            strokeDashoffset={offset - circumference * 0.5}
            className="animate-spin"
            style={{ animationDuration: '2s' }}
          />
        )}
      </svg>
      
      {/* Center Label */}
      {showLabel && (
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-lg font-mono font-bold" style={{ color }}>
            {Math.round(progress)}%
          </span>
          {label && (
            <span className="text-xs text-gray-400 font-mono mt-0.5">{label}</span>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Individual Export Job Card
 */
interface ExportJobCardProps {
  job: ExportJob;
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
}

const ExportJobCard: React.FC<ExportJobCardProps> = ({ job, onCancel, onRetry }) => {
  const config = FORMAT_CONFIG[job.format];
  const isComplete = job.status === 'complete';
  const isError = job.status === 'error';
  const isWriting = job.status === 'writing';
  
  const duration = job.completedAt
    ? formatDuration(job.completedAt - job.createdAt)
    : formatDuration(Date.now() - job.createdAt);

  return (
    <div
      className={`p-4 rounded-lg border transition-all ${
        isError ? 'border-red-500/50 bg-red-500/10' : 'border-white/10 bg-white/5'
      }`}
    >
      <div className="flex items-center gap-4">
        {/* Progress Ring */}
        <ProgressRing
          progress={isError ? 0 : job.progress}
          color={isError ? '#ff0044' : config.color}
          size={60}
          strokeWidth={4}
          showLabel={true}
        />
        
        {/* Job Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">{config.icon}</span>
            <span className="font-mono text-sm text-white truncate">{job.name}</span>
            <span className="text-xs px-1.5 py-0.5 rounded font-mono" style={{ backgroundColor: config.color, color: '#000' }}>
              {config.label}
            </span>
          </div>
          
          <div className="grid grid-cols-3 gap-2 text-xs font-mono text-gray-400">
            <div>
              <span className="text-gray-500">Rows:</span>{' '}
              <span className="text-white">{job.writtenRows.toLocaleString()}/{job.totalRows.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-500">Speed:</span>{' '}
              <span className={isWriting ? 'text-cyan-400' : 'text-gray-400'}>
                {job.writeSpeedMBps.toFixed(1)} MB/s
              </span>
            </div>
            <div>
              <span className="text-gray-500">Time:</span>{' '}
              <span className="text-white">{duration}</span>
            </div>
          </div>
          
          {/* Status Bar */}
          <div className="mt-2 flex items-center gap-2">
            {isWriting && (
              <>
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span className="text-xs text-cyan-400 font-mono">Writing to NVMe...</span>
              </>
            )}
            {isComplete && (
              <>
                <span className="w-2 h-2 rounded-full bg-green-400" />
                <span className="text-xs text-green-400 font-mono">Complete</span>
              </>
            )}
            {isError && (
              <>
                <span className="w-2 h-2 rounded-full bg-red-400" />
                <span className="text-xs text-red-400 font-mono">{job.error || 'Failed'}</span>
              </>
            )}
          </div>
        </div>
        
        {/* Actions */}
        <div className="flex flex-col gap-2">
          {isWriting && (
            <button
              onClick={() => onCancel(job.id)}
              className="px-3 py-1.5 text-xs font-mono rounded border border-red-500/50 text-red-400 hover:bg-red-500/20 transition-colors"
            >
              Cancel
            </button>
          )}
          {isError && (
            <button
              onClick={() => onRetry(job.id)}
              className="px-3 py-1.5 text-xs font-mono rounded border border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
            >
              Retry
            </button>
          )}
          {isComplete && (
            <a
              href="#"
              className="px-3 py-1.5 text-xs font-mono rounded border border-green-500/50 text-green-400 hover:bg-green-500/20 transition-colors text-center"
            >
              Download
            </a>
          )}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const ExportManager: React.FC<ExportManagerProps> = ({
  maxConcurrent = MAX_CONCURRENT_DEFAULT,
  onExportStart,
  onExportComplete,
}) => {
  const [jobs, setJobs] = useState<ExportJob[]>([]);
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('parquet');
  const [exportName, setExportName] = useState('');
  
  /**
   * Creates a new export job
   */
  const createExport = useCallback(() => {
    const newJob: ExportJob = {
      id: generateId(),
      name: exportName || `export-${Date.now()}`,
      format: selectedFormat,
      status: 'pending',
      progress: 0,
      totalRows: Math.floor(Math.random() * 1000000) + 10000,
      writtenRows: 0,
      writeSpeedMBps: 0,
      createdAt: Date.now(),
    };
    
    setJobs((prev) => [newJob, ...prev]);
    onExportStart?.(newJob);
    setExportName('');
    
    // Simulate export progress (in production, this would be driven by Rust backend IPC)
    simulateExportProgress(newJob.id);
  }, [exportName, selectedFormat, onExportStart]);
  
  /**
   * Simulates export progress from Rust backend
   */
  const simulateExportProgress = useCallback((jobId: string) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 5 + 2;
      
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        
        setJobs((prev) =>
          prev.map((job) =>
            job.id === jobId
              ? {
                  ...job,
                  status: 'complete',
                  progress: 100,
                  writtenRows: job.totalRows,
                  writeSpeedMBps: Math.random() * 500 + 200, // Simulated NVMe speed
                  completedAt: Date.now(),
                }
              : job
          )
        );
        
        const completedJob = jobs.find((j) => j.id === jobId);
        if (completedJob) {
          onExportComplete?.(completedJob);
        }
      } else {
        setJobs((prev) =>
          prev.map((job) =>
            job.id === jobId
              ? {
                  ...job,
                  status: 'writing',
                  progress,
                  writtenRows: Math.floor((progress / 100) * job.totalRows),
                  writeSpeedMBps: Math.random() * 500 + 200,
                }
              : job
          )
        );
      }
    }, 200);
  }, [jobs, onExportComplete]);
  
  /**
   * Cancels an export job
   */
  const cancelJob = useCallback((id: string) => {
    setJobs((prev) =>
      prev.map((job) =>
        job.id === id ? { ...job, status: 'error', error: 'Cancelled by user' } : job
      )
    );
  }, []);
  
  /**
   * Retries a failed export job
   */
  const retryJob = useCallback((id: string) => {
    setJobs((prev) =>
      prev.map((job) =>
        job.id === id
          ? { ...job, status: 'pending', progress: 0, error: undefined }
          : job
      )
    );
    
    // Restart progress simulation
    setTimeout(() => simulateExportProgress(id), 100);
  }, [simulateExportProgress]);
  
  // Statistics
  const stats = useMemo(() => {
    const total = jobs.length;
    const complete = jobs.filter((j) => j.status === 'complete').length;
    const writing = jobs.filter((j) => j.status === 'writing').length;
    const error = jobs.filter((j) => j.status === 'error').length;
    const totalRows = jobs.reduce((acc, j) => acc + j.writtenRows, 0);
    
    return { total, complete, writing, error, totalRows };
  }, [jobs]);

  return (
    <div className="w-full rounded-xl overflow-hidden bg-[#0a0a12]/90 backdrop-blur-sm border border-cyan-900/30">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-b from-[#0a0a12] to-transparent border-b border-white/5">
        <h3 className="text-cyan-400 font-mono text-sm tracking-wider uppercase">
          💾 Export Manager <span className="text-xs opacity-70">| NVMe Write</span>
        </h3>
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="text-gray-400">Active: {stats.writing}</span>
          <span className="text-green-400">Complete: {stats.complete}</span>
        </div>
      </div>

      {/* New Export Form */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center gap-3 flex-wrap">
          <input
            type="text"
            value={exportName}
            onChange={(e) => setExportName(e.target.value)}
            placeholder="Export name..."
            className="flex-1 min-w-[200px] bg-white/10 border border-white/20 rounded px-3 py-2 text-sm font-mono text-white focus:outline-none focus:border-cyan-500"
          />
          
          <div className="flex items-center gap-2">
            {(Object.keys(FORMAT_CONFIG) as ExportFormat[]).map((format) => (
              <button
                key={format}
                onClick={() => setSelectedFormat(format)}
                className={`px-3 py-2 rounded border transition-all flex items-center gap-2 ${
                  selectedFormat === format
                    ? 'border-cyan-500 bg-cyan-500/20 text-cyan-400'
                    : 'border-white/20 bg-white/5 text-gray-400 hover:border-white/40'
                }`}
                style={{
                  borderColor: selectedFormat === format ? FORMAT_CONFIG[format].color : undefined,
                }}
              >
                <span>{FORMAT_CONFIG[format].icon}</span>
                <span className="text-xs font-mono">{FORMAT_CONFIG[format].label}</span>
              </button>
            ))}
          </div>
          
          <button
            onClick={createExport}
            disabled={!exportName.trim()}
            className="px-4 py-2 bg-cyan-500/20 border border-cyan-500/50 rounded text-cyan-400 text-sm font-mono hover:bg-cyan-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Start Export
          </button>
        </div>
      </div>

      {/* Export Jobs List */}
      <div className="p-4 space-y-3 max-h-[400px] overflow-y-auto">
        {jobs.length === 0 ? (
          <div className="text-center py-8 text-gray-500 font-mono text-sm">
            No export jobs. Create one to begin.
          </div>
        ) : (
          jobs.map((job) => (
            <ExportJobCard
              key={job.id}
              job={job}
              onCancel={cancelJob}
              onRetry={retryJob}
            />
          ))
        )}
      </div>

      {/* Footer Stats */}
      <div className="px-4 py-2 bg-gradient-to-t from-[#0a0a12] to-transparent border-t border-white/5">
        <div className="flex items-center justify-between text-xs font-mono text-gray-500">
          <span>Total Rows Exported: {stats.totalRows.toLocaleString()}</span>
          <span>Max Concurrent: {maxConcurrent}</span>
        </div>
      </div>
    </div>
  );
};

export default ExportManager;
