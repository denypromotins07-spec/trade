/**
 * File 12: ExportManager.tsx
 * Chapter 4: Data Exploration & Custom Query Builder
 * 
 * Parquet and CSV export manager featuring a cyberpunk-styled progress ring
 * that tracks the Rust backend's asynchronous NVMe disk writes.
 */

import React, { useState, useCallback, useEffect } from 'react';

interface ExportJob {
  id: string;
  name: string;
  format: 'parquet' | 'csv' | 'json';
  status: 'pending' | 'writing' | 'complete' | 'error';
  progress: number; // 0-100
  rows: number;
  sizeBytes?: number;
  writeSpeedMBps?: number;
  error?: string;
}

interface Props {
  onExport?: (job: ExportJob) => void;
  maxConcurrent?: number;
}

const COLORS = {
  bg: '#0a0a0a',
  panel: 'rgba(20, 20, 30, 0.8)',
  border: '#333333',
  text: '#c0c0c0',
  cyan: '#00f3ff',
  green: '#00ff9d',
  orange: '#ffaa00',
  red: '#ff0055',
  purple: '#bd00ff',
};

/**
 * Format bytes to human readable
 */
const formatBytes = (bytes?: number): string => {
  if (!bytes) return 'N/A';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let num = bytes;
  while (num >= 1024 && i < units.length - 1) {
    num /= 1024;
    i++;
  }
  return `${num.toFixed(2)} ${units[i]}`;
};

/**
 * Cyberpunk Progress Ring Component
 */
const ProgressRing: React.FC<{
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
}> = ({ progress, size = 60, strokeWidth = 4, color = COLORS.cyan }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - progress / 100);

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        {/* Background ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#222"
          strokeWidth={strokeWidth}
        />
        {/* Progress ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{
            transition: 'stroke-dashoffset 0.3s ease-out',
            filter: `drop-shadow(0 0 4px ${color})`,
          }}
        />
      </svg>
      {/* Center percentage */}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-[10px] font-mono font-bold text-white">
          {Math.round(progress)}%
        </span>
      </div>
    </div>
  );
};

/**
 * ExportManager Component
 */
export const ExportManager: React.FC<Props> = ({ onExport, maxConcurrent = 3 }) => {
  const [jobs, setJobs] = useState<ExportJob[]>([]);
  const [selectedFormat, setSelectedFormat] = useState<'parquet' | 'csv' | 'json'>('parquet');
  const [fileName, setFileName] = useState('export_data');

  // Simulate export progress (in real app, this would be WebSocket updates from Rust)
  useEffect(() => {
    const interval = setInterval(() => {
      setJobs((prev) =>
        prev.map((job) => {
          if (job.status === 'writing') {
            const newProgress = Math.min(job.progress + Math.random() * 15, 100);
            const isComplete = newProgress >= 100;
            
            return {
              ...job,
              progress: newProgress,
              status: isComplete ? 'complete' : 'writing',
              writeSpeedMBps: isComplete ? job.writeSpeedMBps : 500 + Math.random() * 2000, // Simulate NVMe speeds
              sizeBytes: isComplete ? job.rows * 100 : undefined,
            };
          }
          return job;
        })
      );
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const handleExport = useCallback(() => {
    const newJob: ExportJob = {
      id: `job_${Date.now()}`,
      name: `${fileName}.${selectedFormat}`,
      format: selectedFormat,
      status: 'pending',
      progress: 0,
      rows: Math.floor(Math.random() * 1000000) + 10000, // Simulated row count
    };

    // Start with pending, then move to writing after brief delay
    setJobs((prev) => [...prev, newJob]);
    onExport?.(newJob);

    // Simulate starting the write
    setTimeout(() => {
      setJobs((prev) =>
        prev.map((j) =>
          j.id === newJob.id ? { ...j, status: 'writing', progress: 5 } : j
        )
      );
    }, 500);
  }, [fileName, selectedFormat, onExport]);

  const activeJobs = jobs.filter((j) => j.status === 'writing' || j.status === 'pending').length;
  const canExport = activeJobs < maxConcurrent;

  return (
    <div className="p-4 bg-black/80 backdrop-blur-md border border-cyan-900/50 rounded-xl">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-mono font-bold text-white tracking-wider">
          EXPORT_MANAGER_V4
        </h3>
        <div className="text-[10px] font-mono text-gray-500">
          MAX_CONCURRENT: {maxConcurrent}
        </div>
      </div>

      {/* Export Controls */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="col-span-2">
          <label className="text-[9px] font-mono text-gray-500 block mb-1">FILE_NAME</label>
          <input
            type="text"
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
            placeholder="output_file"
          />
        </div>
        <div>
          <label className="text-[9px] font-mono text-gray-500 block mb-1">FORMAT</label>
          <select
            value={selectedFormat}
            onChange={(e) => setSelectedFormat(e.target.value as typeof selectedFormat)}
            className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none"
          >
            <option value="parquet">PARQUET</option>
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
          </select>
        </div>
      </div>

      <button
        onClick={handleExport}
        disabled={!canExport}
        className={`w-full mb-4 px-4 py-2 text-xs font-mono font-bold rounded transition-all ${
          canExport
            ? 'bg-cyan-600 hover:bg-cyan-500 text-white'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
        }`}
        style={canExport ? { boxShadow: '0 0 15px rgba(0,243,255,0.3)' } : {}}
      >
        {canExport ? 'START_EXPORT' : `MAX_JOBS (${maxConcurrent}) REACHED`}
      </button>

      {/* Active Jobs */}
      <div className="space-y-3">
        <div className="text-[9px] font-mono text-gray-500">ACTIVE_JOBS</div>
        
        {jobs.length === 0 ? (
          <div className="text-center py-8 text-gray-600 text-[10px] font-mono">
            NO ACTIVE EXPORTS
          </div>
        ) : (
          jobs.slice(-5).map((job) => (
            <div
              key={job.id}
              className="flex items-center gap-3 p-3 rounded-lg border bg-gray-900/50"
              style={{
                borderColor:
                  job.status === 'complete'
                    ? COLORS.green
                    : job.status === 'error'
                    ? COLORS.red
                    : COLORS.border,
              }}
            >
              {/* Progress Ring */}
              <ProgressRing
                progress={job.progress}
                color={
                  job.status === 'complete'
                    ? COLORS.green
                    : job.status === 'error'
                    ? COLORS.red
                    : COLORS.cyan
                }
              />

              {/* Job Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-mono font-bold text-white truncate">
                    {job.name}
                  </span>
                  <span
                    className="text-[8px] px-1.5 py-0.5 rounded font-mono"
                    style={{
                      backgroundColor:
                        job.status === 'complete'
                          ? `${COLORS.green}20`
                          : job.status === 'error'
                          ? `${COLORS.red}20`
                          : `${COLORS.cyan}20`,
                      color:
                        job.status === 'complete'
                          ? COLORS.green
                          : job.status === 'error'
                          ? COLORS.red
                          : COLORS.cyan,
                    }}
                  >
                    {job.status.toUpperCase()}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-[9px] font-mono text-gray-400">
                  <div>ROWS: {job.rows.toLocaleString()}</div>
                  <div>SIZE: {formatBytes(job.sizeBytes)}</div>
                  <div>NVMe: {job.writeSpeedMBps?.toFixed(0) || '--'} MB/s</div>
                </div>
              </div>

              {/* Status Icon */}
              {job.status === 'complete' && (
                <span className="text-green-400 text-lg">✓</span>
              )}
              {job.status === 'error' && (
                <span className="text-red-400 text-lg">✗</span>
              )}
              {job.status === 'writing' && (
                <span className="text-cyan-400 animate-pulse">⟳</span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Format Info */}
      <div className="mt-4 pt-3 border-t border-gray-800">
        <div className="text-[9px] font-mono text-gray-500 mb-2">FORMAT_INFO</div>
        <div className="grid grid-cols-3 gap-2 text-[9px] font-mono">
          <div className="p-2 rounded bg-gray-900/50 border border-gray-800">
            <div className="text-cyan-400 mb-1">PARQUET</div>
            <div className="text-gray-500">Columnar • Compressed • Fast</div>
          </div>
          <div className="p-2 rounded bg-gray-900/50 border border-gray-800">
            <div className="text-green-400 mb-1">CSV</div>
            <div className="text-gray-500">Universal • Readable • Large</div>
          </div>
          <div className="p-2 rounded bg-gray-900/50 border border-gray-800">
            <div className="text-purple-400 mb-1">JSON</div>
            <div className="text-gray-500">Nested • Flexible • Verbose</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExportManager;
