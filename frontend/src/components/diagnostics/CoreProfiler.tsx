/**
 * File 4: CoreProfiler.tsx
 * Chapter 2: System Diagnostics & ETW Telemetry
 * 
 * AMD Ryzen core utilization map visualizing thread affinity, interrupt routing,
 * and core parking status via CSS grids and transforms.
 * 
 * Features:
 * - Per-core utilization bars
 * - Thread affinity visualization
 * - Core parking status indicators
 * - AMD DirectML/ROCm queue mapping
 */

import React, { useMemo } from 'react';

// --- Types ---

interface CoreData {
  id: number;
  utilization: number;      // 0-100%
  frequency: number;        // MHz
  temperature: number;      // Celsius
  parked: boolean;          // Is core parked?
  interrupts: number;       // Interrupt count/sec
  threads: string[];        // Affinity thread IDs
  isCCD: boolean;           // Part of CCD complex?
}

interface Props {
  cores: CoreData[];
  showThreadAffinity?: boolean;
  showTemperature?: boolean;
  totalThreads?: number;
}

// --- Constants ---

const COLORS = {
  idle: '#1a3a3a',
  low: '#00ff9d',
  medium: '#00f3ff',
  high: '#ffaa00',
  critical: '#ff0055',
  parked: '#333333',
  ccd: 'rgba(189, 0, 255, 0.2)', // Purple for CCD
  text: '#a0a0a0',
  border: '#444444',
};

/**
 * Get color based on utilization percentage
 */
const getUtilColor = (util: number): string => {
  if (util < 20) return COLORS.low;
  if (util < 50) return COLORS.medium;
  if (util < 80) return COLORS.high;
  return COLORS.critical;
};

/**
 * Single Core Tile Component
 */
const CoreTile: React.FC<{
  core: CoreData;
  showThreadAffinity: boolean;
  showTemperature: boolean;
}> = ({ core, showThreadAffinity, showTemperature }) => {
  const utilColor = getUtilColor(core.utilization);
  
  return (
    <div
      className={`relative p-2 rounded border transition-all duration-200 ${
        core.parked ? 'opacity-50' : 'opacity-100'
      }`}
      style={{
        backgroundColor: core.parked ? COLORS.parked : 'rgba(10, 10, 10, 0.8)',
        borderColor: core.isCCD ? COLORS.ccd : COLORS.border,
        boxShadow: core.parked ? 'none' : `0 0 10px ${utilColor}30`,
      }}
    >
      {/* Core Header */}
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs font-mono font-bold text-white">
          C{core.id.toString().padStart(2, '0')}
        </span>
        {core.parked && (
          <span className="text-[9px] px-1 bg-gray-700 rounded text-gray-400">PARKED</span>
        )}
      </div>

      {/* Utilization Bar */}
      <div className="h-16 flex items-end justify-center mb-2">
        <div
          className="w-full rounded-t transition-all duration-150"
          style={{
            height: `${core.utilization}%`,
            backgroundColor: utilColor,
            boxShadow: `0 0 8px ${utilColor}`,
          }}
        />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-1 text-[9px] font-mono">
        <div className="text-gray-400">
          <span className="text-gray-500">MHz:</span>{' '}
          <span style={{ color: utilColor }}>{core.frequency}</span>
        </div>
        {showTemperature && (
          <div className="text-gray-400">
            <span className="text-gray-500">°C:</span>{' '}
            <span style={{ 
              color: core.temperature > 80 ? COLORS.critical : 
                     core.temperature > 60 ? COLORS.high : COLORS.medium 
            }}>
              {core.temperature}°
            </span>
          </div>
        )}
        <div className="text-gray-400 col-span-2">
          <span className="text-gray-500">IRQ:</span> {core.interrupts}/s
        </div>
      </div>

      {/* Thread Affinity */}
      {showThreadAffinity && core.threads.length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-800">
          <div className="text-[8px] text-gray-500 mb-1">AFFINITY</div>
          <div className="flex flex-wrap gap-1">
            {core.threads.slice(0, 4).map((t) => (
              <span
                key={t}
                className="px-1 bg-cyan-900/30 text-cyan-400 rounded text-[8px]"
              >
                T{t}
              </span>
            ))}
            {core.threads.length > 4 && (
              <span className="text-[8px] text-gray-500">+{core.threads.length - 4}</span>
            )}
          </div>
        </div>
      )}

      {/* CCD Indicator */}
      {core.isCCD && (
        <div className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" />
      )}
    </div>
  );
};

/**
 * CoreProfiler Component
 * Displays AMD Ryzen core topology with real-time metrics.
 */
export const CoreProfiler: React.FC<Props> = ({
  cores,
  showThreadAffinity = true,
  showTemperature = true,
  totalThreads = 0,
}) => {
  // Group cores by CCD (Core Complex Die)
  const ccdGroups = useMemo(() => {
    const groups: Map<number, CoreData[]> = new Map();
    cores.forEach((core) => {
      const ccdId = Math.floor(core.id / 8); // Assuming 8 cores per CCD
      if (!groups.has(ccdId)) groups.set(ccdId, []);
      groups.get(ccdId)!.push(core);
    });
    return groups;
  }, [cores]);

  // Calculate aggregate stats
  const avgUtil = cores.reduce((acc, c) => acc + c.utilization, 0) / cores.length || 0;
  const maxTemp = Math.max(...cores.map((c) => c.temperature), 0);
  const activeCores = cores.filter((c) => !c.parked).length;
  const parkedCores = cores.length - activeCores;

  return (
    <div className="p-4 bg-black/80 backdrop-blur-md border border-cyan-900/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-sm font-mono font-bold text-white tracking-wider">
            AMD_RYZEN_CORE_PROFILER
          </h3>
          <div className="text-[10px] text-gray-400 font-mono">
            {cores.length} CORES • {totalThreads || cores.length * 2} THREADS • {ccdGroups.size} CCDs
          </div>
        </div>
        
        {/* Aggregate Stats */}
        <div className="flex gap-4 text-[10px] font-mono">
          <div className="text-right">
            <div className="text-gray-500">AVG UTIL</div>
            <div className="text-cyan-400 font-bold">{avgUtil.toFixed(1)}%</div>
          </div>
          <div className="text-right">
            <div className="text-gray-500">MAX TEMP</div>
            <div className={maxTemp > 80 ? 'text-red-500 font-bold' : 'text-green-400 font-bold'}>
              {maxTemp}°C
            </div>
          </div>
          <div className="text-right">
            <div className="text-gray-500">ACTIVE/PARKED</div>
            <div className="text-white font-bold">
              {activeCores}/{parkedCores}
            </div>
          </div>
        </div>
      </div>

      {/* CCD Groups Layout */}
      <div className="grid gap-4">
        {Array.from(ccdGroups.entries()).map(([ccdId, groupCores]) => (
          <div key={ccdId} className="relative">
            {/* CCD Label */}
            <div className="absolute -top-3 left-2 px-2 py-0.5 bg-purple-900/50 border border-purple-700 rounded text-[9px] font-mono text-purple-300">
              CCD {ccdId}
            </div>
            
            {/* Cores Grid */}
            <div 
              className="grid gap-2 p-3 rounded-lg border border-purple-900/30"
              style={{
                gridTemplateColumns: `repeat(${Math.min(8, groupCores.length)}, 1fr)`,
                backgroundColor: COLORS.ccd,
              }}
            >
              {groupCores.map((core) => (
                <CoreTile
                  key={core.id}
                  core={core}
                  showThreadAffinity={showThreadAffinity}
                  showTemperature={showTemperature}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* ROCm/DirectML Queue Status */}
      <div className="mt-4 pt-3 border-t border-gray-800">
        <div className="flex justify-between items-center">
          <div className="text-[10px] font-mono text-gray-400">
            GPU_COMPUTE_QUEUE_MAPPING
          </div>
          <div className="flex gap-2">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
              <span className="text-[9px] font-mono text-purple-400">ROCm ACTIVE</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-cyan-500" />
              <span className="text-[9px] font-mono text-cyan-400">DirectML READY</span>
            </div>
          </div>
        </div>
        
        {/* Queue Visualization */}
        <div className="mt-2 flex gap-1">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              className="flex-1 h-1 rounded"
              style={{
                backgroundColor: i < activeCores % 8 ? COLORS.medium : COLORS.parked,
                boxShadow: i < activeCores % 8 ? `0 0 4px ${COLORS.medium}` : 'none',
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default CoreProfiler;
