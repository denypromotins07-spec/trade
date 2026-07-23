/**
 * MutationTrigger.tsx - Manual trigger interface for AutoML sweeps
 * 
 * Features:
 * - GPU-accelerated progress ring tracking Ray worker utilization
 * - Beautiful gradient animations during intense gradient updates
 * - AMD ROCm/DirectML context visualization
 * - Real-time worker status monitoring
 */

import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Zap, Play, Pause, RotateCcw, Activity } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface WorkerStatus {
  id: string;
  status: 'idle' | 'busy' | 'error';
  utilization: number;
  memoryUsage: number;
  currentTask?: string;
}

interface MutationTriggerProps {
  onSweepStarted: (sweepId: string) => void;
  onSweepCompleted: (results: unknown) => void;
}

type SweepState = 'idle' | 'preparing' | 'running' | 'completed' | 'error';

export const MutationTrigger: React.FC<MutationTriggerProps> = ({
  onSweepStarted,
  onSweepCompleted,
}) => {
  const [sweepState, setSweepState] = useState<SweepState>('idle');
  const [progress, setProgress] = useState(0);
  const [workers, setWorkers] = useState<WorkerStatus[]>([]);
  const [currentSweepId, setCurrentSweepId] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  const { sendMessage, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'open';

  // Simulate worker status updates (in production, this comes from WebSocket)
  useEffect(() => {
    if (sweepState !== 'running') return;

    const interval = setInterval(() => {
      setWorkers(prev => prev.map(worker => ({
        ...worker,
        utilization: worker.status === 'busy' ? Math.random() * 40 + 60 : Math.random() * 20,
        memoryUsage: worker.status === 'busy' ? Math.random() * 2 + 6 : Math.random() * 2 + 2,
      })));
      
      setProgress(prev => {
        const newProgress = prev + Math.random() * 2;
        if (newProgress >= 100) {
          setSweepState('completed');
          onSweepCompleted({ sweepId: currentSweepId, timestamp: Date.now() });
          return 100;
        }
        return newProgress;
      });
      
      setElapsedTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [sweepState, currentSweepId, onSweepCompleted]);

  // Initialize workers
  useEffect(() => {
    const initialWorkers: WorkerStatus[] = Array.from({ length: 8 }, (_, i) => ({
      id: `ray-worker-${i}`,
      status: 'idle',
      utilization: 0,
      memoryUsage: 2,
    }));
    setWorkers(initialWorkers);
  }, []);

  const startSweep = useCallback(() => {
    if (!isConnected || sweepState !== 'idle') return;

    const sweepId = `sweep_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setCurrentSweepId(sweepId);
    setSweepState('preparing');
    
    // Send sweep initiation message
    const payload = {
      type: 'START_AUTOML_SWEEP',
      sweepId,
      timestamp: Date.now(),
      data: {
        populationSize: 32,
        generations: 100,
        mutationRate: 0.1,
        crossoverRate: 0.7,
        source: 'UI_MUTATION_TRIGGER',
      },
    };
    
    sendMessage(JSON.stringify(payload));
    onSweepStarted(sweepId);
    
    // Simulate preparation phase
    setTimeout(() => {
      setSweepState('running');
      setWorkers(prev => prev.map(w => ({ ...w, status: 'busy' as const })));
      setProgress(0);
      setElapsedTime(0);
    }, 2000);
  }, [isConnected, sweepState, sendMessage, onSweepStarted]);

  const stopSweep = useCallback(() => {
    if (!currentSweepId) return;
    
    const payload = {
      type: 'STOP_AUTOML_SWEEP',
      sweepId: currentSweepId,
      timestamp: Date.now(),
      reason: 'USER_CANCELLED',
    };
    
    sendMessage(JSON.stringify(payload));
    setSweepState('idle');
    setWorkers(prev => prev.map(w => ({ ...w, status: 'idle' as const })));
    setProgress(0);
  }, [currentSweepId, sendMessage]);

  const resetSweep = useCallback(() => {
    setSweepState('idle');
    setCurrentSweepId(null);
    setProgress(0);
    setElapsedTime(0);
    setWorkers(prev => prev.map(w => ({ ...w, status: 'idle' as const, utilization: 0 })));
  }, []);

  // Progress ring calculations
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  const getProgressColor = () => {
    if (progress >= 80) return '#10b981';
    if (progress >= 50) return '#22d3ee';
    if (progress >= 20) return '#f59e0b';
    return '#ef4444';
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const avgUtilization = workers.reduce((sum, w) => sum + w.utilization, 0) / workers.length;

  return (
    <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          AUTOML MUTATION TRIGGER
        </h3>
        <div className={`px-3 py-1 rounded-full text-xs font-mono font-bold ${
          sweepState === 'running'
            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500 animate-pulse'
            : sweepState === 'preparing'
            ? 'bg-amber-500/20 text-amber-400 border border-amber-500'
            : sweepState === 'completed'
            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500'
            : 'bg-slate-700 text-slate-400'
        }`}>
          {sweepState.toUpperCase()}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Progress Ring */}
        <div className="flex flex-col items-center justify-center p-6 bg-slate-800/50 rounded-xl border border-slate-700">
          <div className="relative">
            <svg width="220" height="220" className="transform -rotate-90">
              {/* Background circle */}
              <circle
                cx="110"
                cy="110"
                r={radius}
                fill="none"
                stroke="#1e293b"
                strokeWidth="12"
              />
              
              {/* Progress circle with gradient */}
              <motion.circle
                cx="110"
                cy="110"
                r={radius}
                fill="none"
                stroke={getProgressColor()}
                strokeWidth="12"
                strokeLinecap="round"
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset }}
                transition={{ duration: 0.3 }}
                style={{
                  filter: `drop-shadow(0 0 15px ${getProgressColor()}66)`,
                }}
              />
              
              {/* Outer glow rings */}
              {sweepState === 'running' && (
                <>
                  <motion.circle
                    cx="110"
                    cy="110"
                    r={radius + 15}
                    fill="none"
                    stroke={getProgressColor()}
                    strokeWidth="2"
                    strokeOpacity="0.3"
                    initial={{ scale: 1, opacity: 0.5 }}
                    animate={{ scale: 1.2, opacity: 0 }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                  <motion.circle
                    cx="110"
                    cy="110"
                    r={radius + 25}
                    fill="none"
                    stroke={getProgressColor()}
                    strokeWidth="1"
                    strokeOpacity="0.2"
                    initial={{ scale: 1, opacity: 0.3 }}
                    animate={{ scale: 1.3, opacity: 0 }}
                    transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                  />
                </>
              )}
            </svg>
            
            {/* Center Content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <motion.div
                className="text-4xl font-mono font-bold"
                style={{ color: getProgressColor() }}
                animate={sweepState === 'running' ? { scale: [1, 1.05, 1] } : {}}
                transition={{ duration: 1, repeat: Infinity }}
              >
                {progress.toFixed(1)}%
              </motion.div>
              <div className="text-xs text-slate-400 mt-1">PROGRESS</div>
              <div className="text-sm font-mono text-cyan-400 mt-2">
                {formatTime(elapsedTime)}
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex gap-3 mt-6">
            {sweepState === 'idle' || sweepState === 'completed' ? (
              <button
                onClick={startSweep}
                disabled={!isConnected}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg font-bold text-white hover:from-cyan-400 hover:to-blue-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <Play className="w-5 h-5" />
                START SWEEP
              </button>
            ) : sweepState === 'running' ? (
              <button
                onClick={stopSweep}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-500 to-orange-500 rounded-lg font-bold text-white hover:from-red-400 hover:to-orange-400 transition-all"
              >
                <Pause className="w-5 h-5" />
                STOP
              </button>
            ) : (
              <button
                onClick={resetSweep}
                className="flex items-center gap-2 px-6 py-3 bg-slate-700 rounded-lg font-bold text-white hover:bg-slate-600 transition-all"
              >
                <RotateCcw className="w-5 h-5" />
                RESET
              </button>
            )}
          </div>
        </div>

        {/* Worker Grid */}
        <div className="p-6 bg-slate-800/50 rounded-xl border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-bold text-slate-300 flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              RAY WORKER STATUS
            </h4>
            <span className="text-xs text-cyan-400 font-mono">
              AVG UTIL: {avgUtilization.toFixed(1)}%
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {workers.map((worker) => (
              <div
                key={worker.id}
                className={`p-3 rounded-lg border transition-all ${
                  worker.status === 'busy'
                    ? 'bg-emerald-500/10 border-emerald-500/50'
                    : worker.status === 'error'
                    ? 'bg-red-500/10 border-red-500/50'
                    : 'bg-slate-700/50 border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-slate-300">{worker.id}</span>
                  <div className={`w-2 h-2 rounded-full ${
                    worker.status === 'busy'
                      ? 'bg-emerald-400 animate-pulse'
                      : worker.status === 'error'
                      ? 'bg-red-400'
                      : 'bg-slate-500'
                  }`} />
                </div>
                
                {/* Utilization Bar */}
                <div className="mb-1">
                  <div className="flex justify-between text-[10px] text-slate-500 mb-0.5">
                    <span>GPU</span>
                    <span>{worker.utilization.toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{
                        backgroundColor: worker.utilization > 80 ? '#ef4444' : '#22d3ee',
                      }}
                      animate={{ width: `${worker.utilization}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                </div>
                
                {/* Memory Bar */}
                <div>
                  <div className="flex justify-between text-[10px] text-slate-500 mb-0.5">
                    <span>VRAM</span>
                    <span>{worker.memoryUsage.toFixed(1)}GB</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-purple-500"
                      animate={{ width: `${(worker.memoryUsage / 16) * 100}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* AMD ROCm Context */}
          <div className="mt-4 pt-4 border-t border-slate-700">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 text-slate-400">
                <Activity className="w-3 h-3" />
                <span>AMD ROCm ACCELERATION</span>
              </div>
              <span className="text-emerald-400 font-mono">ACTIVE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sweep Info Footer */}
      {currentSweepId && (
        <div className="mt-4 pt-4 border-t border-slate-700 flex items-center justify-between text-xs font-mono">
          <div className="text-slate-400">
            SWEEP ID: <span className="text-cyan-400">{currentSweepId}</span>
          </div>
          <div className="text-slate-400">
            POPULATION: <span className="text-white">32</span> | 
            GENERATIONS: <span className="text-white">100</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MutationTrigger;
