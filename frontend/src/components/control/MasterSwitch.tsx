/**
 * MasterSwitch.tsx - Biometric-style /START and /KILL master toggles
 * 
 * Features:
 * - Deliberate swipe-to-confirm interaction to prevent accidental boots
 * - Framer Motion animations for cyberpunk aesthetic
 * - WebSocket disconnect handling during critical boot sequences
 * - PowerShell orchestration compatibility (/START, /KILL commands)
 * - AMD GPU context visualization during system state transitions
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform, useAnimation } from 'framer-motion';
import { Zap, Skull, AlertTriangle, Wifi, WifiOff } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { SystemStatus } from '../../types/system';

interface MasterSwitchProps {
  currentStatus: SystemStatus;
  onStatusChange: (status: SystemStatus) => void;
}

type SwipeState = 'idle' | 'swiping' | 'confirmed' | 'rejected';

export const MasterSwitch: React.FC<MasterSwitchProps> = ({
  currentStatus,
  onStatusChange,
}) => {
  const [swipeState, setSwipeState] = useState<SwipeState>('idle');
  const [isConnected, setIsConnected] = useState(true);
  const { sendMessage, lastMessage, connectionStatus } = useWebSocket();
  const controls = useAnimation();
  
  // Motion values for swipe gesture
  const x = useMotionValue(0);
  const opacity = useTransform(x, [-100, 0, 100], [0.5, 1, 0.5]);
  const backgroundColor = useTransform(x, [-100, 0, 100], ['#ef4444', '#1a1a2e', '#22c55e']);
  const scale = useTransform(x, [-100, 0, 100], [0.95, 1, 0.95]);
  
  const swipeThreshold = 80; // pixels required to confirm
  const isStarting = currentStatus === 'starting' || currentStatus === 'booting';
  const isRunning = currentStatus === 'running' || currentStatus === 'active';
  const isStopping = currentStatus === 'stopping' || currentStatus === 'killing';

  // Handle WebSocket connection status
  useEffect(() => {
    setIsConnected(connectionStatus === 'open');
    
    if (connectionStatus === 'closed' && (isStarting || isStopping)) {
      // Critical: WS disconnected during boot/shutdown sequence
      console.warn('[MasterSwitch] WebSocket disconnected during critical sequence');
      setSwipeState('rejected');
      controls.start({
        x: 0,
        transition: { type: 'spring', stiffness: 300, damping: 20 }
      });
    }
  }, [connectionStatus, isStarting, isStopping, controls]);

  // Handle system status updates from backend
  useEffect(() => {
    if (lastMessage) {
      try {
        const payload = JSON.parse(lastMessage.data as string);
        if (payload.type === 'SYSTEM_STATUS') {
          onStatusChange(payload.status as SystemStatus);
          if (payload.status === 'running' || payload.status === 'stopped') {
            setSwipeState('idle');
          }
        }
      } catch (error) {
        console.error('[MasterSwitch] Failed to parse status update:', error);
      }
    }
  }, [lastMessage, onStatusChange]);

  const sendPowerCommand = useCallback((command: 'START' | 'KILL') => {
    if (!isConnected) {
      console.error('[MasterSwitch] Cannot send command: WebSocket disconnected');
      return;
    }

    // Debounce protection
    if (isStarting || isStopping) {
      console.warn('[MasterSwitch] Command ignored: system already transitioning');
      return;
    }

    const payload = {
      type: 'POWER_COMMAND',
      command,
      timestamp: Date.now(),
      source: 'UI_MASTER_SWITCH',
    };

    sendMessage(JSON.stringify(payload));
    console.log(`[MasterSwitch] Sent ${command} command to Rust core`);
  }, [isConnected, isStarting, isStopping, sendMessage]);

  const handleDragEnd = useCallback(async () => {
    const currentX = x.get();
    
    if (Math.abs(currentX) >= swipeThreshold) {
      // Confirmed swipe
      setSwipeState('confirmed');
      
      if (currentX > 0 && !isRunning) {
        // Swipe right to START
        await controls.start({ x: 100, opacity: 0 });
        sendPowerCommand('START');
      } else if (currentX < 0 && isRunning) {
        // Swipe left to KILL
        await controls.start({ x: -100, opacity: 0 });
        sendPowerCommand('KILL');
      }
      
      // Reset after animation
      setTimeout(() => {
        controls.set({ x: 0, opacity: 1 });
        setSwipeState('idle');
      }, 500);
    } else {
      // Rejected swipe (not far enough)
      setSwipeState('rejected');
      controls.start({
        x: 0,
        transition: { type: 'spring', stiffness: 500, damping: 30 }
      });
      setTimeout(() => setSwipeState('idle'), 300);
    }
  }, [x, swipeThreshold, isRunning, controls, sendPowerCommand]);

  const getStatusLabel = () => {
    switch (currentStatus) {
      case 'running':
      case 'active':
        return 'SYSTEM ACTIVE';
      case 'starting':
      case 'booting':
        return 'INITIALIZING...';
      case 'stopping':
      case 'killing':
        return 'SHUTTING DOWN...';
      case 'stopped':
      case 'offline':
        return 'SYSTEM OFFLINE';
      default:
        return 'UNKNOWN STATE';
    }
  };

  const getStatusColor = () => {
    switch (currentStatus) {
      case 'running':
      case 'active':
        return 'text-emerald-400';
      case 'starting':
      case 'booting':
        return 'text-amber-400';
      case 'stopping':
      case 'killing':
        return 'text-orange-400';
      default:
        return 'text-red-400';
    }
  };

  return (
    <div className="relative w-full max-w-2xl mx-auto p-6">
      {/* Connection Status Indicator */}
      <div className="absolute top-2 right-2 flex items-center gap-2 text-xs">
        {isConnected ? (
          <Wifi className="w-4 h-4 text-emerald-400" />
        ) : (
          <WifiOff className="w-4 h-4 text-red-400" />
        )}
        <span className={isConnected ? 'text-emerald-400' : 'text-red-400'}>
          {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </div>

      {/* Main Switch Container */}
      <motion.div
        className="relative h-24 rounded-2xl overflow-hidden border-2 border-cyan-500/30 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 shadow-[0_0_30px_rgba(6,182,212,0.3)]"
        style={{ scale, backgroundColor }}
        animate={controls}
        drag="x"
        dragConstraints={{ left: -100, right: 100 }}
        onDragEnd={handleDragEnd}
        whileDrag={{ scale: 1.02 }}
      >
        {/* Background Grid Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="w-full h-full" style={{
            backgroundImage: `linear-gradient(rgba(6,182,212,0.3) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(6,182,212,0.3) 1px, transparent 1px)`,
            backgroundSize: '20px 20px'
          }} />
        </div>

        {/* Center Status Display */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <AnimatePresence mode="wait">
            {isRunning ? (
              <motion.div
                key="running"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-3"
              >
                <Zap className="w-8 h-8 text-emerald-400 drop-shadow-[0_0_10px_rgba(52,211,153,0.8)]" />
                <span className={`text-2xl font-bold tracking-widest ${getStatusColor()}`}>
                  {getStatusLabel()}
                </span>
              </motion.div>
            ) : (
              <motion.div
                key="stopped"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-3"
              >
                <Skull className="w-8 h-8 text-red-400 drop-shadow-[0_0_10px_rgba(248,113,113,0.8)]" />
                <span className={`text-2xl font-bold tracking-widest ${getStatusColor()}`}>
                  {getStatusLabel()}
                </span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Swipe Instruction */}
          <motion.p
            className="mt-2 text-sm text-cyan-300/70 font-mono"
            animate={{ opacity: swipeState === 'idle' ? 1 : 0 }}
          >
            {isRunning ? '← SWIPE LEFT TO KILL →' : '← SWIPE RIGHT TO START →'}
          </motion.p>
        </div>

        {/* Swipe Progress Overlay */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-red-500/20 via-transparent to-emerald-500/20 pointer-events-none"
          style={{ opacity }}
        />

        {/* Warning Icon during swipe */}
        <AnimatePresence>
          {swipeState === 'swiping' && (
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
            >
              <AlertTriangle className="w-12 h-12 text-amber-400 drop-shadow-[0_0_15px_rgba(251,191,36,0.8)]" />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Status Details */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-xs font-mono">
        <div className="text-center p-3 rounded-lg bg-slate-800/50 border border-cyan-500/20">
          <div className="text-cyan-400 mb-1">POWER STATE</div>
          <div className={getStatusColor()}>{getStatusLabel()}</div>
        </div>
        <div className="text-center p-3 rounded-lg bg-slate-800/50 border border-cyan-500/20">
          <div className="text-cyan-400 mb-1">WS CONNECTION</div>
          <div className={isConnected ? 'text-emerald-400' : 'text-red-400'}>
            {isConnected ? 'ACTIVE' : 'FAILED'}
          </div>
        </div>
        <div className="text-center p-3 rounded-lg bg-slate-800/50 border border-cyan-500/20">
          <div className="text-cyan-400 mb-1">LAST COMMAND</div>
          <div className="text-slate-300">
            {lastMessage ? new Date(lastMessage.timestamp).toLocaleTimeString() : 'NONE'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MasterSwitch;
