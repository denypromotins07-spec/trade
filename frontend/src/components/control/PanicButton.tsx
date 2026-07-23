/**
 * PanicButton.tsx - Emergency "FLATTEN ALL" button
 * 
 * Features:
 * - Massive, glowing emergency button with confirmation sequence
 * - Sends WebSocket payload to Rust core to cancel all orders and close positions
 * - Debounced emissions to prevent API spam during panic events
 * - Visual feedback for emergency state
 * - AMD GPU context indicator during high-load liquidation
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { AlertTriangle, Skull, XCircle, ShieldAlert, Clock } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface PanicButtonProps {
  onPanicExecuted: (timestamp: number) => void;
  isArmed: boolean;
}

type PanicState = 'idle' | 'arming' | 'confirming' | 'executing' | 'executed';

// Cooldown period after panic execution (ms)
const PANIC_COOLDOWN = 30000; // 30 seconds
// Confirmation hold duration (ms)
const CONFIRM_HOLD_DURATION = 2000;
// Debounce interval for WS messages (ms)
const DEBOUNCE_INTERVAL = 500;

export const PanicButton: React.FC<PanicButtonProps> = ({
  onPanicExecuted,
  isArmed,
}) => {
  const [panicState, setPanicState] = useState<PanicState>('idle');
  const [holdProgress, setHoldProgress] = useState(0);
  const [lastPanicTime, setLastPanicTime] = useState<number>(0);
  const [wsMessageCount, setWsMessageCount] = useState(0);
  
  const { sendMessage, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'open';
  const controls = useAnimation();
  const holdTimerRef = useRef<NodeJS.Timeout | null>(null);
  const progressFrameRef = useRef<number>(0);
  const lastSendTimeRef = useRef<number>(0);

  const isOnCooldown = Date.now() - lastPanicTime < PANIC_COOLDOWN;
  const cooldownRemaining = Math.max(0, PANIC_COOLDOWN - (Date.now() - lastPanicTime));

  // Start confirmation sequence
  const startConfirmation = useCallback(() => {
    if (!isArmed || isOnCooldown || !isConnected) return;
    
    setPanicState('confirming');
    setHoldProgress(0);
    
    const startTime = Date.now();
    
    const animateProgress = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / CONFIRM_HOLD_DURATION, 1);
      setHoldProgress(progress);
      
      if (progress < 1) {
        progressFrameRef.current = requestAnimationFrame(animateProgress);
      } else {
        // Hold completed - execute panic
        executePanic();
      }
    };
    
    progressFrameRef.current = requestAnimationFrame(animateProgress);
  }, [isArmed, isOnCooldown, isConnected]);

  // Cancel confirmation
  const cancelConfirmation = useCallback(() => {
    if (progressFrameRef.current) {
      cancelAnimationFrame(progressFrameRef.current);
    }
    setPanicState('idle');
    setHoldProgress(0);
  }, []);

  // Execute the panic command
  const executePanic = useCallback(async () => {
    const now = Date.now();
    
    // Debounce check
    if (now - lastSendTimeRef.current < DEBOUNCE_INTERVAL) {
      return;
    }
    
    lastSendTimeRef.current = now;
    setPanicState('executing');
    
    // Create panic payload
    const payload = {
      type: 'EMERGENCY_PANIC',
      command: 'FLATTEN_ALL',
      timestamp: now,
      priority: 'CRITICAL',
      data: {
        cancelAllOrders: true,
        closeAllPositions: true,
        source: 'UI_PANIC_BUTTON',
        reason: 'USER_EMERGENCY',
      },
    };

    try {
      // Send multiple confirmations to ensure receipt
      let sendCount = 0;
      const sendInterval = setInterval(() => {
        if (sendCount >= 3 || !isConnected) {
          clearInterval(sendInterval);
          return;
        }
        sendMessage(JSON.stringify(payload));
        setWsMessageCount(prev => prev + 1);
        sendCount++;
      }, 100);

      // Update state
      setLastPanicTime(now);
      onPanicExecuted(now);
      
      // Animation sequence
      await controls.start({
        scale: [1, 1.1, 0.95, 1],
        rotate: [0, -5, 5, 0],
        transition: { duration: 0.5 }
      });
      
      setPanicState('executed');
      
      // Reset to idle after delay
      setTimeout(() => {
        setPanicState('idle');
        setWsMessageCount(0);
      }, 2000);
      
      console.log('[PanicButton] EMERGENCY PANIC EXECUTED');
    } catch (error) {
      console.error('[PanicButton] Failed to execute panic:', error);
      setPanicState('idle');
    }
  }, [isConnected, sendMessage, controls, onPanicExecuted]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (progressFrameRef.current) {
        cancelAnimationFrame(progressFrameRef.current);
      }
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
      }
    };
  }, []);

  // Mouse up handler to cancel hold
  useEffect(() => {
    const handleMouseUp = () => {
      if (panicState === 'confirming') {
        cancelConfirmation();
      }
    };
    
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [panicState, cancelConfirmation]);

  const getButtonColor = () => {
    switch (panicState) {
      case 'executing':
        return '#f59e0b'; // Amber
      case 'executed':
        return '#ef4444'; // Red
      case 'confirming':
        return '#f97316'; // Orange
      default:
        return '#dc2626'; // Dark red
    }
  };

  const getGlowIntensity = () => {
    switch (panicState) {
      case 'executing':
        return '0_0_60px_rgba(245,158,11,0.8)';
      case 'executed':
        return '0_0_80px_rgba(239,68,68,0.9)';
      case 'confirming':
        return '0_0_50px_rgba(249,115,22,0.7)';
      default:
        return '0_0_30px_rgba(220,38,38,0.5)';
    }
  };

  return (
    <div className="w-full p-6 bg-slate-900/90 rounded-2xl border-2 border-red-500/30 shadow-[0_0_40px_rgba(220,38,38,0.3)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <ShieldAlert className="w-6 h-6 text-red-400" />
          <h3 className="text-xl font-bold text-red-400 tracking-wider">
            EMERGENCY PROTOCOL
          </h3>
        </div>
        <div className={`px-3 py-1 rounded-full text-xs font-mono font-bold ${
          isArmed 
            ? 'bg-red-500/20 text-red-400 border border-red-500' 
            : 'bg-slate-700 text-slate-400'
        }`}>
          {isArmed ? 'ARMED' : 'DISARMED'}
        </div>
      </div>

      {/* Main Panic Button */}
      <div className="relative flex flex-col items-center">
        <motion.button
          onMouseDown={startConfirmation}
          onMouseLeave={cancelConfirmation}
          disabled={!isArmed || isOnCooldown || panicState === 'executing'}
          animate={controls}
          whileHover={isArmed && !isOnCooldown ? { scale: 1.05 } : {}}
          whileTap={isArmed && !isOnCooldown ? { scale: 0.95 } : {}}
          className="relative w-48 h-48 rounded-full font-bold text-white overflow-hidden cursor-pointer select-none"
          style={{
            background: `radial-gradient(circle, ${getButtonColor()} 0%, #7f1d1d 100%)`,
            boxShadow: getGlowIntensity(),
          }}
        >
          {/* Animated ring overlay during confirmation */}
          <AnimatePresence>
            {panicState === 'confirming' && (
              <motion.svg
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 200 200"
              >
                <circle
                  cx="100"
                  cy="100"
                  r="90"
                  fill="none"
                  stroke="rgba(255,255,255,0.5)"
                  strokeWidth="4"
                  strokeDasharray={`${2 * Math.PI * 90}`}
                  strokeDashoffset={`${2 * Math.PI * 90 * (1 - holdProgress)}`}
                  strokeLinecap="round"
                  transform="rotate(-90 100 100)"
                />
              </motion.svg>
            )}
          </AnimatePresence>

          {/* Button Content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
            <AnimatePresence mode="wait">
              {panicState === 'executing' ? (
                <motion.div
                  key="executing"
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.5, opacity: 0 }}
                  className="flex flex-col items-center"
                >
                  <Clock className="w-12 h-12 animate-spin" />
                  <span className="mt-2 text-sm font-mono">EXECUTING...</span>
                </motion.div>
              ) : panicState === 'executed' ? (
                <motion.div
                  key="executed"
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.8, opacity: 0 }}
                  className="flex flex-col items-center"
                >
                  <XCircle className="w-12 h-12" />
                  <span className="mt-2 text-sm font-mono">FLATTENED</span>
                </motion.div>
              ) : panicState === 'confirming' ? (
                <motion.div
                  key="confirming"
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  exit={{ scale: 0.8 }}
                  className="flex flex-col items-center"
                >
                  <AlertTriangle className="w-12 h-12 animate-pulse" />
                  <span className="mt-2 text-sm font-mono">
                    HOLD {(1 - holdProgress) * CONFIRM_HOLD_DURATION / 1000}s
                  </span>
                </motion.div>
              ) : (
                <motion.div
                  key="idle"
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  exit={{ scale: 0.8 }}
                  className="flex flex-col items-center"
                >
                  <Skull className="w-12 h-12" />
                  <span className="mt-2 text-lg font-black tracking-wider">
                    FLATTEN ALL
                  </span>
                  <span className="text-xs opacity-70 mt-1">
                    {isOnCooldown ? 'ON COOLDOWN' : 'HOLD TO ACTIVATE'}
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Pulsing outer ring */}
          <motion.div
            className="absolute inset-0 rounded-full border-4 border-white/20"
            animate={{
              scale: [1, 1.1, 1],
              opacity: [0.5, 0.2, 0.5],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        </motion.button>

        {/* Warning Text */}
        <motion.p
          className="mt-6 text-center text-red-400/80 text-sm font-mono max-w-md"
          animate={{ opacity: panicState === 'idle' ? 1 : 0.5 }}
        >
          ⚠️ THIS WILL CANCEL ALL ORDERS AND CLOSE ALL POSITIONS AT MARKET PRICE
        </motion.p>
      </div>

      {/* Status Grid */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700">
          <div className="text-xs text-slate-400 mb-1">CONNECTION</div>
          <div className={`font-mono font-bold ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
            {isConnected ? 'ACTIVE' : 'FAILED'}
          </div>
        </div>
        <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700">
          <div className="text-xs text-slate-400 mb-1">COOLDOWN</div>
          <div className="font-mono font-bold text-slate-300">
            {isOnCooldown ? `${(cooldownRemaining / 1000).toFixed(1)}s` : 'READY'}
          </div>
        </div>
        <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700">
          <div className="text-xs text-slate-400 mb-1">WS MESSAGES</div>
          <div className="font-mono font-bold text-cyan-400">
            {wsMessageCount}
          </div>
        </div>
      </div>

      {/* Last Execution Time */}
      {lastPanicTime > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700 text-center">
          <div className="text-xs text-slate-400">LAST PANIC EXECUTION</div>
          <div className="text-sm font-mono text-red-400">
            {new Date(lastPanicTime).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
};

export default PanicButton;
