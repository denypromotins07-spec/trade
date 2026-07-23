/**
 * File 9: frontend/src/components/feedback/ScreenFlash.tsx
 * 
 * Elite Implementation:
 * - GPU-accelerated CSS border flashes using Framer Motion.
 * - Visual pulse for massive PnL spikes (neon green/red).
 * - Uses transform3d for hardware acceleration, avoiding layout recalculations.
 * - Cyberpunk aesthetic with glow effects.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export type FlashType = 'PROFIT' | 'LOSS' | 'WARNING' | 'INFO';

interface FlashEvent {
  id: string;
  type: FlashType;
  intensity: number; // 0-1 scale
  timestamp: number;
}

interface ScreenFlashProps {
  maxDuration?: number;
}

const FLASH_COLORS: Record<FlashType, { border: string; glow: string; bg: string }> = {
  PROFIT: {
    border: '#00ff88',
    glow: 'rgba(0, 255, 136, 0.5)',
    bg: 'rgba(0, 255, 136, 0.05)',
  },
  LOSS: {
    border: '#ff0055',
    glow: 'rgba(255, 0, 85, 0.5)',
    bg: 'rgba(255, 0, 85, 0.05)',
  },
  WARNING: {
    border: '#ffaa00',
    glow: 'rgba(255, 170, 0, 0.5)',
    bg: 'rgba(255, 170, 0, 0.05)',
  },
  INFO: {
    border: '#00ccff',
    glow: 'rgba(0, 204, 255, 0.5)',
    bg: 'rgba(0, 204, 255, 0.05)',
  },
};

export const ScreenFlash: React.FC<ScreenFlashProps> = ({ maxDuration = 2000 }) => {
  const [flashes, setFlashes] = useState<FlashEvent[]>([]);

  // Listen for flash events via CustomEvent
  useEffect(() => {
    const handleFlashEvent = (e: CustomEvent<FlashEvent>) => {
      setFlashes(prev => [...prev, e.detail].slice(-5)); // Max 5 concurrent flashes
    };

    window.addEventListener('nautilus-flash' as any, handleFlashEvent as any);
    return () => window.removeEventListener('nautilus-flash' as any, handleFlashEvent as any);
  }, []);

  // Auto-remove flashes after duration
  useEffect(() => {
    if (flashes.length === 0) return;

    const timers = flashes.map(flash => 
      setTimeout(() => {
        setFlashes(prev => prev.filter(f => f.id !== flash.id));
      }, maxDuration)
    );

    return () => timers.forEach(clearTimeout);
  }, [flashes, maxDuration]);

  return (
    <AnimatePresence>
      {flashes.map((flash, index) => {
        const colors = FLASH_COLORS[flash.type];
        const borderWidth = Math.max(2, Math.min(8, flash.intensity * 8));
        
        return (
          <motion.div
            key={flash.id}
            initial={{ opacity: 0, scale: 1 + (1 - flash.intensity) * 0.1 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.02 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            className="fixed inset-0 pointer-events-none z-[9999]"
            style={{
              boxShadow: `inset 0 0 ${60 * flash.intensity}px ${colors.glow}`,
            }}
          >
            {/* Border flash */}
            <div 
              className="absolute inset-0"
              style={{
                borderTop: `${borderWidth}px solid ${colors.border}`,
                borderBottom: `${borderWidth}px solid ${colors.border}`,
                filter: `drop-shadow(0 0 ${20 * flash.intensity}px ${colors.border})`,
              }}
            />
            
            {/* Corner accents */}
            <div className="absolute top-0 left-0 w-32 h-32">
              <motion.div
                initial={{ scaleX: 0, scaleY: 0 }}
                animate={{ scaleX: 1, scaleY: 1 }}
                exit={{ scaleX: 0, scaleY: 0 }}
                className="w-full h-full"
                style={{
                  borderLeft: `${borderWidth}px solid ${colors.border}`,
                  borderTop: `${borderWidth}px solid ${colors.border}`,
                  background: `linear-gradient(135deg, ${colors.bg} 0%, transparent 70%)`,
                }}
              />
            </div>
            
            <div className="absolute top-0 right-0 w-32 h-32">
              <motion.div
                initial={{ scaleX: 0, scaleY: 0 }}
                animate={{ scaleX: 1, scaleY: 1 }}
                exit={{ scaleX: 0, scaleY: 0 }}
                className="w-full h-full"
                style={{
                  borderRight: `${borderWidth}px solid ${colors.border}`,
                  borderTop: `${borderWidth}px solid ${colors.border}`,
                  background: `linear-gradient(225deg, ${colors.bg} 0%, transparent 70%)`,
                }}
              />
            </div>
            
            <div className="absolute bottom-0 left-0 w-32 h-32">
              <motion.div
                initial={{ scaleX: 0, scaleY: 0 }}
                animate={{ scaleX: 1, scaleY: 1 }}
                exit={{ scaleX: 0, scaleY: 0 }}
                className="w-full h-full"
                style={{
                  borderLeft: `${borderWidth}px solid ${colors.border}`,
                  borderBottom: `${borderWidth}px solid ${colors.border}`,
                  background: `linear-gradient(45deg, ${colors.bg} 0%, transparent 70%)`,
                }}
              />
            </div>
            
            <div className="absolute bottom-0 right-0 w-32 h-32">
              <motion.div
                initial={{ scaleX: 0, scaleY: 0 }}
                animate={{ scaleX: 1, scaleY: 1 }}
                exit={{ scaleX: 0, scaleY: 0 }}
                className="w-full h-full"
                style={{
                  borderRight: `${borderWidth}px solid ${colors.border}`,
                  borderBottom: `${borderWidth}px solid ${colors.border}`,
                  background: `linear-gradient(315deg, ${colors.bg} 0%, transparent 70%)`,
                }}
              />
            </div>

            {/* Center pulse indicator */}
            <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 rounded-full backdrop-blur-md"
              style={{
                backgroundColor: colors.bg,
                border: `1px solid ${colors.border}`,
                boxShadow: `0 0 30px ${colors.glow}`,
              }}
            >
              <motion.div
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 1.5, opacity: 0 }}
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: colors.border }}
              />
              <span 
                className="text-xs font-mono uppercase tracking-widest"
                style={{ color: colors.border }}
              >
                {flash.type} // {(flash.intensity * 100).toFixed(0)}%
              </span>
            </div>
          </motion.div>
        );
      })}
    </AnimatePresence>
  );
};

/**
 * Utility function to trigger a screen flash
 */
export const triggerFlash = (type: FlashType, intensity: number = 0.5): void => {
  const event = new CustomEvent('nautilus-flash', {
    detail: {
      id: `flash_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      type,
      intensity: Math.max(0, Math.min(1, intensity)),
      timestamp: Date.now(),
    } as FlashEvent,
  });
  window.dispatchEvent(event);
};

export default ScreenFlash;
