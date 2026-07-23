/**
 * ToastSystem.tsx - Advanced Alerting: GPU-Accelerated Non-Blocking Notifications
 * 
 * Implements non-blocking, GPU-accelerated toast notifications using CSS transforms
 * to slide in critical execution fills without triggering layout recalculations.
 * 
 * Features:
 * - CSS transform-based animations (no layout thrashing)
 * - GPU-accelerated compositing via will-change and translateZ
 * - Non-blocking notification queue with auto-dismiss
 * - Priority levels (critical, warning, info, success)
 * - Cyberpunk-styled toast cards with neon accents
 */

'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

type ToastPriority = 'critical' | 'warning' | 'info' | 'success';

interface ToastNotification {
  id: string;
  title: string;
  message: string;
  priority: ToastPriority;
  timestamp: number;
  duration?: number; // ms, 0 = persistent
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastSystemProps {
  maxToasts?: number;
  defaultDuration?: number;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const MAX_TOASTS_DEFAULT = 5;
const DURATION_DEFAULT = 5000;
const TOAST_HEIGHT = 80; // px
const GAP = 12; // px

const PRIORITY_CONFIG: Record<ToastPriority, { color: string; icon: string; bgColor: string }> = {
  critical: { color: '#ff0044', icon: '⚠️', bgColor: 'rgba(255, 0, 68, 0.1)' },
  warning: { color: '#ffcc00', icon: '⚡', bgColor: 'rgba(255, 204, 0, 0.1)' },
  info: { color: '#00ccff', icon: 'ℹ️', bgColor: 'rgba(0, 204, 255, 0.1)' },
  success: { color: '#00ff88', icon: '✓', bgColor: 'rgba(0, 255, 136, 0.1)' },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generates a unique ID for toast notifications
 */
const generateId = (): string => {
  return `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
};

// ============================================================================
// Individual Toast Component
// ============================================================================

interface ToastCardProps {
  toast: ToastNotification;
  index: number;
  onDismiss: (id: string) => void;
  onAnimationComplete: (id: string) => void;
}

const ToastCard: React.FC<ToastCardProps> = ({
  toast,
  index,
  onDismiss,
  onAnimationComplete,
}) => {
  const [isExiting, setIsExiting] = useState(false);
  const [progress, setProgress] = useState(100);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const progressTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  const config = PRIORITY_CONFIG[toast.priority];
  const duration = toast.duration ?? DURATION_DEFAULT;

  // Auto-dismiss timer
  useEffect(() => {
    if (duration === 0) return; // Persistent toast
    
    timerRef.current = setTimeout(() => {
      setIsExiting(true);
    }, duration);
    
    // Progress bar animation
    const startTime = Date.now();
    const updateProgress = () => {
      const elapsed = Date.now() - startTime;
      const newProgress = Math.max(0, 100 - (elapsed / duration) * 100);
      setProgress(newProgress);
      
      if (newProgress > 0) {
        progressTimerRef.current = setTimeout(updateProgress, 16); // ~60fps
      }
    };
    progressTimerRef.current = setTimeout(updateProgress, 16);
    
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (progressTimerRef.current) clearTimeout(progressTimerRef.current);
    };
  }, [duration]);

  // Handle exit animation completion
  useEffect(() => {
    if (isExiting) {
      const timeout = setTimeout(() => {
        onAnimationComplete(toast.id);
      }, 300); // Match CSS transition duration
      return () => clearTimeout(timeout);
    }
  }, [isExiting, toast.id, onAnimationComplete]);

  const handleDismiss = useCallback(() => {
    setIsExiting(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (progressTimerRef.current) clearTimeout(progressTimerRef.current);
  }, []);

  return (
    <div
      className={`absolute right-0 w-full max-w-sm transition-all duration-300 ease-out ${
        isExiting ? 'opacity-0 translate-x-full' : 'opacity-100 translate-x-0'
      }`}
      style={{
        transform: `translateY(${index * -(TOAST_HEIGHT + GAP)}px) translateZ(0)`,
        willChange: 'transform, opacity',
        height: TOAST_HEIGHT,
      }}
      role="alert"
      aria-live={toast.priority === 'critical' ? 'assertive' : 'polite'}
    >
      <div
        className="relative h-full rounded-lg border backdrop-blur-md overflow-hidden shadow-lg"
        style={{
          backgroundColor: config.bgColor,
          borderColor: config.color,
          boxShadow: `0 0 20px ${config.color}33`,
        }}
      >
        {/* Progress bar */}
        {duration > 0 && (
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white/10">
            <div
              className="h-full transition-none"
              style={{
                width: `${progress}%`,
                backgroundColor: config.color,
              }}
            />
          </div>
        )}
        
        {/* Content */}
        <div className="flex items-start gap-3 p-4 h-full">
          {/* Icon */}
          <span className="text-xl flex-shrink-0">{config.icon}</span>
          
          {/* Text */}
          <div className="flex-1 min-w-0 pt-0.5">
            <div className="font-mono text-sm font-bold truncate" style={{ color: config.color }}>
              {toast.title}
            </div>
            <div className="text-xs text-gray-300 font-mono mt-0.5 line-clamp-2">
              {toast.message}
            </div>
          </div>
          
          {/* Actions */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {toast.action && (
              <button
                onClick={toast.action.onClick}
                className="px-2 py-1 text-xs font-mono rounded border transition-colors"
                style={{
                  borderColor: config.color,
                  color: config.color,
                }}
              >
                {toast.action.label}
              </button>
            )}
            <button
              onClick={handleDismiss}
              className="w-5 h-5 rounded flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
              aria-label="Dismiss notification"
            >
              ✕
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const ToastSystem: React.FC<ToastSystemProps> = ({
  maxToasts = MAX_TOASTS_DEFAULT,
  defaultDuration = DURATION_DEFAULT,
  position = 'top-right',
}) => {
  const [toasts, setToasts] = useState<ToastNotification[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  /**
   * Adds a new toast notification
   */
  const addToast = useCallback((toast: Omit<ToastNotification, 'id' | 'timestamp'>) => {
    const newToast: ToastNotification = {
      ...toast,
      id: generateId(),
      timestamp: Date.now(),
    };
    
    setToasts((prev) => {
      const updated = [newToast, ...prev];
      // Trim to maxToasts
      return updated.slice(0, maxToasts);
    });
    
    return newToast.id;
  }, [maxToasts]);

  /**
   * Dismisses a toast by ID
   */
  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  /**
   * Handles animation completion for exiting toasts
   */
  const handleAnimationComplete = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  /**
   * Convenience methods for different priority levels
   */
  const notify = {
    critical: (title: string, message: string, options?: Partial<ToastNotification>) =>
      addToast({ ...options, title, message, priority: 'critical' }),
    warning: (title: string, message: string, options?: Partial<ToastNotification>) =>
      addToast({ ...options, title, message, priority: 'warning' }),
    info: (title: string, message: string, options?: Partial<ToastNotification>) =>
      addToast({ ...options, title, message, priority: 'info' }),
    success: (title: string, message: string, options?: Partial<ToastNotification>) =>
      addToast({ ...options, title, message, priority: 'success' }),
  };

  // Expose notify methods via custom event for external access
  useEffect(() => {
    const handleCustomEvent = (event: CustomEvent) => {
      const { title, message, priority, ...options } = event.detail;
      addToast({ title, message, priority, ...options });
    };
    
    window.addEventListener('toast-notify' as any, handleCustomEvent as any);
    return () => window.removeEventListener('toast-notify' as any, handleCustomEvent as any);
  }, [addToast]);

  // Position classes
  const positionClasses = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
  };

  return (
    <>
      {/* Toast Container */}
      <div
        ref={containerRef}
        className={`fixed z-50 pointer-events-auto ${positionClasses[position]}`}
        style={{
          width: 'min(400px, calc(100vw - 2rem))',
        }}
        aria-label="Notification center"
      >
        <div className="relative" style={{ height: toasts.length * (TOAST_HEIGHT + GAP) }}>
          {toasts.map((toast, index) => (
            <ToastCard
              key={toast.id}
              toast={toast}
              index={index}
              onDismiss={dismissToast}
              onAnimationComplete={handleAnimationComplete}
            />
          ))}
        </div>
      </div>
      
      {/* Expose API globally for external components */}
      <ToastAPIProvider notify={notify} />
    </>
  );
};

// ============================================================================
// Global API Provider
// ============================================================================

interface ToastAPIContext {
  notify: {
    critical: (title: string, message: string, options?: Partial<ToastNotification>) => string;
    warning: (title: string, message: string, options?: Partial<ToastNotification>) => string;
    info: (title: string, message: string, options?: Partial<ToastNotification>) => string;
    success: (title: string, message: string, options?: Partial<ToastNotification>) => string;
  };
}

const ToastContext = React.createContext<ToastAPIContext | null>(null);

const ToastAPIProvider: React.FC<{ notify: ToastAPIContext['notify'] }> = ({ notify }) => {
  useEffect(() => {
    // Store reference globally for imperative usage
    (window as any).__TOAST_API__ = { notify };
  }, [notify]);
  
  return null;
};

/**
 * Hook to access toast API from anywhere in the app
 */
export const useToast = (): ToastAPIContext['notify'] => {
  const context = React.useContext(ToastContext);
  if (!context) {
    // Fallback to global API
    return (window as any).__TOAST_API__?.notify || {
      critical: () => '',
      warning: () => '',
      info: () => '',
      success: () => '',
    };
  }
  return context.notify;
};

export default ToastSystem;
