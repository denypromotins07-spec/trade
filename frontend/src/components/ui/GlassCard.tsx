'use client';

import React, { memo, useMemo } from 'react';
import { motion } from 'framer-motion';

// ============================================================================
// GLASSMORPHISM CARD COMPONENT
// Highly optimized glass card with GPU-accelerated rendering
// Uses CSS backdrop-filter and will-change for compositor isolation
// ============================================================================

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  intensity?: 'light' | 'medium' | 'heavy';
  glowColor?: 'cyan' | 'magenta' | 'lime' | 'amber' | 'none';
  hoverable?: boolean;
  onClick?: () => void;
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  border?: boolean;
  animate?: boolean;
  'data-testid'?: string;
}

// Memoized component to prevent unnecessary re-renders during high-frequency updates
export const GlassCard = memo<GlassCardProps>(function GlassCard({
  children,
  className = '',
  intensity = 'medium',
  glowColor = 'none',
  hoverable = false,
  onClick,
  padding = 'md',
  border = true,
  animate = true,
  'data-testid': dataTestId,
}) {
  // ==========================================================================
  // COMPUTED STYLES - Memoized to prevent recalculation on each render
  // ==========================================================================
  
  const glassStyles = useMemo(() => {
    const baseStyles = {
      light: 'bg-glass-light backdrop-blur-sm',
      medium: 'bg-glass-medium backdrop-blur-md',
      heavy: 'bg-glass-heavy backdrop-blur-lg',
    };
    
    return baseStyles[intensity];
  }, [intensity]);
  
  const glowStyles = useMemo(() => {
    const glows = {
      cyan: 'shadow-neon-cyan hover:shadow-[0_0_20px_rgba(0,243,255,0.6)]',
      magenta: 'shadow-neon-magenta hover:shadow-[0_0_20px_rgba(255,0,255,0.6)]',
      lime: 'shadow-neon-lime hover:shadow-[0_0_20px_rgba(204,255,0,0.6)]',
      amber: 'shadow-neon-amber hover:shadow-[0_0_20px_rgba(255,170,0,0.6)]',
      none: '',
    };
    
    return glows[glowColor];
  }, [glowColor]);
  
  const paddingStyles = useMemo(() => {
    const paddings = {
      none: '',
      sm: 'p-3',
      md: 'p-4',
      lg: 'p-6',
      xl: 'p-8',
    };
    
    return paddings[padding];
  }, [padding]);
  
  // ==========================================================================
  // VARIANTS FOR FRAMER MOTION ANIMATIONS
  // GPU-accelerated transforms only (no layout thrashing)
  // ==========================================================================
  
  const variants = useMemo(() => ({
    initial: {
      opacity: 0,
      y: 10,
      scale: 0.98,
    },
    animate: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1], // Custom cubic-bezier for smooth feel
      },
    },
    hover: {
      scale: hoverable ? 1.02 : 1,
      y: hoverable ? -2 : 0,
      transition: {
        duration: 0.2,
        ease: 'easeOut',
      },
    },
  }), [hoverable]);
  
  // ==========================================================================
  // RENDER
  // Using motion.div for Framer Motion animations with GPU acceleration
  // ==========================================================================
  
  return (
    <motion.div
      data-testid={dataTestId}
      className={`
        relative overflow-hidden
        ${glassStyles}
        ${border ? 'border border-glass-border' : ''}
        ${glowStyles}
        ${paddingStyles}
        ${hoverable ? 'cursor-pointer transition-shadow duration-200' : ''}
        ${className}
        gpu-accelerated
      `}
      initial={animate ? 'initial' : false}
      animate={animate ? 'animate' : false}
      whileHover={hoverable ? 'hover' : undefined}
      variants={variants}
      onClick={onClick}
      // Critical: Force GPU layer isolation for smooth compositing
      style={{
        willChange: animate ? 'transform, opacity' : 'auto',
        transform: 'translateZ(0)',
        backfaceVisibility: 'hidden',
      }}
    >
      {/* Subtle gradient overlay for depth */}
      <div 
        className="absolute inset-0 bg-glass-gradient pointer-events-none" 
        aria-hidden="true"
      />
      
      {/* Content container */}
      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  );
});

// ============================================================================
// DISPLAY NAME FOR DEVTOOLS
// ============================================================================

GlassCard.displayName = 'GlassCard';

// ============================================================================
// EXPORT DEFAULT FOR CONVENIENCE
// ============================================================================

export default GlassCard;
