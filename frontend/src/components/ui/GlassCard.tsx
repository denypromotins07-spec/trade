'use client';

import React, { memo, CSSProperties } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';

/**
 * GlassCard Component
 * 
 * A highly optimized glassmorphism card component utilizing CSS backdrop-filter
 * and will-change properties to isolate rendering layers and optimize compositor performance.
 * 
 * Performance Features:
 * - GPU-accelerated via transform: translateZ(0)
 * - will-change hints for smooth animations
 * - Minimal re-renders via React.memo
 * - Optional Framer Motion animations with hardware acceleration
 */

interface GlassCardProps extends Omit<HTMLMotionProps<'div'>, 'children'> {
  children: React.ReactNode;
  
  /** Glass intensity level */
  variant?: 'light' | 'medium' | 'strong';
  
  /** Enable hover glow effect */
  hoverGlow?: boolean;
  
  /** Glow color variant */
  glowColor?: 'cyan' | 'magenta' | 'green' | 'red' | 'none';
  
  /** Border visibility */
  showBorder?: boolean;
  
  /** Custom padding */
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  
  /** Disable all animations for accessibility */
  reducedMotion?: boolean;
  
  /** Custom className to append */
  className?: string;
}

const variantStyles = {
  light: {
    background: 'rgba(255, 255, 255, 0.03)',
    backdropFilter: 'blur(8px)',
    WebkitBackdropFilter: 'blur(8px)',
  },
  medium: {
    background: 'rgba(10, 10, 15, 0.6)',
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
  },
  strong: {
    background: 'rgba(18, 18, 26, 0.9)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
  },
};

const glowColors = {
  cyan: '0 0 20px rgba(0, 245, 255, 0.4), 0 0 40px rgba(0, 245, 255, 0.2)',
  magenta: '0 0 20px rgba(255, 0, 255, 0.4), 0 0 40px rgba(255, 0, 255, 0.2)',
  green: '0 0 20px rgba(0, 255, 136, 0.4), 0 0 40px rgba(0, 255, 136, 0.2)',
  red: '0 0 20px rgba(255, 51, 102, 0.4), 0 0 40px rgba(255, 51, 102, 0.2)',
  none: 'none',
};

const paddingStyles = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
  xl: 'p-8',
};

/**
 * Base styles object - computed once, reused across instances
 */
const baseStyles: CSSProperties = {
  border: '1px solid rgba(255, 255, 255, 0.08)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
  willChange: 'transform, opacity',
  transform: 'translateZ(0)', // Force GPU layer
  backfaceVisibility: 'hidden',
  transitionProperty: 'box-shadow, border-color, transform',
  transitionDuration: '200ms',
  transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
};

/**
 * Hover animation variants for Framer Motion
 */
const hoverVariants = {
  initial: {
    scale: 1,
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
  },
  hover: {
    scale: 1.01,
    boxShadow: '0 12px 40px rgba(0, 0, 0, 0.4)',
    transition: {
      type: 'spring',
      stiffness: 400,
      damping: 25,
      mass: 0.5,
    },
  },
};

/**
 * GlassCard Component - Memoized for optimal performance
 */
export const GlassCard = memo(function GlassCard({
  children,
  variant = 'medium',
  hoverGlow = false,
  glowColor = 'none',
  showBorder = true,
  padding = 'md',
  reducedMotion = false,
  className = '',
  style,
  ...props
}: GlassCardProps) {
  // Compute dynamic styles
  const dynamicStyles: CSSProperties = {
    ...baseStyles,
    ...variantStyles[variant],
    ...(hoverGlow && glowColor !== 'none' ? { boxShadow: glowColors[glowColor] } : {}),
    ...style,
  };

  // Build className string
  const classes = [
    'relative',
    'overflow-hidden',
    'rounded-xl',
    paddingStyles[padding],
    className,
  ].filter(Boolean).join(' ');

  // Render with or without motion based on reducedMotion prop
  if (reducedMotion) {
    return (
      <div
        className={classes}
        style={dynamicStyles}
        {...(props as React.HTMLAttributes<HTMLDivElement>)}
      >
        {/* Optional decorative scan line */}
        {hoverGlow && (
          <div
            className="absolute inset-0 pointer-events-none opacity-0 hover:opacity-100 transition-opacity duration-300"
            style={{
              background: 'linear-gradient(180deg, transparent 0%, rgba(255, 255, 255, 0.05) 50%, transparent 100%)',
            }}
          />
        )}
        {children}
      </div>
    );
  }

  return (
    <motion.div
      className={classes}
      style={dynamicStyles}
      initial="initial"
      whileHover={hoverGlow ? 'hover' : undefined}
      variants={hoverVariants}
      transition={{
        type: 'spring',
        stiffness: 400,
        damping: 25,
        mass: 0.5,
      }}
      {...props}
    >
      {/* Decorative gradient overlay for cyberpunk aesthetic */}
      <div
        className="absolute inset-0 pointer-events-none opacity-20"
        style={{
          background: `linear-gradient(
            135deg,
            ${glowColor === 'cyan' ? 'rgba(0, 245, 255, 0.1)' : 
              glowColor === 'magenta' ? 'rgba(255, 0, 255, 0.1)' :
              glowColor === 'green' ? 'rgba(0, 255, 136, 0.1)' :
              glowColor === 'red' ? 'rgba(255, 51, 102, 0.1)' :
              'transparent'} 0%,
            transparent 100%
          )`,
        }}
      />
      
      {/* Optional scan line effect on hover */}
      {hoverGlow && (
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'linear-gradient(180deg, transparent 0%, rgba(255, 255, 255, 0.08) 50%, transparent 100%)',
            y: '-100%',
          }}
          whileHover={{
            y: '100%',
            transition: {
              duration: 1.5,
              ease: 'linear',
            },
          }}
        />
      )}
      
      {/* Border highlight */}
      {showBorder && (
        <div
          className="absolute inset-0 rounded-xl pointer-events-none"
          style={{
            border: '1px solid rgba(255, 255, 255, 0.05)',
            boxShadow: 'inset 0 0 0 1px rgba(255, 255, 255, 0.02)',
          }}
        />
      )}
      
      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  );
});

/**
 * Pre-configured GlassCard variants for common use cases
 */
export const DataCard = memo(function DataCard(props: Omit<GlassCardProps, 'variant' | 'padding'>) {
  return <GlassCard variant="medium" padding="lg" {...props} />;
});

export const MetricCard = memo(function MetricCard(props: Omit<GlassCardProps, 'variant' | 'padding' | 'hoverGlow'>) {
  return <GlassCard variant="light" padding="md" hoverGlow glowColor="cyan" {...props} />;
});

export const StatusCard = memo(function StatusCard(props: Omit<GlassCardProps, 'variant' | 'padding' | 'glowColor'>) {
  return <GlassCard variant="strong" padding="lg" glowColor="green" {...props} />;
});

export default GlassCard;
