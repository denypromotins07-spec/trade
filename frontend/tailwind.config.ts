import type { Config } from 'tailwindcss';

// ============================================================================
// CYBERPUNK/QUANTITATIVE TAILWIND THEME CONFIGURATION
// Deep obsidian backgrounds, neon cyan/magenta accents, glassmorphism utilities
// Optimized for hardware-accelerated rendering
// ============================================================================

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      // =========================================================================
      // COLOR PALETTE - Cyberpunk/Quant Aesthetic
      // =========================================================================
      colors: {
        // Primary background layers (deep obsidian)
        obsidian: {
          50: '#0a0a0f',
          100: '#0d0d14',
          200: '#12121a',
          300: '#1a1a25',
          400: '#222230',
          500: '#2a2a3c',
        },
        
        // Neon accent colors
        neon: {
          cyan: {
            DEFAULT: '#00f3ff',
            glow: 'rgba(0, 243, 255, 0.5)',
            dim: '#0099aa',
          },
          magenta: {
            DEFAULT: '#ff00ff',
            glow: 'rgba(255, 0, 255, 0.5)',
            dim: '#aa00aa',
          },
          lime: {
            DEFAULT: '#ccff00',
            glow: 'rgba(204, 255, 0, 0.5)',
            dim: '#88aa00',
          },
          amber: {
            DEFAULT: '#ffaa00',
            glow: 'rgba(255, 170, 0, 0.5)',
            dim: '#aa7700',
          },
        },
        
        // Status colors with cyberpunk twist
        status: {
          success: '#00ff9d',
          warning: '#ffaa00',
          error: '#ff2a6d',
          info: '#00f3ff',
        },
        
        // Glass overlay colors
        glass: {
          light: 'rgba(255, 255, 255, 0.03)',
          medium: 'rgba(255, 255, 255, 0.05)',
          heavy: 'rgba(255, 255, 255, 0.08)',
          border: 'rgba(255, 255, 255, 0.1)',
        },
      },
      
      // =========================================================================
      // BACKDROP FILTER & GLASSMORPHISM
      // =========================================================================
      backdropBlur: {
        xs: '2px',
        sm: '4px',
        md: '8px',
        lg: '16px',
        xl: '24px',
        xxl: '40px',
      },
      
      // =========================================================================
      // BOX SHADOWS - Neon Glow Effects
      // =========================================================================
      boxShadow: {
        'neon-cyan': '0 0 10px rgba(0, 243, 255, 0.5), 0 0 20px rgba(0, 243, 255, 0.3)',
        'neon-magenta': '0 0 10px rgba(255, 0, 255, 0.5), 0 0 20px rgba(255, 0, 255, 0.3)',
        'neon-lime': '0 0 10px rgba(204, 255, 0, 0.5), 0 0 20px rgba(204, 255, 0, 0.3)',
        'neon-amber': '0 0 10px rgba(255, 170, 0, 0.5), 0 0 20px rgba(255, 170, 0, 0.3)',
        'glass-inner': 'inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        'glow-soft': '0 0 30px rgba(0, 243, 255, 0.15)',
      },
      
      // =========================================================================
      // ANIMATIONS - GPU-Accelerated Transitions
      // =========================================================================
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-fast': 'pulse 0.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow-fade': 'glowFade 2s ease-in-out infinite',
        'scan-line': 'scanLine 8s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'blink-slow': 'blink 2s step-end infinite',
      },
      
      keyframes: {
        glowFade: {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
        scanLine: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      
      // =========================================================================
      // GRADIENTS - Cyberpunk Backgrounds
      // =========================================================================
      backgroundImage: {
        'cyber-gradient': 'linear-gradient(135deg, #0a0a0f 0%, #1a1a25 50%, #0d0d14 100%)',
        'neon-gradient': 'linear-gradient(90deg, #00f3ff 0%, #ff00ff 100%)',
        'glass-gradient': 'linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)',
        'holographic': 'linear-gradient(135deg, rgba(0,243,255,0.1) 0%, rgba(255,0,255,0.1) 50%, rgba(204,255,0,0.1) 100%)',
      },
      
      // =========================================================================
      // BORDER RADIUS - Sharp Futuristic Corners
      // =========================================================================
      borderRadius: {
        'none': '0',
        'sm': '2px',
        'DEFAULT': '4px',
        'md': '6px',
        'lg': '8px',
        'xl': '12px',
        '2xl': '16px',
        'full': '9999px',
        'cyber': '2px 8px 2px 8px', // Asymmetric cyber corner
      },
      
      // =========================================================================
      // FONT FAMILY - Monospace for Quant Data
      // =========================================================================
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Orbitron', 'system-ui', 'sans-serif'],
      },
      
      // =========================================================================
      // SPACING - Fine-grained Control
      // =========================================================================
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      
      // =========================================================================
      // Z-INDEX - Layer Management
      // =========================================================================
      zIndex: {
        '-10': '-10',
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
    },
  },
  plugins: [],
};

export default config;
