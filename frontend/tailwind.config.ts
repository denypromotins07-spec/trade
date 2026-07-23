import type { Config } from 'tailwindcss';

/**
 * Nautilus/Ray Cyberpunk/Quantitative Tailwind Configuration
 * 
 * Design Philosophy:
 * - Deep obsidian backgrounds for reduced eye strain during extended trading sessions
 * - Neon cyan/magenta accents for high-contrast data visualization
 * - Custom glassmorphism utilities for modern, layered UI aesthetic
 * - GPU-accelerated transitions via will-change and transform properties
 */
const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      // Cyberpunk Color Palette
      colors: {
        // Primary background - deep obsidian
        obsidian: {
          50: '#0a0a0f',
          100: '#0d0d14',
          200: '#12121a',
          300: '#1a1a25',
          400: '#252535',
          500: '#32324a',
        },
        
        // Neon accent colors
        neon: {
          cyan: {
            DEFAULT: '#00f5ff',
            dim: '#00c4cc',
            bright: '#66f9ff',
            glow: 'rgba(0, 245, 255, 0.5)',
          },
          magenta: {
            DEFAULT: '#ff00ff',
            dim: '#cc00cc',
            bright: '#ff66ff',
            glow: 'rgba(255, 0, 255, 0.5)',
          },
          green: {
            DEFAULT: '#00ff88',
            dim: '#00cc6c',
            bright: '#66ffb3',
            glow: 'rgba(0, 255, 136, 0.5)',
          },
          red: {
            DEFAULT: '#ff3366',
            dim: '#cc2952',
            bright: '#ff6688',
            glow: 'rgba(255, 51, 102, 0.5)',
          },
          amber: {
            DEFAULT: '#ffaa00',
            dim: '#cc8800',
            bright: '#ffcc44',
            glow: 'rgba(255, 170, 0, 0.5)',
          },
        },
        
        // Trading-specific colors
        trading: {
          long: '#00ff88',
          short: '#ff3366',
          neutral: '#6b7280',
        },
      },
      
      // Glassmorphism backdrop blur values
      blur: {
        xs: '2px',
        glass: '12px',
        xl: '24px',
        xxl: '40px',
      },
      
      // Custom animations for cyberpunk aesthetic
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan-line': 'scan-line 8s linear infinite',
        'data-stream': 'data-stream 3s ease-in-out infinite',
        'latency-ping': 'latency-ping 1s ease-out infinite',
        'hologram': 'hologram 4s ease-in-out infinite',
      },
      
      keyframes: {
        'pulse-glow': {
          '0%, 100%': {
            opacity: '1',
            filter: 'drop-shadow(0 0 8px rgba(0, 245, 255, 0.6))',
          },
          '50%': {
            opacity: '0.8',
            filter: 'drop-shadow(0 0 16px rgba(0, 245, 255, 0.3))',
          },
        },
        'scan-line': {
          '0%': {
            transform: 'translateY(-100%)',
          },
          '100%': {
            transform: 'translateY(100vh)',
          },
        },
        'data-stream': {
          '0%, 100%': {
            opacity: '0.3',
          },
          '50%': {
            opacity: '1',
          },
        },
        'latency-ping': {
          '0%': {
            transform: 'scale(0.8)',
            opacity: '1',
          },
          '100%': {
            transform: 'scale(2)',
            opacity: '0',
          },
        },
        'hologram': {
          '0%, 100%': {
            opacity: '0.9',
            filter: 'hue-rotate(0deg)',
          },
          '50%': {
            opacity: '0.7',
            filter: 'hue-rotate(15deg)',
          },
        },
      },
      
      // Hardware-accelerated transitions
      transitionTimingFunction: {
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'snap': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      },
      
      // Spacing for dense trading UI
      spacing: {
        '4.5': '1.125rem',
        '5.5': '1.375rem',
        '6.5': '1.625rem',
      },
      
      // Font families for quantitative display
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        display: ['"Rajdhani"', 'system-ui', 'sans-serif'],
      },
      
      // Box shadows with neon glow effects
      boxShadow: {
        'neon-cyan': '0 0 10px rgba(0, 245, 255, 0.5), 0 0 20px rgba(0, 245, 255, 0.3)',
        'neon-magenta': '0 0 10px rgba(255, 0, 255, 0.5), 0 0 20px rgba(255, 0, 255, 0.3)',
        'neon-green': '0 0 10px rgba(0, 255, 136, 0.5), 0 0 20px rgba(0, 255, 136, 0.3)',
        'neon-red': '0 0 10px rgba(255, 51, 102, 0.5), 0 0 20px rgba(255, 51, 102, 0.3)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3)',
      },
      
      // Border radius for modern UI
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
      },
      
      // Z-index layers for complex dashboard
      zIndex: {
        '-10': '-10',
        '-20': '-20',
        '100': '100',
        '200': '200',
        '300': '300',
      },
    },
  },
  plugins: [],
};

export default config;
