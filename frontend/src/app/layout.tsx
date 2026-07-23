import type { Metadata } from 'next';
import { Inter, JetBrains_Mono, Rajdhani } from 'next/font/google';
import '@/styles/globals.css';

/**
 * Font Configurations - Optimized for Trading Dashboard
 * 
 * - Inter: Primary UI font for readability
 * - JetBrains Mono: Monospace for numerical data, prices, timestamps
 * - Rajdhani: Display font for headers and titles (cyberpunk aesthetic)
 */
const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
  preload: true,
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
  preload: true,
});

const rajdhani = Rajdhani({
  weight: ['400', '500', '600', '700'],
  subsets: ['latin'],
  variable: '--font-display',
  display: 'swap',
  preload: true,
});

/**
 * Metadata for SEO and PWA support
 */
export const metadata: Metadata = {
  title: 'Nautilus/Ray | Crypto Trading Bot',
  description: 'High-frequency crypto trading bot with real-time telemetry and AMD DirectML/ROCm acceleration',
  keywords: [
    'crypto',
    'trading',
    'bot',
    'high-frequency',
    'telemetry',
    'AMD',
    'DirectML',
    'ROCm',
  ],
  authors: [{ name: 'Nautilus/Ray Team' }],
  creator: 'Nautilus/Ray',
  publisher: 'Nautilus/Ray',
  robots: {
    index: false,
    follow: false,
    googleBot: {
      index: false,
      follow: false,
      noimageindex: true,
    },
  },
  manifest: '/manifest.json',
  themeColor: '#0a0a0f',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Nautilus/Ray',
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
    userScalable: false, // Prevent zoom for app-like experience
  },
};

/**
 * Root Layout Component
 * 
 * Scaffolds the Next.js App Router layout with:
 * - Global Zustand providers
 * - Web Worker initializers
 * - Custom dark-mode font configurations
 * - Cyberpunk aesthetic base styles
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} ${rajdhani.variable}`} suppressHydrationWarning>
      <head>
        {/* Preconnect to WebSocket backend for faster initial connection */}
        <link rel="preconnect" href="ws://localhost:8080" />
        
        {/* Meta tags for performance monitoring */}
        <meta name="performance-budget" content="200mb" />
        <meta name="memory-limit" content="8gb" />
      </head>
      <body 
        className={`
          min-h-screen 
          bg-obsidian-50 
          text-gray-100 
          antialiased 
          overflow-hidden
          ${inter.className}
        `}
        style={{
          fontFamily: 'var(--font-inter), system-ui, sans-serif',
        }}
      >
        {/* Global scan line overlay for cyberpunk aesthetic */}
        <div className="scan-line-overlay pointer-events-none fixed inset-0 z-[9999]" />
        
        {/* Main Application Container */}
        <div 
          id="app-container" 
          className="relative flex h-screen w-screen overflow-hidden"
          role="application"
          aria-label="Nautilus/Ray Trading Dashboard"
        >
          {/* 
            Children components will include:
            - Sidebar navigation
            - TopBar with controls
            - Main dashboard content
            - Modal overlays
          */}
          {children}
        </div>
        
        {/* 
          Performance Monitoring Script (optional)
          Injects timing metrics for latency tracking
        */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Mark initial page load
              if (window.performance && window.performance.mark) {
                window.performance.mark('nautilus-app-loaded');
              }
              
              // Track memory usage (Chrome only)
              if (window.performance && window.performance.memory) {
                console.log('[Performance] Initial Memory:', window.performance.memory.usedJSHeapSize / 1048576, 'MB');
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
