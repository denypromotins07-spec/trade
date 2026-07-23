import type { Metadata } from 'next';
import { Inter, JetBrains_Mono, Orbitron } from 'next/font/google';
import '@/styles/globals.css';

// ============================================================================
// ROOT LAYOUT - NEXT.JS APP ROUTER
// Injects global providers, Web Worker initializers, and font configurations
// Optimized for dark-mode cyberpunk aesthetic
// ============================================================================

// ==========================================================================
// FONT CONFIGURATION
// Preloaded fonts with optimized display settings
// ==========================================================================

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
  weight: ['300', '400', '500', '600', '700'],
});

const orbitron = Orbitron({
  subsets: ['latin'],
  variable: '--font-display',
  display: 'swap',
  preload: true,
  weight: ['400', '500', '600', '700', '800', '900'],
});

// ==========================================================================
// METADATA CONFIGURATION
// SEO and PWA settings for the trading bot UI
// ==========================================================================

export const metadata: Metadata = {
  title: 'Nautilus/Ray | Quantum Trading Terminal',
  description: 'Ultra-low latency crypto trading bot with real-time telemetry and AMD DirectML/ROCm acceleration',
  keywords: [
    'crypto',
    'trading',
    'bot',
    'quantum',
    'low-latency',
    'AMD',
    'DirectML',
    'ROCm',
    'Nautilus',
    'Ray',
  ],
  authors: [{ name: 'Nautilus/Ray Team' }],
  creator: 'Nautilus/Ray',
  publisher: 'Nautilus/Ray',
  robots: {
    index: false, // Private trading terminal
    follow: false,
    googleBot: {
      index: false,
      follow: false,
      noimageindex: true,
    },
  },
  
  // Open Graph / Social sharing
  openGraph: {
    title: 'Nautilus/Ray | Quantum Trading Terminal',
    description: 'Professional-grade crypto trading with sub-millisecond execution',
    type: 'website',
    locale: 'en_US',
    siteName: 'Nautilus/Ray Terminal',
  },
  
  // Theme and appearance
  themeColor: [
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0f' },
    { media: '(prefers-color-scheme: light)', color: '#0a0a0f' }, // Always dark mode
  ],
  colorScheme: 'dark',
  
  // Viewport settings for optimal rendering
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1, // Prevent zoom on trading interface
    userScalable: false,
    minimumScale: 1,
  },
  
  // Icons (would be added to public folder)
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
};

// ==========================================================================
// ROOT LAYOUT COMPONENT
// Wraps entire application with providers and base structure
// ==========================================================================

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html 
      lang="en" 
      className={`${inter.variable} ${jetbrainsMono.variable} ${orbitron.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/* Preconnect to WebSocket backend for faster connection */}
        <link rel="preconnect" href="ws://localhost:8080" crossOrigin="anonymous" />
        
        {/* DNS prefetch for API endpoints */}
        <link rel="dns-prefetch" href="//localhost" />
        
        {/* Meta tags for performance */}
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
        <meta name="theme-color" content="#0a0a0f" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        
        {/* Performance optimizations */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      
      <body 
        className={`
          min-h-screen 
          bg-obsidian-50 
          text-gray-100 
          antialiased 
          overflow-x-hidden
          font-sans
        `}
      >
        {/* 
          NOTE: Client-side providers would be wrapped here
          In a real implementation, you'd have:
          - Zustand Provider (if using context)
          - WebSocket Provider
          - Theme Provider
          - Error Boundary
        */}
        
        {/* Main application content */}
        <main 
          id="app-root" 
          className="relative min-h-screen"
          role="application"
          aria-label="Nautilus/Ray Trading Terminal"
        >
          {children}
        </main>
        
        {/* 
          Global event listeners for performance monitoring
          Would be implemented in a client component
        */}
        
        {/* 
          Web Worker initialization script
          Loaded lazily to prevent blocking main thread
        */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Lazy load Web Worker for telemetry parsing
              window.addEventListener('load', function() {
                if ('requestIdleCallback' in window) {
                  requestIdleCallback(function() {
                    // Worker will be initialized by useTelemetry hook when needed
                    console.log('[RootLayout] Ready for Web Worker initialization');
                  });
                }
              });
            `,
          }}
        />
      </body>
    </html>
  );
}
