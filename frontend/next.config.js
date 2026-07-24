/**
 * Next.js Hyper-Optimized Configuration
 * 
 * Enables strict SWC minification, React Server Components, and aggressive chunk splitting
 * to guarantee instant local host boot times while stripping dev-only profiling hooks.
 * 
 * Cyberpunk aesthetic: "Neural network optimization matrix" configuration.
 */

const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React Server Components by default
  experimental: {
    serverComponentsExternalPackages: ['sharp', 'caniuse-lite'],
    optimizePackageImports: [
      '@tanstack/react-query',
      'zustand',
      'three',
      '@react-three/fiber',
      '@react-three/drei',
    ],
    // Enable webpack build optimization
    webpackBuildWorker: true,
    // Enable parallel server builds
    parallelServerBuildTraces: true,
    // Enable parallel server compiles
    parallelServerCompiles: true,
  },

  // Production optimizations
  productionBrowserSourceMaps: false, // Disable source maps in production for security
  
  // Compiler options
  compiler: {
    // Remove console.log in production
    removeConsole: process.env.NODE_ENV === 'production' ? {
      exclude: ['error', 'warn'],
    } : false,
    
    // Enable styled-components optimization if used
    // styledComponents: true,
    
    // Enable react-remove-properties for production
    reactRemoveProperties: process.env.NODE_ENV === 'production' ? {
      properties: ['^data-test$', '^data-testid$'],
    } : false,
  },

  // Webpack configuration
  webpack: (config, { isServer, dev, webpack }) => {
    // Enable tree shaking in production
    if (!dev) {
      config.optimization.usedExports = true;
      config.optimization.concatenateModules = true;
    }

    // Aggressive chunk splitting
    if (!isServer) {
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          // Vendor chunks
          vendors: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: 20,
            chunks: 'all',
          },
          // Framework chunks (React, Next.js)
          framework: {
            test: /[\\/]node_modules[\\/](react|react-dom|next)[\\/]/,
            name: 'framework',
            priority: 30,
            chunks: 'all',
          },
          // WebGL/Three.js chunks
          webgl: {
            test: /[\\/]node_modules[\\/](three|@react-three)[\\/]/,
            name: 'webgl',
            priority: 25,
            chunks: 'all',
          },
          // Trading library chunks
          trading: {
            test: /[\\/]src[\\/](lib|components)[\\/](chart|orderbook|trading)/,
            name: 'trading',
            priority: 15,
            chunks: 'all',
          },
          // Common chunks
          common: {
            minChunks: 2,
            name: 'common',
            priority: 10,
            reuseExistingChunk: true,
            chunks: 'all',
          },
        },
      };
    }

    // Minification settings for production
    if (!dev && !isServer) {
      const TerserPlugin = require('terser-webpack-plugin');
      
      config.optimization.minimizer = [
        new TerserPlugin({
          parallel: true,
          terserOptions: {
            compress: {
              drop_console: true,
              drop_debugger: true,
              pure_funcs: ['console.log', 'console.info', 'console.debug'],
              passes: 2,
            },
            format: {
              comments: false,
            },
            mangle: {
              safari10: false,
            },
          },
          extractComments: false,
        }),
      ];
    }

    // Memory limit enforcement for browser bundles
    if (!isServer) {
      config.performance = {
        hints: 'error',
        maxEntrypointSize: 512000, // 500KB
        maxAssetSize: 512000, // 500KB
      };
    }

    // Add custom aliases for optimized imports
    config.resolve.alias = {
      ...config.resolve.alias,
      '@components': '/src/components',
      '@hooks': '/src/hooks',
      '@lib': '/src/lib',
      '@store': '/src/store',
      '@profiling': '/src/profiling',
      // Use preact in production for smaller bundle (optional)
      // 'react': process.env.NODE_ENV === 'production' ? 'preact/compat' : 'react',
      // 'react-dom': process.env.NODE_ENV === 'production' ? 'preact/compat' : 'react-dom',
    };

    return config;
  },

  // Image optimization
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    minimumCacheTTL: 60,
    dangerouslyAllowSVG: false,
  },

  // Headers for security and caching
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
      {
        // Cache static assets aggressively
        source: '/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ];
  },

  // Redirects
  async redirects() {
    return [
      {
        source: '/dashboard',
        destination: '/',
        permanent: true,
      },
    ];
  },

  // Environment variables
  env: {
    NEXT_PUBLIC_APP_VERSION: process.env.npm_package_version || '0.0.0',
    NEXT_PUBLIC_BUILD_TIME: new Date().toISOString(),
  },

  // Disable powered-by header
  poweredByHeader: false,

  // Compress responses
  compress: true,

  // Generate ETags
  generateEtags: true,

  // TypeScript configuration
  typescript: {
    ignoreBuildErrors: false,
    tsconfigPath: './tsconfig.json',
  },

  // ESLint configuration
  eslint: {
    ignoreDuringBuilds: false,
    dirs: ['src'],
  },

  // Logging
  logging: {
    fetches: {
      fullUrl: false,
    },
  },

  // Output configuration
  output: 'standalone',

  // Transpile packages
  transpilePackages: ['@nautilus', '@ray'],
};

module.exports = withBundleAnalyzer(nextConfig);
