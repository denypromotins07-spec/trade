/**
 * SocialPulse.tsx - Sentiment Analysis: X (Twitter) & Reddit Sentiment Heatmaps
 * 
 * Displays social media sentiment heatmaps and trendlines with WebSocket updates
 * batched via requestAnimationFrame to prevent UI thread jank.
 * 
 * Features:
 * - Real-time WebSocket data batching with RAF synchronization
 * - Heatmap visualization for social sentiment density
 * - Trendline charts using lightweight SVG paths
 * - Platform-specific sentiment breakdown (X vs Reddit)
 * - GPU-accelerated CSS transforms for smooth animations
 */

'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

interface SocialPost {
  id: string;
  platform: 'twitter' | 'reddit';
  content: string;
  author: string;
  timestamp: number;
  sentimentScore: number; // -1 to 1
  engagement: number; // likes + comments + shares
  tickers: string[];
}

interface SentimentBucket {
  timestamp: number;
  positive: number;
  negative: number;
  neutral: number;
}

interface SocialPulseProps {
  data?: SocialPost[];
  width?: number;
  height?: number;
  updateInterval?: number;
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const BUCKET_SIZE_MS = 60000; // 1 minute buckets
const HISTORY_BUCKETS = 60; // Last 60 minutes
const MAX_POSTS = 1000;

const PLATFORM_COLORS = {
  twitter: '#1DA1F2', // Twitter blue
  reddit: '#FF4500',  // Reddit orange
};

const SENTIMENT_COLORS = {
  positive: '#00ff88',
  neutral: '#666666',
  negative: '#ff0088',
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generates mock social media posts
 */
const generateMockPosts = (count: number): SocialPost[] => {
  const platforms: ('twitter' | 'reddit')[] = ['twitter', 'reddit'];
  const tickers = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'XRP'];
  
  const twitterContents = [
    'Just bought more #BTC! To the moon! 🚀',
    'This market dip is a buying opportunity IMO',
    'Bearish on altcoins short term, bullish long term',
    'New ATH incoming, mark my words',
    'Technical analysis suggests breakout pattern forming',
  ];
  
  const redditContents = [
    'DD: Why I think we are at the bottom',
    'Daily discussion thread - share your trades',
    'Warning: FOMO is dangerous in this market',
    'Portfolio update: Up 15% this week',
    'Analysis of on-chain metrics looks bullish',
  ];
  
  return Array.from({ length: count }, (_, i) => {
    const platform = platforms[Math.floor(Math.random() * platforms.length)];
    const sentimentScore = (Math.random() - 0.5) * 2;
    const tickerCount = Math.floor(Math.random() * 2) + 1;
    
    return {
      id: `social-${Date.now()}-${i}`,
      platform,
      content: platform === 'twitter' 
        ? twitterContents[Math.floor(Math.random() * twitterContents.length)]
        : redditContents[Math.floor(Math.random() * redditContents.length)],
      author: `user_${Math.random().toString(36).slice(2, 8)}`,
      timestamp: Date.now() - Math.random() * 3600000, // Last hour
      sentimentScore: parseFloat(sentimentScore.toFixed(3)),
      engagement: Math.floor(Math.random() * 10000),
      tickers: Array.from({ length: tickerCount }, () => tickers[Math.floor(Math.random() * tickers.length)]),
    };
  });
};

/**
 * Buckets posts into time windows for heatmap
 */
const bucketPosts = (posts: SocialPost[], bucketSize: number, bucketCount: number): SentimentBucket[] => {
  const now = Date.now();
  const buckets: SentimentBucket[] = [];
  
  for (let i = 0; i < bucketCount; i++) {
    const bucketStart = now - (bucketCount - i) * bucketSize;
    const bucketEnd = bucketStart + bucketSize;
    
    const bucketPosts = posts.filter(
      (p) => p.timestamp >= bucketStart && p.timestamp < bucketEnd
    );
    
    buckets.push({
      timestamp: bucketStart,
      positive: bucketPosts.filter((p) => p.sentimentScore > 0.2).length,
      negative: bucketPosts.filter((p) => p.sentimentScore < -0.2).length,
      neutral: bucketPosts.filter((p) => Math.abs(p.sentimentScore) <= 0.2).length,
    });
  }
  
  return buckets;
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Heatmap row component for a single time bucket
 */
interface HeatmapCellProps {
  bucket: SentimentBucket;
  maxIntensity: number;
}

const HeatmapCell: React.FC<HeatmapCellProps> = React.memo(({ bucket, maxIntensity }) => {
  const total = bucket.positive + bucket.negative + bucket.neutral;
  const intensity = total / maxIntensity;
  
  // Calculate dominant sentiment color
  let bgColor = 'transparent';
  if (bucket.positive > bucket.negative) {
    bgColor = `rgba(0, 255, 136, ${intensity * 0.8})`;
  } else if (bucket.negative > bucket.positive) {
    bgColor = `rgba(255, 0, 136, ${intensity * 0.8})`;
  } else {
    bgColor = `rgba(102, 102, 102, ${intensity * 0.8})`;
  }
  
  return (
    <div
      className="flex-1 min-w-[4px] h-8 rounded-sm mx-[1px] relative group cursor-pointer"
      style={{
        backgroundColor: bgColor,
        willChange: 'transform',
        transform: 'translateZ(0)',
      }}
      aria-label={`Time bucket: ${bucket.positive} positive, ${bucket.negative} negative, ${bucket.neutral} neutral`}
    >
      {/* Tooltip on hover */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black/90 text-xs font-mono text-white rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-20">
        +{bucket.positive} / -{bucket.negative} / ~{bucket.neutral}
      </div>
    </div>
  );
});

HeatmapCell.displayName = 'HeatmapCell';

/**
 * Simple SVG trendline chart
 */
interface TrendlineProps {
  data: SentimentBucket[];
  metric: 'positive' | 'negative' | 'neutral';
  color: string;
  height?: number;
}

const Trendline: React.FC<TrendlineProps> = React.memo(({ data, metric, color, height = 60 }) => {
  const values = data.map((b) => b[metric]);
  const maxValue = Math.max(...values, 1);
  const width = 100; // Percentage based
  
  // Generate SVG path
  const points = values.map((value, index) => {
    const x = (index / (values.length - 1)) * width;
    const y = height - (value / maxValue) * height;
    return `${x},${y}`;
  }).join(' ');
  
  return (
    <svg 
      viewBox={`0 0 ${width} ${height}`} 
      className="w-full h-full"
      preserveAspectRatio="none"
    >
      {/* Gradient fill */}
      <defs>
        <linearGradient id={`gradient-${metric}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.4" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      
      {/* Area fill */}
      <path
        d={`M 0,${height} L ${points} L ${width},${height} Z`}
        fill={`url(#gradient-${metric})`}
      />
      
      {/* Line */}
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
});

Trendline.displayName = 'Trendline';

// ============================================================================
// Main Component
// ============================================================================

export const SocialPulse: React.FC<SocialPulseProps> = ({
  data,
  width = 800,
  height = 400,
  updateInterval = 1000,
}) => {
  const [posts, setPosts] = useState<SocialPost[]>(data || generateMockPosts(100));
  const rafRef = useRef<number | null>(null);
  const wsBufferRef = useRef<SocialPost[]>([]);
  
  // Simulate WebSocket updates (in production, use actual WS connection)
  useEffect(() => {
    const interval = setInterval(() => {
      // Add new post to buffer
      const newPost = generateMockPosts(1)[0];
      wsBufferRef.current.push(newPost);
    }, updateInterval);
    
    return () => clearInterval(interval);
  }, [updateInterval]);
  
  /**
   * Process WebSocket buffer with RAF batching
   * Prevents UI thread jank by syncing updates to render frame
   */
  const processUpdates = useCallback(() => {
    if (wsBufferRef.current.length > 0) {
      setPosts((prev) => {
        const updated = [...prev, ...wsBufferRef.current.current];
        // Limit to max posts for memory safety
        return updated.slice(-MAX_POSTS);
      });
      wsBufferRef.current = [];
    }
    rafRef.current = requestAnimationFrame(processUpdates);
  }, []);
  
  // Start RAF loop for processing updates
  useEffect(() => {
    rafRef.current = requestAnimationFrame(processUpdates);
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [processUpdates]);
  
  // Update from external data prop
  useEffect(() => {
    if (data) {
      setPosts(data.slice(0, MAX_POSTS));
    }
  }, [data]);
  
  // Compute bucketed data for visualization
  const buckets = useMemo(() => {
    return bucketPosts(posts, BUCKET_SIZE_MS, HISTORY_BUCKETS);
  }, [posts]);
  
  const maxIntensity = useMemo(() => {
    return Math.max(...buckets.map((b) => b.positive + b.negative + b.neutral), 1);
  }, [buckets]);
  
  // Platform breakdown
  const platformStats = useMemo(() => {
    const twitter = posts.filter((p) => p.platform === 'twitter');
    const reddit = posts.filter((p) => p.platform === 'reddit');
    
    return {
      twitter: {
        count: twitter.length,
        avgSentiment: twitter.reduce((acc, p) => acc + p.sentimentScore, 0) / twitter.length || 0,
      },
      reddit: {
        count: reddit.length,
        avgSentiment: reddit.reduce((acc, p) => acc + p.sentimentScore, 0) / reddit.length || 0,
      },
    };
  }, [posts]);

  return (
    <div className="w-full rounded-xl overflow-hidden bg-[#0a0a12]/90 backdrop-blur-sm border border-cyan-900/30">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-b from-[#0a0a12] to-transparent border-b border-white/5">
        <h3 className="text-cyan-400 font-mono text-sm tracking-wider uppercase">
          📡 Social Pulse <span className="text-xs opacity-70">| X & Reddit</span>
        </h3>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-gray-400">LIVE WS</span>
          </div>
          <span className="text-xs text-gray-500 font-mono">
            {posts.length.toLocaleString()} posts
          </span>
        </div>
      </div>
      
      {/* Content Grid */}
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Heatmap */}
        <div className="space-y-2">
          <div className="text-xs font-mono text-gray-400 uppercase tracking-wider">
            Sentiment Heatmap (60min)
          </div>
          <div className="flex items-end h-12 gap-0">
            {buckets.map((bucket, index) => (
              <HeatmapCell 
                key={bucket.timestamp} 
                bucket={bucket} 
                maxIntensity={maxIntensity}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs font-mono text-gray-500">
            <span>60m ago</span>
            <span>Now</span>
          </div>
        </div>
        
        {/* Platform Stats */}
        <div className="space-y-3">
          {/* Twitter */}
          <div className="p-3 rounded-lg bg-[#1DA1F2]/10 border border-[#1DA1F2]/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-[#1DA1F2]">🐦 Twitter</span>
              <span className="text-xs font-mono text-gray-400">{platformStats.twitter.count} posts</span>
            </div>
            <div className="h-10">
              <Trendline 
                data={buckets} 
                metric="positive" 
                color={PLATFORM_COLORS.twitter}
                height={40}
              />
            </div>
            <div className="flex justify-between text-xs mt-1">
              <span className="font-mono text-gray-400">
                Avg Sentiment: {platformStats.twitter.avgSentiment > 0 ? '+' : ''}{platformStats.twitter.avgSentiment.toFixed(2)}
              </span>
            </div>
          </div>
          
          {/* Reddit */}
          <div className="p-3 rounded-lg bg-[#FF4500]/10 border border-[#FF4500]/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-[#FF4500]">🤖 Reddit</span>
              <span className="text-xs font-mono text-gray-400">{platformStats.reddit.count} posts</span>
            </div>
            <div className="h-10">
              <Trendline 
                data={buckets} 
                metric="positive" 
                color={PLATFORM_COLORS.reddit}
                height={40}
              />
            </div>
            <div className="flex justify-between text-xs mt-1">
              <span className="font-mono text-gray-400">
                Avg Sentiment: {platformStats.reddit.avgSentiment > 0 ? '+' : ''}{platformStats.reddit.avgSentiment.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer Legend */}
      <div className="px-4 py-2 bg-gradient-to-t from-[#0a0a12] to-transparent border-t border-white/5">
        <div className="flex items-center justify-between text-xs font-mono text-gray-500">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: SENTIMENT_COLORS.positive }} />
              Bullish
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: SENTIMENT_COLORS.neutral }} />
              Neutral
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: SENTIMENT_COLORS.negative }} />
              Bearish
            </span>
          </div>
          <span>RAF Batched Updates</span>
        </div>
      </div>
    </div>
  );
};

export default SocialPulse;
