// ============================================================================
// WEB WORKER FOR OFF-MAIN-THREAD TELEMETRY PARSING
// Dedicated worker for parsing heavy JSON/MessagePack order book snapshots
// Ensures 60FPS rendering by preventing main thread blocking
// ============================================================================

import { unpack } from 'msgpackr';

// Message types for worker communication
export interface WorkerRequest {
  type: 'PARSE_MESSAGEPACK' | 'PARSE_JSON' | 'AGGREGATE_ORDERBOOK' | 'CALCULATE_DEPTH';
  id: string;
  payload: unknown;
}

export interface WorkerResponse {
  type: 'PARSE_RESULT' | 'AGGREGATE_RESULT' | 'DEPTH_RESULT' | 'ERROR';
  id: string;
  data?: unknown;
  error?: string;
  memoryUsage?: number;
}

// Memory limit enforcement (8GB global RAM constraint)
const MAX_MEMORY_THRESHOLD_MB = 512; // Worker-specific limit
const STALE_SNAPSHOT_TTL_MS = 100;   // Drop snapshots older than 100ms

// Order book snapshot cache with TTL-based eviction
interface CachedSnapshot {
  timestamp: number;
  data: Record<string, unknown>;
  symbol: string;
}

const snapshotCache = new Map<string, CachedSnapshot>();

// ============================================================================
// MESSAGEPACK PARSING (BINARY DATA)
// ============================================================================

function parseMessagePack(buffer: ArrayBuffer): unknown {
  try {
    const uint8Array = new Uint8Array(buffer);
    return unpack(uint8Array);
  } catch (error) {
    throw new Error(`MessagePack decode failed: ${error}`);
  }
}

// ============================================================================
// JSON PARSING WITH ERROR HANDLING
// ============================================================================

function parseJsonString(jsonString: string): unknown {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    throw new Error(`JSON parse failed: ${error}`);
  }
}

// ============================================================================
// ORDER BOOK AGGREGATION
// Consolidates multiple price levels to reduce rendering overhead
// ============================================================================

interface PriceLevel {
  price: number;
  size: number;
  count?: number;
}

interface AggregatedOrderBook {
  bids: PriceLevel[];
  asks: PriceLevel[];
  timestamp: number;
  symbol: string;
}

function aggregateOrderBook(
  rawBook: { bids: [number, number][]; asks: [number, number][] },
  aggregationFactor: number = 10
): AggregatedOrderBook {
  const aggregateLevels = (levels: [number, number][]): PriceLevel[] => {
    if (levels.length === 0) return [];
    
    const aggregated: Map<number, { price: number; size: number; count: number }> = new Map();
    
    for (const [price, size] of levels) {
      const bucket = Math.floor(price / aggregationFactor) * aggregationFactor;
      const existing = aggregated.get(bucket);
      
      if (existing) {
        existing.size += size;
        existing.count += 1;
      } else {
        aggregated.set(bucket, { price: bucket, size, count: 1 });
      }
    }
    
    return Array.from(aggregated.values())
      .sort((a, b) => a.price - b.price);
  };

  return {
    bids: aggregateLevels(rawBook.bids).reverse(), // Highest first
    asks: aggregateLevels(rawBook.asks),           // Lowest first
    timestamp: Date.now(),
    symbol: 'UNKNOWN',
  };
}

// ============================================================================
// ORDER BOOK DEPTH CALCULATION
// Computes cumulative depth for visualization
// ============================================================================

interface DepthData {
  bids: { price: number; cumulative: number }[];
  asks: { price: number; cumulative: number }[];
}

function calculateDepth(orderBook: AggregatedOrderBook): DepthData {
  let bidCumulative = 0;
  let askCumulative = 0;

  const bids = orderBook.bids.map(level => {
    bidCumulative += level.size;
    return { price: level.price, cumulative: bidCumulative };
  });

  const asks = orderBook.asks.map(level => {
    askCumulative += level.size;
    return { price: level.price, cumulative: askCumulative };
  });

  return { bids, asks };
}

// ============================================================================
// MEMORY MANAGEMENT & STALE SNAPSHOT EVICTION
// Enforces memory limits by dropping outdated order book data
// ============================================================================

function enforceMemoryLimit(): void {
  const now = Date.now();
  let evictedCount = 0;

  // Remove stale snapshots
  for (const [key, snapshot] of snapshotCache.entries()) {
    if (now - snapshot.timestamp > STALE_SNAPSHOT_TTL_MS) {
      snapshotCache.delete(key);
      evictedCount++;
    }
  }

  // Estimate memory usage (rough approximation)
  const estimatedSizeMB = (snapshotCache.size * 1024) / (1024 * 1024); // Assume ~1KB per snapshot
  
  // Aggressive eviction if approaching threshold
  if (estimatedSizeMB > MAX_MEMORY_THRESHOLD_MB) {
    console.warn('[Worker] Memory threshold exceeded, clearing cache');
    snapshotCache.clear();
  }

  if (evictedCount > 0) {
    console.debug(`[Worker] Evicted ${evictedCount} stale snapshots`);
  }
}

function cacheSnapshot(symbol: string, data: Record<string, unknown>): void {
  const key = `${symbol}_${Date.now()}`;
  
  // Evict old entries for this symbol first
  for (const [k, v] of snapshotCache.entries()) {
    if (v.symbol === symbol) {
      snapshotCache.delete(k);
    }
  }
  
  snapshotCache.set(key, {
    timestamp: Date.now(),
    data,
    symbol,
  });
  
  // Periodic memory enforcement
  if (snapshotCache.size % 100 === 0) {
    enforceMemoryLimit();
  }
}

// ============================================================================
// MAIN MESSAGE HANDLER
// ============================================================================

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const { type, id, payload } = event.data;
  
  try {
    let result: unknown;
    let response: WorkerResponse;

    switch (type) {
      case 'PARSE_MESSAGEPACK': {
        const buffer = payload as ArrayBuffer;
        result = parseMessagePack(buffer);
        
        // Cache if it's an order book snapshot
        if (result && typeof result === 'object' && 'bids' in result && 'asks' in result) {
          const symbol = (result as { symbol?: string }).symbol || 'UNKNOWN';
          cacheSnapshot(symbol, result as Record<string, unknown>);
        }
        
        response = {
          type: 'PARSE_RESULT',
          id,
          data: result,
          memoryUsage: snapshotCache.size,
        };
        break;
      }

      case 'PARSE_JSON': {
        const jsonString = payload as string;
        result = parseJsonString(jsonString);
        
        if (result && typeof result === 'object' && 'bids' in result && 'asks' in result) {
          const symbol = (result as { symbol?: string }).symbol || 'UNKNOWN';
          cacheSnapshot(symbol, result as Record<string, unknown>);
        }
        
        response = {
          type: 'PARSE_RESULT',
          id,
          data: result,
          memoryUsage: snapshotCache.size,
        };
        break;
      }

      case 'AGGREGATE_ORDERBOOK': {
        const rawBook = payload as { bids: [number, number][]; asks: [number, number][] };
        const aggregationFactor = (payload as { aggregationFactor?: number }).aggregationFactor || 10;
        result = aggregateOrderBook(rawBook, aggregationFactor);
        
        response = {
          type: 'AGGREGATE_RESULT',
          id,
          data: result,
        };
        break;
      }

      case 'CALCULATE_DEPTH': {
        const orderBook = payload as AggregatedOrderBook;
        result = calculateDepth(orderBook);
        
        response = {
          type: 'DEPTH_RESULT',
          id,
          data: result,
        };
        break;
      }

      default:
        throw new Error(`Unknown request type: ${type}`);
    }

    self.postMessage(response);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    self.postMessage({
      type: 'ERROR',
      id,
      error: errorMessage,
    } as WorkerResponse);
  }
};

// ============================================================================
// PERIODIC CLEANUP
// Run memory enforcement every 5 seconds
// ============================================================================

setInterval(() => {
  enforceMemoryLimit();
}, 5000);

// Export for type checking (worker doesn't use these exports directly)
export {};
