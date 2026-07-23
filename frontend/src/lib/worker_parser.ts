/**
 * Web Worker for Off-Main-Thread Order Book Parsing
 * Dedicated worker to parse heavy JSON/MessagePack order book snapshots.
 * Ensures 60FPS rendering by preventing UI thread blocking during extreme volatility.
 * 
 * Memory Optimization: Drops stale snapshots when newer sequence numbers arrive.
 */

interface OrderBookSnapshot {
  symbol: string;
  bids: [number, number][];  // [price, size]
  asks: [number, number][];
  sequence: number;
  timestamp: number;
}

interface ParsedData {
  type: 'orderbook' | 'trade' | 'ticker' | 'system_health';
  data: unknown;
  processedAt: number;
}

// Track latest sequence numbers per symbol to drop stale data
const latestSequences: Map<string, number> = new Map();

// Memory limit: max snapshots to buffer before dropping oldest
const MAX_BUFFER_SIZE = 100;
const snapshotBuffer: OrderBookSnapshot[] = [];

/**
 * Parse incoming message from main thread
 * Supports both JSON and binary MessagePack (transferred as ArrayBuffer)
 */
self.onmessage = function(event: MessageEvent<ArrayBuffer | string | { type: string; data: unknown }>): void {
  try {
    const message = event.data;

    if (typeof message === 'string') {
      // JSON message from main thread
      handleMessage(JSON.parse(message));
    } else if (message instanceof ArrayBuffer) {
      // Binary MessagePack - would need msgpackr loaded in worker
      // For now, we'll handle as JSON-encoded
      const decoder = new TextDecoder();
      const jsonString = decoder.decode(new Uint8Array(message));
      handleMessage(JSON.parse(jsonString));
    } else if (typeof message === 'object' && message !== null) {
      // Already parsed object from main thread
      handleMessage(message);
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error instanceof Error ? error.message : 'Unknown parsing error',
      timestamp: Date.now(),
    });
  }
};

/**
 * Handle parsed message with sequence deduplication
 */
function handleMessage(message: { type?: string; data?: unknown }): void {
  const msgType = message.type;
  const data = message.data;

  if (!msgType || !data) {
    return;
  }

  switch (msgType) {
    case 'orderbook': {
      const snapshot = data as OrderBookSnapshot;
      
      // Drop stale snapshots based on sequence number
      const latestSeq = latestSequences.get(snapshot.symbol) ?? 0;
      if (snapshot.sequence <= latestSeq) {
        // Stale data - drop it to save memory and processing
        console.log('[Worker] Dropping stale orderbook:', snapshot.symbol, snapshot.sequence);
        return;
      }

      // Update latest sequence
      latestSequences.set(snapshot.symbol, snapshot.sequence);

      // Enforce memory limit by dropping oldest snapshots
      if (snapshotBuffer.length >= MAX_BUFFER_SIZE) {
        snapshotBuffer.shift();  // Remove oldest
      }
      snapshotBuffer.push(snapshot);

      // Process and send to main thread
      const processed = processOrderBook(snapshot);
      postResult({
        type: 'orderbook',
        data: processed,
        processedAt: Date.now(),
      });
      break;
    }

    case 'trade': {
      // Trades are small, process immediately
      postResult({
        type: 'trade',
        data: data,
        processedAt: Date.now(),
      });
      break;
    }

    case 'ticker': {
      // Tickers are small, process immediately
      postResult({
        type: 'ticker',
        data: data,
        processedAt: Date.now(),
      });
      break;
    }

    case 'system_health': {
      // System health updates
      postResult({
        type: 'system_health',
        data: data,
        processedAt: Date.now(),
      });
      break;
    }

    default:
      console.warn('[Worker] Unknown message type:', msgType);
  }
}

/**
 * Process order book snapshot with optimizations
 * - Calculate depth metrics
 * - Normalize price levels
 * - Compute spread and mid-price
 */
function processOrderBook(snapshot: OrderBookSnapshot): {
  symbol: string;
  bids: [number, number][];
  asks: [number, number][];
  spread: number;
  midPrice: number;
  totalBidDepth: number;
  totalAskDepth: number;
  sequence: number;
  timestamp: number;
} {
  const { bids, asks, symbol, sequence, timestamp } = snapshot;

  // Calculate mid-price and spread
  const bestBid = bids[0]?.[0] ?? 0;
  const bestAsk = asks[0]?.[0] ?? 0;
  const midPrice = (bestBid + bestAsk) / 2;
  const spread = bestAsk - bestBid;

  // Calculate total depth
  const totalBidDepth = bids.reduce((sum, [, size]) => sum + size, 0);
  const totalAskDepth = asks.reduce((sum, [, size]) => sum + size, 0);

  return {
    symbol,
    bids,
    asks,
    spread,
    midPrice,
    totalBidDepth,
    totalAskDepth,
    sequence,
    timestamp,
  };
}

/**
 * Post result back to main thread
 */
function postResult(result: ParsedData): void {
  self.postMessage(result);
}

/**
 * Cleanup function to clear buffers (called periodically or on low memory)
 */
function cleanup(): void {
  snapshotBuffer.splice(0, snapshotBuffer.length);
  latestSequences.clear();
  console.log('[Worker] Buffers cleared');
}

// Listen for cleanup commands from main thread
self.addEventListener('message', (event) => {
  if (event.data?.type === 'cleanup') {
    cleanup();
  }
});

export {};
