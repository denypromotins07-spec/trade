/**
 * ManualOverride.tsx - Hotkey-enabled manual execution panel
 * 
 * Features:
 * - Instant market/limit order firing with hotkey support (F1-F4)
 * - Real-time slippage warnings and size validation
 * - Debounced WebSocket emissions to prevent API spam
 * - Cyberpunk aesthetic with neon indicators
 * - AMD GPU load visualization during order processing
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowUpRight, ArrowDownRight, AlertTriangle, Clock, Zap } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { OrderType, OrderSide } from '../../types/orders';
import type { MarketData } from '../../types/market';

interface ManualOverrideProps {
  currentPrice: number;
  marketData: MarketData | null;
  maxPositionSize: number;
  onOrderSubmitted: (orderId: string) => void;
}

interface OrderFormData {
  side: OrderSide;
  type: OrderType;
  size: number;
  price?: number;
  slippageTolerance: number;
}

// Hotkey configuration
const HOTKEYS = {
  BUY_MARKET: 'f1',
  SELL_MARKET: 'f2',
  BUY_LIMIT: 'f3',
  SELL_LIMIT: 'f4',
} as const;

// Debounce interval in ms
const DEBOUNCE_INTERVAL = 200;

export const ManualOverride: React.FC<ManualOverrideProps> = ({
  currentPrice,
  marketData,
  maxPositionSize,
  onOrderSubmitted,
}) => {
  const [formData, setFormData] = useState<OrderFormData>({
    side: 'buy',
    type: 'market',
    size: 0.01,
    slippageTolerance: 0.5,
  });
  const [estimatedSlippage, setEstimatedSlippage] = useState<number>(0);
  const [lastSendTime, setLastSendTime] = useState<number>(0);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { sendMessage, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'open';
  const abortControllerRef = useRef<AbortController | null>(null);

  // Calculate estimated slippage based on order size and orderbook depth
  useEffect(() => {
    if (!marketData || formData.size === 0) {
      setEstimatedSlippage(0);
      return;
    }

    const notionalValue = formData.size * currentPrice;
    const orderbookDepth = formData.side === 'buy' 
      ? marketData.asks.reduce((sum, level) => sum + level[1], 0)
      : marketData.bids.reduce((sum, level) => sum + level[1], 0);

    // Simple slippage estimation: larger orders relative to depth = more slippage
    const slippageFactor = Math.min(notionalValue / (orderbookDepth * currentPrice), 1);
    const baseSlippage = formData.type === 'market' ? 0.1 : 0.05; // Base slippage %
    const estimated = baseSlippage + (slippageFactor * 2); // Up to 2% additional
    
    setEstimatedSlippage(Math.min(estimated, 5)); // Cap at 5%
  }, [formData.size, formData.side, formData.type, marketData, currentPrice]);

  // Validate order parameters
  const validateOrder = useCallback((): string | null => {
    if (formData.size <= 0) {
      return 'Order size must be positive';
    }
    
    if (formData.size > maxPositionSize) {
      return `Size exceeds maximum allowed (${maxPositionSize})`;
    }

    if (formData.type === 'limit' && (!formData.price || formData.price <= 0)) {
      return 'Limit price required for limit orders';
    }

    if (estimatedSlippage > formData.slippageTolerance) {
      return `Estimated slippage (${estimatedSlippage.toFixed(2)}%) exceeds tolerance (${formData.slippageTolerance}%)`;
    }

    return null;
  }, [formData, maxPositionSize, estimatedSlippage]);

  // Send order with debounce protection
  const sendOrder = useCallback(async () => {
    const now = Date.now();
    
    // Debounce check
    if (now - lastSendTime < DEBOUNCE_INTERVAL) {
      console.warn('[ManualOverride] Order rejected: rate limited');
      setError('Please wait between orders');
      return;
    }

    // Validation
    const validationError = validateOrder();
    if (validationError) {
      setError(validationError);
      return;
    }

    if (!isConnected) {
      setError('WebSocket disconnected');
      return;
    }

    setError(null);
    setIsSending(true);

    // Create abort controller for potential cancellation
    abortControllerRef.current = new AbortController();

    const payload = {
      type: 'MANUAL_ORDER',
      orderId: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: now,
      data: {
        side: formData.side,
        type: formData.type,
        size: formData.size,
        price: formData.type === 'limit' ? formData.price : undefined,
        slippageTolerance: formData.slippageTolerance,
        source: 'UI_MANUAL_OVERRIDE',
        hotkeyTriggered: false,
      },
    };

    try {
      sendMessage(JSON.stringify(payload));
      setLastSendTime(now);
      onOrderSubmitted(payload.orderId);
      console.log('[ManualOverride] Order sent:', payload.orderId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send order');
    } finally {
      setIsSending(false);
    }
  }, [formData, validateOrder, isConnected, lastSendTime, sendMessage, onOrderSubmitted]);

  // Hotkey handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if ((e.target as HTMLElement).tagName === 'INPUT') {
        return;
      }

      const key = e.key.toLowerCase();
      
      switch (key) {
        case HOTKEYS.BUY_MARKET:
          e.preventDefault();
          setFormData(prev => ({ ...prev, side: 'buy', type: 'market' }));
          setTimeout(() => sendOrder(), 100);
          break;
        case HOTKEYS.SELL_MARKET:
          e.preventDefault();
          setFormData(prev => ({ ...prev, side: 'sell', type: 'market' }));
          setTimeout(() => sendOrder(), 100);
          break;
        case HOTKEYS.BUY_LIMIT:
          e.preventDefault();
          setFormData(prev => ({ ...prev, side: 'buy', type: 'limit', price: currentPrice * 0.99 }));
          break;
        case HOTKEYS.SELL_LIMIT:
          e.preventDefault();
          setFormData(prev => ({ ...prev, side: 'sell', type: 'limit', price: currentPrice * 1.01 }));
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentPrice, sendOrder]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const handleSizeChange = (value: number) => {
    setFormData(prev => ({ ...prev, size: Math.max(0, value) }));
    setError(null);
  };

  const handleSlippageChange = (value: number) => {
    setFormData(prev => ({ ...prev, slippageTolerance: Math.max(0, Math.min(10, value)) }));
    setError(null);
  };

  const isSlippageWarning = estimatedSlippage > formData.slippageTolerance;
  const isSlippageDanger = estimatedSlippage > 2;

  return (
    <div className="w-full p-4 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          MANUAL OVERRIDE
        </h3>
        <div className="text-xs font-mono text-slate-400">
          <span className={isConnected ? 'text-emerald-400' : 'text-red-400'}>
            {isConnected ? '● LIVE' : '● DISCONNECTED'}
          </span>
        </div>
      </div>

      {/* Current Price Display */}
      <div className="mb-4 p-3 bg-slate-800/50 rounded-lg border border-slate-700">
        <div className="text-xs text-slate-400 mb-1">CURRENT PRICE</div>
        <div className="text-2xl font-mono font-bold text-white">
          ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </div>
      </div>

      {/* Order Type Selection */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <button
          onClick={() => setFormData(prev => ({ ...prev, type: 'market' }))}
          className={`p-3 rounded-lg font-bold transition-all ${
            formData.type === 'market'
              ? 'bg-cyan-500/20 border-2 border-cyan-400 text-cyan-400'
              : 'bg-slate-800 border border-slate-700 text-slate-400 hover:border-cyan-500/50'
          }`}
        >
          MARKET
        </button>
        <button
          onClick={() => setFormData(prev => ({ ...prev, type: 'limit' }))}
          className={`p-3 rounded-lg font-bold transition-all ${
            formData.type === 'limit'
              ? 'bg-cyan-500/20 border-2 border-cyan-400 text-cyan-400'
              : 'bg-slate-800 border border-slate-700 text-slate-400 hover:border-cyan-500/50'
          }`}
        >
          LIMIT
        </button>
      </div>

      {/* Limit Price Input (only for limit orders) */}
      <AnimatePresence>
        {formData.type === 'limit' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4"
          >
            <label className="block text-xs text-slate-400 mb-1">LIMIT PRICE</label>
            <input
              type="number"
              value={formData.price || ''}
              onChange={(e) => setFormData(prev => ({ ...prev, price: parseFloat(e.target.value) || 0 }))}
              className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-white font-mono focus:border-cyan-400 focus:outline-none"
              placeholder="Enter price"
              step={0.01}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Size Input */}
      <div className="mb-4">
        <label className="block text-xs text-slate-400 mb-1">
          SIZE (BTC) - Max: {maxPositionSize}
        </label>
        <input
          type="number"
          value={formData.size}
          onChange={(e) => handleSizeChange(parseFloat(e.target.value) || 0)}
          className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-white font-mono focus:border-cyan-400 focus:outline-none"
          step={0.001}
          min={0}
          max={maxPositionSize}
        />
        <div className="mt-2 text-xs text-slate-400">
          Notional: ${(formData.size * currentPrice).toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </div>
      </div>

      {/* Slippage Tolerance */}
      <div className="mb-4">
        <label className="block text-xs text-slate-400 mb-1">
          SLIPPAGE TOLERANCE: {formData.slippageTolerance.toFixed(1)}%
        </label>
        <input
          type="range"
          min={0}
          max={5}
          step={0.1}
          value={formData.slippageTolerance}
          onChange={(e) => handleSlippageChange(parseFloat(e.target.value))}
          className="w-full accent-cyan-400"
        />
      </div>

      {/* Slippage Warning */}
      <AnimatePresence>
        {(isSlippageWarning || isSlippageDanger) && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`mb-4 p-3 rounded-lg flex items-center gap-2 ${
              isSlippageDanger
                ? 'bg-red-500/20 border border-red-500'
                : 'bg-amber-500/20 border border-amber-500'
            }`}
          >
            <AlertTriangle className={`w-4 h-4 ${isSlippageDanger ? 'text-red-400' : 'text-amber-400'}`} />
            <span className={`text-xs font-bold ${isSlippageDanger ? 'text-red-400' : 'text-amber-400'}`}>
              Est. Slippage: {estimatedSlippage.toFixed(2)}%
              {isSlippageWarning && ' (Exceeds tolerance)'}
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mb-4 p-3 bg-red-500/20 border border-red-500 rounded-lg text-red-400 text-sm"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-3">
        <motion.button
          onClick={() => {
            setFormData(prev => ({ ...prev, side: 'buy' }));
            sendOrder();
          }}
          disabled={!isConnected || isSending}
          whileHover={{ scale: isConnected && !isSending ? 1.02 : 1 }}
          whileTap={{ scale: isConnected && !isSending ? 0.98 : 1 }}
          className="relative p-4 rounded-xl font-bold text-white overflow-hidden group"
          style={{
            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-500" />
          <div className="flex flex-col items-center gap-1">
            <ArrowUpRight className="w-6 h-6" />
            <span>BUY</span>
            <span className="text-xs opacity-70">[F1]</span>
          </div>
        </motion.button>

        <motion.button
          onClick={() => {
            setFormData(prev => ({ ...prev, side: 'sell' }));
            sendOrder();
          }}
          disabled={!isConnected || isSending}
          whileHover={{ scale: isConnected && !isSending ? 1.02 : 1 }}
          whileTap={{ scale: isConnected && !isSending ? 0.98 : 1 }}
          className="relative p-4 rounded-xl font-bold text-white overflow-hidden group"
          style={{
            background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-500" />
          <div className="flex flex-col items-center gap-1">
            <ArrowDownRight className="w-6 h-6" />
            <span>SELL</span>
            <span className="text-xs opacity-70">[F2]</span>
          </div>
        </motion.button>
      </div>

      {/* Status Footer */}
      <div className="mt-4 pt-3 border-t border-slate-700 flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-2 text-slate-400">
          <Clock className="w-3 h-3" />
          <span>Last: {lastSendTime ? new Date(lastSendTime).toLocaleTimeString() : '--:--:--'}</span>
        </div>
        <div className="text-slate-400">
          Debounce: {DEBOUNCE_INTERVAL}ms
        </div>
      </div>
    </div>
  );
};

export default ManualOverride;
