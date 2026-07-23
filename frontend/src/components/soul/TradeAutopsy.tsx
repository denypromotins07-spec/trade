/**
 * TradeAutopsy.tsx - Deep-dive Modal for Losing Trade Analysis
 * 
 * Shows the exact microsecond order book state and RL latent state
 * at the time of execution for post-mortem analysis.
 * 
 * Features:
 * - Detailed trade reconstruction
 * - Order book snapshot visualization
 * - RL agent state at execution time
 * - Slippage and timing analysis
 * - Cyberpunk forensic aesthetic
 */

import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSoulStore } from '../../store/soulStore';

interface TradeAutopsyProps {
  tradeId?: string;
  onClose?: () => void;
}

export const TradeAutopsy: React.FC<TradeAutopsyProps> = ({ tradeId, onClose }) => {
  const { selectedTrade, tradeHistory } = useSoulStore();

  // Use provided tradeId or fall back to selected trade from store
  const trade = useMemo(() => {
    if (tradeId) {
      return tradeHistory?.find(t => t.id === tradeId);
    }
    return selectedTrade;
  }, [tradeId, selectedTrade, tradeHistory]);

  if (!trade) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '400px',
          color: 'rgba(139, 155, 180, 0.4)',
        }}
      >
        <span style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.3 }}>🔬</span>
        <p style={{ fontSize: '12px' }}>No trade selected for autopsy</p>
        <p style={{ fontSize: '10px', marginTop: '8px' }}>Select a losing trade from the history</p>
      </div>
    );
  }

  const pnlPercent = ((trade.exitPrice - trade.entryPrice) / trade.entryPrice) * 100 * (trade.side === 'BUY' ? 1 : -1);
  const isLoss = pnlPercent < 0;

  const modalVariants = {
    hidden: { opacity: 0, scale: 0.9, y: 20 },
    visible: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.3, type: 'spring' } },
    exit: { opacity: 0, scale: 0.9, y: -20, transition: { duration: 0.2 } },
  };

  return (
    <AnimatePresence>
      <motion.div
        initial="hidden"
        animate="visible"
        exit="exit"
        variants={modalVariants}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95 }}
          animate={{ scale: 1 }}
          style={{
            width: '90%',
            maxWidth: '900px',
            maxHeight: '85vh',
            overflow: 'auto',
            background: 'linear-gradient(135deg, rgba(15, 25, 45, 0.98) 0%, rgba(10, 15, 30, 0.98) 100%)',
            borderRadius: '12px',
            border: `2px solid ${isLoss ? '#ff3366' : '#00ff88'}`,
            boxShadow: `0 0 60px ${isLoss ? 'rgba(255, 51, 102, 0.3)' : 'rgba(0, 255, 136, 0.2)'}, inset 0 0 40px rgba(0, 0, 0, 0.4)`,
            fontFamily: '"JetBrains Mono", monospace',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '16px 20px',
              borderBottom: `1px solid ${isLoss ? 'rgba(255, 51, 102, 0.3)' : 'rgba(0, 255, 136, 0.3)'}`,
              background: isLoss 
                ? 'linear-gradient(90deg, rgba(255, 51, 102, 0.1) 0%, transparent 100%)'
                : 'linear-gradient(90deg, rgba(0, 255, 136, 0.1) 0%, transparent 100%)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '24px' }}>{isLoss ? '💀' : '✅'}</span>
              <div>
                <h2
                  style={{
                    margin: 0,
                    fontSize: '14px',
                    fontWeight: 700,
                    color: isLoss ? '#ff3366' : '#00ff88',
                    textTransform: 'uppercase',
                    letterSpacing: '1.5px',
                    textShadow: `0 0 15px ${isLoss ? 'rgba(255, 51, 102, 0.6)' : 'rgba(0, 255, 136, 0.6)'}`,
                  }}
                >
                  TRADE AUTOPSY #{trade.id?.slice(-6).toUpperCase()}
                </h2>
                <p style={{ margin: '4px 0 0', fontSize: '9px', color: 'rgba(139, 155, 180, 0.6)' }}>
                  {new Date(trade.timestamp).toLocaleString()} • {trade.symbol}
                </p>
              </div>
            </div>
            
            <button
              onClick={onClose}
              style={{
                padding: '8px 16px',
                background: 'rgba(139, 155, 180, 0.1)',
                border: '1px solid rgba(139, 155, 180, 0.3)',
                borderRadius: '6px',
                color: '#8b9bb4',
                fontSize: '10px',
                fontFamily: '"JetBrains Mono", monospace',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 51, 102, 0.2)';
                e.currentTarget.style.borderColor = '#ff3366';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(139, 155, 180, 0.1)';
                e.currentTarget.style.borderColor = 'rgba(139, 155, 180, 0.3)';
              }}
            >
              ✕ CLOSE
            </button>
          </div>

          {/* Content */}
          <div style={{ padding: '20px' }}>
            {/* Trade Summary */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(6, 1fr)',
                gap: '12px',
                marginBottom: '20px',
                padding: '16px',
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '8px',
                border: `1px solid ${isLoss ? 'rgba(255, 51, 102, 0.2)' : 'rgba(0, 255, 136, 0.2)'}`,
              }}
            >
              <div>
                <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '4px' }}>SIDE</div>
                <div
                  style={{
                    fontSize: '12px',
                    fontWeight: 700,
                    color: trade.side === 'BUY' ? '#00ff88' : '#ff3366',
                  }}
                >
                  {trade.side}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '4px' }}>ENTRY</div>
                <div style={{ fontSize: '12px', color: '#c0c5ce' }}>${trade.entryPrice.toFixed(2)}</div>
              </div>
              <div>
                <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '4px' }}>EXIT</div>
                <div style={{ fontSize: '12px', color: '#c0c5ce' }}>${trade.exitPrice.toFixed(2)}</div>
              </div>
              <div>
                <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '4px' }}>SIZE</div>
                <div style={{ fontSize: '12px', color: '#00ffff' }}>{trade.quantity}</div>
              </div>
              <div>
                <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '4px' }}>P&L</div>
                <div
                  style={{
                    fontSize: '12px',
                    fontWeight: 700,
                    color: isLoss ? '#ff3366' : '#00ff88',
                    textShadow: `0 0 8px ${isLoss ? 'rgba(255, 51, 102, 0.5)' : 'rgba(0, 255, 136, 0.5)'}`,
                  }}
                >
                  {pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
                </div>
              </div>
              <div>
                <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '4px' }}>DURATION</div>
                <div style={{ fontSize: '12px', color: '#bd93f9' }}>{trade.duration || 'N/A'}</div>
              </div>
            </div>

            {/* Two Column Layout */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              {/* Left Column - Order Book State */}
              <div>
                <h3
                  style={{
                    margin: '0 0 12px',
                    fontSize: '11px',
                    fontWeight: 600,
                    color: '#00ffff',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  📊 ORDER BOOK SNAPSHOT
                </h3>
                <div
                  style={{
                    background: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '8px',
                    padding: '12px',
                    border: '1px solid rgba(0, 255, 255, 0.15)',
                  }}
                >
                  {/* Bids */}
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '8px', color: '#00ff88', marginBottom: '6px' }}>BIDS (BUY)</div>
                    {trade.orderBookSnapshot?.bids?.slice(0, 5).map((bid, idx) => (
                      <div
                        key={`bid-${idx}`}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '4px 8px',
                          marginBottom: '2px',
                          background: 'rgba(0, 255, 136, 0.05)',
                          borderRadius: '3px',
                          fontSize: '9px',
                        }}
                      >
                        <span style={{ color: '#00ff88' }}>${bid.price?.toFixed(2)}</span>
                        <span style={{ color: 'rgba(139, 155, 180, 0.7)' }}>{bid.size}</span>
                      </div>
                    ))}
                  </div>
                  
                  {/* Asks */}
                  <div>
                    <div style={{ fontSize: '8px', color: '#ff3366', marginBottom: '6px' }}>ASKS (SELL)</div>
                    {trade.orderBookSnapshot?.asks?.slice(0, 5).map((ask, idx) => (
                      <div
                        key={`ask-${idx}`}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '4px 8px',
                          marginBottom: '2px',
                          background: 'rgba(255, 51, 102, 0.05)',
                          borderRadius: '3px',
                          fontSize: '9px',
                        }}
                      >
                        <span style={{ color: '#ff3366' }}>${ask.price?.toFixed(2)}</span>
                        <span style={{ color: 'rgba(139, 155, 180, 0.7)' }}>{ask.size}</span>
                      </div>
                    ))}
                  </div>
                  
                  {/* Spread */}
                  <div
                    style={{
                      marginTop: '12px',
                      paddingTop: '8px',
                      borderTop: '1px solid rgba(139, 155, 180, 0.2)',
                      textAlign: 'center',
                    }}
                  >
                    <span style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)' }}>SPREAD: </span>
                    <span style={{ fontSize: '10px', color: '#ffaa00' }}>
                      ${((trade.orderBookSnapshot?.asks?.[0]?.price || 0) - (trade.orderBookSnapshot?.bids?.[0]?.price || 0)).toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Right Column - RL Agent State */}
              <div>
                <h3
                  style={{
                    margin: '0 0 12px',
                    fontSize: '11px',
                    fontWeight: 600,
                    color: '#bd93f9',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  🧠 RL AGENT STATE
                </h3>
                <div
                  style={{
                    background: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '8px',
                    padding: '12px',
                    border: '1px solid rgba(189, 147, 249, 0.15)',
                  }}
                >
                  {/* Action Taken */}
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '6px' }}>ACTION SELECTED</div>
                    <div
                      style={{
                        display: 'inline-block',
                        padding: '6px 12px',
                        background: `rgba(${trade.agentState?.action === 'BUY' ? '0, 255, 136' : trade.agentState?.action === 'SELL' ? '255, 51, 102' : '139, 155, 180'}, 0.15)`,
                        border: `1px solid ${trade.agentState?.action === 'BUY' ? '#00ff88' : trade.agentState?.action === 'SELL' ? '#ff3366' : '#8b9bb4'}`,
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 700,
                        color: trade.agentState?.action === 'BUY' ? '#00ff88' : trade.agentState?.action === 'SELL' ? '#ff3366' : '#8b9bb4',
                      }}
                    >
                      {trade.agentState?.action || 'UNKNOWN'}
                    </div>
                  </div>

                  {/* Confidence */}
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '6px' }}>CONFIDENCE</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div
                        style={{
                          flex: 1,
                          height: '8px',
                          background: 'rgba(20, 30, 48, 0.8)',
                          borderRadius: '4px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${(trade.agentState?.confidence || 0) * 100}%`,
                            height: '100%',
                            background: `linear-gradient(90deg, 
                              ${(trade.agentState?.confidence || 0) > 0.7 ? '#00ff88' : (trade.agentState?.confidence || 0) > 0.4 ? '#ffaa00' : '#ff3366'}, 
                              ${(trade.agentState?.confidence || 0) > 0.7 ? '#00ffff' : '#ffcc00'})`,
                          }}
                        />
                      </div>
                      <span style={{ fontSize: '10px', color: '#bd93f9' }}>
                        {((trade.agentState?.confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Latent State */}
                  <div>
                    <div style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '6px' }}>LATENT VECTOR (Z)</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px' }}>
                      {trade.agentState?.latentState?.slice(0, 8).map((val, idx) => (
                        <div
                          key={`latent-${idx}`}
                          style={{
                            padding: '6px 4px',
                            background: 'rgba(189, 147, 249, 0.1)',
                            borderRadius: '3px',
                            textAlign: 'center',
                          }}
                        >
                          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.5)' }}>Z{idx}</div>
                          <div style={{ fontSize: '9px', color: '#bd93f9', fontWeight: 600 }}>
                            {val?.toFixed(3)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Notes */}
            {trade.analysisNotes && (
              <div style={{ marginTop: '20px' }}>
                <h3
                  style={{
                    margin: '0 0 12px',
                    fontSize: '11px',
                    fontWeight: 600,
                    color: '#ffaa00',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  📝 ANALYSIS NOTES
                </h3>
                <div
                  style={{
                    background: 'rgba(255, 170, 0, 0.05)',
                    borderRadius: '8px',
                    padding: '12px',
                    border: '1px solid rgba(255, 170, 0, 0.2)',
                  }}
                >
                  <p style={{ fontSize: '10px', color: '#c0c5ce', lineHeight: 1.6, margin: 0 }}>
                    {trade.analysisNotes}
                  </p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default TradeAutopsy;
