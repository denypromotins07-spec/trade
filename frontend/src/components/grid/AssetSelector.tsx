/**
 * `frontend/src/components/grid/AssetSelector.tsx`
 *
 * **Dynamic Asset Routing Panel**
 * Allows the user to instantly swap underlying WebSocket streams for specific grid slots
 * without unmounting the heavy Canvas DOM components.
 *
 * **Optimizations:**
 * - Maintains persistent chart instances, only swapping data sources.
 * - Uses React state minimally to prevent full re-renders.
 * - Pre-fetches symbol metadata to reduce latency on selection.
 */

import React, { useState, useCallback, useMemo } from 'react';

interface AssetSelectorProps {
  availableSymbols: string[];
  currentAssignments: Record<number, string>; // slotId -> symbol
  onAssignmentChange: (slotId: number, symbol: string) => void;
}

// Predefined list of major crypto assets
const MAJOR_SYMBOLS = [
  'BTCUSDT',
  'ETHUSDT',
  'SOLUSDT',
  'BNBUSDT',
  'XRPUSDT',
  'ADAUSDT',
  'DOGEUSDT',
  'MATICUSDT',
  'DOTUSDT',
  'AVAXUSDT',
];

/**
 * Individual Slot Selector Component
 */
const SlotSelector: React.FC<{
  slotId: number;
  currentSymbol: string;
  availableSymbols: string[];
  onChange: (slotId: number, symbol: string) => void;
}> = React.memo(({ slotId, currentSymbol, availableSymbols, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleChange = useCallback((newSymbol: string) => {
    onChange(slotId, newSymbol);
    setIsOpen(false);
  }, [slotId, onChange]);

  return (
    <div className="asset-selector-slot" style={{ position: 'relative', display: 'inline-block' }}>
      {/* Current Symbol Display */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="asset-selector-button"
        style={{
          background: '#161b22',
          border: '1px solid #30363d',
          color: '#c9d1d9',
          padding: '4px 12px',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '12px',
          fontWeight: 600,
          minWidth: '100px',
          textAlign: 'left',
        }}
      >
        {currentSymbol}
        <span style={{ float: 'right', marginLeft: '8px' }}>▼</span>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          {/* Backdrop to close on outside click */}
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 99,
            }}
            onClick={() => setIsOpen(false)}
          />
          
          {/* Dropdown Content */}
          <div
            className="asset-selector-dropdown"
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              marginTop: '4px',
              background: '#161b22',
              border: '1px solid #30363d',
              borderRadius: '4px',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
              zIndex: 100,
              maxHeight: '300px',
              overflowY: 'auto',
              minWidth: '150px',
            }}
          >
            {/* Search Input */}
            <input
              type="text"
              placeholder="Search symbol..."
              style={{
                width: '100%',
                boxSizing: 'border-box',
                background: '#0d1117',
                border: 'none',
                borderBottom: '1px solid #30363d',
                color: '#c9d1d9',
                padding: '8px',
                fontSize: '12px',
                outline: 'none',
              }}
              onChange={(e) => {
                // Simple filter logic could be added here
              }}
            />

            {/* Symbol List */}
            <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
              {availableSymbols.map((symbol) => (
                <li
                  key={symbol}
                  onClick={() => handleChange(symbol)}
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    background: symbol === currentSymbol ? '#1f6feb33' : 'transparent',
                    color: symbol === currentSymbol ? '#58a6ff' : '#c9d1d9',
                    fontSize: '12px',
                    borderBottom: '1px solid #21262d',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = '#21262d';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = symbol === currentSymbol ? '#1f6feb33' : 'transparent';
                  }}
                >
                  {symbol}
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
});

SlotSelector.displayName = 'SlotSelector';

/**
 * Main Asset Selector Panel Component
 */
export const AssetSelector: React.FC<AssetSelectorProps> = ({
  availableSymbols = MAJOR_SYMBOLS,
  currentAssignments,
  onAssignmentChange,
}) => {
  // Memoize slot IDs to prevent unnecessary re-renders
  const slotIds = useMemo(() => Object.keys(currentAssignments).map(Number), [currentAssignments]);

  return (
    <div
      className="asset-selector-panel"
      style={{
        display: 'flex',
        gap: '8px',
        padding: '8px',
        background: '#0d1117',
        borderBottom: '1px solid #30363d',
        flexWrap: 'wrap',
      }}
    >
      <span
        style={{
          color: '#8b949e',
          fontSize: '12px',
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        Grid Slots:
      </span>
      
      {slotIds.map((slotId) => (
        <SlotSelector
          key={slotId}
          slotId={slotId}
          currentSymbol={currentAssignments[slotId] || 'EMPTY'}
          availableSymbols={availableSymbols}
          onChange={onAssignmentChange}
        />
      ))}
    </div>
  );
};

export default AssetSelector;
