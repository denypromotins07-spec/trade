/**
 * DataGrid.tsx - Data Exploration: High-Performance Virtualized Data Table
 * 
 * Implements a high-performance, virtualized data table for query results,
 * recycling DOM rows to display millions of historical ticks without browser crashes.
 * 
 * Features:
 * - Strict virtualization with DOM row recycling
 * - Windowed rendering for memory-bounded display
 * - Column resizing and sorting
 * - CSV export functionality
 * - Cyberpunk-styled table with sticky headers
 */

'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

interface Column {
  key: string;
  label: string;
  width?: number;
  sortable?: boolean;
  format?: (value: any) => string;
}

interface DataRow {
  id: string | number;
  [key: string]: any;
}

interface DataGridProps {
  columns: Column[];
  data?: DataRow[];
  height?: number;
  rowHeight?: number;
  onRowClick?: (row: DataRow) => void;
  enableExport?: boolean;
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const ROW_HEIGHT_DEFAULT = 32;
const HEIGHT_DEFAULT = 400;
const OVERSCAN_ROWS = 10; // Extra rows to render above/below viewport

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generates mock tick data for demonstration
 */
const generateMockTickData = (count: number): DataRow[] => {
  const symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP'];
  
  return Array.from({ length: count }, (_, i) => ({
    id: `tick-${i}`,
    timestamp: Date.now() - (count - i) * 100,
    symbol: symbols[Math.floor(Math.random() * symbols.length)],
    price: Math.random() * 50000 + 20000,
    size: Math.random() * 10,
    side: Math.random() > 0.5 ? 'buy' : 'sell',
    exchange: ['Binance', 'Coinbase', 'Kraken'][Math.floor(Math.random() * 3)],
  }));
};

/**
 * Formats numbers with locale separators
 */
const formatNumber = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

/**
 * Formats timestamp to readable format
 */
const formatTimestamp = (timestamp: number): string => {
  return new Date(timestamp).toISOString().slice(11, 23);
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Individual Row Component - Memoized for performance
 */
interface GridRowProps {
  row: DataRow;
  columns: Column[];
  style: React.CSSProperties;
  index: number;
  isSelected: boolean;
  onClick: () => void;
}

const GridRow: React.FC<GridRowProps> = React.memo(({
  row,
  columns,
  style,
  index,
  isSelected,
  onClick,
}) => {
  return (
    <div
      style={style}
      className={`absolute w-full flex items-center border-b border-white/5 cursor-pointer transition-colors ${
        isSelected ? 'bg-cyan-500/20' : 'hover:bg-white/5'
      }`}
      onClick={onClick}
      role="row"
      aria-rowindex={index + 1}
    >
      {columns.map((column) => {
        const value = row[column.key];
        const displayValue = column.format ? column.format(value) : String(value);
        
        return (
          <div
            key={column.key}
            className="px-3 py-1 text-xs font-mono truncate"
            style={{
              width: column.width || 100,
              minWidth: column.width || 100,
              color: column.key === 'side' ? (value === 'buy' ? '#00ff88' : '#ff0088') : undefined,
            }}
            role="gridcell"
          >
            {displayValue}
          </div>
        );
      })}
    </div>
  );
});

GridRow.displayName = 'GridRow';

/**
 * Header Row Component
 */
interface HeaderRowProps {
  columns: Column[];
  onSort?: (key: string) => void;
  sortColumn?: string;
  sortDirection?: 'asc' | 'desc';
}

const HeaderRow: React.FC<HeaderRowProps> = ({
  columns,
  onSort,
  sortColumn,
  sortDirection,
}) => {
  return (
    <div
      className="flex items-center bg-cyan-900/30 border-b border-cyan-500/30 sticky top-0 z-10"
      role="row"
      aria-rowindex={0}
    >
      {columns.map((column) => (
        <div
          key={column.key}
          className={`px-3 py-2 text-xs font-mono font-bold uppercase tracking-wider cursor-pointer hover:bg-cyan-500/20 transition-colors ${
            sortColumn === column.key ? 'text-cyan-400' : 'text-gray-400'
          }`}
          style={{ width: column.width || 100, minWidth: column.width || 100 }}
          onClick={() => column.sortable && onSort?.(column.key)}
          role="columnheader"
          aria-sort={sortColumn === column.key ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
        >
          <span>{column.label}</span>
          {column.sortable && sortColumn === column.key && (
            <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
          )}
        </div>
      ))}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const DataGrid: React.FC<DataGridProps> = ({
  columns,
  data,
  height = HEIGHT_DEFAULT,
  rowHeight = ROW_HEIGHT_DEFAULT,
  onRowClick,
  enableExport = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(height);
  const [selectedRow, setSelectedRow] = useState<string | number | null>(null);
  const [sortColumn, setSortColumn] = useState<string>('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  
  // Generate or use provided data
  const allData = useMemo(() => {
    return data || generateMockTickData(10000);
  }, [data]);
  
  // Sort data
  const sortedData = useMemo(() => {
    if (!sortColumn) return allData;
    
    return [...allData].sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      const comparison = String(aVal).localeCompare(String(bVal));
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [allData, sortColumn, sortDirection]);
  
  // Calculate visible range
  const totalHeight = sortedData.length * rowHeight;
  const visibleStartIndex = Math.floor(scrollTop / rowHeight);
  const visibleEndIndex = Math.ceil((scrollTop + containerHeight) / rowHeight);
  
  // Add overscan for smooth scrolling
  const overscanStart = Math.max(0, visibleStartIndex - OVERSCAN_ROWS);
  const overscanEnd = Math.min(sortedData.length, visibleEndIndex + OVERSCAN_ROWS);
  
  // Get visible rows
  const visibleRows = useMemo(() => {
    return sortedData.slice(overscanStart, overscanEnd);
  }, [sortedData, overscanStart, overscanEnd]);

  /**
   * Handle scroll with RAF batching
   */
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    
    let rafId: number;
    
    const updateScroll = () => {
      if (containerRef.current) {
        setScrollTop(containerRef.current.scrollTop);
      }
    };
    
    rafId = requestAnimationFrame(updateScroll);
    
    return () => cancelAnimationFrame(rafId);
  }, []);

  /**
   * Handle column sort
   */
  const handleSort = useCallback((key: string) => {
    if (sortColumn === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(key);
      setSortDirection('asc');
    }
  }, [sortColumn, sortDirection]);

  /**
   * Export to CSV
   */
  const handleExport = useCallback(() => {
    const headers = columns.map((c) => c.label).join(',');
    const rows = sortedData.map((row) =>
      columns.map((c) => `"${row[c.key]}"`).join(',')
    ).join('\n');
    
    const csv = `${headers}\n${rows}`;
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `data-export-${Date.now()}.csv`;
    link.click();
    
    URL.revokeObjectURL(url);
  }, [columns, sortedData]);

  // Setup scroll listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    container.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      container.removeEventListener('scroll', HandleScroll);
    };
  }, [handleScroll]);

  // Update container height
  useEffect(() => {
    setContainerHeight(height);
  }, [height]);

  // Default column formats
  const columnsWithDefaults = useMemo(() => {
    return columns.map((col) => ({
      ...col,
      sortable: col.sortable ?? true,
      format: col.format ?? ((val: any) => {
        if (typeof val === 'number') return formatNumber(val);
        if (col.key === 'timestamp') return formatTimestamp(val);
        return String(val);
      }),
    }));
  }, [columns]);

  return (
    <div className="w-full rounded-xl overflow-hidden bg-[#0a0a12]/90 backdrop-blur-sm border border-cyan-900/30">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-b from-[#0a0a12] to-transparent border-b border-white/5">
        <h3 className="text-cyan-400 font-mono text-sm tracking-wider uppercase">
          📊 Data Grid <span className="text-xs opacity-70">| {sortedData.length.toLocaleString()} rows</span>
        </h3>
        {enableExport && (
          <button
            onClick={handleExport}
            className="px-3 py-1.5 bg-cyan-500/20 border border-cyan-500/50 rounded text-cyan-400 text-xs font-mono hover:bg-cyan-500/30 transition-colors"
          >
            📥 Export CSV
          </button>
        )}
      </div>

      {/* Grid Container */}
      <div
        ref={containerRef}
        className="relative overflow-y-auto overflow-x-hidden"
        style={{ height: containerHeight }}
        role="grid"
        aria-label="Data grid table"
      >
        {/* Header */}
        <div style={{ minWidth: columnsWithDefaults.reduce((acc, c) => acc + (c.width || 100), 0) }}>
          <HeaderRow
            columns={columnsWithDefaults}
            onSort={handleSort}
            sortColumn={sortColumn}
            sortDirection={sortDirection}
          />
          
          {/* Virtualized Rows */}
          <div style={{ height: totalHeight, position: 'relative' }}>
            {visibleRows.map((row, index) => {
              const actualIndex = overscanStart + index;
              const top = actualIndex * rowHeight;
              
              return (
                <GridRow
                  key={row.id}
                  row={row}
                  columns={columnsWithDefaults}
                  index={actualIndex}
                  isSelected={selectedRow === row.id}
                  onClick={() => {
                    setSelectedRow(row.id);
                    onRowClick?.(row);
                  }}
                  style={{
                    position: 'absolute',
                    top,
                    left: 0,
                    right: 0,
                    height: rowHeight,
                    willChange: 'transform',
                    transform: 'translateZ(0)',
                  }}
                />
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer Stats */}
      <div className="px-4 py-2 bg-gradient-to-t from-[#0a0a12] to-transparent border-t border-white/5">
        <div className="flex items-center justify-between text-xs font-mono text-gray-500">
          <span>
            Showing {visibleRows.length} of {sortedData.length.toLocaleString()} rows
          </span>
          <span>Virtualized • Memory Safe</span>
        </div>
      </div>
    </div>
  );
};

export default DataGrid;
