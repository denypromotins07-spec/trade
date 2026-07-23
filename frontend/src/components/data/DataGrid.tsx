/**
 * File 11: DataGrid.tsx
 * Chapter 4: Data Exploration & Custom Query Builder
 * 
 * High-performance, virtualized data table for query results,
 * recycling DOM rows to display millions of historical ticks without browser crashes.
 */

import React, { useState, useMemo, useCallback, useRef } from 'react';

interface RowData {
  id: string;
  [key: string]: unknown;
}

interface Column {
  key: string;
  label: string;
  width?: number;
  align?: 'left' | 'right' | 'center';
  format?: (value: unknown) => string;
}

interface Props {
  data: RowData[];
  columns: Column[];
  rowHeight?: number;
  height?: number;
  onRowClick?: (row: RowData) => void;
}

const COLORS = {
  bg: '#0a0a0a',
  headerBg: '#111111',
  rowEven: 'rgba(255,255,255,0.02)',
  rowOdd: 'rgba(255,255,255,0.04)',
  rowHover: 'rgba(0,243,255,0.1)',
  border: '#222222',
  text: '#c0c0c0',
  headerText: '#00f3ff',
};

export const DataGrid: React.FC<Props> = ({
  data,
  columns,
  rowHeight = 28,
  height = 500,
  onRowClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  const visibleRows = Math.ceil(height / rowHeight) + 2;
  const startIndex = Math.floor(scrollTop / rowHeight);
  const endIndex = Math.min(startIndex + visibleRows, data.length);

  const sortedData = useMemo(() => {
    if (!sortKey) return data;
    return [...data].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortAsc ? aVal - bVal : bVal - aVal;
      }
      const aStr = String(aVal);
      const bStr = String(bVal);
      return sortAsc ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
    });
  }, [data, sortKey, sortAsc]);

  const handleSort = useCallback((key: string) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  }, [sortKey, sortAsc]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  const offsetY = startIndex * rowHeight;

  return (
    <div className="border border-cyan-900/50 rounded-lg overflow-hidden bg-black/80 backdrop-blur">
      {/* Header */}
      <div className="flex border-b border-gray-800 bg-[#111]" style={{ height: rowHeight }}>
        {columns.map((col) => (
          <div
            key={col.key}
            onClick={() => handleSort(col.key)}
            className="flex items-center px-3 cursor-pointer hover:bg-gray-800 transition-colors select-none"
            style={{
              width: col.width || 'auto',
              flex: col.width ? 'none' : 1,
              justifyContent: col.align === 'right' ? 'flex-end' : col.align === 'center' ? 'center' : 'flex-start',
            }}
          >
            <span className="text-[10px] font-mono font-bold" style={{ color: COLORS.headerText }}>
              {col.label}
            </span>
            {sortKey === col.key && (
              <span className="ml-1 text-[10px]">{sortAsc ? '▲' : '▼'}</span>
            )}
          </div>
        ))}
      </div>

      {/* Virtualized Body */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="overflow-auto"
        style={{ height }}
      >
        <div style={{ height: data.length * rowHeight, position: 'relative' }}>
          {sortedData.slice(startIndex, endIndex).map((row, idx) => {
            const actualIndex = startIndex + idx;
            const isEven = actualIndex % 2 === 0;
            const isHovered = hoveredRow === row.id;

            return (
              <div
                key={row.id}
                onClick={() => onRowClick?.(row)}
                onMouseEnter={() => setHoveredRow(row.id)}
                onMouseLeave={() => setHoveredRow(null)}
                className="flex border-b border-gray-800/50 cursor-pointer transition-colors"
                style={{
                  position: 'absolute',
                  top: offsetY + idx * rowHeight,
                  left: 0,
                  right: 0,
                  height: rowHeight,
                  backgroundColor: isHovered
                    ? COLORS.rowHover
                    : isEven
                    ? COLORS.rowEven
                    : COLORS.rowOdd,
                }}
              >
                {columns.map((col) => {
                  const value = row[col.key];
                  const display = col.format ? col.format(value) : String(value);
                  return (
                    <div
                      key={col.key}
                      className="flex items-center px-3 overflow-hidden text-ellipsis whitespace-nowrap"
                      style={{
                        width: col.width || 'auto',
                        flex: col.width ? 'none' : 1,
                        justifyContent: col.align === 'right' ? 'flex-end' : col.align === 'center' ? 'center' : 'flex-start',
                      }}
                    >
                      <span className="text-[10px] font-mono" style={{ color: COLORS.text }}>
                        {display}
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Stats */}
      <div className="flex justify-between items-center px-3 py-2 border-t border-gray-800 bg-[#0d0d0d]">
        <span className="text-[9px] font-mono text-gray-500">
          SHOWING {startIndex + 1}-{Math.min(endIndex, data.length)} OF {data.length.toLocaleString()} ROWS
        </span>
        <span className="text-[9px] font-mono text-cyan-600">
          VIRTUALIZED • {((endIndex - startIndex) / data.length * 100).toFixed(4)}% RENDERED
        </span>
      </div>
    </div>
  );
};

export default DataGrid;
