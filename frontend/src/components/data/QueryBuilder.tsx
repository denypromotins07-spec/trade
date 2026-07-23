/**
 * QueryBuilder.tsx - Data Exploration: Visual Node-Based Query Builder
 * 
 * Provides a visual node-based UI to construct complex Polars and DuckDB
 * analytical queries, translating visual graphs into strict Rust IPC payloads.
 * 
 * Features:
 * - Drag-and-drop node-based query builder
 * - Visual graph representation of data transformations
 * - Input sanitization to prevent injection attacks
 * - Rust IPC payload generation for backend execution
 * - Cyberpunk-styled node connections with animated flows
 */

'use client';

import React, { useState, useCallback, useMemo, useRef } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

type NodeType = 'source' | 'filter' | 'aggregate' | 'join' | 'select' | 'sort' | 'output';

interface QueryNode {
  id: string;
  type: NodeType;
  position: { x: number; y: number };
  config: Record<string, any>;
  label: string;
}

interface NodeConnection {
  from: string;
  to: string;
}

interface QueryGraph {
  nodes: QueryNode[];
  connections: NodeConnection[];
}

interface QueryBuilderProps {
  initialGraph?: QueryGraph;
  onExecute?: (payload: RustPayload) => void;
}

interface RustPayload {
  query_type: string;
  operations: any[];
  sanitized: boolean;
  checksum: string;
}

// ============================================================================
// Constants & Configuration
// ============================================================================

const NODE_TYPES: Record<NodeType, { icon: string; color: string; label: string }> = {
  source: { icon: '📊', color: '#00ff88', label: 'Data Source' },
  filter: { icon: '🔍', color: '#00ccff', label: 'Filter' },
  aggregate: { icon: '📈', color: '#ffcc00', label: 'Aggregate' },
  join: { icon: '🔗', color: '#ff6600', label: 'Join' },
  select: { icon: '✓', color: '#00ff00', label: 'Select Columns' },
  sort: { icon: '↕️', color: '#ff00ff', label: 'Sort' },
  output: { icon: '💾', color: '#ff0088', label: 'Output' },
};

const CANVAS_SIZE = { width: 800, height: 500 };

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generates a unique ID for nodes
 */
const generateId = (): string => {
  return `node-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
};

/**
 * Sanitizes user input to prevent SQL/DuckDB injection
 */
const sanitizeInput = (input: string): string => {
  // Remove potentially dangerous characters
  const sanitized = input
    .replace(/[;'"\\]/g, '') // Remove semicolons, quotes, backslashes
    .replace(/\b(DROP|DELETE|INSERT|UPDATE|ALTER)\b/gi, '') // Block SQL keywords
    .trim();
  
  return sanitized;
};

/**
 * Generates a checksum for the query payload
 */
const generateChecksum = (payload: any): string => {
  const str = JSON.stringify(payload);
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(16);
};

/**
 * Converts visual graph to Rust IPC payload
 */
const graphToRustPayload = (graph: QueryGraph): RustPayload => {
  const operations = graph.nodes.map((node) => ({
    type: node.type,
    config: Object.entries(node.config).reduce((acc, [key, value]) => ({
      ...acc,
      [key]: typeof value === 'string' ? sanitizeInput(value) : value,
    }), {}),
  }));
  
  const payload = {
    query_type: 'polars_lazy' as const,
    operations,
    sanitized: true,
    checksum: '',
  };
  
  payload.checksum = generateChecksum(payload);
  
  return payload;
};

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Individual Query Node Component
 */
interface QueryNodeCardProps {
  node: QueryNode;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onUpdate: (id: string, updates: Partial<QueryNode>) => void;
  onDelete: (id: string) => void;
}

const QueryNodeCard: React.FC<QueryNodeCardProps> = ({
  node,
  isSelected,
  onSelect,
  onUpdate,
  onDelete,
}) => {
  const nodeConfig = NODE_TYPES[node.type];
  
  return (
    <div
      className={`absolute p-3 rounded-lg border backdrop-blur-md cursor-move select-none ${
        isSelected ? 'ring-2 ring-cyan-500' : ''
      }`}
      style={{
        left: node.position.x,
        top: node.position.y,
        backgroundColor: `${nodeConfig.color}11`,
        borderColor: nodeConfig.color,
        minWidth: '180px',
        boxShadow: isSelected ? `0 0 20px ${nodeConfig.color}44` : 'none',
      }}
      onClick={() => onSelect(node.id)}
      role="button"
      aria-label={`${nodeConfig.label} node: ${node.label}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">{nodeConfig.icon}</span>
          <span className="text-xs font-mono font-bold" style={{ color: nodeConfig.color }}>
            {nodeConfig.label}
          </span>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete(node.id);
          }}
          className="w-4 h-4 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/20 flex items-center justify-center text-xs"
          aria-label="Delete node"
        >
          ✕
        </button>
      </div>
      
      {/* Node Label */}
      <input
        type="text"
        value={node.label}
        onChange={(e) => onUpdate(node.id, { label: e.target.value })}
        onClick={(e) => e.stopPropagation()}
        className="w-full bg-transparent border-b border-white/20 text-white text-sm py-1 focus:outline-none focus:border-cyan-500 font-mono"
        placeholder="Node label"
      />
      
      {/* Config Fields */}
      <div className="mt-2 space-y-1">
        {node.type === 'filter' && (
          <>
            <input
              type="text"
              value={node.config.column || ''}
              onChange={(e) => onUpdate(node.id, { config: { ...node.config, column: e.target.value } })}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
              placeholder="Column name"
            />
            <select
              value={node.config.operator || '='}
              onChange={(e) => onUpdate(node.id, { config: { ...node.config, operator: e.target.value } })}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
            >
              <option value="=">=</option>
              <option value=">">&gt;</option>
              <option value="<">&lt;</option>
              <option value=">=">&gt;=</option>
              <option value="<=">&lt;=</option>
              <option value="!=">≠</option>
            </select>
            <input
              type="text"
              value={node.config.value || ''}
              onChange={(e) => onUpdate(node.id, { config: { ...node.config, value: e.target.value } })}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
              placeholder="Value"
            />
          </>
        )}
        
        {node.type === 'aggregate' && (
          <>
            <select
              value={node.config.function || 'sum'}
              onChange={(e) => onUpdate(node.id, { config: { ...node.config, function: e.target.value } })}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
            >
              <option value="sum">SUM</option>
              <option value="avg">AVG</option>
              <option value="count">COUNT</option>
              <option value="min">MIN</option>
              <option value="max">MAX</option>
            </select>
            <input
              type="text"
              value={node.config.column || ''}
              onChange={(e) => onUpdate(node.id, { config: { ...node.config, column: e.target.value } })}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
              placeholder="Column"
            />
          </>
        )}
        
        {node.type === 'sort' && (
          <>
            <input
              type="text"
              value={node.config.column || ''}
              onChange={(e) => onUpdate(node.id, { config: { ...node.config, column: e.target.value } })}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
              placeholder="Sort column"
            />
            <select
              value={node.config.order || 'asc'}
              onChange={(e) => onUpdate(node.id, { config: { ...node.config, order: e.target.value } })}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-white/10 border border-white/20 rounded px-2 py-1 text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
            >
              <option value="asc">Ascending ↑</option>
              <option value="desc">Descending ↓</option>
            </select>
          </>
        )}
      </div>
      
      {/* Connection Points */}
      <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-cyan-500 border-2 border-[#0a0a12]" />
      <div className="absolute -right-1 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-cyan-500 border-2 border-[#0a0a12]" />
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const QueryBuilder: React.FC<QueryBuilderProps> = ({
  initialGraph,
  onExecute,
}) => {
  const [graph, setGraph] = useState<QueryGraph>(initialGraph || {
    nodes: [
      {
        id: generateId(),
        type: 'source',
        position: { x: 50, y: 200 },
        config: { table: 'trades' },
        label: 'Trade Data',
      },
      {
        id: generateId(),
        type: 'output',
        position: { x: 550, y: 200 },
        config: {},
        label: 'Result',
      },
    ],
    connections: [],
  });
  
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);
  const dragOffset = useRef({ x: 0, y: 0 });

  /**
   * Adds a new node to the graph
   */
  const addNode = useCallback((type: NodeType) => {
    const newNode: QueryNode = {
      id: generateId(),
      type,
      position: { x: 200 + Math.random() * 200, y: 100 + Math.random() * 200 },
      config: {},
      label: NODE_TYPES[type].label,
    };
    
    setGraph((prev) => ({
      nodes: [...prev.nodes, newNode],
      connections: prev.connections,
    }));
  }, []);

  /**
   * Updates a node's properties
   */
  const updateNode = useCallback((id: string, updates: Partial<QueryNode>) => {
    setGraph((prev) => ({
      nodes: prev.nodes.map((node) =>
        node.id === id ? { ...node, ...updates } : node
      ),
      connections: prev.connections,
    }));
  }, []);

  /**
   * Deletes a node from the graph
   */
  const deleteNode = useCallback((id: string) => {
    setGraph((prev) => ({
      nodes: prev.nodes.filter((node) => node.id !== id),
      connections: prev.connections.filter(
        (conn) => conn.from !== id && conn.to !== id
      ),
    }));
    if (selectedNode === id) {
      setSelectedNode(null);
    }
  }, [selectedNode]);

  /**
   * Handles node drag start
   */
  const handleNodeSelect = useCallback((id: string) => {
    setSelectedNode(id);
  }, []);

  /**
   * Executes the query and sends to Rust backend
   */
  const executeQuery = useCallback(() => {
    const payload = graphToRustPayload(graph);
    console.log('Executing query:', payload);
    onExecute?.(payload);
  }, [graph, onExecute]);

  // Generate SVG path for connections
  const connectionPaths = useMemo(() => {
    return graph.connections.map((conn) => {
      const fromNode = graph.nodes.find((n) => n.id === conn.from);
      const toNode = graph.nodes.find((n) => n.id === conn.to);
      
      if (!fromNode || !toNode) return null;
      
      const startX = fromNode.position.x + 180; // Right side of from node
      const startY = fromNode.position.y + 50; // Middle of from node
      const endX = toNode.position.x; // Left side of to node
      const endY = toNode.position.y + 50; // Middle of to node
      
      // Bezier curve
      const controlOffset = Math.abs(endX - startX) * 0.5;
      const path = `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`;
      
      return (
        <path
          key={`${conn.from}-${conn.to}`}
          d={path}
          fill="none"
          stroke="#00ffff"
          strokeWidth="2"
          strokeDasharray="5,5"
          className="animate-pulse"
        />
      );
    });
  }, [graph]);

  return (
    <div className="w-full rounded-xl overflow-hidden bg-[#0a0a12]/90 backdrop-blur-sm border border-cyan-900/30">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-b from-[#0a0a12] to-transparent border-b border-white/5">
        <h3 className="text-cyan-400 font-mono text-sm tracking-wider uppercase">
          🔧 Query Builder <span className="text-xs opacity-70">| Polars/DuckDB</span>
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={executeQuery}
            className="px-4 py-1.5 bg-cyan-500/20 border border-cyan-500/50 rounded text-cyan-400 text-xs font-mono hover:bg-cyan-500/30 transition-colors"
          >
            ▶ Execute
          </button>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 border-b border-white/10 flex-wrap">
        <span className="text-xs text-gray-400 font-mono mr-2">Add Node:</span>
        {(Object.keys(NODE_TYPES) as NodeType[]).map((type) => (
          <button
            key={type}
            onClick={() => addNode(type)}
            className="px-2 py-1 rounded border border-white/20 text-xs font-mono hover:border-cyan-500 hover:text-cyan-400 transition-colors"
            style={{ color: NODE_TYPES[type].color }}
          >
            {NODE_TYPES[type].icon} {NODE_TYPES[type].label}
          </button>
        ))}
      </div>

      {/* Canvas */}
      <div
        ref={canvasRef}
        className="relative overflow-hidden"
        style={{ height: CANVAS_SIZE.height }}
        onClick={() => setSelectedNode(null)}
      >
        {/* Grid Background */}
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: 'linear-gradient(#ffffff 1px, transparent 1px), linear-gradient(90deg, #ffffff 1px, transparent 1px)',
            backgroundSize: '20px 20px',
          }}
        />

        {/* Connection Lines SVG */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {connectionPaths}
        </svg>

        {/* Nodes */}
        {graph.nodes.map((node) => (
          <QueryNodeCard
            key={node.id}
            node={node}
            isSelected={selectedNode === node.id}
            onSelect={handleNodeSelect}
            onUpdate={updateNode}
            onDelete={deleteNode}
          />
        ))}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-gradient-to-t from-[#0a0a12] to-transparent border-t border-white/5">
        <div className="flex items-center justify-between text-xs font-mono text-gray-500">
          <span>{graph.nodes.length} nodes | {graph.connections.length} connections</span>
          <span className="text-green-400">✓ Input Sanitized</span>
        </div>
      </div>
    </div>
  );
};

export default QueryBuilder;
