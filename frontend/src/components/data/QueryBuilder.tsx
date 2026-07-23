/**
 * File 10: QueryBuilder.tsx
 * Chapter 4: Data Exploration & Custom Query Builder
 * 
 * Visual node-based UI to construct complex Polars and DuckDB analytical queries,
 * translating visual graphs into strict Rust IPC payloads.
 * 
 * Features:
 * - Node-based visual query builder
 * - Input sanitization (prevents SQL/DuckDB injection)
 * - Real-time query preview
 * - Rust IPC payload generation
 */

import React, { useState, useCallback, useMemo } from 'react';

// --- Types ---

interface Node {
  id: string;
  type: 'source' | 'filter' | 'aggregate' | 'transform' | 'output';
  position: { x: number; y: number };
  config: Record<string, unknown>;
}

interface Edge {
  source: string;
  target: string;
}

interface QueryGraph {
  nodes: Node[];
  edges: Edge[];
}

interface Props {
  onGenerate?: (payload: RustPayload) => void;
}

interface RustPayload {
  query_type: 'polars' | 'duckdb';
  operations: Operation[];
  sanitized: boolean;
  checksum: string;
}

interface Operation {
  op: string;
  params: Record<string, unknown>;
}

// --- Constants ---

const NODE_TYPES = {
  source: { label: 'DATA_SOURCE', color: '#00f3ff', icon: '📊' },
  filter: { label: 'FILTER', color: '#00ff9d', icon: '🔍' },
  aggregate: { label: 'AGGREGATE', color: '#ffaa00', icon: '∑' },
  transform: { label: 'TRANSFORM', color: '#bd00ff', icon: '🔄' },
  output: { label: 'OUTPUT', color: '#ff0055', icon: '📤' },
};

const COLORS = {
  bg: '#0a0a0a',
  panel: 'rgba(20, 20, 30, 0.8)',
  border: '#333333',
  text: '#c0c0c0',
  highlight: '#00f3ff',
  error: '#ff0055',
};

/**
 * Sanitize user input to prevent injection attacks
 */
const sanitizeInput = (input: string): string => {
  const dangerous = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'TRUNCATE', 'ALTER'];
  let sanitized = input.toUpperCase();
  
  for (const pattern of dangerous) {
    if (sanitized.includes(pattern)) {
      console.warn(`[SECURITY] Blocked potential injection: ${pattern}`);
      return '';
    }
  }
  
  return input.replace(/[^a-zA-Z0-9_.,\s*+\-/%=<>!]/g, '');
};

/**
 * Generate a checksum for the query
 */
const generateChecksum = (ops: Operation[]): string => {
  const str = JSON.stringify(ops);
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
};

/**
 * Convert graph to Rust IPC payload
 */
const graphToPayload = (graph: QueryGraph): RustPayload => {
  const operations: Operation[] = [];
  const sortedNodes = [...graph.nodes].sort((a, b) => a.position.x - b.position.x);
  
  for (const node of sortedNodes) {
    const op: Operation = { op: node.type, params: {} };
    
    switch (node.type) {
      case 'source':
        op.params.table = sanitizeInput(String(node.config.table || ''));
        op.params.columns = Array.isArray(node.config.columns) 
          ? node.config.columns.map((c: string) => sanitizeInput(c))
          : ['*'];
        break;
      case 'filter':
        op.params.column = sanitizeInput(String(node.config.column || ''));
        op.params.operator = ['=', '!=', '>', '<', '>=', '<=', 'LIKE'].includes(String(node.config.operator))
          ? String(node.config.operator)
          : '=';
        op.params.value = sanitizeInput(String(node.config.value || ''));
        break;
      case 'aggregate':
        op.params.function = ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX'].includes(String(node.config.function))
          ? String(node.config.function)
          : 'SUM';
        op.params.column = sanitizeInput(String(node.config.column || ''));
        op.params.groupBy = Array.isArray(node.config.groupBy)
          ? node.config.groupBy.map((c: string) => sanitizeInput(c))
          : [];
        break;
      case 'transform':
        op.params.expression = sanitizeInput(String(node.config.expression || ''));
        op.params.alias = sanitizeInput(String(node.config.alias || 'result'));
        break;
      case 'output':
        op.params.format = ['parquet', 'csv', 'json'].includes(String(node.config.format))
          ? String(node.config.format)
          : 'parquet';
        op.params.path = sanitizeInput(String(node.config.path || '/tmp/output'));
        break;
    }
    
    operations.push(op);
  }
  
  return {
    query_type: 'polars',
    operations,
    sanitized: true,
    checksum: generateChecksum(operations),
  };
};

/**
 * QueryBuilder Component
 */
export const QueryBuilder: React.FC<Props> = ({ onGenerate }) => {
  const [graph, setGraph] = useState<QueryGraph>({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [preview, setPreview] = useState<RustPayload | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const addNode = useCallback((type: Node['type']) => {
    const newNode: Node = {
      id: `node_${Date.now()}`,
      type,
      position: { x: 100 + graph.nodes.length * 150, y: 200 },
      config: {},
    };
    setGraph((prev) => ({
      nodes: [...prev.nodes, newNode],
      edges: prev.edges,
    }));
    setSelectedNode(newNode.id);
  }, [graph.nodes.length]);

  const updateNodeConfig = useCallback((nodeId: string, config: Record<string, unknown>) => {
    setGraph((prev) => ({
      nodes: prev.nodes.map((n) =>
        n.id === nodeId ? { ...n, config: { ...n.config, ...config } } : n
      ),
      edges: prev.edges,
    }));
  }, []);

  const deleteNode = useCallback((nodeId: string) => {
    setGraph((prev) => ({
      nodes: prev.nodes.filter((n) => n.id !== nodeId),
      edges: prev.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
    }));
    if (selectedNode === nodeId) setSelectedNode(null);
  }, [selectedNode]);

  const handleGenerate = useCallback(() => {
    try {
      const payload = graphToPayload(graph);
      setPreview(payload);
      setValidationError(null);
      onGenerate?.(payload);
    } catch (error) {
      setValidationError('Failed to generate query payload');
    }
  }, [graph, onGenerate]);

  const selectedNodeData = useMemo(
    () => graph.nodes.find((n) => n.id === selectedNode),
    [graph.nodes, selectedNode]
  );

  return (
    <div className="p-4 bg-black/80 backdrop-blur-md border border-cyan-900/50 rounded-xl h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-mono font-bold text-white tracking-wider">QUERY_BUILDER_V4</h3>
        <button
          onClick={handleGenerate}
          disabled={graph.nodes.length === 0}
          className="px-4 py-1.5 text-xs font-mono font-bold bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded transition-colors"
        >
          GENERATE_PAYLOAD
        </button>
      </div>

      <div className="flex-1 flex gap-4 overflow-hidden">
        <div className="w-48 flex-shrink-0 p-3 bg-gray-900/50 rounded-lg border border-gray-800">
          <div className="text-[10px] font-mono text-gray-500 mb-2">NODE_PALETTE</div>
          <div className="space-y-2">
            {Object.entries(NODE_TYPES).map(([type, config]) => (
              <button
                key={type}
                onClick={() => addNode(type as Node['type'])}
                className="w-full flex items-center gap-2 px-2 py-2 rounded border border-gray-700 hover:border-cyan-500 transition-colors group"
              >
                <span className="text-lg">{config.icon}</span>
                <span className="text-[10px] font-mono text-gray-400 group-hover:text-cyan-400">{config.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 relative bg-gray-900/30 rounded-lg border border-gray-800 overflow-auto">
          {graph.nodes.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <div className="text-3xl mb-2">📐</div>
                <div className="text-[10px] font-mono">DRAG NODES FROM PALETTE</div>
              </div>
            </div>
          ) : (
            <div className="p-8 min-w-max min-h-max">
              {graph.nodes.map((node) => {
                const config = NODE_TYPES[node.type];
                const isSelected = selectedNode === node.id;
                return (
                  <div
                    key={node.id}
                    onClick={() => setSelectedNode(node.id)}
                    className={`absolute p-3 rounded-lg border cursor-pointer transition-all ${isSelected ? 'ring-2 ring-cyan-500 scale-105' : ''}`}
                    style={{
                      left: node.position.x,
                      top: node.position.y,
                      backgroundColor: COLORS.panel,
                      borderColor: isSelected ? config.color : COLORS.border,
                      boxShadow: `0 0 10px ${config.color}20`,
                    }}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-lg">{config.icon}</span>
                      <span className="text-[10px] font-mono font-bold" style={{ color: config.color }}>{config.label}</span>
                    </div>
                    <div className="text-[9px] font-mono text-gray-500">
                      {Object.keys(node.config).length > 0
                        ? Object.entries(node.config).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(', ')
                        : '(unconfigured)'}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteNode(node.id);
                      }}
                      className="absolute -top-2 -right-2 w-5 h-5 bg-red-900/80 hover:bg-red-700 rounded-full text-[10px] text-white flex items-center justify-center"
                    >
                      ×
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="w-64 flex-shrink-0 p-3 bg-gray-900/50 rounded-lg border border-gray-800 overflow-y-auto">
          <div className="text-[10px] font-mono text-gray-500 mb-2">NODE_CONFIG</div>
          {selectedNodeData ? (
            <div className="space-y-3">
              <div className="text-xs font-mono font-bold text-white mb-2">{NODE_TYPES[selectedNodeData.type].label}</div>
              
              {selectedNodeData.type === 'source' && (
                <>
                  <input type="text" placeholder="Table name" value={String(selectedNodeData.config.table || '')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { table: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none" />
                  <input type="text" placeholder="Columns (comma-separated)"
                    value={Array.isArray(selectedNodeData.config.columns) ? selectedNodeData.config.columns.join(', ') : ''}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { columns: e.target.value.split(',').map((c) => c.trim()) })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none" />
                </>
              )}
              
              {selectedNodeData.type === 'filter' && (
                <>
                  <input type="text" placeholder="Column" value={String(selectedNodeData.config.column || '')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { column: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none" />
                  <select value={String(selectedNodeData.config.operator || '=')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { operator: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none">
                    <option value="=">=</option><option value="!=">!=</option><option value=">">&gt;</option>
                    <option value="<">&lt;</option><option value=">=">&gt;=</option><option value="<=">&lt;=</option>
                  </select>
                  <input type="text" placeholder="Value" value={String(selectedNodeData.config.value || '')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { value: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none" />
                </>
              )}

              {selectedNodeData.type === 'aggregate' && (
                <>
                  <select value={String(selectedNodeData.config.function || 'SUM')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { function: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none">
                    <option value="SUM">SUM</option><option value="AVG">AVG</option><option value="COUNT">COUNT</option>
                    <option value="MIN">MIN</option><option value="MAX">MAX</option>
                  </select>
                  <input type="text" placeholder="Column" value={String(selectedNodeData.config.column || '')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { column: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none" />
                </>
              )}

              {selectedNodeData.type === 'output' && (
                <>
                  <select value={String(selectedNodeData.config.format || 'parquet')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { format: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none">
                    <option value="parquet">Parquet</option><option value="csv">CSV</option><option value="json">JSON</option>
                  </select>
                  <input type="text" placeholder="Output path" value={String(selectedNodeData.config.path || '')}
                    onChange={(e) => updateNodeConfig(selectedNodeData.id, { path: e.target.value })}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white focus:border-cyan-500 focus:outline-none" />
                </>
              )}

              <button onClick={() => deleteNode(selectedNodeData.id)}
                className="w-full mt-4 px-2 py-1 bg-red-900/50 hover:bg-red-800 border border-red-700 rounded text-xs font-mono text-red-400 transition-colors">
                DELETE_NODE
              </button>
            </div>
          ) : (
            <div className="text-[10px] font-mono text-gray-600 text-center py-8">SELECT A NODE TO CONFIGURE</div>
          )}
        </div>
      </div>

      {preview && (
        <div className="mt-4 p-3 bg-gray-900/80 rounded-lg border border-cyan-900/50">
          <div className="flex justify-between items-center mb-2">
            <div className="text-[10px] font-mono text-cyan-400">RUST_IPC_PAYLOAD_PREVIEW</div>
            <div className="text-[9px] font-mono text-gray-500">CHECKSUM: {preview.checksum}</div>
          </div>
          <pre className="text-[9px] font-mono text-gray-300 overflow-x-auto whitespace-pre-wrap">{JSON.stringify(preview, null, 2)}</pre>
        </div>
      )}

      {validationError && (
        <div className="mt-2 px-3 py-2 bg-red-900/30 border border-red-700 rounded text-[10px] font-mono text-red-400">⚠️ {validationError}</div>
      )}
    </div>
  );
};

export default QueryBuilder;
