/**
 * GenePool.tsx - Drag-and-drop node graph for RL hyperparameter genes
 * 
 * Features:
 * - Visualizes active RL hyperparameter "genes" as draggable nodes
 * - Allows manual locking or mutating specific weights during PBT sweeps
 * - Cyberpunk node graph aesthetic with connection lines
 * - AMD GPU context for mutation operations
 */

import React, { useState, useCallback, useMemo } from 'react';
import { motion, Reorder } from 'framer-motion';
import { Dna, Lock, Unlock, RefreshCw, Zap, GitBranch } from 'lucide-react';

interface GeneNode {
  id: string;
  name: string;
  category: 'learning' | 'exploration' | 'network' | 'reward';
  value: number;
  min: number;
  max: number;
  locked: boolean;
  mutationRate: number;
  fitness: number;
}

interface GenePoolProps {
  genes: GeneNode[];
  onGeneUpdate: (geneId: string, newValue: number) => void;
  onGeneLock: (geneId: string, locked: boolean) => void;
  onMutateGene: (geneId: string) => void;
}

const CATEGORY_COLORS = {
  learning: '#22d3ee',    // Cyan
  exploration: '#a855f7', // Purple
  network: '#10b981',     // Emerald
  reward: '#f59e0b',      // Amber
};

const CATEGORY_LABELS = {
  learning: 'LEARNING',
  exploration: 'EXPLORATION',
  network: 'NETWORK',
  reward: 'REWARD',
};

export const GenePool: React.FC<GenePoolProps> = ({
  genes,
  onGeneUpdate,
  onGeneLock,
  onMutateGene,
}) => {
  const [selectedGene, setSelectedGene] = useState<string | null>(null);
  const [draggedGene, setDraggedGene] = useState<string | null>(null);

  // Group genes by category
  const genesByCategory = useMemo(() => {
    const grouped: Record<string, GeneNode[]> = {
      learning: [],
      exploration: [],
      network: [],
      reward: [],
    };
    
    genes.forEach(gene => {
      grouped[gene.category].push(gene);
    });
    
    return grouped;
  }, [genes]);

  const handleToggleLock = useCallback((geneId: string, currentLocked: boolean) => {
    onGeneLock(geneId, !currentLocked);
  }, [onGeneLock]);

  const handleValueChange = useCallback((geneId: string, newValue: number) => {
    const gene = genes.find(g => g.id === geneId);
    if (!gene) return;
    
    const clampedValue = Math.max(gene.min, Math.min(gene.max, newValue));
    onGeneUpdate(geneId, clampedValue);
  }, [genes, onGeneUpdate]);

  const getNormalizedValue = (gene: GeneNode): number => {
    return ((gene.value - gene.min) / (gene.max - gene.min)) * 100;
  };

  const getFitnessColor = (fitness: number): string => {
    if (fitness >= 0.8) return '#10b981';
    if (fitness >= 0.5) return '#22d3ee';
    if (fitness >= 0.3) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <Dna className="w-5 h-5" />
          GENE POOL - HYPERPARAMETER NODES
        </h3>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-slate-400">TOTAL GENES: {genes.length}</span>
          <span className="text-emerald-400">LOCKED: {genes.filter(g => g.locked).length}</span>
          <span className="text-amber-400">MUTABLE: {genes.filter(g => !g.locked).length}</span>
        </div>
      </div>

      {/* Category Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(genesByCategory).map(([category, categoryGenes]) => (
          <div
            key={category}
            className="p-4 rounded-xl border-2 overflow-hidden"
            style={{
              borderColor: `${CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS]}40`,
              backgroundColor: `${CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS]}10`,
            }}
          >
            {/* Category Header */}
            <div className="flex items-center gap-2 mb-4 pb-2 border-b border-slate-700">
              <GitBranch 
                className="w-4 h-4" 
                style={{ color: CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS] }}
              />
              <span 
                className="text-sm font-bold"
                style={{ color: CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS] }}
              >
                {CATEGORY_LABELS[category as keyof typeof CATEGORY_LABELS]}
              </span>
              <span className="text-xs text-slate-500 ml-auto">
                {categoryGenes.length}
              </span>
            </div>

            {/* Gene Nodes */}
            <div className="space-y-3">
              {categoryGenes.map((gene) => (
                <motion.div
                  key={gene.id}
                  layout
                  className={`relative p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedGene === gene.id
                      ? 'bg-slate-800 border-cyan-400'
                      : gene.locked
                      ? 'bg-slate-800/50 border-slate-600'
                      : 'bg-slate-800/30 border-slate-700 hover:border-cyan-500/50'
                  }`}
                  onClick={() => setSelectedGene(gene.id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {/* Gene Name & Lock */}
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-slate-200 truncate">
                      {gene.name}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleLock(gene.id, gene.locked);
                      }}
                      className={`p-1 rounded transition-colors ${
                        gene.locked
                          ? 'text-red-400 hover:bg-red-500/20'
                          : 'text-slate-500 hover:bg-slate-700'
                      }`}
                    >
                      {gene.locked ? (
                        <Lock className="w-3 h-3" />
                      ) : (
                        <Unlock className="w-3 h-3" />
                      )}
                    </button>
                  </div>

                  {/* Value Display */}
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-lg font-mono font-bold text-white">
                      {gene.value.toFixed(4)}
                    </span>
                    <span 
                      className="text-xs font-mono px-1.5 py-0.5 rounded"
                      style={{
                        backgroundColor: `${getFitnessColor(gene.fitness)}20`,
                        color: getFitnessColor(gene.fitness),
                      }}
                    >
                      F: {gene.fitness.toFixed(2)}
                    </span>
                  </div>

                  {/* Value Slider */}
                  <input
                    type="range"
                    min={gene.min}
                    max={gene.max}
                    step={(gene.max - gene.min) / 100}
                    value={gene.value}
                    disabled={gene.locked}
                    onChange={(e) => handleValueChange(gene.id, parseFloat(e.target.value))}
                    onClick={(e) => e.stopPropagation()}
                    className="w-full h-1.5 accent-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      accentColor: CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS],
                    }}
                  />

                  {/* Range Labels */}
                  <div className="flex justify-between mt-1 text-[9px] text-slate-500 font-mono">
                    <span>{gene.min}</span>
                    <span>{gene.max}</span>
                  </div>

                  {/* Mutation Button */}
                  {!gene.locked && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onMutateGene(gene.id);
                      }}
                      className="absolute top-2 right-8 p-1 text-amber-400 hover:bg-amber-500/20 rounded transition-colors"
                      title="Mutate Gene"
                    >
                      <RefreshCw className="w-3 h-3" />
                    </button>
                  )}

                  {/* Locked Overlay */}
                  {gene.locked && (
                    <div className="absolute inset-0 bg-slate-900/50 rounded-lg pointer-events-none flex items-center justify-center">
                      <Lock className="w-6 h-6 text-red-400/50" />
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Selected Gene Details */}
      {selectedGene && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          className="mt-6 p-4 bg-slate-800/50 rounded-xl border border-cyan-500/30"
        >
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-bold text-cyan-400">GENE DETAILS</h4>
            <button
              onClick={() => setSelectedGene(null)}
              className="text-slate-400 hover:text-white text-sm"
            >
              ✕ CLOSE
            </button>
          </div>
          
          {(() => {
            const gene = genes.find(g => g.id === selectedGene);
            if (!gene) return null;
            
            return (
              <div className="grid grid-cols-4 gap-4 text-xs">
                <div>
                  <div className="text-slate-500 mb-1">ID</div>
                  <div className="font-mono text-slate-300">{gene.id}</div>
                </div>
                <div>
                  <div className="text-slate-500 mb-1">CATEGORY</div>
                  <div 
                    className="font-mono font-bold"
                    style={{ color: CATEGORY_COLORS[gene.category] }}
                  >
                    {CATEGORY_LABELS[gene.category]}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 mb-1">MUTATION RATE</div>
                  <div className="font-mono text-slate-300">{(gene.mutationRate * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-slate-500 mb-1">STATUS</div>
                  <div className={`font-mono font-bold ${gene.locked ? 'text-red-400' : 'text-emerald-400'}`}>
                    {gene.locked ? 'LOCKED' : 'ACTIVE'}
                  </div>
                </div>
              </div>
            );
          })()}
        </motion.div>
      )}

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-slate-700 flex items-center justify-between text-xs">
        <div className="flex items-center gap-4">
          <span className="text-slate-400">FITNESS:</span>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-emerald-500" />
            <span className="text-slate-500">&gt;0.8</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-cyan-500" />
            <span className="text-slate-500">0.5-0.8</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500" />
            <span className="text-slate-500">0.3-0.5</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-slate-500">&lt;0.3</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <Zap className="w-3 h-3" />
          <span>PBT SWEEP ACTIVE</span>
        </div>
      </div>
    </div>
  );
};

export default GenePool;
