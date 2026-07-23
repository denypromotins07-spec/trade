/**
 * MutationLog.tsx - Timeline Visualization of Evolutionary Breeding Events
 * 
 * Displays hyperparameter shifts and strategy mutations as the bot learns.
 * Uses Framer Motion for smooth entry animations.
 * 
 * Features:
 * - Animated timeline of mutation events
 * - Hyperparameter change visualization
 * - Fitness score tracking
 * - Generation counter
 * - Cyberpunk aesthetic with glowing evolution indicators
 */

import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSoulStore } from '../../store/soulStore';

interface MutationEvent {
  id: string;
  timestamp: number;
  generation: number;
  type: 'MUTATION' | 'CROSSOVER' | 'SELECTION' | 'ELITISM';
  description: string;
  paramsChanged: Record<string, { before: number; after: number }>;
  fitnessBefore: number;
  fitnessAfter: number;
  accepted: boolean;
}

const getTypeColor = (type: string): string => {
  switch (type) {
    case 'MUTATION': return '#00ffff';
    case 'CROSSOVER': return '#bd93f9';
    case 'SELECTION': return '#00ff88';
    case 'ELITISM': return '#ffaa00';
    default: return '#8b9bb4';
  }
};

const getTypeIcon = (type: string): string => {
  switch (type) {
    case 'MUTATION': return '🧬';
    case 'CROSSOVER': return '⚡';
    case 'SELECTION': return '🎯';
    case 'ELITISM': return '👑';
    default: return '•';
  }
};

interface MutationCardProps {
  event: MutationEvent;
  index: number;
}

const MutationCard: React.FC<MutationCardProps> = ({ event, index }) => {
  const typeColor = getTypeColor(event.type);
  const fitnessImprovement = ((event.fitnessAfter - event.fitnessBefore) / Math.abs(event.fitnessBefore || 1)) * 100;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -50, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 50, scale: 0.9 }}
      transition={{ 
        duration: 0.4, 
        delay: index * 0.05,
        type: 'spring',
        stiffness: 100,
      }}
      style={{
        position: 'relative',
        marginBottom: '12px',
        padding: '10px 14px',
        background: `linear-gradient(135deg, rgba(20, 30, 50, 0.8) 0%, rgba(10, 15, 30, 0.9) 100%)`,
        borderRadius: '6px',
        border: `1px solid ${typeColor}30`,
        borderLeft: `3px solid ${typeColor}`,
        boxShadow: event.accepted 
          ? `0 0 15px ${typeColor}15, inset 0 0 20px rgba(0, 0, 0, 0.3)`
          : 'none',
        opacity: event.accepted ? 1 : 0.6,
      }}
    >
      {/* Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '14px' }}>{getTypeIcon(event.type)}</span>
          <span
            style={{
              fontSize: '9px',
              fontWeight: 700,
              color: typeColor,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            {event.type}
          </span>
          <span
            style={{
              fontSize: '8px',
              color: 'rgba(139, 155, 180, 0.5)',
              padding: '2px 6px',
              background: 'rgba(139, 155, 180, 0.1)',
              borderRadius: '3px',
            }}
          >
            GEN {event.generation}
          </span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)' }}>
            {new Date(event.timestamp).toLocaleTimeString()}
          </span>
          {event.accepted && (
            <span
              style={{
                fontSize: '7px',
                color: '#00ff88',
                padding: '2px 4px',
                background: 'rgba(0, 255, 136, 0.1)',
                borderRadius: '3px',
              }}
            >
              ✓ ACCEPTED
            </span>
          )}
        </div>
      </div>

      {/* Description */}
      <p style={{ fontSize: '10px', color: '#c0c5ce', margin: '0 0 8px 0', lineHeight: 1.5 }}>
        {event.description}
      </p>

      {/* Parameters Changed */}
      {Object.keys(event.paramsChanged).length > 0 && (
        <div style={{ marginBottom: '8px' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.5)', marginBottom: '4px' }}>
            PARAMETERS MODIFIED
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {Object.entries(event.paramsChanged).map(([param, values]) => {
              const change = ((values.after - values.before) / Math.abs(values.before || 1)) * 100;
              const isIncrease = change >= 0;
              
              return (
                <div
                  key={param}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '3px 6px',
                    background: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '3px',
                    fontSize: '8px',
                    fontFamily: '"JetBrains Mono", monospace',
                  }}
                >
                  <span style={{ color: 'rgba(139, 155, 180, 0.7)' }}>{param}</span>
                  <span style={{ color: '#8b9bb4' }}>
                    {values.before.toFixed(3)}
                  </span>
                  <span style={{ color: isIncrease ? '#00ff88' : '#ff3366' }}>
                    {isIncrease ? '→' : '↓'}
                  </span>
                  <span style={{ color: typeColor, fontWeight: 600 }}>
                    {values.after.toFixed(3)}
                  </span>
                  <span
                    style={{
                      fontSize: '7px',
                      color: isIncrease ? '#00ff88' : '#ff3366',
                    }}
                  >
                    ({isIncrease ? '+' : ''}{change.toFixed(1)}%)
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Fitness Change */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingTop: '8px',
          borderTop: '1px solid rgba(139, 155, 180, 0.1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div>
            <span style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.5)' }}>FITNESS</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '10px', color: '#8b9bb4' }}>
                {event.fitnessBefore.toFixed(4)}
              </span>
              <span style={{ fontSize: '10px', color: fitnessImprovement >= 0 ? '#00ff88' : '#ff3366' }}>
                {fitnessImprovement >= 0 ? '→' : '↘'}
              </span>
              <span style={{ fontSize: '10px', color: typeColor, fontWeight: 600 }}>
                {event.fitnessAfter.toFixed(4)}
              </span>
            </div>
          </div>
        </div>
        
        <div
          style={{
            padding: '3px 8px',
            background: fitnessImprovement >= 0 
              ? 'rgba(0, 255, 136, 0.1)' 
              : 'rgba(255, 51, 102, 0.1)',
            borderRadius: '3px',
            fontSize: '8px',
            color: fitnessImprovement >= 0 ? '#00ff88' : '#ff3366',
            fontWeight: 600,
          }}
        >
          {fitnessImprovement >= 0 ? '+' : ''}{fitnessImprovement.toFixed(2)}%
        </div>
      </div>
    </motion.div>
  );
};

export const MutationLog: React.FC = () => {
  const { mutationEvents, currentGeneration, bestFitness } = useSoulStore();

  const sortedEvents = useMemo(() => {
    if (!mutationEvents) return [];
    return [...mutationEvents].sort((a, b) => b.timestamp - a.timestamp);
  }, [mutationEvents]);

  const stats = useMemo(() => {
    if (!mutationEvents || mutationEvents.length === 0) {
      return { accepted: 0, rejected: 0, avgImprovement: 0 };
    }
    
    const accepted = mutationEvents.filter(e => e.accepted).length;
    const improvements = mutationEvents.map(e => 
      ((e.fitnessAfter - e.fitnessBefore) / Math.abs(e.fitnessBefore || 1)) * 100
    );
    const avgImprovement = improvements.reduce((a, b) => a + b, 0) / improvements.length;
    
    return { accepted, rejected: mutationEvents.length - accepted, avgImprovement };
  }, [mutationEvents]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'linear-gradient(135deg, rgba(10, 15, 30, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%)',
        borderRadius: '8px',
        border: '1px solid rgba(0, 255, 255, 0.15)',
        boxShadow: '0 0 20px rgba(0, 255, 255, 0.05), inset 0 0 30px rgba(0, 0, 0, 0.3)',
        fontFamily: '"JetBrains Mono", monospace',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px 14px',
          borderBottom: '1px solid rgba(0, 255, 255, 0.2)',
          background: 'rgba(0, 255, 255, 0.03)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '14px' }}>🧬</span>
          <h3
            style={{
              margin: 0,
              fontSize: '12px',
              fontWeight: 600,
              color: '#00ffff',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              textShadow: '0 0 10px rgba(0, 255, 255, 0.5)',
            }}
          >
            Evolution Log
          </h3>
        </div>
        
        <div style={{ display: 'flex', gap: '12px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 8px',
              background: 'rgba(0, 255, 255, 0.1)',
              borderRadius: '4px',
            }}
          >
            <span style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.6)' }}>GEN</span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#00ffff' }}>
              {currentGeneration ?? 0}
            </span>
          </div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 8px',
              background: 'rgba(0, 255, 136, 0.1)',
              borderRadius: '4px',
            }}
          >
            <span style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.6)' }}>BEST</span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#00ff88' }}>
              {(bestFitness ?? 0).toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Stats Bar */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '8px',
          padding: '8px 14px',
          borderBottom: '1px solid rgba(139, 155, 180, 0.1)',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.5)' }}>ACCEPTED</div>
          <div style={{ fontSize: '12px', fontWeight: 700, color: '#00ff88' }}>
            {stats.accepted}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.5)' }}>REJECTED</div>
          <div style={{ fontSize: '12px', fontWeight: 700, color: '#ff3366' }}>
            {stats.rejected}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '7px', color: 'rgba(139, 155, 180, 0.5)' }}>AVG Δ</div>
          <div style={{ fontSize: '12px', fontWeight: 700, color: stats.avgImprovement >= 0 ? '#00ffff' : '#ff3366' }}>
            {stats.avgImprovement >= 0 ? '+' : ''}{stats.avgImprovement.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '14px',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(0, 255, 255, 0.3) transparent',
        }}
      >
        <AnimatePresence mode="popLayout">
          {sortedEvents.length > 0 ? (
            sortedEvents.map((event, idx) => (
              <MutationCard key={event.id} event={event} index={idx} />
            ))
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '150px',
                color: 'rgba(139, 155, 180, 0.4)',
              }}
            >
              <span style={{ fontSize: '32px', marginBottom: '12px', opacity: 0.3 }}>🧬</span>
              <p style={{ fontSize: '11px', textAlign: 'center' }}>
                No mutations yet...<br/>
                The evolutionary algorithm will log changes here.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default MutationLog;
