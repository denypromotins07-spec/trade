/**
 * RiskExposure.tsx - Real-time Portfolio Risk Metrics Dashboard
 * 
 * Displays portfolio beta, delta, and cross-asset correlation matrix
 * rendered via CSS transforms. Highlights concentration risks in neon red.
 * 
 * Features:
 * - Real-time Greeks (Delta, Gamma, Theta, Vega)
 * - Cross-asset correlation heatmap using CSS grid
 * - Concentration risk alerts with visual indicators
 * - Cyberpunk aesthetic with glowing risk thresholds
 */

import React, { useMemo } from 'react';
import { useMetricsStore } from '../../store/metricsStore';

interface RiskMetricCardProps {
  label: string;
  value: number;
  threshold: number;
  unit?: string;
  colorGood: string;
  colorBad: string;
}

const RiskMetricCard: React.FC<RiskMetricCardProps> = ({
  label,
  value,
  threshold,
  unit = '',
  colorGood,
  colorBad,
}) => {
  const isRisky = Math.abs(value) > threshold;
  const displayColor = isRisky ? colorBad : colorGood;
  
  return (
    <div
      style={{
        background: `linear-gradient(135deg, rgba(20, 30, 50, 0.9) 0%, rgba(10, 15, 30, 0.95) 100%)`,
        borderRadius: '6px',
        padding: '10px 12px',
        border: `1px solid ${isRisky ? 'rgba(255, 51, 102, 0.4)' : 'rgba(0, 255, 136, 0.2)'}`,
        boxShadow: isRisky 
          ? '0 0 15px rgba(255, 51, 102, 0.2), inset 0 0 20px rgba(0, 0, 0, 0.3)'
          : '0 0 10px rgba(0, 255, 136, 0.1), inset 0 0 20px rgba(0, 0, 0, 0.3)',
        transition: 'all 0.2s ease',
      }}
    >
      <div
        style={{
          fontSize: '8px',
          color: 'rgba(139, 155, 180, 0.7)',
          marginBottom: '4px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: '16px',
          fontWeight: 700,
          color: displayColor,
          fontFamily: '"JetBrains Mono", monospace',
          textShadow: `0 0 8px ${displayColor}40`,
        }}
      >
        {value >= 0 ? '+' : ''}{value.toFixed(3)}{unit}
      </div>
      {isRisky && (
        <div
          style={{
            fontSize: '7px',
            color: '#ff3366',
            marginTop: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}
        >
          <span style={{ animation: 'pulse 1s infinite' }}>⚠</span>
          THRESHOLD EXCEEDED
        </div>
      )}
    </div>
  );
};

interface CorrelationMatrixProps {
  assets: string[];
  correlations: number[][];
}

const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({ assets, correlations }) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `40px repeat(${assets.length}, 1fr)`,
        gap: '2px',
        marginTop: '8px',
      }}
    >
      {/* Header row */}
      <div />
      {assets.map((asset) => (
        <div
          key={`header-${asset}`}
          style={{
            fontSize: '8px',
            color: '#8b9bb4',
            textAlign: 'center',
            padding: '4px 2px',
            fontFamily: '"JetBrains Mono", monospace',
            writingMode: 'vertical-rl',
            transform: 'rotate(180deg)',
          }}
        >
          {asset}
        </div>
      ))}

      {/* Matrix rows */}
      {assets.map((rowAsset, rowIdx) => (
        <React.Fragment key={`row-${rowAsset}`}>
          {/* Row label */}
          <div
            style={{
              fontSize: '8px',
              color: '#8b9bb4',
              textAlign: 'right',
              padding: '4px 6px',
              fontFamily: '"JetBrains Mono", monospace',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
            }}
          >
            {rowAsset}
          </div>
          
          {/* Correlation cells */}
          {assets.map((colAsset, colIdx) => {
            const corr = correlations[rowIdx]?.[colIdx] ?? 0;
            const intensity = Math.abs(corr);
            const isPositive = corr >= 0;
            
            // Color based on correlation strength and direction
            const bgColor = isPositive
              ? `rgba(0, 255, 136, ${intensity * 0.4})`
              : `rgba(255, 51, 102, ${intensity * 0.4})`;
            
            const borderColor = isPositive
              ? `rgba(0, 255, 136, ${0.2 + intensity * 0.3})`
              : `rgba(255, 51, 102, ${0.2 + intensity * 0.3})`;

            return (
              <div
                key={`cell-${rowAsset}-${colAsset}`}
                style={{
                  aspectRatio: '1',
                  background: bgColor,
                  border: `1px solid ${borderColor}`,
                  borderRadius: '2px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '7px',
                  fontFamily: '"JetBrains Mono", monospace',
                  color: isPositive ? '#00ff88' : '#ff3366',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                title={`${rowAsset} ↔ ${colAsset}: ${corr.toFixed(3)}`}
              >
                {rowIdx === colIdx ? '1.0' : corr.toFixed(2)}
              </div>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );
};

export const RiskExposure: React.FC = () => {
  const { 
    portfolioDelta, 
    portfolioGamma, 
    portfolioTheta, 
    portfolioVega,
    portfolioBeta,
    assetCorrelations,
    concentrationRisk,
    var95,
    expectedShortfall,
  } = useMetricsStore();

  const assets = useMemo(() => {
    if (!assetCorrelations?.assets) return ['BTC', 'ETH', 'SOL'];
    return assetCorrelations.assets;
  }, [assetCorrelations]);

  const correlations = useMemo(() => {
    if (!assetCorrelations?.matrix) {
      // Default correlation matrix
      return [
        [1.0, 0.72, 0.58],
        [0.72, 1.0, 0.65],
        [0.58, 0.65, 1.0],
      ];
    }
    return assetCorrelations.matrix;
  }, [assetCorrelations]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        padding: '12px',
        background: 'linear-gradient(135deg, rgba(10, 15, 30, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%)',
        borderRadius: '8px',
        border: '1px solid rgba(255, 170, 0, 0.15)',
        boxShadow: '0 0 20px rgba(255, 170, 0, 0.05), inset 0 0 30px rgba(0, 0, 0, 0.3)',
        fontFamily: '"JetBrains Mono", monospace',
        minHeight: '320px',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingBottom: '8px',
          borderBottom: '1px solid rgba(255, 170, 0, 0.2)',
        }}
      >
        <h3
          style={{
            margin: 0,
            fontSize: '12px',
            fontWeight: 600,
            color: '#ffaa00',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            textShadow: '0 0 10px rgba(255, 170, 0, 0.5)',
          }}
        >
          📊 Risk Exposure Dashboard
        </h3>
        {concentrationRisk && concentrationRisk > 0.3 && (
          <div
            style={{
              padding: '4px 8px',
              background: 'rgba(255, 51, 102, 0.2)',
              border: '1px solid rgba(255, 51, 102, 0.4)',
              borderRadius: '4px',
              fontSize: '8px',
              color: '#ff3366',
              animation: 'pulse 2s infinite',
            }}
          >
            ⚠ HIGH CONCENTRATION
          </div>
        )}
      </div>

      {/* Greeks Grid */}
      <div>
        <div
          style={{
            fontSize: '9px',
            color: '#64ffda',
            marginBottom: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          Portfolio Greeks
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '8px',
          }}
        >
          <RiskMetricCard
            label="Delta"
            value={portfolioDelta ?? 0}
            threshold={0.5}
            colorGood="#00ff88"
            colorBad="#ff3366"
          />
          <RiskMetricCard
            label="Gamma"
            value={portfolioGamma ?? 0}
            threshold={0.1}
            colorGood="#00ffff"
            colorBad="#ff3366"
          />
          <RiskMetricCard
            label="Theta"
            value={portfolioTheta ?? 0}
            threshold={0.05}
            unit="/d"
            colorGood="#bd93f9"
            colorBad="#ffaa00"
          />
          <RiskMetricCard
            label="Vega"
            value={portfolioVega ?? 0}
            threshold={0.2}
            colorGood="#00ff88"
            colorBad="#ff3366"
          />
        </div>
      </div>

      {/* Beta & VaR */}
      <div>
        <div
          style={{
            fontSize: '9px',
            color: '#64ffda',
            marginBottom: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          Market Risk Metrics
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '8px',
          }}
        >
          <RiskMetricCard
            label="Beta (vs BTC)"
            value={portfolioBeta ?? 1}
            threshold={1.5}
            colorGood="#00ffff"
            colorBad="#ff3366"
          />
          <RiskMetricCard
            label="VaR (95%)"
            value={-(var95 ?? 0)}
            threshold={0.05}
            unit="%"
            colorGood="#00ff88"
            colorBad="#ff3366"
          />
          <RiskMetricCard
            label="Exp. Shortfall"
            value={-(expectedShortfall ?? 0)}
            threshold={0.08}
            unit="%"
            colorGood="#bd93f9"
            colorBad="#ffaa00"
          />
        </div>
      </div>

      {/* Correlation Matrix */}
      <div>
        <div
          style={{
            fontSize: '9px',
            color: '#64ffda',
            marginBottom: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>Cross-Asset Correlation</span>
          <span
            style={{
              fontSize: '7px',
              color: 'rgba(139, 155, 180, 0.5)',
            }}
          >
            (+) Positive | (-) Negative
          </span>
        </div>
        <CorrelationMatrix assets={assets} correlations={correlations} />
      </div>

      {/* Concentration Warning */}
      {concentrationRisk && (
        <div
          style={{
            marginTop: 'auto',
            paddingTop: '8px',
            borderTop: '1px solid rgba(139, 155, 180, 0.2)',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span
              style={{
                fontSize: '9px',
                color: 'rgba(139, 155, 180, 0.7)',
              }}
            >
              TOP POSITION CONCENTRATION
            </span>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <div
                style={{
                  width: '100px',
                  height: '6px',
                  background: 'rgba(20, 30, 48, 0.8)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${Math.min(concentrationRisk * 100, 100)}%`,
                    height: '100%',
                    background: `linear-gradient(90deg, 
                      ${concentrationRisk > 0.3 ? '#ff3366' : concentrationRisk > 0.2 ? '#ffaa00' : '#00ff88'}, 
                      ${concentrationRisk > 0.3 ? '#ff6688' : '#ffcc00'})`,
                    boxShadow: `0 0 6px ${concentrationRisk > 0.3 ? '#ff3366' : '#ffaa00'}`,
                  }}
                />
              </div>
              <span
                style={{
                  fontSize: '10px',
                  color: concentrationRisk > 0.3 ? '#ff3366' : concentrationRisk > 0.2 ? '#ffaa00' : '#00ff88',
                  minWidth: '35px',
                  textAlign: 'right',
                }}
              >
                {(concentrationRisk * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RiskExposure;
