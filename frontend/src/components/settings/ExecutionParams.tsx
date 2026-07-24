/**
 * ExecutionParams.tsx - Cyberpunk-styled Configuration Sliders
 * 
 * Provides configuration sliders for TWAP/VWAP participation rates,
 * max slippage tolerances, and smart order routing venue preferences.
 * Designed with a sexy cyberpunk/quant aesthetic.
 * 
 * Features:
 * - Interactive range sliders with visual feedback
 * - TWAP/VWAP participation rate configuration
 * - Max slippage tolerance settings
 * - Smart order routing venue selection
 * - Real-time parameter validation
 */

import React, { useState, useCallback } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

interface ExecutionConfig {
  // TWAP Settings
  twapParticipationRate: number; // 0-100%
  twapDuration: number; // minutes
  twapSliceCount: number;
  
  // VWAP Settings
  vwapParticipationRate: number; // 0-100%
  vwapVolumeProfile: 'historical' | 'realtime' | 'hybrid';
  
  // Slippage Tolerance
  maxSlippageBps: number; // basis points (1 bp = 0.01%)
  slippageMode: 'fixed' | 'dynamic' | 'adaptive';
  
  // Smart Order Routing
  venues: VenueConfig[];
  routingStrategy: 'best_price' | 'lowest_fee' | 'fastest_fill' | 'balanced';
  darkPoolEnabled: boolean;
  
  // Risk Limits
  maxOrderSize: number; // USD
  dailyVolumeLimit: number; // USD
  positionSizeLimit: number; // % of portfolio
}

interface VenueConfig {
  id: string;
  name: string;
  enabled: boolean;
  priority: number;
  feeTier: number; // bps
  latency: number; // ms
}

interface ExecutionParamsProps {
  config: ExecutionConfig;
  onConfigChange?: (config: ExecutionConfig) => void;
  onApply?: () => void;
  onReset?: () => void;
  className?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Slider Component
// ─────────────────────────────────────────────────────────────────────────────

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
  color?: 'cyan' | 'purple' | 'green' | 'pink';
  showValue?: boolean;
  disabled?: boolean;
}

const CyberSlider: React.FC<SliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
  color = 'cyan',
  showValue = true,
  disabled = false
}) => {
  const percentage = ((value - min) / (max - min)) * 100;
  
  const colorClasses = {
    cyan: 'accent-cyan-500 from-cyan-600 to-cyan-400',
    purple: 'accent-purple-500 from-purple-600 to-purple-400',
    green: 'accent-green-500 from-green-600 to-green-400',
    pink: 'accent-pink-500 from-pink-600 to-pink-400'
  };
  
  const glowColors = {
    cyan: 'shadow-cyan-500/50',
    purple: 'shadow-purple-500/50',
    green: 'shadow-green-500/50',
    pink: 'shadow-pink-500/50'
  };
  
  return (
    <div className={`mb-4 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex items-center justify-between mb-2">
        <label className="text-xs font-mono text-gray-400">{label}</label>
        {showValue && (
          <span className={`text-sm font-mono font-bold bg-gradient-to-r ${colorClasses[color].split(' ').slice(1).join(' ')} bg-clip-text text-transparent`}>
            {value.toFixed(step < 1 ? 2 : 0)}{unit}
          </span>
        )}
      </div>
      
      <div className="relative h-8 flex items-center">
        {/* Track */}
        <div className="absolute w-full h-2 bg-gray-800 rounded-full overflow-hidden">
          {/* Fill */}
          <div 
            className={`h-full bg-gradient-to-r ${colorClasses[color].split(' ').slice(1).join(' ')} transition-all duration-100`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        
        {/* Range Input */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className={`absolute w-full h-2 opacity-0 cursor-pointer ${colorClasses[color].split(' ')[0]}`}
        />
        
        {/* Thumb Indicator */}
        <div 
          className={`absolute w-4 h-4 bg-white rounded-full shadow-lg ${glowColors[color]} pointer-events-none transition-all duration-100`}
          style={{ 
            left: `calc(${percentage}% - 8px)`,
            boxShadow: `0 0 10px var(--tw-shadow-color)`
          }}
        />
      </div>
      
      {/* Scale markers */}
      <div className="flex justify-between mt-1 text-xs text-gray-600 font-mono">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Toggle Switch Component
// ─────────────────────────────────────────────────────────────────────────────

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

const CyberToggle: React.FC<ToggleProps> = ({ label, checked, onChange, disabled = false }) => {
  return (
    <label className={`flex items-center justify-between py-2 cursor-pointer ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
      <span className="text-xs font-mono text-gray-400">{label}</span>
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
          disabled={disabled}
          className="sr-only"
        />
        <div className={`w-10 h-5 rounded-full transition-colors ${
          checked ? 'bg-cyan-600' : 'bg-gray-700'
        }`}>
          <div className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
            checked ? 'translate-x-5' : 'translate-x-0'
          }`} />
        </div>
      </div>
    </label>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const ExecutionParams: React.FC<ExecutionParamsProps> = ({
  config,
  onConfigChange,
  onApply,
  onReset,
  className = ''
}) => {
  const [localConfig, setLocalConfig] = useState<ExecutionConfig>(config);
  const [hasChanges, setHasChanges] = useState(false);

  // Update local config when prop changes
  React.useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  // Handle config change
  const handleChange = useCallback(<K extends keyof ExecutionConfig>(key: K, value: ExecutionConfig[K]) => {
    setLocalConfig(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  }, []);

  // Handle venue toggle
  const handleVenueToggle = useCallback((venueId: string, enabled: boolean) => {
    setLocalConfig(prev => ({
      ...prev,
      venues: prev.venues.map(v => v.id === venueId ? { ...v, enabled } : v)
    }));
    setHasChanges(true);
  }, []);

  // Handle venue priority change
  const handleVenuePriority = useCallback((venueId: string, priority: number) => {
    setLocalConfig(prev => ({
      ...prev,
      venues: prev.venues.map(v => v.id === venueId ? { ...v, priority } : v)
    }));
    setHasChanges(true);
  }, []);

  // Apply changes
  const handleApply = () => {
    onConfigChange?.(localConfig);
    onApply?.();
    setHasChanges(false);
  };

  // Reset to defaults
  const handleReset = () => {
    onReset?.();
    setHasChanges(false);
  };

  // Calculate estimated impact
  const estimatedImpact = {
    avgFillTime: `${(config.twapDuration / config.twapSliceCount).toFixed(1)}s per slice`,
    expectedSlippage: `${(config.maxSlippageBps / 100).toFixed(2)}% max`,
    venueCoverage: `${config.venues.filter(v => v.enabled).length}/${config.venues.length} venues active`
  };

  return (
    <div className={`p-6 ${className}`}>
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-cyan-400 font-mono">EXECUTION PARAMETERS</h2>
          <p className="text-gray-500 text-sm mt-1">TWAP/VWAP, slippage & smart routing</p>
        </div>
        
        {/* Change indicator */}
        {hasChanges && (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
            <span className="text-xs font-mono text-yellow-400">UNSAVED CHANGES</span>
          </div>
        )}
      </div>
      
      <div className="grid grid-cols-3 gap-6">
        {/* Left Column - TWAP/VWAP */}
        <div className="space-y-6">
          {/* TWAP Settings */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-mono text-cyan-400 mb-4 flex items-center gap-2">
              <span>⏱</span> TWAP SETTINGS
            </h3>
            
            <CyberSlider
              label="Participation Rate"
              value={localConfig.twapParticipationRate}
              min={1}
              max={100}
              step={1}
              unit="%"
              onChange={v => handleChange('twapParticipationRate', v)}
              color="cyan"
            />
            
            <CyberSlider
              label="Duration"
              value={localConfig.twapDuration}
              min={5}
              max={480}
              step={5}
              unit="m"
              onChange={v => handleChange('twapDuration', v)}
              color="purple"
            />
            
            <CyberSlider
              label="Slice Count"
              value={localConfig.twapSliceCount}
              min={10}
              max={100}
              step={1}
              unit=""
              onChange={v => handleChange('twapSliceCount', v)}
              color="green"
            />
          </div>
          
          {/* VWAP Settings */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-mono text-purple-400 mb-4 flex items-center gap-2">
              <span>📊</span> VWAP SETTINGS
            </h3>
            
            <CyberSlider
              label="Participation Rate"
              value={localConfig.vwapParticipationRate}
              min={1}
              max={50}
              step={1}
              unit="%"
              onChange={v => handleChange('vwapParticipationRate', v)}
              color="purple"
            />
            
            <div className="mt-4">
              <label className="block text-xs font-mono text-gray-400 mb-2">Volume Profile</label>
              <div className="flex gap-2">
                {(['historical', 'realtime', 'hybrid'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => handleChange('vwapVolumeProfile', mode)}
                    className={`flex-1 py-2 text-xs font-mono rounded transition-colors ${
                      localConfig.vwapVolumeProfile === mode
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    {mode.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
        
        {/* Middle Column - Slippage */}
        <div className="space-y-6">
          {/* Slippage Settings */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-mono text-pink-400 mb-4 flex items-center gap-2">
              <span>⚠️</span> SLIPPAGE TOLERANCE
            </h3>
            
            <CyberSlider
              label="Max Slippage"
              value={localConfig.maxSlippageBps}
              min={1}
              max={100}
              step={1}
              unit="bps"
              onChange={v => handleChange('maxSlippageBps', v)}
              color="pink"
            />
            
            <div className="mt-4">
              <label className="block text-xs font-mono text-gray-400 mb-2">Slippage Mode</label>
              <div className="flex gap-2">
                {(['fixed', 'dynamic', 'adaptive'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => handleChange('slippageMode', mode)}
                    className={`flex-1 py-2 text-xs font-mono rounded transition-colors ${
                      localConfig.slippageMode === mode
                        ? 'bg-pink-600 text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    {mode.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Slippage info */}
            <div className="mt-4 p-3 bg-pink-500/10 border border-pink-500/30 rounded">
              <div className="text-xs font-mono text-pink-400">
                Current setting allows up to {(localConfig.maxSlippageBps / 100).toFixed(2)}% price deviation
              </div>
            </div>
          </div>
          
          {/* Risk Limits */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-mono text-green-400 mb-4 flex items-center gap-2">
              <span>🛡️</span> RISK LIMITS
            </h3>
            
            <CyberSlider
              label="Max Order Size"
              value={localConfig.maxOrderSize}
              min={100}
              max={1000000}
              step={100}
              unit="$"
              onChange={v => handleChange('maxOrderSize', v)}
              color="green"
            />
            
            <CyberSlider
              label="Daily Volume Limit"
              value={localConfig.dailyVolumeLimit}
              min={1000}
              max={10000000}
              step={1000}
              unit="$"
              onChange={v => handleChange('dailyVolumeLimit', v)}
              color="green"
            />
            
            <CyberSlider
              label="Position Size Limit"
              value={localConfig.positionSizeLimit}
              min={1}
              max={100}
              step={1}
              unit="%"
              onChange={v => handleChange('positionSizeLimit', v)}
              color="green"
            />
          </div>
        </div>
        
        {/* Right Column - Smart Routing */}
        <div className="space-y-6">
          {/* Smart Order Routing */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-mono text-yellow-400 mb-4 flex items-center gap-2">
              <span>🔀</span> SMART ORDER ROUTING
            </h3>
            
            <div className="mb-4">
              <label className="block text-xs font-mono text-gray-400 mb-2">Routing Strategy</label>
              <select
                value={localConfig.routingStrategy}
                onChange={e => handleChange('routingStrategy', e.target.value as ExecutionConfig['routingStrategy'])}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:border-yellow-500 focus:outline-none"
              >
                <option value="best_price">Best Price</option>
                <option value="lowest_fee">Lowest Fee</option>
                <option value="fastest_fill">Fastest Fill</option>
                <option value="balanced">Balanced</option>
              </select>
            </div>
            
            <CyberToggle
              label="Dark Pool Enabled"
              checked={localConfig.darkPoolEnabled}
              onChange={v => handleChange('darkPoolEnabled', v)}
            />
            
            {/* Venues */}
            <div className="mt-4">
              <label className="block text-xs font-mono text-gray-400 mb-2">Trading Venues</label>
              <div className="space-y-2">
                {localConfig.venues.map(venue => (
                  <div
                    key={venue.id}
                    className={`flex items-center justify-between p-2 rounded ${
                      venue.enabled ? 'bg-gray-800' : 'bg-gray-800/50'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={venue.enabled}
                        onChange={e => handleVenueToggle(venue.id, e.target.checked)}
                        className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-yellow-500 focus:ring-yellow-500"
                      />
                      <span className={`text-xs font-mono ${venue.enabled ? 'text-white' : 'text-gray-500'}`}>
                        {venue.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-xs font-mono">
                      <span className="text-gray-500">{venue.feeTier}bps</span>
                      <span className="text-gray-500">{venue.latency}ms</span>
                      <input
                        type="number"
                        value={venue.priority}
                        onChange={e => handleVenuePriority(venue.id, parseInt(e.target.value))}
                        className="w-12 bg-gray-700 border border-gray-600 rounded px-1 text-center text-white"
                        min={1}
                        max={10}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Estimated Impact */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-mono text-gray-400 mb-3">ESTIMATED IMPACT</h3>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-gray-500">Avg Fill Time:</span>
                <span className="text-cyan-400">{estimatedImpact.avgFillTime}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Expected Slippage:</span>
                <span className="text-pink-400">{estimatedImpact.expectedSlippage}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Venue Coverage:</span>
                <span className="text-yellow-400">{estimatedImpact.venueCoverage}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Action Buttons */}
      <div className="mt-6 flex items-center justify-end gap-4">
        <button
          onClick={handleReset}
          className="px-6 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm font-mono text-gray-400 transition-colors"
        >
          RESET TO DEFAULTS
        </button>
        <button
          onClick={handleApply}
          disabled={!hasChanges}
          className={`px-6 py-2 rounded text-sm font-mono font-bold transition-all ${
            hasChanges
              ? 'bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 text-white shadow-lg shadow-cyan-500/30'
              : 'bg-gray-800 text-gray-600 cursor-not-allowed'
          }`}
        >
          APPLY CHANGES
        </button>
      </div>
    </div>
  );
};

export default ExecutionParams;
