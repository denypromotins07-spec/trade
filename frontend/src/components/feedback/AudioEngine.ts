/**
 * File 7: frontend/src/components/feedback/AudioEngine.ts
 * 
 * Elite Implementation:
 * - Web Audio API synthesizer for ultra-low latency auditory cues.
 * - Lazy initialization to respect 8GB RAM system limit.
 * - Generates subtle sounds for limit fills, circuit breakers, and alerts.
 * - Uses oscillators and gain nodes for minimal memory footprint.
 */

export type SoundType = 
  | 'LIMIT_FILL'
  | 'MARKET_FILL'
  | 'CIRCUIT_BREAKER'
  | 'ALERT_TRIGGER'
  | 'CONNECTION_LOST'
  | 'CONNECTION_RESTORED'
  | 'EXECUTION_COMPLETE';

interface SoundConfig {
  frequency: number;
  duration: number;
  type: OscillatorType;
  volume: number;
  envelope: {
    attack: number;
    decay: number;
    sustain: number;
    release: number;
  };
}

class AudioEngine {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private isInitialized: boolean = false;
  private isEnabled: boolean = true;
  private volume: number = 0.3;
  
  // Sound presets with cyberpunk aesthetic
  private readonly SOUND_PRESETS: Record<SoundType, SoundConfig> = {
    LIMIT_FILL: {
      frequency: 880,
      duration: 0.15,
      type: 'sine',
      volume: 0.4,
      envelope: { attack: 0.01, decay: 0.05, sustain: 0.3, release: 0.09 },
    },
    MARKET_FILL: {
      frequency: 1200,
      duration: 0.1,
      type: 'triangle',
      volume: 0.5,
      envelope: { attack: 0.005, decay: 0.03, sustain: 0.2, release: 0.065 },
    },
    CIRCUIT_BREAKER: {
      frequency: 220,
      duration: 0.5,
      type: 'sawtooth',
      volume: 0.6,
      envelope: { attack: 0.01, decay: 0.1, sustain: 0.5, release: 0.29 },
    },
    ALERT_TRIGGER: {
      frequency: 1760,
      duration: 0.2,
      type: 'square',
      volume: 0.4,
      envelope: { attack: 0.01, decay: 0.05, sustain: 0.3, release: 0.14 },
    },
    CONNECTION_LOST: {
      frequency: 330,
      duration: 0.3,
      type: 'sine',
      volume: 0.3,
      envelope: { attack: 0.05, decay: 0.1, sustain: 0.2, release: 0.15 },
    },
    CONNECTION_RESTORED: {
      frequency: 660,
      duration: 0.2,
      type: 'sine',
      volume: 0.3,
      envelope: { attack: 0.02, decay: 0.05, sustain: 0.3, release: 0.13 },
    },
    EXECUTION_COMPLETE: {
      frequency: 440,
      duration: 0.25,
      type: 'triangle',
      volume: 0.4,
      envelope: { attack: 0.01, decay: 0.08, sustain: 0.4, release: 0.16 },
    },
  };

  /**
   * Lazy initialize audio context (must be called after user interaction)
   */
  public async initialize(): Promise<boolean> {
    if (this.isInitialized) return true;

    try {
      // Check if AudioContext is supported
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextClass) {
        console.warn('[AudioEngine] Web Audio API not supported');
        return false;
      }

      this.context = new AudioContextClass();
      this.masterGain = this.context.createGain();
      this.masterGain.connect(this.context.destination);
      this.masterGain.gain.value = this.volume;
      
      this.isInitialized = true;
      console.log('[AudioEngine] Initialized successfully');
      return true;
    } catch (error) {
      console.error('[AudioEngine] Initialization failed:', error);
      return false;
    }
  }

  /**
   * Play a sound by type
   */
  public play(soundType: SoundType): void {
    if (!this.isEnabled || !this.isInitialized || !this.context || !this.masterGain) {
      return;
    }

    const config = this.SOUND_PRESETS[soundType];
    if (!config) {
      console.warn(`[AudioEngine] Unknown sound type: ${soundType}`);
      return;
    }

    const now = this.context.currentTime;
    
    // Create oscillator
    const oscillator = this.context.createOscillator();
    oscillator.type = config.type;
    oscillator.frequency.setValueAtTime(config.frequency, now);

    // Create gain node for envelope
    const gainNode = this.context.createGain();
    gainNode.connect(this.masterGain);

    // Apply ADSR envelope
    const { attack, decay, sustain, release } = config.envelope;
    const peakVolume = config.volume * this.volume;
    const sustainVolume = peakVolume * sustain;

    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(peakVolume, now + attack);
    gainNode.gain.linearRampToValueAtTime(sustainVolume, now + attack + decay);
    gainNode.gain.setValueAtTime(sustainVolume, now + config.duration - release);
    gainNode.gain.linearRampToValueAtTime(0, now + config.duration);

    // Connect and start
    oscillator.connect(gainNode);
    oscillator.start(now);
    oscillator.stop(now + config.duration);

    // Cleanup
    setTimeout(() => {
      oscillator.disconnect();
      gainNode.disconnect();
    }, config.duration * 1000 + 100);
  }

  /**
   * Play a sequence of sounds (for complex alerts)
   */
  public playSequence(soundTypes: SoundType[], interval: number = 150): void {
    soundTypes.forEach((type, index) => {
      setTimeout(() => this.play(type), index * interval);
    });
  }

  /**
   * Set master volume
   */
  public setVolume(level: number): void {
    this.volume = Math.max(0, Math.min(1, level));
    if (this.masterGain) {
      this.masterGain.gain.value = this.volume;
    }
  }

  /**
   * Enable/disable audio
   */
  public setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    if (!enabled && this.context?.state === 'running') {
      this.context.suspend();
    } else if (enabled && this.context?.state === 'suspended') {
      this.context.resume();
    }
  }

  /**
   * Get initialization status
   */
  public getIsInitialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Get enabled status
   */
  public getIsEnabled(): boolean {
    return this.isEnabled;
  }

  /**
   * Cleanup resources
   */
  public dispose(): void {
    if (this.context) {
      this.context.close();
      this.context = null;
      this.masterGain = null;
      this.isInitialized = false;
    }
  }
}

// Singleton instance with lazy initialization
let audioEngineInstance: AudioEngine | null = null;

export const getAudioEngine = (): AudioEngine => {
  if (!audioEngineInstance) {
    audioEngineInstance = new AudioEngine();
  }
  return audioEngineInstance;
};

export default getAudioEngine;
