/**
 * File 8: frontend/src/components/feedback/HapticPulse.ts
 * 
 * Elite Implementation:
 * - Gamepad Vibration API for physical haptic feedback.
 * - Triggers on high-risk institutional block executions.
 * - Lazy initialization to respect system RAM limits.
 * - Pattern-based vibration for different event types.
 */

export type HapticPattern = 
  | 'LIGHT_PULSE'
  | 'MEDIUM_PULSE'
  | 'HEAVY_PULSE'
  | 'DOUBLE_TAP'
  | 'RAPID_FIRE'
  | 'INSTITUTIONAL_BLOCK';

interface VibrationPattern {
  duration: number;
  weakMagnitude: number;
  strongMagnitude: number;
}

class HapticEngine {
  private gamepads: Map<number, Gamepad> = new Map();
  private isEnabled: boolean = true;
  private isInitialized: boolean = false;
  private pollInterval: NodeJS.Timeout | null = null;
  
  // Vibration presets
  private readonly PATTERNS: Record<HapticPattern, VibrationPattern> = {
    LIGHT_PULSE: {
      duration: 100,
      weakMagnitude: 0.3,
      strongMagnitude: 0,
    },
    MEDIUM_PULSE: {
      duration: 200,
      weakMagnitude: 0.5,
      strongMagnitude: 0.2,
    },
    HEAVY_PULSE: {
      duration: 300,
      weakMagnitude: 0.7,
      strongMagnitude: 0.5,
    },
    DOUBLE_TAP: {
      duration: 150,
      weakMagnitude: 0.6,
      strongMagnitude: 0.4,
    },
    RAPID_FIRE: {
      duration: 80,
      weakMagnitude: 0.8,
      strongMagnitude: 0.6,
    },
    INSTITUTIONAL_BLOCK: {
      duration: 500,
      weakMagnitude: 1.0,
      strongMagnitude: 1.0,
    },
  };

  constructor() {
    this.init();
  }

  private init() {
    if (typeof navigator === 'undefined' || !navigator.getGamepads) {
      console.warn('[HapticEngine] Gamepad API not supported');
      return;
    }

    // Listen for gamepad connections
    window.addEventListener('gamepadconnected', (e) => {
      console.log(`[HapticEngine] Gamepad connected: ${e.gamepad.id}`);
      this.gamepads.set(e.gamepad.index, e.gamepad);
      this.isInitialized = true;
    });

    window.addEventListener('gamepaddisconnected', (e) => {
      console.log(`[HapticEngine] Gamepad disconnected: ${e.gamepad.id}`);
      this.gamepads.delete(e.gamepad.index);
      if (this.gamepads.size === 0) {
        this.isInitialized = false;
      }
    });

    // Poll for gamepads periodically
    this.pollInterval = setInterval(() => this.pollGamepads(), 500);
    
    console.log('[HapticEngine] Initialized');
  }

  private pollGamepads() {
    if (!navigator.getGamepads) return;
    
    const gamepads = navigator.getGamepads();
    for (let i = 0; i < gamepads.length; i++) {
      const gamepad = gamepads[i];
      if (gamepad) {
        this.gamepads.set(i, gamepad);
        this.isInitialized = true;
      }
    }
  }

  /**
   * Trigger a haptic pulse with the specified pattern
   */
  public trigger(pattern: HapticPattern): boolean {
    if (!this.isEnabled || !this.isInitialized || this.gamepads.size === 0) {
      return false;
    }

    const config = this.PATTERNS[pattern];
    if (!config) {
      console.warn(`[HapticEngine] Unknown pattern: ${pattern}`);
      return false;
    }

    // Trigger on all connected gamepads
    let triggered = false;
    this.gamepads.forEach((gamepad, index) => {
      const vibrationActuator = (gamepad as any).vibrationActuator;
      if (vibrationActuator && typeof vibrationActuator.playEffect === 'function') {
        try {
          vibrationActuator.playEffect('dual-rumble', {
            startDelay: 0,
            duration: config.duration,
            weakMagnitude: config.weakMagnitude,
            strongMagnitude: config.strongMagnitude,
          });
          triggered = true;
        } catch (error) {
          console.error(`[HapticEngine] Failed to trigger gamepad ${index}:`, error);
        }
      }
    });

    if (triggered) {
      console.debug(`[HapticEngine] Triggered pattern: ${pattern}`);
    }

    return triggered;
  }

  /**
   * Trigger a custom vibration pattern
   */
  public triggerCustom(
    duration: number,
    weakMagnitude: number,
    strongMagnitude: number
  ): boolean {
    if (!this.isEnabled || !this.isInitialized || this.gamepads.size === 0) {
      return false;
    }

    let triggered = false;
    this.gamepads.forEach((gamepad) => {
      const vibrationActuator = (gamepad as any).vibrationActuator;
      if (vibrationActuator) {
        try {
          vibrationActuator.playEffect('dual-rumble', {
            startDelay: 0,
            duration,
            weakMagnitude: Math.max(0, Math.min(1, weakMagnitude)),
            strongMagnitude: Math.max(0, Math.min(1, strongMagnitude)),
          });
          triggered = true;
        } catch (error) {
          console.error('[HapticEngine] Custom trigger failed:', error);
        }
      }
    });

    return triggered;
  }

  /**
   * Trigger haptic feedback for institutional block execution
   */
  public triggerInstitutionalBlock(quantity: number, value: number): void {
    // Scale intensity based on order size
    const baseValue = 1000000; // 1M USD base
    const intensity = Math.min(1, value / baseValue);
    
    if (intensity > 0.5) {
      // Heavy pulse for large blocks
      this.trigger('INSTITUTIONAL_BLOCK');
    } else if (intensity > 0.2) {
      // Medium pulse for moderate blocks
      this.trigger('HEAVY_PULSE');
    } else {
      // Light pulse for smaller blocks
      this.trigger('MEDIUM_PULSE');
    }
  }

  /**
   * Enable/disable haptic feedback
   */
  public setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    if (!enabled) {
      // Stop all vibrations
      this.gamepads.forEach((gamepad) => {
        const vibrationActuator = (gamepad as any).vibrationActuator;
        if (vibrationActuator) {
          vibrationActuator.reset();
        }
      });
    }
  }

  /**
   * Get connected gamepads count
   */
  public getConnectedCount(): number {
    return this.gamepads.size;
  }

  /**
   * Check if haptics are available
   */
  public isAvailable(): boolean {
    return this.isInitialized && this.gamepads.size > 0;
  }

  /**
   * Cleanup
   */
  public dispose(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    
    // Reset all gamepads
    this.gamepads.forEach((gamepad) => {
      const vibrationActuator = (gamepad as any).vibrationActuator;
      if (vibrationActuator) {
        vibrationActuator.reset();
      }
    });
    
    this.gamepads.clear();
    this.isInitialized = false;
  }
}

// Singleton instance
let hapticEngineInstance: HapticEngine | null = null;

export const getHapticEngine = (): HapticEngine => {
  if (!hapticEngineInstance) {
    hapticEngineInstance = new HapticEngine();
  }
  return hapticEngineInstance;
};

export default getHapticEngine;
