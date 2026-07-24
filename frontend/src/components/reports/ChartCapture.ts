/**
 * ChartCapture - OffscreenCanvas Utility
 * Safely snapshots WebGL and 3D volatility surfaces for PDF injection
 * Does not freeze the main rendering thread by using OffscreenCanvas
 */

export interface CaptureOptions {
  width?: number;
  height?: number;
  format?: 'image/png' | 'image/jpeg' | 'image/webp';
  quality?: number;
  preserveDrawingBuffer?: boolean;
}

export interface CaptureResult {
  blob: Blob;
  dataUrl: string;
  width: number;
  height: number;
  timestamp: number;
  source: 'webgl' | 'canvas2d' | 'offscreen';
}

export interface VolatilitySurfaceData {
  strikes: number[];
  expiries: number[];
  volatilities: number[][];
  underlyingPrice: number;
  timestamp: number;
}

/**
 * ChartCapture - High-performance chart snapshot utility
 * Uses OffscreenCanvas to avoid blocking the main thread
 */
export class ChartCapture {
  private offscreenCanvas: OffscreenCanvas | null = null;
  private offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;
  private isSupported: boolean;

  constructor() {
    this.isSupported = typeof OffscreenCanvas !== 'undefined';
    
    if (this.isSupported) {
      this.offscreenCanvas = new OffscreenCanvas(1, 1);
      this.offscreenCtx = this.offscreenCanvas.getContext('2d');
    }
  }

  /**
   * Check if OffscreenCanvas is supported
   */
  static isOffscreenCanvasSupported(): boolean {
    return typeof OffscreenCanvas !== 'undefined';
  }

  /**
   * Capture a WebGL canvas without blocking the main thread
   * Waits for the next animation frame before extracting pixel data
   */
  async captureWebGL(
    canvas: HTMLCanvasElement,
    options: CaptureOptions = {}
  ): Promise<CaptureResult> {
    const {
      width = canvas.width,
      height = canvas.height,
      format = 'image/png',
      quality = 0.92,
    } = options;

    // Wait for the next rendered frame
    await this.waitForNextFrame(canvas);

    // Create offscreen canvas for processing
    const offscreen = new OffscreenCanvas(width, height);
    const ctx = offscreen.getContext('2d');

    if (!ctx) {
      throw new Error('Failed to get 2D context from OffscreenCanvas');
    }

    // Draw the WebGL canvas content to offscreen
    ctx.drawImage(canvas, 0, 0, width, height);

    // Convert to blob asynchronously
    const blob = await offscreen.convertToBlob({
      type: format,
      quality,
    });

    // Convert blob to data URL
    const dataUrl = await this.blobToDataURL(blob);

    return {
      blob,
      dataUrl,
      width,
      height,
      timestamp: Date.now(),
      source: 'webgl',
    };
  }

  /**
   * Capture a 2D canvas
   */
  async captureCanvas2D(
    canvas: HTMLCanvasElement,
    options: CaptureOptions = {}
  ): Promise<CaptureResult> {
    const {
      width = canvas.width,
      height = canvas.height,
      format = 'image/png',
      quality = 0.92,
    } = options;

    // Wait for any pending renders
    await this.waitForNextFrame(canvas);

    // Create offscreen canvas
    const offscreen = new OffscreenCanvas(width, height);
    const ctx = offscreen.getContext('2d');

    if (!ctx) {
      throw new Error('Failed to get 2D context from OffscreenCanvas');
    }

    // Draw the source canvas to offscreen
    ctx.drawImage(canvas, 0, 0, width, height);

    // Convert to blob
    const blob = await offscreen.convertToBlob({
      type: format,
      quality,
    });

    const dataUrl = await this.blobToDataURL(blob);

    return {
      blob,
      dataUrl,
      width,
      height,
      timestamp: Date.now(),
      source: 'canvas2d',
    };
  }

  /**
   * Render a volatility surface to an offscreen canvas and capture it
   * This creates a 3D-like visualization of the volatility smile/skew
   */
  async captureVolatilitySurface(
    data: VolatilitySurfaceData,
    options: CaptureOptions & {
      colorScheme?: 'cyberpunk' | 'classic' | 'heatmap';
      showGrid?: boolean;
      showLabels?: boolean;
    } = {}
  ): Promise<CaptureResult> {
    const {
      width = 800,
      height = 600,
      format = 'image/png',
      quality = 0.92,
      colorScheme = 'cyberpunk',
      showGrid = true,
      showLabels = true,
    } = options;

    const offscreen = new OffscreenCanvas(width, height);
    const ctx = offscreen.getContext('2d');

    if (!ctx) {
      throw new Error('Failed to get 2D context from OffscreenCanvas');
    }

    // Clear with cyberpunk background
    ctx.fillStyle = '#050510';
    ctx.fillRect(0, 0, width, height);

    // Calculate dimensions
    const padding = showLabels ? 60 : 20;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Find min/max for normalization
    const allVols = data.volatilities.flat();
    const minVol = Math.min(...allVols);
    const maxVol = Math.max(...allVols);
    const volRange = maxVol - minVol || 1;

    // Color schemes
    const colors = this.getColorScheme(colorScheme);

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = 'rgba(0, 243, 255, 0.15)';
      ctx.lineWidth = 1;

      // Vertical lines (strikes)
      for (let i = 0; i <= data.strikes.length; i++) {
        const x = padding + (i / data.strikes.length) * chartWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
      }

      // Horizontal lines (volatility levels)
      for (let i = 0; i <= 5; i++) {
        const y = padding + (i / 5) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
      }
    }

    // Draw volatility surface as colored rectangles
    const strikeStep = chartWidth / data.strikes.length;
    const expiryStep = chartHeight / data.expiries.length;

    for (let i = 0; i < data.strikes.length; i++) {
      for (let j = 0; j < data.expiries.length; j++) {
        const vol = data.volatilities[j][i];
        const normalizedVol = (vol - minVol) / volRange;

        const x = padding + i * strikeStep;
        const y = padding + (data.expiries.length - 1 - j) * expiryStep;

        // Get color based on volatility level
        const color = this.interpolateColor(colors, normalizedVol);
        ctx.fillStyle = color;
        ctx.fillRect(x, y, strikeStep + 1, expiryStep + 1);
      }
    }

    // Draw axes labels
    if (showLabels) {
      ctx.fillStyle = '#00f3ff';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';

      // Strike labels (x-axis)
      const labelInterval = Math.ceil(data.strikes.length / 5);
      data.strikes.forEach((strike, i) => {
        if (i % labelInterval === 0 || i === data.strikes.length - 1) {
          const x = padding + i * strikeStep + strikeStep / 2;
          ctx.fillText(`$${strike}`, x, height - padding + 20);
        }
      });

      // Expiry labels (y-axis)
      ctx.textAlign = 'right';
      data.expiries.forEach((expiry, j) => {
        if (j % labelInterval === 0 || j === data.expiries.length - 1) {
          const y = padding + (data.expiries.length - 1 - j) * expiryStep + expiryStep / 2;
          ctx.fillText(`${expiry}D`, padding - 10, y + 4);
        }
      });

      // Title
      ctx.fillStyle = '#00f3ff';
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(
        `VOLATILITY SURFACE - ${data.underlyingPrice.toFixed(2)}`,
        width / 2,
        25
      );

      // Timestamp
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#666';
      ctx.fillText(
        new Date(data.timestamp).toLocaleString(),
        width / 2,
        height - 10
      );
    }

    // Add glow effect border
    ctx.strokeStyle = 'rgba(0, 243, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00f3ff';
    ctx.shadowBlur = 10;
    ctx.strokeRect(padding, padding, chartWidth, chartHeight);
    ctx.shadowBlur = 0;

    // Convert to blob
    const blob = await offscreen.convertToBlob({
      type: format,
      quality,
    });

    const dataUrl = await this.blobToDataURL(blob);

    return {
      blob,
      dataUrl,
      width,
      height,
      timestamp: Date.now(),
      source: 'offscreen',
    };
  }

  /**
   * Capture multiple charts and return as array
   */
  async captureMultiple(
    captures: Array<{
      canvas: HTMLCanvasElement;
      options?: CaptureOptions;
    }>
  ): Promise<CaptureResult[]> {
    return Promise.all(
      captures.map(({ canvas, options }) => this.captureWebGL(canvas, options))
    );
  }

  /**
   * Wait for the next animation frame to ensure render is complete
   */
  private waitForNextFrame(canvas: HTMLCanvasElement): Promise<void> {
    return new Promise((resolve) => {
      // Request two frames - first to complete current render, second to ensure it's displayed
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          resolve();
        });
      });
    });
  }

  /**
   * Convert Blob to DataURL
   */
  private blobToDataURL(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Get color scheme configuration
   */
  private getColorScheme(
    scheme: 'cyberpunk' | 'classic' | 'heatmap'
  ): { low: string; mid: string; high: string } {
    switch (scheme) {
      case 'cyberpunk':
        return {
          low: '#001a33',
          mid: '#00f3ff',
          high: '#ff0055',
        };
      case 'classic':
        return {
          low: '#006400',
          mid: '#ffff00',
          high: '#ff0000',
        };
      case 'heatmap':
        return {
          low: '#0000ff',
          mid: '#ffff00',
          high: '#ff0000',
        };
    }
  }

  /**
   * Interpolate between colors based on value
   */
  private interpolateColor(
    colors: { low: string; mid: string; high: string },
    value: number
  ): string {
    if (value < 0.5) {
      // Interpolate between low and mid
      const ratio = value * 2;
      return this.lerpColor(colors.low, colors.mid, ratio);
    } else {
      // Interpolate between mid and high
      const ratio = (value - 0.5) * 2;
      return this.lerpColor(colors.mid, colors.high, ratio);
    }
  }

  /**
   * Linear interpolation between two hex colors
   */
  private lerpColor(color1: string, color2: string, t: number): string {
    const r1 = parseInt(color1.slice(1, 3), 16);
    const g1 = parseInt(color1.slice(3, 5), 16);
    const b1 = parseInt(color1.slice(5, 7), 16);

    const r2 = parseInt(color2.slice(1, 3), 16);
    const g2 = parseInt(color2.slice(3, 5), 16);
    const b2 = parseInt(color2.slice(5, 7), 16);

    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);

    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  }

  /**
   * Create a thumbnail from a capture result
   */
  async createThumbnail(
    capture: CaptureResult,
    maxWidth: number = 200,
    maxHeight: number = 150
  ): Promise<string> {
    const img = await this.createImageBitmap(capture.blob);
    
    const scale = Math.min(maxWidth / img.width, maxHeight / img.height);
    const width = Math.floor(img.width * scale);
    const height = Math.floor(img.height * scale);

    const offscreen = new OffscreenCanvas(width, height);
    const ctx = offscreen.getContext('2d');
    
    if (!ctx) {
      throw new Error('Failed to get context for thumbnail');
    }

    ctx.drawImage(img, 0, 0, width, height);
    
    const blob = await offscreen.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
    return this.blobToDataURL(blob);
  }

  /**
   * Create ImageBitmap from blob (more efficient than Image element)
   */
  private createImageBitmap(blob: Blob): Promise<ImageBitmap> {
    return createImageBitmap(blob);
  }
}

// Export singleton instance
export const chartCapture = new ChartCapture();

/**
 * React Hook for using ChartCapture in components
 */
export function useChartCapture() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const capture = useCallback(
    async (
      canvas: HTMLCanvasElement,
      options?: CaptureOptions
    ): Promise<CaptureResult | null> => {
      setIsCapturing(true);
      setError(null);

      try {
        const result = await chartCapture.captureWebGL(canvas, options);
        return result;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Capture failed';
        setError(message);
        return null;
      } finally {
        setIsCapturing(false);
      }
    },
    []
  );

  const captureVolatilitySurface = useCallback(
    async (
      data: VolatilitySurfaceData,
      options?: Parameters<ChartCapture['captureVolatilitySurface']>[1]
    ): Promise<CaptureResult | null> => {
      setIsCapturing(true);
      setError(null);

      try {
        const result = await chartCapture.captureVolatilitySurface(data, options);
        return result;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Capture failed';
        setError(message);
        return null;
      } finally {
        setIsCapturing(false);
      }
    },
    []
  );

  return {
    isCapturing,
    error,
    capture,
    captureVolatilitySurface,
    isSupported: ChartCapture.isOffscreenCanvasSupported(),
  };
}

// React import for hook
import { useState, useCallback } from 'react';

export default ChartCapture;
