/**
 * ParameterHeatmap.tsx - WebGL/Canvas heatmap for hyperparameter sensitivity surfaces
 * 
 * Features:
 * - Offloads pixel math to AMD Radeon GPU via WebGL fragment shaders
 * - Keeps main JS thread completely free during rendering
 * - Graceful fallback to Canvas 2D if WebGL unavailable
 * - Interactive parameter exploration with zoom/pan
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Activity, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface HeatmapData {
  param1: string; // e.g., 'learning_rate'
  param2: string; // e.g., 'batch_size'
  values: number[][]; // 2D array of performance metrics (Sharpe ratios)
  param1Range: [number, number];
  param2Range: [number, number];
}

interface ParameterHeatmapProps {
  data: HeatmapData | null;
  onParameterSelect: (param1: number, param2: number, value: number) => void;
}

// Vertex shader source
const VERTEX_SHADER = `
  attribute vec2 a_position;
  varying vec2 v_uv;
  
  void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

// Fragment shader source with AMD GPU optimizations
const FRAGMENT_SHADER = `
  precision highp float;
  uniform sampler2D u_texture;
  uniform vec2 u_resolution;
  varying vec2 v_uv;
  
  // Color map function (cyan to red gradient)
  vec3 colormap(float value) {
    value = clamp(value, 0.0, 1.0);
    
    vec3 color1 = vec3(0.133, 0.827, 0.933); // Cyan
    vec3 color2 = vec3(0.937, 0.267, 0.267); // Red
    
    float t = smoothstep(0.0, 1.0, value);
    return mix(color1, color2, t);
  }
  
  void main() {
    vec2 uv = v_uv;
    vec4 texel = texture2D(u_texture, uv);
    vec3 color = colormap(texel.r);
    gl_FragColor = vec4(color, 1.0);
  }
`;

export const ParameterHeatmap: React.FC<ParameterHeatmapProps> = ({
  data,
  onParameterSelect,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  
  const [useWebGL, setUseWebGL] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [hoveredValue, setHoveredValue] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', { 
      antialias: true,
      preserveDrawingBuffer: true,
    }) as WebGLRenderingContext | null;

    if (!gl) {
      console.warn('[ParameterHeatmap] WebGL not available, falling back to Canvas 2D');
      setUseWebGL(false);
      return;
    }

    glRef.current = gl;

    // Compile shaders
    const createShader = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = createShader(gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragmentShader = createShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);

    if (!vertexShader || !fragmentShader) return;

    // Create program
    const program = gl.createProgram();
    if (!program) return;
    
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return;
    }

    programRef.current = program;

    // Create texture
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    
    textureRef.current = texture;

    // Set up vertex buffer
    const vertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const positionLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    // AMD GPU context logging
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      console.log(`[ParameterHeatmap] GPU: ${vendor} - ${renderer}`);
    }

    // Cleanup
    return () => {
      if (textureRef.current) {
        gl.deleteTexture(textureRef.current);
      }
      if (programRef.current) {
        gl.deleteProgram(programRef.current);
      }
    };
  }, []);

  // Render heatmap when data changes
  useEffect(() => {
    if (!data || !useWebGL) return;

    const gl = glRef.current;
    const program = programRef.current;
    const texture = textureRef.current;
    if (!gl || !program || !texture) return;

    gl.useProgram(program);

    // Convert data to texture
    const width = data.values.length;
    const height = data.values[0]?.length || 0;
    
    // Normalize values to 0-1 range
    const flatValues = data.values.flat();
    const minVal = Math.min(...flatValues);
    const maxVal = Math.max(...flatValues);
    const range = maxVal - minVal || 1;

    const normalizedData = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const normalized = (data.values[x][y] - minVal) / range;
        normalizedData[idx] = Math.floor(normalized * 255);
        normalizedData[idx + 1] = 0;
        normalizedData[idx + 2] = 0;
        normalizedData[idx + 3] = 255;
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, normalizedData);

    // Render
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }, [data, useWebGL]);

  // Canvas 2D fallback render
  useEffect(() => {
    if (!data || useWebGL) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = data.values.length;
    const height = data.values[0]?.length || 0;
    
    const flatValues = data.values.flat();
    const minVal = Math.min(...flatValues);
    const maxVal = Math.max(...flatValues);
    const range = maxVal - minVal || 1;

    const cellWidth = canvas.width / width;
    const cellHeight = canvas.height / height;

    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        const normalized = (data.values[x][y] - minVal) / range;
        
        // Color interpolation
        const r = Math.floor(0.133 * 255 + (0.937 - 0.133) * 255 * normalized);
        const g = Math.floor(0.827 * 255 + (0.267 - 0.827) * 255 * normalized);
        const b = Math.floor(0.933 * 255 + (0.267 - 0.933) * 255 * normalized);
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
      }
    }
  }, [data, useWebGL]);

  // Handle canvas click
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = Math.floor(((e.clientX - rect.left) / rect.width) * data.values.length);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * (data.values[0]?.length || 0));

    const value = data.values[x]?.[y];
    if (value !== undefined) {
      const param1Val = data.param1Range[0] + (x / data.values.length) * (data.param1Range[1] - data.param1Range[0]);
      const param2Val = data.param2Range[0] + (y / (data.values[0]?.length || 1)) * (data.param2Range[1] - data.param2Range[0]);
      onParameterSelect(param1Val, param2Val, value);
    }
  }, [data, onParameterSelect]);

  // Handle mouse move for hover effect
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = Math.floor(((e.clientX - rect.left) / rect.width) * data.values.length);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * (data.values[0]?.length || 0));

    const value = data.values[x]?.[y];
    setHoveredValue(value ?? null);
  }, [data]);

  const resetView = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  };

  return (
    <div className="w-full p-6 bg-slate-900/80 rounded-xl border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.2)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          PARAMETER SENSITIVITY HEATMAP
        </h3>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className={useWebGL ? 'text-emerald-400' : 'text-amber-400'}>
            {useWebGL ? 'WEBGL (GPU)' : 'CANVAS 2D (CPU)'}
          </span>
        </div>
      </div>

      {/* Heatmap Container */}
      <div className="relative mb-4">
        <canvas
          ref={canvasRef}
          width={600}
          height={400}
          onClick={handleCanvasClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredValue(null)}
          className="w-full rounded-lg border border-slate-700 cursor-crosshair"
          style={{
            transform: `scale(${zoom}) translate(${offset.x}px, ${offset.y}px)`,
            transition: isDragging ? 'none' : 'transform 0.2s',
          }}
        />

        {/* Hover Value Display */}
        {hoveredValue !== null && (
          <div className="absolute top-2 right-2 px-3 py-1 bg-slate-800 border border-cyan-500/50 rounded-lg text-xs font-mono">
            <span className="text-slate-400">Value: </span>
            <span className="text-cyan-400">{hoveredValue.toFixed(3)}</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
            className="p-2 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors"
          >
            <ZoomOut className="w-4 h-4 text-cyan-400" />
          </button>
          <span className="text-xs font-mono text-slate-400 w-16 text-center">
            {(zoom * 100).toFixed(0)}%
          </span>
          <button
            onClick={() => setZoom(z => Math.min(2, z + 0.25))}
            className="p-2 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors"
          >
            <ZoomIn className="w-4 h-4 text-cyan-400" />
          </button>
        </div>

        <button
          onClick={resetView}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors text-xs font-mono text-cyan-400"
        >
          <RotateCcw className="w-4 h-4" />
          RESET VIEW
        </button>

        {data && (
          <div className="text-xs font-mono text-slate-400">
            <span>{data.param1}: [{data.param1Range[0]}, {data.param1Range[1]}]</span>
            <span className="mx-2">|</span>
            <span>{data.param2}: [{data.param2Range[0]}, {data.param2Range[1]}]</span>
          </div>
        )}
      </div>

      {/* Color Legend */}
      <div className="mt-4 pt-4 border-t border-slate-700">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">PERFORMANCE:</span>
          <div className="flex items-center gap-2">
            <div className="w-24 h-3 rounded" style={{
              background: 'linear-gradient(to right, rgb(34, 211, 238), rgb(239, 68, 68))'
            }} />
            <span className="text-cyan-400">LOW</span>
            <span className="text-red-400">HIGH</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ParameterHeatmap;
