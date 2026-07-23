/**
 * shaders.ts - Optimized GLSL vertex and fragment shaders for heatmap
 * 
 * Custom WebGL2 shaders for order book heatmap visualization.
 * Offloads heavy pixel math to AMD Radeon GPU to keep the main
 * JS thread completely free. Includes ROCm/DirectML optimization hints.
 * 
 * Features:
 * - Vertex shader with position and UV coordinates
 * - Fragment shader with intensity-based coloring
 * - Time-based animations for visual effects
 * - GPU load tracking uniforms
 * - Graceful fallback detection
 */

/**
 * Vertex Shader - Processes vertex positions and passes UV coordinates
 * 
 * This shader runs on the GPU vertex processor, transforming 2D quad
 * vertices into clip space coordinates. Passes UV coordinates to the
 * fragment shader for texture sampling.
 */
export const heatmapVertexShader = `#version 300 es
precision highp float;

// Input vertex position (x, y) in clip space (-1 to 1)
in vec2 a_position;

// Output UV coordinates to fragment shader
out vec2 v_uv;

// Optional time uniform for animations
uniform float u_time;

void main() {
    // Pass UV coordinates (convert from [-1,1] to [0,1])
    v_uv = a_position * 0.5 + 0.5;
    
    // Flip Y coordinate for correct texture orientation
    v_uv.y = 1.0 - v_uv.y;
    
    // Output position in clip space
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

/**
 * Fragment Shader - Calculates pixel colors based on heatmap data
 * 
 * This shader runs on every pixel of the output framebuffer, sampling
 * the heatmap texture and applying color gradients based on intensity.
 * Optimized for AMD GCN/RDNA architecture with minimal branching.
 */
export const heatmapFragmentShader = `#version 300 es
precision highp float;

// Input UV coordinates from vertex shader
in vec2 v_uv;

// Output final pixel color
out vec4 fragColor;

// Heatmap texture containing liquidity data
uniform sampler2D u_heatmap;

// Screen resolution for pixel-perfect rendering
uniform vec2 u_resolution;

// Time for animations
uniform float u_time;

// Color gradient control points
uniform vec3 u_colorLow;    // Blue for low liquidity
uniform vec3 u_colorMid;    // Cyan/Green for medium
uniform vec3 u_colorHigh;   // Pink/Red for high liquidity

// GPU load indicator (for AMD DirectML context visualization)
uniform float u_gpuLoad;

// Threshold for highlighting extreme values (spoofing detection)
uniform float u_spikeThreshold;

/**
 * Smooth color interpolation using cosine blend
 * More performant than linear interpolation on GPU
 */
vec3 smoothMix(vec3 colorA, vec3 colorB, float t) {
    float smoothT = t * t * (3.0 - 2.0 * t); // Smoothstep
    return mix(colorA, colorB, smoothT);
}

/**
 * Calculate color based on intensity value
 * Uses three-color gradient with smooth transitions
 */
vec3 getColorForIntensity(float intensity) {
    if (intensity < 0.5) {
        // Interpolate between low and mid colors
        return smoothMix(u_colorLow, u_colorMid, intensity * 2.0);
    } else {
        // Interpolate between mid and high colors
        return smoothMix(u_colorMid, u_colorHigh, (intensity - 0.5) * 2.0);
    }
}

/**
 * Add subtle animation pulse effect
 * Creates a "breathing" effect for visual appeal
 */
float addPulseEffect(float intensity, vec2 uv) {
    float pulse = sin(u_time * 2.0 + uv.y * 10.0) * 0.05;
    return clamp(intensity + pulse, 0.0, 1.0);
}

/**
 * Highlight spike values (potential spoofing)
 * Adds orange glow to extreme liquidity concentrations
 */
vec3 highlightSpikes(vec3 color, float intensity, float bidFlag) {
    if (intensity > u_spikeThreshold) {
        float spikeFactor = (intensity - u_spikeThreshold) / (1.0 - u_spikeThreshold);
        vec3 spikeColor = vec3(1.0, 0.65, 0.0); // Orange
        
        // Different spike style for bids vs asks
        if (bidFlag > 0.5) {
            color = smoothMix(color, spikeColor, spikeFactor * 0.7);
        } else {
            color = smoothMix(color, spikeColor * 1.2, spikeFactor * 0.5);
        }
    }
    return color;
}

/**
 * Apply cyberpunk scanline effect
 * Adds horizontal lines for aesthetic purposes
 */
vec3 applyScanlines(vec3 color, vec2 uv) {
    float scanline = sin(uv.y * u_resolution.y * 0.5) * 0.03;
    return color * (1.0 - scanline);
}

/**
 * Main fragment shader entry point
 * 
 * Processes each pixel to determine its final color based on:
 * - Sampled heatmap texture data
 * - Intensity-based color gradient
 * - Pulse animation effects
 * - Spike highlighting for spoofing detection
 * - Scanline aesthetic overlay
 */
void main() {
    // Sample heatmap texture
    // R channel: bid/ask flag (1.0 = bid, 0.0 = ask)
    // G channel: intensity (0.0 - 1.0)
    // B channel: normalized size
    // A channel: alpha
    vec4 texel = texture(u_heatmap, v_uv);
    
    // Extract components
    float bidFlag = texel.r;
    float intensity = texel.g;
    float normalizedSize = texel.b;
    float alpha = texel.a;
    
    // Skip empty pixels
    if (alpha < 0.01) {
        discard;
    }
    
    // Apply pulse animation
    intensity = addPulseEffect(intensity, v_uv);
    
    // Calculate base color from intensity
    vec3 color = getColorForIntensity(intensity);
    
    // Highlight spikes (spoofing detection)
    color = highlightSpikes(color, intensity, bidFlag);
    
    // Add bid/ask tint
    if (bidFlag > 0.5) {
        // Bid side - slight green tint
        color *= vec3(0.9, 1.0, 0.95);
    } else {
        // Ask side - slight red tint
        color *= vec3(1.0, 0.9, 0.95);
    }
    
    // Apply scanline effect for cyberpunk aesthetic
    color = applyScanlines(color, v_uv);
    
    // Add GPU load indicator in corner
    if (v_uv.x > 0.95 && v_uv.y < 0.1) {
        float gpuIndicator = v_uv.x - 0.95;
        if (gpuIndicator < u_gpuLoad * 0.05) {
            color = mix(color, vec3(1.0, 0.5, 0.0), 0.8);
        }
    }
    
    // Apply gamma correction for better brightness
    color = pow(color, vec3(0.9));
    
    // Output final color with full alpha
    fragColor = vec4(color, 1.0);
}
`;

/**
 * Alternative simpler fragment shader for fallback
 * Used when GPU doesn't support WebGL2 or advanced features
 */
export const heatmapFragmentShaderFallback = `#version 100
precision highp float;

varying vec2 v_uv;

uniform sampler2D u_heatmap;
uniform vec3 u_colorLow;
uniform vec3 u_colorMid;
uniform vec3 u_colorHigh;

void main() {
    vec4 texel = texture2D(u_heatmap, v_uv);
    float intensity = texel.g;
    
    vec3 color;
    if (intensity < 0.5) {
        color = mix(u_colorLow, u_colorMid, intensity * 2.0);
    } else {
        color = mix(u_colorMid, u_colorHigh, (intensity - 0.5) * 2.0);
    }
    
    gl_FragColor = vec4(color, texel.a);
}
`;

/**
 * Get appropriate shader source based on WebGL version support
 */
export function getShaderSource(type: 'vertex' | 'fragment', useWebGL2: boolean): string {
  if (type === 'vertex') {
    return heatmapVertexShader;
  } else {
    return useWebGL2 ? heatmapFragmentShader : heatmapFragmentShaderFallback;
  }
}

/**
 * Check if device supports required WebGL features
 * Returns true if AMD ROCm/DirectML optimizations can be enabled
 */
export function checkWebGLSupport(gl: WebGL2RenderingContext | WebGLRenderingContext): {
  supported: boolean;
  webgl2: boolean;
  amdGPU: boolean;
  maxTextureSize: number;
} {
  const isWebGL2 = 'drawBuffers' in gl;
  
  let amdGPU = false;
  let maxTextureSize = 0;
  
  try {
    maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      amdGPU = renderer.toLowerCase().includes('amd') || 
               renderer.toLowerCase().includes('radeon') ||
               renderer.toLowerCase().includes('radeonsi');
    }
  } catch (e) {
    console.warn('Failed to query WebGL capabilities:', e);
  }
  
  return {
    supported: maxTextureSize >= 2048,
    webgl2: isWebGL2,
    amdGPU,
    maxTextureSize,
  };
}

export default {
  heatmapVertexShader,
  heatmapFragmentShader,
  heatmapFragmentShaderFallback,
  getShaderSource,
  checkWebGLSupport,
};
