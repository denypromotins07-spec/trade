/**
 * PipManager Component
 * Document Picture-in-Picture API integration for detaching WebGL order book heatmaps
 * Handles floating OS-level overlay windows with graceful context loss recovery
 */

import React, { useState, useEffect, useCallback, useRef, useImperativeHandle } from 'react';

export interface PipContent {
  id: string;
  title: string;
  element: HTMLElement | null;
  width?: number;
  height?: number;
}

export interface PipManagerRef {
  openPip: (content: PipContent) => Promise<boolean>;
  closePip: (id: string) => void;
  closeAll: () => void;
  isPipOpen: (id: string) => boolean;
  getPipWindow: (id: string) => DocumentPictureInPicture | null;
}

interface PipManagerProps {
  defaultWidth?: number;
  defaultHeight?: number;
  onContextLoss?: (id: string) => void;
  onContextRestore?: (id: string) => void;
  debugMode?: boolean;
}

interface PipState {
  id: string;
  title: string;
  pipWindow: DocumentPictureInPicture | null;
  container: HTMLDivElement | null;
  isActive: boolean;
  hasContextLoss: boolean;
}

/**
 * PipManager - Manages Picture-in-Picture windows for trading visualizations
 */
export const PipManager = React.forwardRef<PipManagerRef, PipManagerProps>(
  (
    {
      defaultWidth = 800,
      defaultHeight = 600,
      onContextLoss,
      onContextRestore,
      debugMode = false,
    },
    ref
  ) => {
    const [pipStates, setPipStates] = useState<Map<string, PipState>>(new Map());
    const contentRefs = useRef<Map<string, HTMLElement>>(new Map());
    const canvasRefs = useRef<Map<string, HTMLCanvasElement>>(new Map());
    const animationFrameRefs = useRef<Map<string, number>>(new Map());

    // Check if Document Picture-in-Picture is supported
    const isSupported = typeof window !== 'undefined' && 'documentPictureInPicture' in window;

    /**
     * Open a new PiP window
     */
    const openPip = useCallback(
      async (content: PipContent): Promise<boolean> => {
        if (!isSupported) {
          console.warn('[PipManager] Document Picture-in-Picture not supported');
          return false;
        }

        try {
          const pipWindow = await (window as unknown as {
            documentPictureInPicture: {
              requestWindow: (options?: {
                width?: number;
                height?: number;
              }) => Promise<unknown & { document: Document }>;
            };
          }).documentPictureInPicture.requestWindow({
            width: content.width || defaultWidth,
            height: content.height || defaultHeight,
          });

          // Create container in main document
          const container = document.createElement('div');
          container.id = `pip-container-${content.id}`;
          container.style.display = 'none';
          document.body.appendChild(container);

          // Store reference
          contentRefs.current.set(content.id, container);

          // Setup PiP document
          const pipDoc = pipWindow.document;
          
          // Apply cyberpunk styles to PiP window
          const style = pipDoc.createElement('style');
          style.textContent = `
            * {
              margin: 0;
              padding: 0;
              box-sizing: border-box;
            }
            body {
              background: #050510;
              color: #00f3ff;
              font-family: 'JetBrains Mono', 'Fira Code', monospace;
              overflow: hidden;
            }
            .pip-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              padding: 8px 12px;
              background: linear-gradient(90deg, #0a0a1a, #0f0f2a);
              border-bottom: 1px solid rgba(0, 243, 255, 0.3);
              user-select: none;
            }
            .pip-title {
              font-size: 12px;
              font-weight: 600;
              text-transform: uppercase;
              letter-spacing: 1px;
              color: #00f3ff;
              text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
            }
            .pip-close {
              background: transparent;
              border: 1px solid #ff0055;
              color: #ff0055;
              padding: 4px 8px;
              cursor: pointer;
              font-size: 11px;
              border-radius: 4px;
              transition: all 0.2s;
            }
            .pip-close:hover {
              background: #ff0055;
              color: #fff;
              box-shadow: 0 0 15px rgba(255, 0, 85, 0.5);
            }
            .pip-content {
              width: 100%;
              height: calc(100vh - 40px);
              position: relative;
            }
            canvas {
              width: 100%;
              height: 100%;
            }
            .context-lost-overlay {
              position: absolute;
              inset: 0;
              display: flex;
              align-items: center;
              justify-content: center;
              background: rgba(5, 5, 16, 0.9);
              backdrop-filter: blur(4px);
              z-index: 100;
            }
            .context-lost-message {
              text-align: center;
              padding: 20px;
            }
            .context-lost-message h3 {
              color: #ff0055;
              margin-bottom: 8px;
              font-size: 14px;
            }
            .context-lost-message p {
              color: #666;
              font-size: 11px;
            }
            .restore-btn {
              margin-top: 12px;
              padding: 8px 16px;
              background: linear-gradient(135deg, #00f3ff, #0066ff);
              border: none;
              color: #050510;
              font-weight: bold;
              cursor: pointer;
              border-radius: 4px;
              text-transform: uppercase;
              letter-spacing: 1px;
            }
          `;
          pipDoc.head.appendChild(style);

          // Create header
          const header = pipDoc.createElement('div');
          header.className = 'pip-header';
          
          const titleEl = pipDoc.createElement('span');
          titleEl.className = 'pip-title';
          titleEl.textContent = content.title;
          
          const closeBtn = pipDoc.createElement('button');
          closeBtn.className = 'pip-close';
          closeBtn.textContent = '✕ CLOSE';
          closeBtn.onclick = () => {
            closePip(content.id);
          };
          
          header.appendChild(titleEl);
          header.appendChild(closeBtn);
          pipDoc.body.appendChild(header);

          // Create content area
          const contentEl = pipDoc.createElement('div');
          contentEl.className = 'pip-content';
          contentEl.id = `pip-content-${content.id}`;
          pipDoc.body.appendChild(contentEl);

          // Move the actual content (canvas or element) to PiP
          const sourceElement = content.element;
          if (sourceElement) {
            const clonedElement = sourceElement.cloneNode(true) as HTMLElement;
            contentEl.appendChild(clonedElement);

            // Handle canvas specially for WebGL contexts
            if (clonedElement instanceof HTMLCanvasElement) {
              canvasRefs.current.set(content.id, clonedElement);
              
              // Monitor for context loss
              clonedElement.addEventListener('webglcontextlost', (e) => {
                e.preventDefault();
                console.log(`[PipManager] Context lost for ${content.id}`);
                
                setPipStates((prev) => {
                  const newMap = new Map(prev);
                  const state = newMap.get(content.id);
                  if (state) {
                    newMap.set(content.id, { ...state, hasContextLoss: true });
                  }
                  return newMap;
                });
                
                onContextLoss?.(content.id);
              });

              clonedElement.addEventListener('webglcontextrestored', () => {
                console.log(`[PipManager] Context restored for ${content.id}`);
                
                setPipStates((prev) => {
                  const newMap = new Map(prev);
                  const state = newMap.get(content.id);
                  if (state) {
                    newMap.set(content.id, { ...state, hasContextLoss: false });
                  }
                  return newMap;
                });
                
                onContextRestore?.(content.id);
              });
            }
          }

          // Handle PiP window close
          pipWindow.addEventListener('pagehide', () => {
            closePip(content.id);
          });

          // Update state
          setPipStates((prev) => {
            const newMap = new Map(prev);
            newMap.set(content.id, {
              id: content.id,
              title: content.title,
              pipWindow: pipWindow as unknown as DocumentPictureInPicture,
              container,
              isActive: true,
              hasContextLoss: false,
            });
            return newMap;
          });

          if (debugMode) {
            console.log(`[PipManager] Opened PiP window: ${content.id}`);
          }

          return true;
        } catch (error) {
          console.error('[PipManager] Failed to open PiP window:', error);
          return false;
        }
      },
      [isSupported, defaultWidth, defaultHeight, onContextLoss, onContextRestore, debugMode]
    );

    /**
     * Close a specific PiP window
     */
    const closePip = useCallback(
      (id: string): void => {
        setPipStates((prev) => {
          const newMap = new Map(prev);
          const state = newMap.get(id);
          
          if (state) {
            // Clean up container
            state.container?.remove();
            contentRefs.current.delete(id);
            
            // Cancel animation frames
            const frameId = animationFrameRefs.current.get(id);
            if (frameId) {
              cancelAnimationFrame(frameId);
              animationFrameRefs.current.delete(id);
            }
            
            newMap.delete(id);
            
            if (debugMode) {
              console.log(`[PipManager] Closed PiP window: ${id}`);
            }
          }
          
          return newMap;
        });
      },
      [debugMode]
    );

    /**
     * Close all PiP windows
     */
    const closeAll = useCallback((): void => {
      setPipStates((prev) => {
        prev.forEach((_, id) => {
          const state = prev.get(id);
          state?.container?.remove();
        });
        contentRefs.current.clear();
        canvasRefs.current.clear();
        return new Map();
      });
      
      if (debugMode) {
        console.log('[PipManager] Closed all PiP windows');
      }
    }, [debugMode]);

    /**
     * Check if a specific PiP is open
     */
    const isPipOpen = useCallback((id: string): boolean => {
      return pipStates.has(id);
    }, [pipStates]);

    /**
     * Get the PiP window instance
     */
    const getPipWindow = useCallback(
      (id: string): DocumentPictureInPicture | null => {
        const state = pipStates.get(id);
        return state?.pipWindow ?? null;
      },
      [pipStates]
    );

    /**
     * Restore WebGL context after loss
     */
    const restoreContext = useCallback((id: string): void => {
      const canvas = canvasRefs.current.get(id);
      if (canvas) {
        // Force context recreation
        const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
        if (gl) {
          // Reset canvas dimensions to trigger context restoration
          const { width, height } = canvas;
          canvas.width = 0;
          canvas.height = 0;
          canvas.width = width;
          canvas.height = height;
          
          setPipStates((prev) => {
            const newMap = new Map(prev);
            const state = newMap.get(id);
            if (state) {
              newMap.set(id, { ...state, hasContextLoss: false });
            }
            return newMap;
          });
        }
      }
    }, []);

    // Expose methods via ref
    useImperativeHandle(
      ref,
      () => ({
        openPip,
        closePip,
        closeAll,
        isPipOpen,
        getPipWindow,
      }),
      [openPip, closePip, closeAll, isPipOpen, getPipWindow]
    );

    // Cleanup on unmount
    useEffect(() => {
      return () => {
        closeAll();
      };
    }, [closeAll]);

    // Render context loss overlays
    return (
      <>
        {Array.from(pipStates.values()).map(
          (state) =>
            state.hasContextLoss && (
              <div
                key={state.id}
                className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
                onClick={() => restoreContext(state.id)}
              >
                <div className="text-center p-6 rounded-lg border border-red-500/50 bg-[#0a0a1a]">
                  <h3 className="text-red-400 font-bold mb-2">WebGL Context Lost</h3>
                  <p className="text-gray-400 text-sm mb-4">
                    The graphics context was evicted due to memory pressure.
                  </p>
                  <button
                    className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-black font-semibold rounded"
                    onClick={() => restoreContext(state.id)}
                  >
                    Restore Context
                  </button>
                </div>
              </div>
            )
        )}
      </>
    );
  }
);

PipManager.displayName = 'PipManager';

/**
 * Hook to use PipManager in other components
 */
export function usePipManager() {
  const managerRef = React.useRef<PipManagerRef>(null);

  return {
    ref: managerRef,
    openPip: async (content: PipContent): Promise<boolean> => {
      return managerRef.current?.openPip(content) ?? false;
    },
    closePip: (id: string): void => {
      managerRef.current?.closePip(id);
    },
    closeAll: (): void => {
      managerRef.current?.closeAll();
    },
    isPipOpen: (id: string): boolean => {
      return managerRef.current?.isPipOpen(id) ?? false;
    },
  };
}

export default PipManager;
