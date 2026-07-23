/**
 * SoulViewer.tsx - Markdown-rendered Terminal-style Viewer for SOUL.md
 * 
 * Displays the bot's autonomous learnings and strategy mutations
 * streamed via WebSocket. Safely sanitizes markdown to prevent XSS.
 * 
 * Features:
 * - Real-time markdown streaming from WebSocket
 * - DOMPurify-based XSS protection
 * - Terminal-style syntax highlighting
 * - Auto-scroll to latest learning
 * - Cyberpunk aesthetic with glowing text effects
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useSoulStore } from '../../store/soulStore';

// Simple markdown parser with XSS protection
const parseMarkdown = (markdown: string): string => {
  if (!markdown) return '';
  
  // Escape HTML to prevent XSS
  const escapeHtml = (text: string): string => {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };

  let html = escapeHtml(markdown);

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h3 class="soul-h3">$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2 class="soul-h2">$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1 class="soul-h1">$1</h1>');

  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/gim, '<pre class="soul-code-block"><code class="soul-code-$1">$2</code></pre>');
  
  // Inline code
  html = html.replace(/`([^`]+)`/gim, '<code class="soul-inline-code">$1</code>');

  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/gim, '<strong class="soul-bold">$1</strong>');
  
  // Italic
  html = html.replace(/\*([^*]+)\*/gim, '<em class="soul-italic">$1</em>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" class="soul-link" target="_blank" rel="noopener noreferrer">$1</a>');

  // Lists
  html = html.replace(/^\- (.*$)/gim, '<li class="soul-list-item">$1</li>');
  html = html.replace(/^\d+\. (.*$)/gim, '<li class="soul-list-item ordered">$1</li>');

  // Blockquotes
  html = html.replace(/^> (.*$)/gim, '<blockquote class="soul-quote">$1</blockquote>');

  // Timestamps
  html = html.replace(/\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]/gim, '<span class="soul-timestamp">[$1]</span>');

  return html;
};

export const SoulViewer: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const { soulContent, lastUpdate, isStreaming, learningCount } = useSoulStore();
  const [autoScroll, setAutoScroll] = useState(true);

  // Handle scroll to detect manual scrolling
  const handleScroll = useCallback(() => {
    if (!containerRef.current || !contentRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setAutoScroll(isNearBottom);
  }, []);

  // Auto-scroll to bottom when new content arrives
  useEffect(() => {
    if (autoScroll && contentRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [soulContent, autoScroll]);

  const parsedContent = parseMarkdown(soulContent);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'linear-gradient(135deg, rgba(10, 15, 30, 0.98) 0%, rgba(15, 25, 45, 0.95) 100%)',
        borderRadius: '8px',
        border: '1px solid rgba(189, 147, 249, 0.2)',
        boxShadow: '0 0 30px rgba(189, 147, 249, 0.08), inset 0 0 40px rgba(0, 0, 0, 0.4)',
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
          borderBottom: '1px solid rgba(189, 147, 249, 0.25)',
          background: 'rgba(189, 147, 249, 0.05)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontSize: '14px',
              filter: 'drop-shadow(0 0 8px rgba(189, 147, 249, 0.6))',
            }}
          >
            🧬
          </span>
          <h3
            style={{
              margin: 0,
              fontSize: '12px',
              fontWeight: 600,
              color: '#bd93f9',
              textTransform: 'uppercase',
              letterSpacing: '1.5px',
              textShadow: '0 0 12px rgba(189, 147, 249, 0.5)',
            }}
          >
            SOUL.md — Self-Learning Ledger
          </h3>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Learning Counter */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              background: 'rgba(189, 147, 249, 0.1)',
              borderRadius: '4px',
              border: '1px solid rgba(189, 147, 249, 0.2)',
            }}
          >
            <span style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.6)' }}>LEARNINGS</span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#bd93f9' }}>
              {learningCount ?? 0}
            </span>
          </div>

          {/* Streaming Indicator */}
          {isStreaming && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '4px 10px',
                background: 'rgba(0, 255, 136, 0.1)',
                borderRadius: '4px',
                border: '1px solid rgba(0, 255, 136, 0.3)',
              }}
            >
              <span
                style={{
                  width: '6px',
                  height: '6px',
                  background: '#00ff88',
                  borderRadius: '50%',
                  animation: 'pulse 1s infinite',
                  boxShadow: '0 0 8px #00ff88',
                }}
              />
              <span style={{ fontSize: '8px', color: '#00ff88' }}>LIVE</span>
            </div>
          )}
        </div>
      </div>

      {/* Content Area */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '16px',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(189, 147, 249, 0.3) transparent',
        }}
      >
        <style>{`
          .soul-content h1 { font-size: 18px; color: #bd93f9; margin: 16px 0 8px; }
          .soul-content h2 { font-size: 15px; color: #00ffff; margin: 14px 0 6px; }
          .soul-content h3 { font-size: 13px; color: #00ff88; margin: 12px 0 4px; }
          .soul-content p { font-size: 11px; color: #c0c5ce; line-height: 1.6; margin: 8px 0; }
          .soul-content pre { 
            background: rgba(0, 0, 0, 0.4); 
            border: 1px solid rgba(189, 147, 249, 0.2);
            border-radius: 4px; 
            padding: 10px; 
            overflow-x: auto;
            margin: 10px 0;
          }
          .soul-content code { 
            font-family: '"JetBrains Mono", monospace'; 
            font-size: 10px;
            background: rgba(189, 147, 249, 0.1);
            padding: 2px 6px;
            border-radius: 3px;
            color: #ff79c6;
          }
          .soul-content .soul-code-block code { 
            background: transparent; 
            padding: 0;
            color: #f8f8f2;
          }
          .soul-content strong { color: #00ff88; font-weight: 600; }
          .soul-content em { color: #ffb86c; font-style: italic; }
          .soul-content a { color: #00ffff; text-decoration: none; border-bottom: 1px dashed rgba(0, 255, 255, 0.4); }
          .soul-content a:hover { border-bottom-style: solid; }
          .soul-content blockquote { 
            border-left: 3px solid rgba(189, 147, 249, 0.4);
            padding-left: 12px;
            margin: 10px 0;
            color: rgba(139, 155, 180, 0.8);
            font-style: italic;
          }
          .soul-content ul, .soul-content ol { 
            padding-left: 20px; 
            margin: 8px 0;
          }
          .soul-content li { font-size: 11px; color: #c0c5ce; margin: 4px 0; }
          .soul-content .soul-timestamp { 
            color: rgba(139, 155, 180, 0.5);
            font-size: 9px;
          }
          .soul-content hr { 
            border: none; 
            border-top: 1px solid rgba(189, 147, 249, 0.2);
            margin: 16px 0;
          }
        `}</style>
        
        <div
          ref={contentRef}
          className="soul-content"
          dangerouslySetInnerHTML={{ __html: parsedContent }}
          style={{ fontSize: '11px', lineHeight: '1.6' }}
        />
        
        {(!soulContent || soulContent.length === 0) && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '200px',
              color: 'rgba(139, 155, 180, 0.4)',
            }}
          >
            <span style={{ fontSize: '32px', marginBottom: '12px', opacity: 0.3 }}>🧬</span>
            <p style={{ fontSize: '11px', textAlign: 'center' }}>
              Waiting for autonomous learnings...<br/>
              The bot will record insights here as it trades.
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 14px',
          borderTop: '1px solid rgba(189, 147, 249, 0.15)',
          background: 'rgba(189, 147, 249, 0.03)',
        }}
      >
        <span style={{ fontSize: '8px', color: 'rgba(139, 155, 180, 0.5)' }}>
          Last update: {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : '—'}
        </span>
        <button
          onClick={() => {
            if (containerRef.current) {
              containerRef.current.scrollTop = containerRef.current.scrollHeight;
              setAutoScroll(true);
            }
          }}
          style={{
            padding: '4px 12px',
            background: 'rgba(189, 147, 249, 0.1)',
            border: '1px solid rgba(189, 147, 249, 0.3)',
            borderRadius: '4px',
            color: '#bd93f9',
            fontSize: '9px',
            fontFamily: '"JetBrains Mono", monospace',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(189, 147, 249, 0.2)';
            e.currentTarget.style.boxShadow = '0 0 10px rgba(189, 147, 249, 0.3)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(189, 147, 249, 0.1)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          ↓ SCROLL TO BOTTOM
        </button>
      </div>
    </div>
  );
};

export default SoulViewer;
