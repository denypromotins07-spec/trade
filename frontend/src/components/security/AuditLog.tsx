/**
 * AuditLog Component - Immutable Security Audit Log Viewer
 * 
 * Displays cryptographically verified security audit logs including:
 * - Blocked syscall attempts
 * - Memory scrubbing statistics
 * - Core dump prevention events
 * - WER interception records
 * 
 * Features immutable log verification using Merkle proofs.
 */

import React, { useState, useEffect, useCallback } from 'react';

interface AuditLogEntry {
  id: string;
  timestamp: Date;
  category: AuditCategory;
  eventType: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  details?: Record<string, unknown>;
  merkleProof?: string;
  verified: boolean;
}

type AuditCategory = 
  | 'SYSCALL_BLOCK'
  | 'MEMORY_SCRUB'
  | 'CORE_DUMP_PREVENTED'
  | 'WER_INTERCEPTED'
  | 'PANIC_HANDLED'
  | 'SECRET_SCRUBBED';

interface ScrubStats {
  totalBytesScrubbed: number;
  scrubCycles: number;
  gpuVramScrubs: number;
  averageLatencyUs: number;
}

interface SyscallStats {
  totalBlocked: number;
  totalAllowed: number;
  alertsTriggered: number;
  blockedCalls: BlockedCall[];
}

interface BlockedCall {
  category: string;
  api: string;
  pid: number;
  timestamp: Date;
}

interface AuditLogProps {
  autoRefreshMs?: number;
  showVerifiedOnly?: boolean;
  onSecurityAlert?: (entry: AuditLogEntry) => void;
}

const AuditLog: React.FC<AuditLogProps> = ({
  autoRefreshMs = 5000,
  showVerifiedOnly = false,
  onSecurityAlert,
}) => {
  const [entries, setEntries] = useState<AuditLogEntry[]>([]);
  const [scrubStats, setScrubStats] = useState<ScrubStats | null>(null);
  const [syscallStats, setSyscallStats] = useState<SyscallStats | null>(null);
  const [filter, setFilter] = useState<AuditCategory | 'ALL'>('ALL');
  const [verificationStatus, setVerificationStatus] = useState<'verifying' | 'verified' | 'failed'>('verifying');

  // Fetch audit logs
  const fetchAuditLogs = useCallback(async () => {
    try {
      const response = await fetch('/api/security/audit-logs');
      if (response.ok) {
        const data = await response.json();
        setEntries(data.entries || []);
        
        // Check for new critical entries
        const criticalEntries = data.entries.filter(
          (e: AuditLogEntry) => e.severity === 'critical' && !e.verified
        );
        criticalEntries.forEach((entry: AuditLogEntry) => {
          onSecurityAlert?.(entry);
        });
      }
    } catch (error) {
      console.warn('Failed to fetch audit logs:', error);
    }
  }, [onSecurityAlert]);

  // Fetch scrub stats
  const fetchScrubStats = useCallback(async () => {
    try {
      const response = await fetch('/api/security/scrub/stats');
      if (response.ok) {
        const data = await response.json();
        setScrubStats(data);
      }
    } catch (error) {
      console.warn('Failed to fetch scrub stats:', error);
    }
  }, []);

  // Fetch syscall stats
  const fetchSyscallStats = useCallback(async () => {
    try {
      const response = await fetch('/api/security/syscall/stats');
      if (response.ok) {
        const data = await response.json();
        setSyscallStats(data);
      }
    } catch (error) {
      console.warn('Failed to fetch syscall stats:', error);
    }
  }, []);

  // Verify log integrity using Merkle proof
  const verifyLogIntegrity = useCallback(async (entry: AuditLogEntry): Promise<boolean> => {
    if (!entry.merkleProof) return false;

    try {
      const response = await fetch('/api/security/verify-proof', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entryId: entry.id,
          proof: entry.merkleProof,
          data: JSON.stringify(entry),
        }),
      });

      if (response.ok) {
        const result = await response.json();
        return result.verified;
      }
    } catch (error) {
      console.warn('Verification failed:', error);
    }
    return false;
  }, []);

  // Initial fetch and polling
  useEffect(() => {
    fetchAuditLogs();
    fetchScrubStats();
    fetchSyscallStats();

    const interval = setInterval(() => {
      fetchAuditLogs();
      fetchScrubStats();
      fetchSyscallStats();
    }, autoRefreshMs);

    return () => clearInterval(interval);
  }, [autoRefreshMs, fetchAuditLogs, fetchScrubStats, fetchSyscallStats]);

  // Verify all entries on mount
  useEffect(() => {
    if (entries.length === 0) return;

    const verifyAll = async () => {
      setVerificationStatus('verifying');
      
      let allVerified = true;
      for (const entry of entries) {
        if (!await verifyLogIntegrity(entry)) {
          allVerified = false;
          break;
        }
      }

      setVerificationStatus(allVerified ? 'verified' : 'failed');
    };

    verifyAll();
  }, [entries, verifyLogIntegrity]);

  // Filter entries
  const filteredEntries = entries.filter(entry => {
    if (showVerifiedOnly && !entry.verified) return false;
    if (filter !== 'ALL' && entry.category !== filter) return false;
    return true;
  });

  // Format bytes to human readable
  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  // Get severity color
  const getSeverityColor = (severity: string): string => {
    switch (severity) {
      case 'critical': return '#f56565';
      case 'warning': return '#ed8936';
      case 'info': return '#4299e1';
      default: return '#a0aec0';
    }
  };

  // Get category icon
  const getCategoryIcon = (category: AuditCategory): string => {
    switch (category) {
      case 'SYSCALL_BLOCK': return '🚫';
      case 'MEMORY_SCRUB': return '🧹';
      case 'CORE_DUMP_PREVENTED': return '🛡️';
      case 'WER_INTERCEPTED': return '⛔';
      case 'PANIC_HANDLED': return '⚠️';
      case 'SECRET_SCRUBBED': return '🔒';
      default: return '📋';
    }
  };

  return (
    <div className="audit-log-container" style={{ padding: '20px', backgroundColor: '#1a202c', borderRadius: '8px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ color: '#ffffff', margin: 0 }}>Security Audit Log</h2>
        
        {/* Verification Status Badge */}
        <div style={{
          padding: '8px 16px',
          backgroundColor: verificationStatus === 'verified' ? 'rgba(72, 187, 120, 0.2)' : 
                           verificationStatus === 'failed' ? 'rgba(245, 101, 101, 0.2)' : 
                           'rgba(237, 137, 54, 0.2)',
          borderRadius: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: verificationStatus === 'verified' ? '#48bb78' : 
                            verificationStatus === 'failed' ? '#f56565' : '#ed8936',
            animation: verificationStatus === 'verifying' ? 'pulse 1s infinite' : 'none',
          }} />
          <span style={{ 
            color: verificationStatus === 'verified' ? '#48bb78' : 
                   verificationStatus === 'failed' ? '#f56565' : '#ed8936',
            fontSize: '12px',
            fontWeight: 'bold',
          }}>
            {verificationStatus === 'verified' ? 'CRYPTOGRAPHICALLY VERIFIED' :
             verificationStatus === 'failed' ? 'VERIFICATION FAILED' : 'VERIFYING...'}
          </span>
        </div>
      </div>

      {/* Statistics Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px', marginBottom: '20px' }}>
        {/* Syscall Stats Card */}
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#2d3748', 
          borderRadius: '8px',
        }}>
          <div style={{ color: '#a0aec0', fontSize: '12px', marginBottom: '8px' }}>BLOCKED SYSCALLS</div>
          <div style={{ color: '#f56565', fontSize: '24px', fontWeight: 'bold' }}>
            {syscallStats?.totalBlocked ?? '-'}
          </div>
          <div style={{ color: '#48bb78', fontSize: '12px', marginTop: '4px' }}>
            Allowed: {syscallStats?.totalAllowed ?? '-'}
          </div>
        </div>

        {/* Memory Scrub Card */}
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#2d3748', 
          borderRadius: '8px',
        }}>
          <div style={{ color: '#a0aec0', fontSize: '12px', marginBottom: '8px' }}>MEMORY SCRUBBED</div>
          <div style={{ color: '#4299e1', fontSize: '24px', fontWeight: 'bold' }}>
            {scrubStats ? formatBytes(scrubStats.totalBytesScrubbed) : '-'}
          </div>
          <div style={{ color: '#a0aec0', fontSize: '12px', marginTop: '4px' }}>
            {scrubStats?.scrubCycles ?? '-'} cycles
          </div>
        </div>

        {/* GPU VRAM Card */}
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#2d3748', 
          borderRadius: '8px',
        }}>
          <div style={{ color: '#a0aec0', fontSize: '12px', marginBottom: '8px' }}>GPU VRAM SCRUBS</div>
          <div style={{ color: '#ed8936', fontSize: '24px', fontWeight: 'bold' }}>
            {scrubStats?.gpuVramScrubs ?? '-'}
          </div>
          <div style={{ color: '#a0aec0', fontSize: '12px', marginTop: '4px' }}>
            Avg latency: {scrubStats?.averageLatencyUs.toFixed(1) ?? '-'} μs
          </div>
        </div>
      </div>

      {/* Filter Controls */}
      <div style={{ marginBottom: '15px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        {(['ALL', 'SYSCALL_BLOCK', 'MEMORY_SCRUB', 'CORE_DUMP_PREVENTED', 'WER_INTERCEPTED', 'PANIC_HANDLED', 'SECRET_SCRUBBED'] as const).map(cat => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            style={{
              padding: '8px 16px',
              backgroundColor: filter === cat ? '#4299e1' : '#4a5568',
              color: '#ffffff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              transition: 'background-color 0.2s',
            }}
          >
            {cat.replace(/_/g, ' ')}
          </button>
        ))}
      </div>

      {/* Log Entries Table */}
      <div style={{ 
        backgroundColor: '#000000', 
        borderRadius: '8px', 
        overflow: 'hidden',
        maxHeight: '400px',
        overflowY: 'auto',
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ backgroundColor: '#2d3748' }}>
              <th style={{ padding: '12px', textAlign: 'left', color: '#a0aec0', fontSize: '12px' }}>Time</th>
              <th style={{ padding: '12px', textAlign: 'left', color: '#a0aec0', fontSize: '12px' }}>Category</th>
              <th style={{ padding: '12px', textAlign: 'left', color: '#a0aec0', fontSize: '12px' }}>Event</th>
              <th style={{ padding: '12px', textAlign: 'left', color: '#a0aec0', fontSize: '12px' }}>Message</th>
              <th style={{ padding: '12px', textAlign: 'center', color: '#a0aec0', fontSize: '12px' }}>Verified</th>
            </tr>
          </thead>
          <tbody>
            {filteredEntries.map((entry) => (
              <tr 
                key={entry.id} 
                style={{ borderBottom: '1px solid #2d3748', backgroundColor: entry.severity === 'critical' ? 'rgba(245, 101, 101, 0.1)' : 'transparent' }}
              >
                <td style={{ padding: '12px', color: '#a0aec0', fontSize: '12px', fontFamily: 'monospace' }}>
                  {entry.timestamp.toLocaleTimeString()}
                </td>
                <td style={{ padding: '12px', color: '#ffffff', fontSize: '12px' }}>
                  <span style={{ marginRight: '8px' }}>{getCategoryIcon(entry.category)}</span>
                  {entry.category.replace(/_/g, ' ')}
                </td>
                <td style={{ padding: '12px', color: getSeverityColor(entry.severity), fontSize: '12px', fontWeight: 'bold' }}>
                  {entry.eventType}
                </td>
                <td style={{ padding: '12px', color: '#e2e8f0', fontSize: '12px', maxWidth: '300px' }}>
                  {entry.message}
                </td>
                <td style={{ padding: '12px', textAlign: 'center' }}>
                  {entry.verified ? (
                    <span style={{ color: '#48bb78' }}>✓</span>
                  ) : (
                    <span style={{ color: '#f56565' }}>✗</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filteredEntries.length === 0 && (
          <div style={{ padding: '40px', textAlign: 'center', color: '#a0aec0' }}>
            No audit entries found
          </div>
        )}
      </div>
    </div>
  );
};

export default AuditLog;
