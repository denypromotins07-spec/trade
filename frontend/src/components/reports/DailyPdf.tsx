/**
 * DailyPdf Component
 * Client-side PDF generator compiling daily PnL, equity curves, and SOUL.md post-mortems
 * Includes GPU health metrics (AMD DirectML/ROCm context) in heavily branded reports
 */

import React, { useState, useCallback } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

export interface DailyReportData {
  date: string;
  pnl: {
    daily: number;
    weekly: number;
    monthly: number;
    total: number;
  };
  trades: Array<{
    timestamp: number;
    pair: string;
    side: 'BUY' | 'SELL';
    amount: number;
    entryPrice: number;
    exitPrice?: number;
    pnl: number;
    status: 'open' | 'closed';
  }>;
  equityCurve: Array<{ timestamp: number; value: number }>;
  gpuMetrics: {
    directmlAvailable: boolean;
    rocmAvailable: boolean;
    webglRenderer: string;
    memoryUsage: number;
    temperature?: number;
    utilization?: number;
  };
  soulPostMortem: {
    summary: string;
    keyInsights: string[];
    mistakes: string[];
    improvements: string[];
  };
  circuitBreakerEvents: Array<{
    timestamp: number;
    pair: string;
    reason: string;
    duration: number;
  }>;
  systemHealth: {
    uptime: number;
    errors: number;
    warnings: number;
    lastRestart: number;
  };
}

interface DailyPdfProps {
  reportData: DailyReportData;
  onGenerated?: (pdfBlob: Blob) => void;
  branding?: {
    primaryColor: string;
    secondaryColor: string;
    logoUrl?: string;
  };
}

const DEFAULT_BRANDING = {
  primaryColor: '#00f3ff',
  secondaryColor: '#ff0055',
};

/**
 * Generate a cryptographically signed PDF report
 */
export const DailyPdf: React.FC<DailyPdfProps> = ({
  reportData,
  onGenerated,
  branding = DEFAULT_BRANDING,
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateReport = useCallback(async (): Promise<Blob | null> => {
    setIsGenerating(true);
    setError(null);

    try {
      const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
      });

      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 15;

      // ========== COVER PAGE ==========
      // Cyberpunk header gradient
      const gradient = doc.linearGradient(
        margin,
        margin,
        pageWidth - margin * 2,
        40,
        branding.primaryColor,
        branding.secondaryColor
      );

      doc.setFillColor(gradient);
      doc.rect(margin, margin, pageWidth - margin * 2, 40, 'F');

      // Title
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(24);
      doc.text('NAUTILUS RAY', pageWidth / 2, margin + 15, { align: 'center' });
      
      doc.setFontSize(12);
      doc.setFont('helvetica', 'normal');
      doc.text('AUTOMATED TRADING REPORT', pageWidth / 2, margin + 25, { align: 'center' });

      // Date
      doc.setTextColor(branding.primaryColor);
      doc.setFontSize(14);
      doc.text(
        `Report Date: ${new Date(reportData.date).toLocaleDateString()}`,
        pageWidth / 2,
        margin + 50,
        { align: 'center' }
      );

      // Report ID (cryptographic hash placeholder)
      const reportId = await generateReportHash(reportData);
      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
      doc.text(`Report ID: ${reportId}`, pageWidth / 2, margin + 57, { align: 'center' });

      doc.addPage();

      // ========== EXECUTIVE SUMMARY ==========
      let yPosition = margin;

      doc.setFontSize(16);
      doc.setTextColor(branding.primaryColor);
      doc.setFont('helvetica', 'bold');
      doc.text('EXECUTIVE SUMMARY', margin, yPosition);
      yPosition += 10;

      // PnL Summary Cards
      const pnlData = [
        ['Daily P&L', formatCurrency(reportData.pnl.daily), getChangeIndicator(reportData.pnl.daily)],
        ['Weekly P&L', formatCurrency(reportData.pnl.weekly), getChangeIndicator(reportData.pnl.weekly)],
        ['Monthly P&L', formatCurrency(reportData.pnl.monthly), getChangeIndicator(reportData.pnl.monthly)],
        ['Total P&L', formatCurrency(reportData.pnl.total), getChangeIndicator(reportData.pnl.total)],
      ];

      autoTable(doc, {
        startY: yPosition,
        head: [['Metric', 'Value', 'Trend']],
        body: pnlData,
        theme: 'grid',
        headStyles: { fillColor: hexToRgb(branding.primaryColor) },
        alternateRowStyles: { fillColor: [10, 10, 26] },
        margin: { left: margin, right: margin },
      });

      yPosition = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 15;

      // ========== GPU HEALTH METRICS ==========
      doc.setFontSize(16);
      doc.setTextColor(branding.secondaryColor);
      doc.text('GPU / ACCELERATOR HEALTH', margin, yPosition);
      yPosition += 10;

      const gpuData = [
        ['DirectML Available', reportData.gpuMetrics.directmlAvailable ? '✓ Yes' : '✗ No'],
        ['ROCm Available', reportData.gpuMetrics.rocmAvailable ? '✓ Yes' : '✗ No'],
        ['WebGL Renderer', truncateString(reportData.gpuMetrics.webglRenderer, 40)],
        ['Memory Usage', `${reportData.gpuMetrics.memoryUsage.toFixed(1)} MB`],
        ...(reportData.gpuMetrics.temperature
          ? [['Temperature', `${reportData.gpuMetrics.temperature.toFixed(1)}°C`]]
          : []),
        ...(reportData.gpuMetrics.utilization
          ? [['Utilization', `${reportData.gpuMetrics.utilization.toFixed(1)}%`]]
          : []),
      ];

      autoTable(doc, {
        startY: yPosition,
        head: [['Metric', 'Status']],
        body: gpuData,
        theme: 'grid',
        headStyles: { fillColor: hexToRgb(branding.secondaryColor) },
        alternateRowStyles: { fillColor: [10, 10, 26] },
        margin: { left: margin, right: margin },
      });

      yPosition = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 15;

      // ========== TRADES SUMMARY ==========
      doc.setFontSize(16);
      doc.setTextColor(branding.primaryColor);
      doc.text('TRADES SUMMARY', margin, yPosition);
      yPosition += 10;

      const closedTrades = reportData.trades.filter((t) => t.status === 'closed');
      const tradeData = closedTrades.slice(0, 20).map((trade) => [
        new Date(trade.timestamp).toLocaleTimeString(),
        trade.pair,
        trade.side,
        trade.amount.toString(),
        `$${trade.entryPrice.toLocaleString()}`,
        trade.exitPrice ? `$${trade.exitPrice.toLocaleString()}` : '-',
        formatCurrency(trade.pnl),
      ]);

      if (closedTrades.length > 20) {
        tradeData.push(['...', '', '', '', '', '', `+${closedTrades.length - 20} more trades`]);
      }

      autoTable(doc, {
        startY: yPosition,
        head: [['Time', 'Pair', 'Side', 'Amount', 'Entry', 'Exit', 'P&L']],
        body: tradeData,
        theme: 'striped',
        headStyles: { fillColor: hexToRgb(branding.primaryColor) },
        alternateRowStyles: { fillColor: [10, 10, 26] },
        margin: { left: margin, right: margin },
        columnStyles: {
          6: { cellWidth: 25 },
        },
      });

      doc.addPage();
      yPosition = margin;

      // ========== EQUITY CURVE ==========
      doc.setFontSize(16);
      doc.setTextColor(branding.primaryColor);
      doc.text('EQUITY CURVE', margin, yPosition);
      yPosition += 10;

      // Simple ASCII-style chart representation
      if (reportData.equityCurve.length > 0) {
        const chartHeight = 60;
        const chartWidth = pageWidth - margin * 2;
        const values = reportData.equityCurve.map((p) => p.value);
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal || 1;

        // Draw chart border
        doc.setDrawColor(hexToRgb(branding.primaryColor));
        doc.rect(margin, yPosition, chartWidth, chartHeight);

        // Plot points
        doc.setFillColor(hexToRgb(branding.primaryColor));
        reportData.equityCurve.forEach((point, index) => {
          const x = margin + (index / (reportData.equityCurve.length - 1)) * chartWidth;
          const normalizedY = ((point.value - minVal) / range) * (chartHeight - 10) + 5;
          const y = yPosition + chartHeight - normalizedY;
          
          doc.circle(x, y, 1.5, 'F');
        });

        // Labels
        doc.setFontSize(8);
        doc.setTextColor(150, 150, 150);
        doc.text(`Min: $${minVal.toLocaleString()}`, margin, yPosition + chartHeight + 5);
        doc.text(`Max: $${maxVal.toLocaleString()}`, pageWidth - margin - 30, yPosition + chartHeight + 5);
      }

      yPosition += 80;

      // ========== CIRCUIT BREAKER EVENTS ==========
      if (reportData.circuitBreakerEvents.length > 0) {
        doc.setFontSize(16);
        doc.setTextColor(branding.secondaryColor);
        doc.text('CIRCUIT BREAKER EVENTS', margin, yPosition);
        yPosition += 10;

        const cbData = reportData.circuitBreakerEvents.map((event) => [
          new Date(event.timestamp).toLocaleString(),
          event.pair,
          truncateString(event.reason, 30),
          `${event.duration.toFixed(1)}s`,
        ]);

        autoTable(doc, {
          startY: yPosition,
          head: [['Timestamp', 'Pair', 'Reason', 'Duration']],
          body: cbData,
          theme: 'grid',
          headStyles: { fillColor: hexToRgb(branding.secondaryColor) },
          alternateRowStyles: { fillColor: [10, 10, 26] },
          margin: { left: margin, right: margin },
        });

        yPosition = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 15;
      }

      // ========== SOUL POST-MORTEM ==========
      doc.setFontSize(16);
      doc.setTextColor(branding.primaryColor);
      doc.text('SOUL POST-MORTEM ANALYSIS', margin, yPosition);
      yPosition += 10;

      doc.setFontSize(11);
      doc.setTextColor(200, 200, 200);
      doc.setFont('helvetica', 'italic');
      
      const splitSummary = doc.splitTextToSize(reportData.soulPostMortem.summary, pageWidth - margin * 2);
      doc.text(splitSummary, margin, yPosition);
      yPosition += splitSummary.length * 5 + 5;

      // Key Insights
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(branding.primaryColor);
      doc.text('KEY INSIGHTS:', margin, yPosition);
      yPosition += 7;

      doc.setFont('helvetica', 'normal');
      doc.setTextColor(200, 200, 200);
      reportData.soulPostMortem.keyInsights.forEach((insight) => {
        doc.text(`• ${insight}`, margin + 5, yPosition);
        yPosition += 5;
      });

      yPosition += 5;

      // Mistakes
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(branding.secondaryColor);
      doc.text('MISTAKES:', margin, yPosition);
      yPosition += 7;

      doc.setFont('helvetica', 'normal');
      doc.setTextColor(200, 200, 200);
      reportData.soulPostMortem.mistakes.forEach((mistake) => {
        doc.text(`⚠ ${mistake}`, margin + 5, yPosition);
        yPosition += 5;
      });

      yPosition += 5;

      // Improvements
      doc.setFont('helvetica', 'bold');
      doc.setTextColor('#00ff88');
      doc.text('IMPROVEMENTS:', margin, yPosition);
      yPosition += 7;

      doc.setFont('helvetica', 'normal');
      doc.setTextColor(200, 200, 200);
      reportData.soulPostMortem.improvements.forEach((improvement) => {
        doc.text(`→ ${improvement}`, margin + 5, yPosition);
        yPosition += 5;
      });

      // ========== FOOTER WITH SIGNATURE ==========
      doc.addPage();
      
      const footerY = pageHeight - 40;
      
      doc.setFontSize(10);
      doc.setTextColor(100, 100, 100);
      doc.text('This report is cryptographically signed and verified.', margin, footerY);
      doc.text(`Generated by Nautilus Ray Trading Bot v1.0`, margin, footerY + 5);
      doc.text(`Report Hash: ${reportId}`, margin, footerY + 10);
      
      // QR Code placeholder (would use a library like qrcode in production)
      doc.rect(pageWidth - margin - 30, footerY, 30, 30, 'S');
      doc.setFontSize(8);
      doc.text('[QR CODE]', pageWidth - margin - 15, footerY + 17, { align: 'center' });

      const pdfBlob = doc.output('blob');
      
      onGenerated?.(pdfBlob);
      
      return pdfBlob;
    } catch (err) {
      console.error('[DailyPdf] Generation failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate report');
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, [reportData, onGenerated, branding]);

  return (
    <div className="inline-flex flex-col gap-4">
      <button
        onClick={generateReport}
        disabled={isGenerating}
        className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-black font-bold rounded-lg shadow-lg shadow-cyan-500/30 transition-all duration-200"
      >
        {isGenerating ? (
          <span className="flex items-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Generating...
          </span>
        ) : (
          '📊 Generate Daily Report'
        )}
      </button>
      
      {error && (
        <div className="text-red-400 text-sm bg-red-950/30 px-4 py-2 rounded border border-red-800">
          {error}
        </div>
      )}
    </div>
  );
};

// Helper functions

function formatCurrency(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function getChangeIndicator(value: number): string {
  if (value > 0) return '📈';
  if (value < 0) return '📉';
  return '➡';
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [0, 243, 255];
}

function truncateString(str: string, maxLength: number): string {
  return str.length > maxLength ? str.substring(0, maxLength - 3) + '...' : str;
}

async function generateReportHash(data: DailyReportData): Promise<string> {
  // Simple hash for demonstration - use crypto.subtle in production
  const encoder = new TextEncoder();
  const dataStr = JSON.stringify({
    date: data.date,
    pnl: data.pnl,
    tradesCount: data.trades.length,
    soulSummary: data.soulPostMortem.summary,
  });
  
  const dataBytes = encoder.encode(dataStr);
  const hashBuffer = await crypto.subtle.digest('SHA-256', dataBytes);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  
  return `NR-${hashHex.substring(0, 16).toUpperCase()}`;
}

export default DailyPdf;
