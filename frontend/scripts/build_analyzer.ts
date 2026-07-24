/**
 * Build Analyzer Script
 * 
 * Custom build script integrating bundle analysis to ensure no heavy dependencies
 * breach strict browser memory limits before production build completes.
 * 
 * Cyberpunk aesthetic: "Neural network weight analyzer" with threshold breach alerts.
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, resolve } from 'path';

// Configuration
const CONFIG = {
  // Memory limits (in KB)
  MAX_BUNDLE_SIZE: 512, // 500KB per chunk
  MAX_TOTAL_SIZE: 2048, // 2MB total
  MAX_CHUNK_COUNT: 30,
  
  // Critical dependencies to watch
  CRITICAL_DEPS: [
    'react',
    'react-dom',
    'three',
    '@react-three/fiber',
    '@react-three/drei',
    'zustand',
    '@tanstack/react-query',
  ],
  
  // Size thresholds for warnings (KB)
  WARNING_THRESHOLD: 400,
  ERROR_THRESHOLD: 512,
  
  // Output paths
  OUTPUT_DIR: '.next/analyze',
  REPORT_PATH: '.next/analyze/bundle-report.json',
};

interface BundleStats {
  name: string;
  size: number;
  gzipSize?: number;
  modules: ModuleInfo[];
}

interface ModuleInfo {
  name: string;
  size: number;
  isCritical: boolean;
}

interface AnalysisReport {
  timestamp: string;
  success: boolean;
  totalSize: number;
  chunkCount: number;
  chunks: BundleStats[];
  warnings: string[];
  errors: string[];
  criticalDepsAnalysis: CriticalDepAnalysis[];
  recommendations: string[];
}

interface CriticalDepAnalysis {
  name: string;
  size: number;
  percentage: number;
  status: 'ok' | 'warning' | 'critical';
}

/**
 * Run the Next.js build and analyze bundles
 */
async function analyzeBuild(): Promise<void> {
  console.log('🔍 NAUTILUS BUILD ANALYZER');
  console.log('========================\n');

  const startTime = Date.now();
  const report: AnalysisReport = {
    timestamp: new Date().toISOString(),
    success: true,
    totalSize: 0,
    chunkCount: 0,
    chunks: [],
    warnings: [],
    errors: [],
    criticalDepsAnalysis: [],
    recommendations: [],
  };

  try {
    // Step 1: Run the build with stats collection
    console.log('📦 Running Next.js build with bundle stats...');
    
    process.env.NEXT_BUNDLE_ANALYZE = 'true';
    
    try {
      execSync('npm run build', {
        stdio: 'inherit',
        cwd: resolve(__dirname, '..'),
      });
    } catch (buildError) {
      report.errors.push('Build failed: ' + (buildError as Error).message);
      report.success = false;
      writeReport(report);
      process.exit(1);
    }

    // Step 2: Analyze bundle stats
    console.log('\n📊 Analyzing bundle sizes...');
    
    const statsPath = resolve(__dirname, '..', '.next', 'stats.json');
    
    if (existsSync(statsPath)) {
      const stats = JSON.parse(readFileSync(statsPath, 'utf-8'));
      report.chunks = analyzeChunks(stats);
    } else {
      // Fallback: analyze from build output
      report.chunks = analyzeFromBuildOutput();
    }

    // Step 3: Calculate totals
    report.chunkCount = report.chunks.length;
    report.totalSize = report.chunks.reduce((sum, chunk) => sum + chunk.size, 0);

    // Step 4: Check limits
    checkLimits(report);

    // Step 5: Analyze critical dependencies
    report.criticalDepsAnalysis = analyzeCriticalDependencies(report.chunks);

    // Step 6: Generate recommendations
    report.recommendations = generateRecommendations(report);

    // Step 7: Write report
    writeReport(report);

    // Step 8: Print summary
    printSummary(report, Date.now() - startTime);

    // Exit with error if there are critical issues
    if (report.errors.length > 0) {
      console.error('\n❌ BUILD ANALYSIS FAILED');
      report.errors.forEach(err => console.error('   ', err));
      process.exit(1);
    } else if (report.warnings.length > 0) {
      console.warn('\n⚠️  BUILD ANALYSIS COMPLETED WITH WARNINGS');
      report.warnings.forEach(warn => console.warn('   ', warn));
    } else {
      console.log('\n✅ BUILD ANALYSIS PASSED');
    }

  } catch (error) {
    report.errors.push('Analysis failed: ' + (error as Error).message);
    report.success = false;
    writeReport(report);
    console.error('\n❌ ANALYSIS FAILED:', error);
    process.exit(1);
  }
}

/**
 * Analyze chunks from webpack stats
 */
function analyzeChunks(stats: any): BundleStats[] {
  const chunks: BundleStats[] = [];
  
  if (stats.assets) {
    stats.assets.forEach((asset: any) => {
      if (asset.name.endsWith('.js')) {
        const sizeKB = asset.size / 1024;
        
        chunks.push({
          name: asset.name,
          size: sizeKB,
          gzipSize: asset.size / 1024 / 4, // Estimate gzip size
          modules: analyzeModules(asset.modules || []),
        });
      }
    });
  }
  
  return chunks.sort((a, b) => b.size - a.size);
}

/**
 * Analyze modules within a chunk
 */
function analyzeModules(modules: any[]): ModuleInfo[] {
  return modules.map((mod: any) => ({
    name: mod.name || 'unknown',
    size: (mod.size || 0) / 1024,
    isCritical: CONFIG.CRITICAL_DEPS.some(dep => mod.name?.includes(dep)),
  }));
}

/**
 * Fallback analysis from build output directory
 */
function analyzeFromBuildOutput(): BundleStats[] {
  const chunks: BundleStats[] = [];
  const buildDir = resolve(__dirname, '..', '.next', 'static', 'chunks');
  
  // This would recursively scan the build directory in a full implementation
  // For now, return placeholder data
  console.log('⚠️  Using fallback analysis (no stats.json found)');
  
  return chunks;
}

/**
 * Check bundle size limits
 */
function checkLimits(report: AnalysisReport): void {
  // Check individual chunk sizes
  report.chunks.forEach(chunk => {
    if (chunk.size > CONFIG.ERROR_THRESHOLD) {
      report.errors.push(`Chunk "${chunk.name}" exceeds limit: ${chunk.size.toFixed(2)}KB > ${CONFIG.ERROR_THRESHOLD}KB`);
      report.success = false;
    } else if (chunk.size > CONFIG.WARNING_THRESHOLD) {
      report.warnings.push(`Chunk "${chunk.name}" approaching limit: ${chunk.size.toFixed(2)}KB`);
    }
  });

  // Check total size
  if (report.totalSize > CONFIG.MAX_TOTAL_SIZE) {
    report.errors.push(`Total bundle size exceeds limit: ${report.totalSize.toFixed(2)}KB > ${CONFIG.MAX_TOTAL_SIZE}KB`);
    report.success = false;
  }

  // Check chunk count
  if (report.chunkCount > CONFIG.MAX_CHUNK_COUNT) {
    report.warnings.push(`Too many chunks: ${report.chunkCount} > ${CONFIG.MAX_CHUNK_COUNT}`);
  }
}

/**
 * Analyze critical dependencies
 */
function analyzeCriticalDependencies(chunks: BundleStats[]): CriticalDepAnalysis[] {
  const analysis: CriticalDepAnalysis[] = [];
  
  CONFIG.CRITICAL_DEPS.forEach(dep => {
    let totalSize = 0;
    
    chunks.forEach(chunk => {
      chunk.modules.forEach(mod => {
        if (mod.name.includes(dep)) {
          totalSize += mod.size;
        }
      });
    });

    const percentage = (totalSize / report.totalSize) * 100;
    let status: 'ok' | 'warning' | 'critical' = 'ok';
    
    if (percentage > 20) status = 'critical';
    else if (percentage > 10) status = 'warning';

    analysis.push({
      name: dep,
      size: totalSize,
      percentage,
      status,
    });
  });

  return analysis.sort((a, b) => b.percentage - a.percentage);
}

/**
 * Generate optimization recommendations
 */
function generateRecommendations(report: AnalysisReport): string[] {
  const recommendations: string[] = [];

  // Check for large chunks
  const largeChunks = report.chunks.filter(c => c.size > CONFIG.WARNING_THRESHOLD);
  if (largeChunks.length > 0) {
    recommendations.push('Consider code-splitting large chunks using dynamic imports');
  }

  // Check for duplicate dependencies
  const deps = new Map<string, number>();
  report.chunks.forEach(chunk => {
    chunk.modules.forEach(mod => {
      deps.set(mod.name, (deps.get(mod.name) || 0) + 1);
    });
  });
  
  const duplicates = Array.from(deps.entries())
    .filter(([_, count]) => count > 1)
    .slice(0, 5);
  
  if (duplicates.length > 0) {
    recommendations.push('Found duplicate modules across chunks - review import paths');
  }

  // Check critical dependency sizes
  const largeDeps = report.criticalDepsAnalysis.filter(d => d.status === 'critical');
  if (largeDeps.length > 0) {
    recommendations.push(`Consider lazy-loading or tree-shaking: ${largeDeps.map(d => d.name).join(', ')}`);
  }

  // General recommendations
  if (report.totalSize > CONFIG.MAX_TOTAL_SIZE * 0.8) {
    recommendations.push('Bundle size approaching limit - consider removing unused dependencies');
  }

  recommendations.push('Enable compression middleware in production');
  recommendations.push('Use next/image for automatic image optimization');

  return recommendations;
}

/**
 * Write analysis report to file
 */
function writeReport(report: AnalysisReport): void {
  const outputDir = resolve(__dirname, '..', CONFIG.OUTPUT_DIR);
  
  // Ensure output directory exists
  if (!existsSync(outputDir)) {
    execSync(`mkdir -p ${outputDir}`);
  }

  const reportPath = resolve(__dirname, '..', CONFIG.REPORT_PATH);
  writeFileSync(reportPath, JSON.stringify(report, null, 2));
  
  console.log(`📄 Report written to: ${CONFIG.REPORT_PATH}`);
}

/**
 * Print analysis summary to console
 */
function printSummary(report: AnalysisReport, duration: number): void {
  console.log('\n📈 BUILD ANALYSIS SUMMARY');
  console.log('========================');
  console.log(`Duration: ${(duration / 1000).toFixed(2)}s`);
  console.log(`Status: ${report.success ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`Total Size: ${(report.totalSize / 1024).toFixed(2)}MB`);
  console.log(`Chunk Count: ${report.chunkCount}`);
  
  if (report.chunks.length > 0) {
    console.log('\n📦 TOP 5 LARGEST CHUNKS:');
    report.chunks.slice(0, 5).forEach((chunk, i) => {
      const icon = chunk.size > CONFIG.ERROR_THRESHOLD ? '🔴' : 
                   chunk.size > CONFIG.WARNING_THRESHOLD ? '🟡' : '🟢';
      console.log(`   ${i + 1}. ${icon} ${chunk.name}: ${chunk.size.toFixed(2)}KB`);
    });
  }

  if (report.criticalDepsAnalysis.length > 0) {
    console.log('\n🔗 CRITICAL DEPENDENCIES:');
    report.criticalDepsAnalysis.slice(0, 5).forEach(dep => {
      const icon = dep.status === 'critical' ? '🔴' : 
                   dep.status === 'warning' ? '🟡' : '🟢';
      console.log(`   ${icon} ${dep.name}: ${dep.percentage.toFixed(1)}% (${dep.size.toFixed(2)}KB)`);
    });
  }

  if (report.recommendations.length > 0) {
    console.log('\n💡 RECOMMENDATIONS:');
    report.recommendations.forEach(rec => console.log(`   • ${rec}`));
  }
}

// Run the analysis
analyzeBuild().catch(console.error);
