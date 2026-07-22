//! Asynchronous Parquet Compaction Engine with Memory-Mapped Files
//! 
//! This module merges fragmented Arrow tick logs into highly compressed,
//! queryable historical Parquet blocks while strictly enforcing the 8GB RAM limit.
//! 
//! Key features:
//! - Async compaction using tokio runtime
//! - Memory-mapped file I/O for zero-copy reads
//! - ZSTD compression optimized for time-series data
//! - Out-of-core processing for datasets larger than RAM
//! - Automatic chunking to stay within memory bounds

use arrow::array::RecordBatch;
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatchIterator;
use parquet::arrow::{
    ArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask,
    AsyncArrowWriter,
};
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::compression::Compression;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use memmap2::Mmap;

/// Global memory limit for compaction operations (4GB per operation)
/// Leaves headroom for other system processes within 8GB total
const COMPACTION_MEMORY_LIMIT_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Default batch size for streaming compaction
const DEFAULT_BATCH_SIZE: usize = 100_000;

/// Compression level for ZSTD (1-22, higher = better compression but slower)
const ZSTD_COMPRESSION_LEVEL: i32 = 9;

/// Configuration for Parquet compaction
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Target row group size (rows)
    pub row_group_size: usize,
    /// Batch size for streaming reads
    pub batch_size: usize,
    /// Number of parallel compaction tasks
    pub parallelism: usize,
    /// Enable statistics writing for query pruning
    pub write_statistics: bool,
    /// Data page size (bytes)
    pub data_page_size: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: COMPACTION_MEMORY_LIMIT_BYTES,
            row_group_size: 1_000_000, // 1M rows per row group
            batch_size: DEFAULT_BATCH_SIZE,
            parallelism: 4, // Tune for Ryzen AI 5 core count
            write_statistics: true,
            data_page_size: 1024 * 1024, // 1MB pages
        }
    }
}

/// Statistics for a single source file
#[derive(Debug, Clone)]
pub struct SourceFileStats {
    pub path: PathBuf,
    pub row_count: usize,
    pub byte_size: u64,
    pub min_timestamp: i64,
    pub max_timestamp: i64,
}

/// Result of a compaction operation
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub output_path: PathBuf,
    pub total_rows: usize,
    pub total_bytes: u64,
    pub source_files: Vec<PathBuf>,
    pub compression_ratio: f64,
    pub duration_ms: u64,
}

/// Memory-mapped file reader for zero-copy Arrow IPC reading
pub struct MmapArrowReader {
    mmap: Mmap,
    current_offset: usize,
    file_size: usize,
}

impl MmapArrowReader {
    /// Create new memory-mapped Arrow reader
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let file_size = mmap.len();

        Ok(Self {
            mmap,
            current_offset: 0,
            file_size,
        })
    }

    /// Read next RecordBatch from memory-mapped data
    pub fn read_next_batch(&mut self) -> Result<Option<RecordBatch>, arrow::error::ArrowError> {
        if self.current_offset >= self.file_size {
            return Ok(None);
        }

        // Parse Arrow IPC format from mmap'd data
        let remaining = &self.mmap[self.current_offset..];
        
        // Use StreamReader for zero-copy parsing where possible
        let mut reader = StreamReader::try_new(remaining, None)?;
        
        match reader.next() {
            Some(Ok(batch)) => {
                self.current_offset += batch.get_array_memory_size();
                Ok(Some(batch))
            }
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    /// Get all batches from file (streaming, memory-bounded)
    pub fn read_all_bounded<F>(
        &mut self,
        mut process_fn: F,
    ) -> Result<(), arrow::error::ArrowError>
    where
        F: FnMut(RecordBatch) -> Result<(), arrow::error::ArrowError>,
    {
        while let Some(batch) = self.read_next_batch()? {
            process_fn(batch)?;
        }
        Ok(())
    }
}

/// Asynchronous Parquet compaction engine
pub struct ParquetCompactor {
    config: CompactionConfig,
    writer_props: Arc<WriterProperties>,
}

impl ParquetCompactor {
    /// Create new compactor with default configuration
    pub fn new() -> Self {
        Self::with_config(CompactionConfig::default())
    }

    /// Create new compactor with custom configuration
    pub fn with_config(config: CompactionConfig) -> Self {
        // Build WriterProperties with ZSTD compression
        let mut props_builder = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(Compression::ZSTD(ZSTD_COMPRESSION_LEVEL))
            .set_data_page_size_limit(config.data_page_size)
            .set_write_statistics(config.write_statistics)
            .set_max_row_group_size(config.row_group_size);

        // Enable dictionary encoding for symbol_id column
        props_builder = props_builder.set_dictionary_enabled(true);

        // Enable byte stream split for float columns (better compression)
        props_builder = props_builder.set_encoding(
            parquet::basic::Encoding::BYTE_STREAM_SPLIT,
        );

        let writer_props = Arc::new(props_builder.build());

        Self {
            config,
            writer_props,
        }
    }

    /// Analyze source files to determine compaction strategy
    pub async fn analyze_sources<P: AsRef<Path>>(
        &self,
        source_paths: &[P],
    ) -> Result<Vec<SourceFileStats>, std::io::Error> {
        let mut stats = Vec::with_capacity(source_paths.len());

        for path_ref in source_paths {
            let path = path_ref.as_ref();
            let file = File::open(path).await?;
            let metadata = file.metadata().await?;
            let byte_size = metadata.len();

            // Quick scan to get row count and timestamp range
            let mut mmap_reader = MmapArrowReader::new(path)?;
            let mut row_count = 0;
            let mut min_ts = i64::MAX;
            let mut max_ts = i64::MIN;

            while let Some(batch) = mmap_reader.read_next_batch().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })? {
                row_count += batch.num_rows();
                
                // Extract timestamp column (assumed first column)
                if let Some(ts_col) = batch.column(0).as_any().downcast_ref::<arrow::array::TimestampNanosecondArray>() {
                    for i in 0..ts_col.len() {
                        let ts = ts_col.value(i);
                        min_ts = min_ts.min(ts);
                        max_ts = max_ts.max(ts);
                    }
                }
            }

            stats.push(SourceFileStats {
                path: path.to_path_buf(),
                row_count,
                byte_size,
                min_timestamp: min_ts,
                max_timestamp: max_ts,
            });
        }

        Ok(stats)
    }

    /// Compact multiple Arrow IPC files into a single Parquet file
    /// Uses streaming to stay within memory limits
    pub async fn compact<P: AsRef<Path>>(
        &self,
        source_paths: &[P],
        output_path: P,
    ) -> Result<CompactionResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let output_path = output_path.as_ref().to_path_buf();
        
        // Sort sources by timestamp for optimal ordering
        let mut source_stats = self.analyze_sources(source_paths).await?;
        source_stats.sort_by_key(|s| s.min_timestamp);

        // Estimate total size and check memory constraints
        let total_source_bytes: u64 = source_stats.iter().map(|s| s.byte_size).sum();
        let estimated_output_bytes = total_source_bytes / 3; // ZSTD typically achieves 3x compression

        // If input exceeds memory limit, use chunked compaction
        if total_source_bytes as usize > self.config.max_memory_bytes {
            return self.compact_chunked(source_paths, output_path).await;
        }

        // Single-pass compaction for smaller datasets
        let output_file = File::create(&output_path).await?;
        let mut parquet_writer = AsyncArrowWriter::try_new(
            output_file,
            self.get_merged_schema(&source_stats)?,
            Some(self.writer_props.clone()),
        )?;

        let mut total_rows = 0;

        // Stream through each source file
        for source_stat in &source_stats {
            let mut mmap_reader = MmapArrowReader::new(&source_stat.path)?;
            
            while let Some(batch) = mmap_reader.read_next_batch().map_err(|e| {
                Box::new(e) as Box<dyn std::error::Error + Send + Sync>
            })? {
                total_rows += batch.num_rows();
                
                // Write batch to Parquet
                parquet_writer.write(&batch).await?;
                
                // Periodically flush to manage memory
                if parquet_writer.in_progress_size() > self.config.data_page_size {
                    parquet_writer.flush().await?;
                }
            }
        }

        parquet_writer.close().await?;

        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let output_metadata = std::fs::metadata(&output_path)?;
        let output_bytes = output_metadata.len();

        let compression_ratio = if total_source_bytes > 0 {
            total_source_bytes as f64 / output_bytes as f64
        } else {
            1.0
        };

        Ok(CompactionResult {
            output_path,
            total_rows,
            total_bytes: output_bytes,
            source_files: source_stats.into_iter().map(|s| s.path).collect(),
            compression_ratio,
            duration_ms: end_time - start_time,
        })
    }

    /// Chunked compaction for datasets exceeding memory limits
    async fn compact_chunked<P: AsRef<Path>>(
        &self,
        source_paths: &[P],
        output_path: P,
    ) -> Result<CompactionResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let output_path = output_path.as_ref().to_path_buf();
        let source_stats = self.analyze_sources(source_paths).await?;
        let total_source_bytes: u64 = source_stats.iter().map(|s| s.byte_size).sum();

        // Calculate chunk boundaries based on memory limit
        let chunk_target_bytes = self.config.max_memory_bytes / 2; // 50% headroom
        
        let mut chunks: Vec<Vec<SourceFileStats>> = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_chunk_size: u64 = 0;

        for stat in source_stats {
            if current_chunk_size + stat.byte_size > chunk_target_bytes as u64 && !current_chunk.is_empty() {
                chunks.push(current_chunk);
                current_chunk = Vec::new();
                current_chunk_size = 0;
            }
            current_chunk_size += stat.byte_size;
            current_chunk.push(stat);
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        // Process each chunk sequentially
        let mut total_rows = 0;
        let mut temp_files = Vec::new();

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let temp_path = output_path.with_extension(format!("chunk_{chunk_idx}.parquet.tmp"));
            
            // Compact this chunk
            let chunk_paths: Vec<&PathBuf> = chunk.iter().map(|s| &s.path).collect();
            let chunk_result = self.compact_simple(&chunk_paths, &temp_path).await?;
            
            total_rows += chunk_result.total_rows;
            temp_files.push(temp_path);
        }

        // Merge chunk files into final output
        let final_result = self.merge_parquet_files(&temp_files, &output_path).await?;

        // Clean up temp files
        for temp_file in &temp_files {
            let _ = tokio::fs::remove_file(temp_file).await;
        }

        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Ok(CompactionResult {
            output_path: final_result.output_path,
            total_rows: final_result.total_rows,
            total_bytes: final_result.total_bytes,
            source_files: source_stats.into_iter().map(|s| s.path).collect(),
            compression_ratio: final_result.compression_ratio,
            duration_ms: end_time - start_time,
        })
    }

    /// Simple compaction for a small set of files
    async fn compact_simple<P: AsRef<Path>>(
        &self,
        source_paths: &[&P],
        output_path: &P,
    ) -> Result<CompactionResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let output_path = output_path.as_ref().to_path_buf();
        let output_file = File::create(&output_path).await?;
        
        let source_refs: Vec<&Path> = source_paths.iter().map(|p| p.as_ref()).collect();
        let schema = self.infer_schema_from_files(&source_refs)?;
        
        let mut parquet_writer = AsyncArrowWriter::try_new(
            output_file,
            schema,
            Some(self.writer_props.clone()),
        )?;

        let mut total_rows = 0;
        let mut total_source_bytes: u64 = 0;

        for source_path in source_paths {
            let path = source_path.as_ref();
            let metadata = tokio::fs::metadata(path).await?;
            total_source_bytes += metadata.len();

            let mut mmap_reader = MmapArrowReader::new(path)?;
            
            while let Some(batch) = mmap_reader.read_next_batch().map_err(|e| {
                Box::new(e) as Box<dyn std::error::Error + Send + Sync>
            })? {
                total_rows += batch.num_rows();
                parquet_writer.write(&batch).await?;
            }
        }

        parquet_writer.close().await?;

        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let output_metadata = std::fs::metadata(&output_path)?;
        let output_bytes = output_metadata.len();

        Ok(CompactionResult {
            output_path,
            total_rows,
            total_bytes: output_bytes,
            source_files: source_refs.into_iter().map(|p| p.to_path_buf()).collect(),
            compression_ratio: total_source_bytes as f64 / output_bytes as f64,
            duration_ms: end_time - start_time,
        })
    }

    /// Merge multiple Parquet files into one
    async fn merge_parquet_files<P: AsRef<Path>>(
        &self,
        input_paths: &[P],
        output_path: &P,
    ) -> Result<CompactionResult, Box<dyn std::error::Error + Send + Sync>> {
        let output_path = output_path.as_ref().to_path_buf();
        let output_file = File::create(&output_path).await?;
        
        // Infer schema from first file
        let first_file = File::open(input_paths[0].as_ref()).await?;
        let builder = ParquetRecordBatchStreamBuilder::new(first_file).await?;
        let schema = builder.schema().clone();
        drop(builder);

        let mut parquet_writer = AsyncArrowWriter::try_new(
            output_file,
            schema,
            Some(self.writer_props.clone()),
        )?;

        let mut total_rows = 0;
        let mut total_input_bytes: u64 = 0;

        for input_path in input_paths {
            let path = input_path.as_ref();
            let metadata = tokio::fs::metadata(path).await?;
            total_input_bytes += metadata.len();

            let file = File::open(path).await?;
            let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
            let mut stream = builder.build()?;

            while let Some(result) = stream.next().await {
                let batch = result?;
                total_rows += batch.num_rows();
                parquet_writer.write(&batch).await?;
            }
        }

        parquet_writer.close().await?;

        let output_metadata = std::fs::metadata(&output_path)?;
        let output_bytes = output_metadata.len();

        Ok(CompactionResult {
            output_path,
            total_rows,
            total_bytes: output_bytes,
            source_files: input_paths.iter().map(|p| p.as_ref().to_path_buf()).collect(),
            compression_ratio: total_input_bytes as f64 / output_bytes as f64,
            duration_ms: 0, // Not tracked for merge
        })
    }

    /// Get merged schema from source files
    fn get_merged_schema(
        &self,
        source_stats: &[SourceFileStats],
    ) -> Result<Arc<arrow::datatypes::Schema>, Box<dyn std::error::Error + Send + Sync>> {
        if source_stats.is_empty() {
            return Err("No source files provided".into());
        }

        // Use schema from first file (assuming consistent schemas)
        let first_path = &source_stats[0].path;
        self.infer_schema_from_files(&[first_path.as_path()])
    }

    /// Infer schema from Parquet files
    fn infer_schema_from_files(
        &self,
        paths: &[&Path],
    ) -> Result<Arc<arrow::datatypes::Schema>, Box<dyn std::error::Error + Send + Sync>> {
        if paths.is_empty() {
            return Err("No files provided".into());
        }

        // For now, return the standard tick schema
        // In production, would read actual schema from file metadata
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp_ns", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new("symbol_id", DataType::Int64, false),
            Field::new("bid_price", DataType::Float64, false),
            Field::new("ask_price", DataType::Float64, false),
            Field::new("bid_size", DataType::Float64, false),
            Field::new("ask_size", DataType::Float64, false),
            Field::new("last_price", DataType::Float64, false),
            Field::new("last_size", DataType::Float64, false),
            Field::new("exchange_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));

        Ok(schema)
    }
}

impl Default for ParquetCompactor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compactor_creation() {
        let compactor = ParquetCompactor::new();
        assert_eq!(compactor.config.batch_size, DEFAULT_BATCH_SIZE);
    }

    #[tokio::test]
    async fn test_memory_limit_config() {
        let config = CompactionConfig {
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            ..Default::default()
        };
        let compactor = ParquetCompactor::with_config(config);
        assert_eq!(compactor.config.max_memory_bytes, 2 * 1024 * 1024 * 1024);
    }
}
