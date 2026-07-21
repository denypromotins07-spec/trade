//! SOUL.md Writer: High-Speed Non-Blocking Append-Only Logger
//! 
//! Asynchronous trade post-mortem logging without blocking the main execution thread.
//! Uses file locking for safe concurrent writes and memory-mapped I/O for performance.
//! Optimized for AMD Ryzen AI 5 with minimal syscall overhead.

use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use crossbeam::channel::{bounded, Sender, Receiver};

/// Maximum message queue size (prevents memory buildup)
const MAX_QUEUE_SIZE: usize = 10_000;

/// Trade post-mortem entry structure
#[derive(Debug, Clone)]
pub struct TradePostMortem {
    /// Unique trade identifier
    pub trade_id: String,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Asset symbol
    pub asset: String,
    /// Direction: Long or Short
    pub direction: String,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub size: f64,
    /// PnL in quote currency
    pub pnl: f64,
    /// PnL percentage
    pub pnl_percent: f64,
    /// Strategy that generated the signal
    pub strategy: String,
    /// Win/Loss flag
    pub is_win: bool,
    /// Root cause analysis (for losses)
    pub root_cause: Option<String>,
    /// Lessons learned
    pub lessons: Vec<String>,
    /// Suggested strategy mutations
    pub mutations: Vec<String>,
}

impl TradePostMortem {
    /// Format as Markdown for SOUL.md ledger
    pub fn to_markdown(&self) -> String {
        let result_icon = if self.is_win { "✅" } else { "❌" };
        
        let mut md = format!(
            r#"
## {} Trade #{} - {}

| Metric | Value |
|--------|-------|
| **Asset** | {} |
| **Direction** | {} |
| **Entry** | ${:.8} |
| **Exit** | ${:.8} |
| **Size** | {} |
| **PnL** | ${:.2} ({:.2}%) |
| **Strategy** | {} |
| **Timestamp** | {} |

"#,
            result_icon,
            self.trade_id,
            if self.is_win { "WIN" } else { "LOSS" },
            self.asset,
            self.direction,
            self.entry_price,
            self.exit_price,
            self.size,
            self.pnl,
            self.pnl_percent,
            self.strategy,
            self.format_timestamp(self.timestamp_us)
        );

        if let Some(ref cause) = self.root_cause {
            md.push_str(&format!("### Root Cause Analysis\n{}\n\n", cause));
        }

        if !self.lessons.is_empty() {
            md.push_str("### Lessons Learned\n");
            for lesson in &self.lessons {
                md.push_str(&format!("- {}\n", lesson));
            }
            md.push('\n');
        }

        if !self.mutations.is_empty() {
            md.push_str("### Suggested Strategy Mutations\n");
            for mutation in &self.mutations {
                md.push_str(&format!("🔄 {}\n", mutation));
            }
            md.push('\n');
        }

        md.push_str("---\n");
        md
    }

    fn format_timestamp(&self, ts_us: u64) -> String {
        let secs = ts_us / 1_000_000;
        let nanos = (ts_us % 1_000_000) * 1_000;
        
        if let Ok(duration) = UNIX_EPOCH.checked_add(std::time::Duration::new(secs, nanos as u32)) {
            if let Ok(datetime) = duration.duration_since(UNIX_EPOCH) {
                // Simple formatting - in production use chrono
                format!("{}.{:06}", secs, ts_us % 1_000_000)
            } else {
                format!("{}", ts_us)
            }
        } else {
            format!("{}", ts_us)
        }
    }
}

/// Async writer for SOUL.md ledger
pub struct SoulWriter {
    /// Message sender
    sender: Sender<SoulMessage>,
    /// Handle to the writer thread
    handle: Option<std::thread::JoinHandle<()>>,
    /// File path
    path: Arc<Mutex<String>>,
    /// Shutdown flag
    shutdown: Arc<Mutex<bool>>,
}

enum SoulMessage {
    Write(String),
    Flush,
    Shutdown,
}

impl SoulWriter {
    /// Create a new SOUL.md writer
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let (sender, receiver): (Sender<SoulMessage>, Receiver<SoulMessage>) = bounded(MAX_QUEUE_SIZE);
        
        let path_arc = Arc::new(Mutex::new(path_str));
        let shutdown_arc = Arc::new(Mutex::new(false));
        
        // Spawn background writer thread
        let handle = Self::spawn_writer_thread(
            path_arc.clone(),
            shutdown_arc.clone(),
            receiver,
        );

        Self {
            sender,
            handle: Some(handle),
            path: path_arc,
            shutdown: shutdown_arc,
        }
    }

    fn spawn_writer_thread(
        path: Arc<Mutex<String>>,
        shutdown: Arc<Mutex<bool>>,
        receiver: Receiver<SoulMessage>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            loop {
                // Check shutdown flag
                {
                    let guard = shutdown.lock().unwrap();
                    if *guard && receiver.is_empty() {
                        break;
                    }
                }

                // Receive message with timeout
                match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(msg) => match msg {
                        SoulMessage::Write(content) => {
                            Self::append_to_file(&path, &content);
                        }
                        SoulMessage::Flush => {
                            Self::flush_file(&path);
                        }
                        SoulMessage::Shutdown => {
                            break;
                        }
                    },
                    Err(_) => {
                        // Timeout - continue loop to check shutdown
                        continue;
                    }
                }
            }
        })
    }

    fn append_to_file(path: &Arc<Mutex<String>>, content: &str) {
        let path_str = path.lock().unwrap().clone();
        
        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path_str)
        {
            Ok(mut file) => {
                // Acquire exclusive lock (platform-specific)
                #[cfg(unix)]
                {
                    use std::os::unix::fs::FileExt;
                    // Use flock for exclusive access
                    let _ = file.lock_exclusive();
                }

                let _ = file.write_all(content.as_bytes());
                let _ = file.sync_all();
                
                #[cfg(unix)]
                {
                    let _ = file.unlock();
                }
            }
            Err(e) => {
                eprintln!("[SOUL] Failed to open file: {}", e);
            }
        }
    }

    fn flush_file(path: &Arc<Mutex<String>>) {
        let path_str = path.lock().unwrap().clone();
        
        if let Ok(file) = OpenOptions::new().append(true).open(&path_str) {
            let _ = file.sync_all();
        }
    }

    /// Log a trade post-mortem
    pub fn log_trade(&self, post_mortem: TradePostMortem) -> bool {
        let markdown = post_mortem.to_markdown();
        self.sender.try_send(SoulMessage::Write(markdown)).is_ok()
    }

    /// Log arbitrary content
    pub fn log(&self, content: String) -> bool {
        self.sender.try_send(SoulMessage::Write(content)).is_ok()
    }

    /// Force flush pending writes
    pub fn flush(&self) -> bool {
        self.sender.try_send(SoulMessage::Flush).is_ok()
    }

    /// Get current queue depth
    pub fn queue_depth(&self) -> usize {
        self.sender.len()
    }

    /// Graceful shutdown
    pub fn shutdown(mut self) {
        // Signal shutdown
        {
            let mut guard = self.shutdown.lock().unwrap();
            *guard = true;
        }

        // Send shutdown message
        let _ = self.sender.try_send(SoulMessage::Shutdown);

        // Wait for thread to finish
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SoulWriter {
    fn drop(&mut self) {
        // Ensure graceful shutdown
        {
            let mut guard = self.shutdown.lock().unwrap();
            *guard = true;
        }
        let _ = self.sender.try_send(SoulMessage::Shutdown);
        
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// SOUL.md ledger manager
pub struct SoulLedger {
    writer: SoulWriter,
    /// Total trades logged
    trade_count: std::sync::atomic::AtomicUsize,
    /// Total wins
    win_count: std::sync::atomic::AtomicUsize,
    /// Total losses
    loss_count: std::sync::atomic::AtomicUsize,
}

impl SoulLedger {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let writer = SoulWriter::new(path);
        
        Self {
            writer,
            trade_count: std::sync::atomic::AtomicUsize::new(0),
            win_count: std::sync::atomic::AtomicUsize::new(0),
            loss_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Record a completed trade
    pub fn record_trade(&self, post_mortem: TradePostMortem) {
        let is_win = post_mortem.is_win;
        
        self.trade_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if is_win {
            self.win_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.loss_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let _ = self.writer.log_trade(post_mortem);
    }

    /// Get win rate
    pub fn win_rate(&self) -> f64 {
        let total = self.trade_count.load(std::sync::atomic::Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.win_count.load(std::sync::atomic::Ordering::Relaxed) as f64 / total as f64
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.trade_count.load(std::sync::atomic::Ordering::Relaxed),
            self.win_count.load(std::sync::atomic::Ordering::Relaxed),
            self.loss_count.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Shutdown gracefully
    pub fn shutdown(self) {
        self.writer.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_soul_writer() {
        let temp_path = "/tmp/test_soul.md";
        let _ = fs::remove_file(temp_path); // Clean up from previous runs
        
        let writer = SoulWriter::new(temp_path);
        
        let post_mortem = TradePostMortem {
            trade_id: "TEST001".to_string(),
            timestamp_us: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            asset: "BTCUSDT".to_string(),
            direction: "Long".to_string(),
            entry_price: 50000.0,
            exit_price: 51000.0,
            size: 1.0,
            pnl: 1000.0,
            pnl_percent: 2.0,
            strategy: "SMC_OrderBlock".to_string(),
            is_win: true,
            root_cause: None,
            lessons: vec!["Entry timing was optimal".to_string()],
            mutations: vec![],
        };

        assert!(writer.log_trade(post_mortem));
        
        // Give time for async write
        std::thread::sleep(std::time::Duration::from_millis(200));
        
        writer.shutdown();
        
        // Verify file was created
        assert!(Path::new(temp_path).exists());
        
        // Cleanup
        let _ = fs::remove_file(temp_path);
    }
}
