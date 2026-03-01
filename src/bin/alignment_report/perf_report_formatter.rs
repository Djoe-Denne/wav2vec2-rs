use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::Serialize;

const PERF_SCHEMA_VERSION: u32 = 1;
const JSONL_FLUSH_EVERY: usize = 262144;
/// BufWriter capacity so one record (~1–2 KB) doesn't fill the buffer and trigger implicit flushes.
const JSONL_BUFFER_CAPACITY: usize = 2 * 1024 * 1024;

/// GPU memory snapshot at a point in time (used and total device memory in bytes).
#[derive(Debug, Clone, Serialize)]
pub struct GpuMemorySnapshot {
    pub gpu_used: u64,
    pub gpu_total: u64,
}

/// Per-stage GPU memory for one run; matches Python reference shape.
#[derive(Debug, Clone, Serialize)]
pub struct PerfMemory {
    pub forward: Option<GpuMemorySnapshot>,
    pub post: Option<GpuMemorySnapshot>,
    pub dp: Option<GpuMemorySnapshot>,
    pub group: Option<GpuMemorySnapshot>,
    pub conf: Option<GpuMemorySnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerfRunConfig {
    pub warmup: usize,
    pub repeats: usize,
    pub aggregate: String,
    pub append: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerfUtteranceRecord {
    pub utterance_id: String,
    pub audio_path: String,
    pub duration_ms: u64,
    pub num_frames_t: usize,
    pub state_len: usize,
    pub ts_product: u64,
    pub vocab_size: usize,
    pub dtype: String,
    pub device: String,
    pub frame_stride_ms: f64,
    pub warmup: usize,
    pub repeats: usize,
    pub aggregate: String,
    pub forward_ms: f64,
    pub post_ms: f64,
    pub dp_ms: f64,
    pub group_ms: f64,
    pub conf_ms: f64,
    pub align_ms: f64,
    pub align_ms_per_ts: f64,
    pub align_ms_per_t: f64,
    pub total_ms: f64,
    pub forward_ms_repeats: Vec<f64>,
    pub post_ms_repeats: Vec<f64>,
    pub dp_ms_repeats: Vec<f64>,
    pub group_ms_repeats: Vec<f64>,
    pub conf_ms_repeats: Vec<f64>,
    pub align_ms_repeats: Vec<f64>,
    pub total_ms_repeats: Vec<f64>,
    /// Per-stage GPU memory (gpu_used, gpu_total); present only when benchmark ran with memory profiling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<PerfMemory>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerfMetricStats {
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerfAggregateStats {
    pub utterance_count: usize,
    pub forward_ms: PerfMetricStats,
    pub post_ms: PerfMetricStats,
    pub dp_ms: PerfMetricStats,
    pub group_ms: PerfMetricStats,
    pub conf_ms: PerfMetricStats,
    pub align_ms: PerfMetricStats,
    pub align_ms_per_ts: PerfMetricStats,
    pub align_ms_per_t: PerfMetricStats,
    pub total_ms: PerfMetricStats,
    /// Aggregate GPU memory stats; present when records include memory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<PerfAggregateMemory>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerfAggregateMemory {
    pub forward_gpu_used: PerfMetricStats,
    pub dp_gpu_used: PerfMetricStats,
    pub gpu_total: u64,
}

#[derive(Debug, Serialize)]
struct PerfRunReport<'a> {
    schema_version: u32,
    generated_at: String,
    config: &'a PerfRunConfig,
    records: &'a [PerfUtteranceRecord],
    aggregate: &'a PerfAggregateStats,
}

#[derive(Debug, Serialize)]
struct PerfSummaryReport<'a> {
    schema_version: u32,
    generated_at: String,
    config: &'a PerfRunConfig,
    aggregate: &'a PerfAggregateStats,
}

pub struct PerfJsonlAppender {
    writer: BufWriter<File>,
    writes_since_flush: usize,
}

impl PerfJsonlAppender {
    pub fn open(path: &Path) -> Result<Self, String> {
        ensure_parent_directory(path)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|err| {
                format!(
                    "Failed to open perf JSONL file '{}' for append: {err}",
                    path.display()
                )
            })?;
        Ok(Self {
            writer: BufWriter::with_capacity(JSONL_BUFFER_CAPACITY, file),
            writes_since_flush: 0,
        })
    }

    pub fn append(&mut self, record: &PerfUtteranceRecord) -> Result<(), String> {
        serde_json::to_writer(&mut self.writer, record).map_err(|err| {
            format!(
                "Failed to serialize perf JSONL record for utterance '{}': {err}",
                record.utterance_id
            )
        })?;
        self.writer.write_all(b"\n").map_err(|err| {
            format!(
                "Failed to write trailing newline for utterance '{}': {err}",
                record.utterance_id
            )
        })?;
        self.writes_since_flush += 1;
        if self.writes_since_flush >= JSONL_FLUSH_EVERY {
            self.writer
                .flush()
                .map_err(|err| format!("Failed flushing perf JSONL writer: {err}"))?;
            self.writes_since_flush = 0;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|err| format!("Failed finalizing perf JSONL writer: {err}"))
    }
}

pub fn write_json_report(
    path: &Path,
    config: &PerfRunConfig,
    records: &[PerfUtteranceRecord],
    aggregate: &PerfAggregateStats,
) -> Result<(), String> {
    ensure_parent_directory(path)?;
    let mut file = File::create(path).map_err(|err| {
        format!(
            "Failed to create perf report file '{}': {err}",
            path.display()
        )
    })?;
    let payload = PerfRunReport {
        schema_version: PERF_SCHEMA_VERSION,
        generated_at: Utc::now().to_rfc3339(),
        config,
        records,
        aggregate,
    };
    serde_json::to_writer_pretty(&mut file, &payload).map_err(|err| {
        format!(
            "Failed to serialize perf report JSON '{}': {err}",
            path.display()
        )
    })?;
    file.write_all(b"\n").map_err(|err| {
        format!(
            "Failed finalizing perf report file '{}': {err}",
            path.display()
        )
    })
}

pub fn write_summary_report(
    path: &Path,
    config: &PerfRunConfig,
    aggregate: &PerfAggregateStats,
) -> Result<(), String> {
    ensure_parent_directory(path)?;
    let mut file = File::create(path).map_err(|err| {
        format!(
            "Failed to create perf summary report '{}': {err}",
            path.display()
        )
    })?;
    let payload = PerfSummaryReport {
        schema_version: PERF_SCHEMA_VERSION,
        generated_at: Utc::now().to_rfc3339(),
        config,
        aggregate,
    };
    serde_json::to_writer_pretty(&mut file, &payload).map_err(|err| {
        format!(
            "Failed to serialize perf summary report '{}': {err}",
            path.display()
        )
    })?;
    file.write_all(b"\n").map_err(|err| {
        format!(
            "Failed finalizing perf summary report '{}': {err}",
            path.display()
        )
    })
}

pub fn summary_path_for(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(".summary.json");
    PathBuf::from(value)
}

fn ensure_parent_directory(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create perf output directory '{}': {err}",
                parent.display()
            )
        })?;
    }
    Ok(())
}
