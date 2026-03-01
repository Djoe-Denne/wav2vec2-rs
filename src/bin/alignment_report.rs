use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::Utc;
use clap::{Parser, ValueEnum};
use claxon::FlacReader;
use indicatif::{ProgressBar, ProgressStyle};
use textgrid::{TextGrid, TierType};
use wav2vec2_rs::{
    aggregate_reports, attach_outlier_traces, compute_sentence_report, infer_split, AlignmentInput,
    ForcedAligner, ForcedAlignerBuilder, Meta, ReferenceWord, Report, RuntimeKind, SentenceReport,
    Wav2Vec2Config, WordTiming,
};

#[path = "alignment_report/json_report_formatter.rs"]
mod json_report_formatter;
#[cfg(feature = "alignment-profiling")]
#[path = "alignment_report/perf_report_formatter.rs"]
mod perf_report_formatter;
#[cfg(feature = "alignment-profiling")]
use wav2vec2_rs::pipeline::memory_tracker::{MemoryTracker, StageMemoryMap};
#[path = "alignment_report/text_grid_report_formatter.rs"]
mod text_grid_report_formatter;

#[cfg(feature = "alignment-profiling")]
fn stage_memory_map_to_perf_memory(
    m: &StageMemoryMap,
) -> Option<perf_report_formatter::PerfMemory> {
    use perf_report_formatter::GpuMemorySnapshot;
    use wav2vec2_rs::pipeline::memory_tracker::StageMemory;
    fn to_snapshot(s: &StageMemory) -> Option<GpuMemorySnapshot> {
        if s.peak_gpu_allocated_bytes > 0 || s.gpu_total_bytes > 0 {
            Some(GpuMemorySnapshot {
                gpu_used: s.peak_gpu_allocated_bytes,
                gpu_total: s.gpu_total_bytes,
            })
        } else {
            None
        }
    }
    let forward = to_snapshot(&m.forward);
    let post = to_snapshot(&m.post);
    let dp = to_snapshot(&m.dp);
    let group = to_snapshot(&m.group);
    let conf = to_snapshot(&m.conf);
    if forward.is_none() && post.is_none() && dp.is_none() && group.is_none() && conf.is_none() {
        return None;
    }
    Some(perf_report_formatter::PerfMemory {
        forward,
        post,
        dp,
        group,
        conf,
    })
}

const LIBRISPEECH_SUBSETS: [&str; 2] = ["test-clean", "test-other"];
const OUTLIER_TRACE_TOP_N: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Json,
    #[value(name = "textgrid")]
    TextGrid,
    /// Run alignment for performance measurement only; write perf JSON/JSONL but no report or TextGrid files.
    #[value(name = "perf")]
    Perf,
}

#[cfg(feature = "alignment-profiling")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PerfAggregate {
    Median,
    Mean,
}

#[cfg(feature = "alignment-profiling")]
impl PerfAggregate {
    fn as_str(self) -> &'static str {
        match self {
            Self::Median => "median",
            Self::Mean => "mean",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RuntimeChoice {
    Onnx,
    Candle,
}

impl RuntimeChoice {
    fn model_filename(self) -> &'static str {
        match self {
            Self::Onnx => "model.onnx",
            Self::Candle => "model.safetensors",
        }
    }

    fn runtime_kind(self) -> RuntimeKind {
        match self {
            Self::Onnx => RuntimeKind::Onnx,
            Self::Candle => RuntimeKind::Candle,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Candle => "candle",
        }
    }
}

fn default_runtime_choice() -> RuntimeChoice {
    #[cfg(feature = "onnx")]
    {
        RuntimeChoice::Onnx
    }

    #[cfg(not(feature = "onnx"))]
    {
        RuntimeChoice::Candle
    }
}

#[derive(Debug, Parser)]
#[command(name = "alignment_report")]
#[command(about = "Generate deterministic forced-alignment quality reports")]
struct Args {
    #[arg(
        long,
        env = "WAV2VEC2_REPORT_MODEL_DIR",
        default_value = "models/onnx_wav2vec2_base_960h"
    )]
    model_dir: PathBuf,
    #[arg(
        long,
        env = "WAV2VEC2_REPORT_DATASET_ROOT",
        default_value = "test-data"
    )]
    dataset_root: PathBuf,
    #[arg(long, env = "WAV2VEC2_REPORT_CASES_FILE")]
    cases_file: Option<PathBuf>,
    #[arg(long, env = "WAV2VEC2_REPORT_OUT")]
    out: Option<PathBuf>,
    #[arg(long, env = "WAV2VEC2_REPORT_LIMIT")]
    limit: Option<usize>,
    #[arg(long, env = "WAV2VEC2_REPORT_OFFSET", default_value_t = 0)]
    offset: usize,
    #[arg(long, env = "WAV2VEC2_REPORT_DEVICE", default_value = "cpu")]
    device: String,
    #[arg(
        long,
        env = "WAV2VEC2_REPORT_RUNTIME",
        value_enum,
        default_value_t = default_runtime_choice()
    )]
    runtime: RuntimeChoice,
    #[arg(
        long,
        env = "WAV2VEC2_REPORT_FORMAT",
        value_enum,
        default_value_t = OutputFormat::Json
    )]
    output_format: OutputFormat,
    #[arg(long, env = "WAV2VEC2_REPORT_TEXTGRID_SUFFIX", default_value = "")]
    textgrid_suffix: String,
    #[cfg(feature = "alignment-profiling")]
    #[arg(long, env = "WAV2VEC2_REPORT_PERF_OUT")]
    perf_out: Option<PathBuf>,
    #[cfg(feature = "alignment-profiling")]
    #[arg(long, env = "WAV2VEC2_REPORT_PERF_WARMUP", default_value_t = 10)]
    perf_warmup: usize,
    #[cfg(feature = "alignment-profiling")]
    #[arg(long, env = "WAV2VEC2_REPORT_PERF_REPEATS", default_value_t = 30)]
    perf_repeats: usize,
    #[cfg(feature = "alignment-profiling")]
    #[arg(
        long,
        env = "WAV2VEC2_REPORT_PERF_AGGREGATE",
        value_enum,
        default_value_t = PerfAggregate::Median
    )]
    perf_aggregate: PerfAggregate,
    #[cfg(feature = "alignment-profiling")]
    #[arg(long, env = "WAV2VEC2_REPORT_PERF_APPEND", default_value_t = false)]
    perf_append: bool,
    #[cfg(feature = "alignment-profiling")]
    #[arg(
        long,
        env = "WAV2VEC2_REPORT_PERF_SCALING_REPORT",
        default_value_t = false
    )]
    perf_scaling_report: bool,
}

#[derive(Debug, Clone)]
struct Case {
    id: String,
    audio_path: String,
    transcript: String,
    reference_words: Vec<ReferenceWord>,
}

#[cfg(feature = "alignment-profiling")]
#[derive(Debug, Clone)]
struct ScalingSample {
    utterance_id: String,
    num_frames_t: usize,
    state_len: usize,
    ts_product: u64,
    dp_ms: f64,
    group_ms: f64,
    conf_ms: f64,
}

fn main() {
    if run().is_err() {
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let model_dir = resolve_path(&repo_root, &args.model_dir);
    let dataset_root = resolve_path(&repo_root, &args.dataset_root);
    #[cfg(feature = "alignment-profiling")]
    if args.perf_repeats == 0 {
        return Err("--perf-repeats must be >= 1.".to_string());
    }
    let out_path = match args.output_format {
        OutputFormat::Json => Some(resolve_out_path(&repo_root, args.out.as_ref())),
        OutputFormat::TextGrid | OutputFormat::Perf => None,
    };
    #[cfg(feature = "alignment-profiling")]
    let perf_out_path = args
        .perf_out
        .as_ref()
        .map(|path| resolve_path(&repo_root, path));
    #[cfg(not(feature = "alignment-profiling"))]
    let perf_out_path: Option<PathBuf> = None;
    #[cfg(feature = "alignment-profiling")]
    if args.output_format == OutputFormat::Perf && perf_out_path.is_none() {
        return Err("--output-format perf requires --perf-out.".to_string());
    }

    let include_ids = load_case_filter(args.cases_file.as_ref(), &repo_root)?;
    let mut cases = match args.output_format {
        OutputFormat::Json => load_all_cases(&dataset_root)?,
        OutputFormat::TextGrid | OutputFormat::Perf => {
            load_all_cases_from_transcripts(&dataset_root)?
        }
    };

    if let Some(ids) = include_ids.as_ref() {
        let known: HashSet<&str> = cases.iter().map(|case| case.id.as_str()).collect();
        let mut missing = ids
            .iter()
            .filter(|id| !known.contains(id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        missing.sort();

        cases.retain(|case| ids.contains(&case.id));
    }

    if args.offset > 0 {
        cases = cases.into_iter().skip(args.offset).collect();
    }
    if let Some(limit) = args.limit {
        cases.truncate(limit);
    }

    if cases.is_empty() {
        return Err("No cases selected after applying filters/offset/limit.".to_string());
    }
    let selected_case_count = cases.len();

    let aligner = build_aligner(&model_dir, &args.device, args.runtime)?;
    let frame_stride_ms = aligner.frame_stride_ms();
    if !frame_stride_ms.is_finite() {
        return Err(format!(
            "aligner produced invalid frame_stride_ms: {frame_stride_ms}"
        ));
    }
    if frame_stride_ms < f32::MIN as f64 || frame_stride_ms > f32::MAX as f64 {
        return Err(format!(
            "aligner frame_stride_ms is out of range for report serialization: {frame_stride_ms}"
        ));
    }

    let mut sentence_reports: Vec<SentenceReport> = Vec::with_capacity(cases.len());
    let mut predicted_by_id: HashMap<String, Vec<WordTiming>> = HashMap::with_capacity(cases.len());
    let mut references_by_id: HashMap<String, Vec<ReferenceWord>> =
        HashMap::with_capacity(cases.len());
    let mut written_textgrids = 0usize;
    let mut lib_work_elapsed = Duration::ZERO;
    let perf_enabled = perf_out_path.is_some();
    #[cfg(feature = "alignment-profiling")]
    let perf_config = perf_report_formatter::PerfRunConfig {
        warmup: args.perf_warmup,
        repeats: args.perf_repeats,
        aggregate: args.perf_aggregate.as_str().to_string(),
        append: args.perf_append,
    };
    #[cfg(feature = "alignment-profiling")]
    let mut perf_jsonl_appender = if perf_enabled && args.perf_append {
        let perf_path = perf_out_path
            .as_ref()
            .ok_or_else(|| "internal error: missing --perf-out path".to_string())?;
        Some(perf_report_formatter::PerfJsonlAppender::open(perf_path)?)
    } else {
        None
    };
    #[cfg(feature = "alignment-profiling")]
    let mut perf_records = if perf_enabled && !args.perf_append {
        Vec::with_capacity(cases.len())
    } else {
        Vec::new()
    };
    #[cfg(feature = "alignment-profiling")]
    let mut perf_forward_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_post_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_dp_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_group_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_conf_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_align_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_align_per_ts_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_align_per_t_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_total_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_forward_gpu_used_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_dp_gpu_used_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_gpu_total: Option<u64> = None;
    #[cfg(feature = "alignment-profiling")]
    let mut scaling_samples = Vec::new();
    #[cfg(feature = "alignment-profiling")]
    let mut perf_warmup_done = false;
    let progress = ProgressBar::new(cases.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    progress.set_message("starting...");

    for case in &cases {
        progress.set_message(case.id.clone());
        let audio_file = dataset_root.join(&case.audio_path);
        require_path_exists(
            &audio_file,
            "Missing audio file referenced by alignment case.",
        )?;

        let (sample_rate_hz, samples) = read_flac_mono(&audio_file)?;
        let duration_ms = if sample_rate_hz == 0 {
            0
        } else {
            ((samples.len() as u128) * 1000 / sample_rate_hz as u128) as u64
        };

        let normalized = if perf_enabled {
            #[cfg(feature = "alignment-profiling")]
            {
                Some(wav2vec2_rs::normalize_audio(&samples))
            }
            #[cfg(not(feature = "alignment-profiling"))]
            {
                None
            }
        } else {
            None
        };
        let alignment_input = AlignmentInput {
            sample_rate_hz,
            samples,
            transcript: case.transcript.clone(),
            normalized,
        };
        let output_words = if perf_enabled {
            #[cfg(feature = "alignment-profiling")]
            {
                if !perf_warmup_done {
                    for _ in 0..args.perf_warmup {
                        aligner
                            .align_profiled(&alignment_input)
                            .map_err(|err| format!("{}: warm-up align() failed: {err}", case.id))?;
                    }
                    perf_warmup_done = true;
                }

                let mut forward_ms_repeats = Vec::with_capacity(args.perf_repeats);
                let mut post_ms_repeats = Vec::with_capacity(args.perf_repeats);
                let mut dp_ms_repeats = Vec::with_capacity(args.perf_repeats);
                let mut group_ms_repeats = Vec::with_capacity(args.perf_repeats);
                let mut conf_ms_repeats = Vec::with_capacity(args.perf_repeats);
                let mut align_ms_repeats = Vec::with_capacity(args.perf_repeats);
                let mut total_ms_repeats = Vec::with_capacity(args.perf_repeats);
                let mut selected_words: Option<Vec<WordTiming>> = None;
                let mut selected_num_frames_t = 0usize;
                let mut selected_state_len = 0usize;
                let mut selected_ts_product = 0u64;
                let mut selected_vocab_size = 0usize;
                let mut selected_dtype = String::new();
                let mut selected_device = String::new();
                let mut process_memory: Option<StageMemoryMap> = None;
                // Use GPU memory probe for both cuda and generic-gpu (ORT CUDA EP loads libcudart).
                let gpu_reader = if args.device.eq_ignore_ascii_case("cpu") {
                    None
                } else {
                    wav2vec2_rs::pipeline::memory_tracker::cuda_gpu_reader()
                };
                let mut tracker = MemoryTracker::new(gpu_reader);

                for repeat_idx in 0..args.perf_repeats {
                    let profiled = if repeat_idx == 0 {
                        let (prof, mem) = aligner
                            .align_profiled_with_memory(&alignment_input, &mut tracker)
                            .map_err(|err| {
                                format!("{}: perf align_with_memory() failed: {err}", case.id)
                            })?;
                        process_memory = Some(mem);
                        prof
                    } else {
                        aligner
                            .align_profiled(&alignment_input)
                            .map_err(|err| format!("{}: perf align() failed: {err}", case.id))?
                    };
                    let timings = profiled.timings;
                    forward_ms_repeats.push(timings.forward_ms);
                    post_ms_repeats.push(timings.post_ms);
                    dp_ms_repeats.push(timings.dp_ms);
                    group_ms_repeats.push(timings.group_ms);
                    conf_ms_repeats.push(timings.conf_ms);
                    align_ms_repeats.push(timings.align_ms);
                    total_ms_repeats.push(timings.total_ms);

                    if repeat_idx == 0 {
                        selected_words = Some(profiled.output.words);
                        selected_num_frames_t = profiled.num_frames_t;
                        selected_state_len = profiled.state_len;
                        selected_ts_product = profiled.ts_product;
                        selected_vocab_size = profiled.vocab_size;
                        selected_dtype = profiled.dtype;
                        selected_device = profiled.device;
                    }
                }

                let forward_ms = aggregate_measurements(&forward_ms_repeats, args.perf_aggregate);
                let post_ms = aggregate_measurements(&post_ms_repeats, args.perf_aggregate);
                let dp_ms = aggregate_measurements(&dp_ms_repeats, args.perf_aggregate);
                let group_ms = aggregate_measurements(&group_ms_repeats, args.perf_aggregate);
                let conf_ms = aggregate_measurements(&conf_ms_repeats, args.perf_aggregate);
                // Keep record-level substage timings internally consistent even when
                // the selected aggregate is median (median(a+b) != median(a)+median(b)).
                let align_ms = dp_ms + group_ms + conf_ms;
                let total_ms = aggregate_measurements(&total_ms_repeats, args.perf_aggregate);
                let align_ms_per_ts = if selected_ts_product > 0 {
                    align_ms / selected_ts_product as f64
                } else {
                    0.0
                };
                let align_ms_per_t = if selected_num_frames_t > 0 {
                    align_ms / selected_num_frames_t as f64
                } else {
                    0.0
                };

                let record = perf_report_formatter::PerfUtteranceRecord {
                    utterance_id: case.id.clone(),
                    audio_path: case.audio_path.clone(),
                    duration_ms,
                    num_frames_t: selected_num_frames_t,
                    state_len: selected_state_len,
                    ts_product: selected_ts_product,
                    vocab_size: selected_vocab_size,
                    dtype: selected_dtype,
                    device: selected_device,
                    frame_stride_ms,
                    warmup: args.perf_warmup,
                    repeats: args.perf_repeats,
                    aggregate: args.perf_aggregate.as_str().to_string(),
                    forward_ms,
                    post_ms,
                    dp_ms,
                    group_ms,
                    conf_ms,
                    align_ms,
                    align_ms_per_ts,
                    align_ms_per_t,
                    total_ms,
                    forward_ms_repeats,
                    post_ms_repeats,
                    dp_ms_repeats,
                    group_ms_repeats,
                    conf_ms_repeats,
                    align_ms_repeats,
                    total_ms_repeats,
                    memory: process_memory
                        .as_ref()
                        .and_then(stage_memory_map_to_perf_memory),
                };

                perf_forward_samples.push(record.forward_ms);
                perf_post_samples.push(record.post_ms);
                perf_dp_samples.push(record.dp_ms);
                perf_group_samples.push(record.group_ms);
                perf_conf_samples.push(record.conf_ms);
                perf_align_samples.push(record.align_ms);
                perf_align_per_ts_samples.push(record.align_ms_per_ts);
                perf_align_per_t_samples.push(record.align_ms_per_t);
                perf_total_samples.push(record.total_ms);
                if let Some(ref mem) = record.memory {
                    if let Some(ref f) = mem.forward {
                        perf_forward_gpu_used_samples.push(f.gpu_used as f64);
                        if perf_gpu_total.is_none() {
                            perf_gpu_total = Some(f.gpu_total);
                        }
                    }
                    if let Some(ref d) = mem.dp {
                        perf_dp_gpu_used_samples.push(d.gpu_used as f64);
                    }
                }
                scaling_samples.push(ScalingSample {
                    utterance_id: record.utterance_id.clone(),
                    num_frames_t: record.num_frames_t,
                    state_len: record.state_len,
                    ts_product: record.ts_product,
                    dp_ms: record.dp_ms,
                    group_ms: record.group_ms,
                    conf_ms: record.conf_ms,
                });
                lib_work_elapsed += Duration::from_secs_f64(record.total_ms / 1000.0);

                if args.perf_append {
                    let appender = perf_jsonl_appender
                        .as_mut()
                        .ok_or_else(|| "internal error: missing perf JSONL appender".to_string())?;
                    appender.append(&record)?;
                } else {
                    perf_records.push(record);
                }

                selected_words
                    .ok_or_else(|| format!("{}: missing profiled output words", case.id))?
            }
            #[cfg(not(feature = "alignment-profiling"))]
            {
                unreachable!(
                    "perf_enabled is only true when alignment-profiling feature is enabled"
                )
            }
        } else {
            let lib_started = Instant::now();
            let output = aligner
                .align(&alignment_input)
                .map_err(|err| format!("{}: align() failed: {err}", case.id))?;
            lib_work_elapsed += lib_started.elapsed();
            output.words
        };

        match args.output_format {
            OutputFormat::Json => {
                let split = infer_split(&case.audio_path);
                let sentence = compute_sentence_report(
                    &case.id,
                    split,
                    &output_words,
                    Some(&case.reference_words),
                    duration_ms,
                )
                .map_err(|err| format!("{}: metric computation failed: {err}", case.id))?;

                predicted_by_id.insert(case.id.clone(), output_words);
                references_by_id.insert(case.id.clone(), case.reference_words.clone());
                sentence_reports.push(sentence);
            }
            OutputFormat::TextGrid => {
                text_grid_report_formatter::write_textgrid(
                    &dataset_root,
                    &case.audio_path,
                    &case.transcript,
                    &output_words,
                    duration_ms,
                    &args.textgrid_suffix,
                )?;
                written_textgrids += 1;
            }
            OutputFormat::Perf => {}
        }
        progress.inc(1);
    }
    progress.finish_with_message("alignment pass complete");
    let lib_work_seconds = lib_work_elapsed.as_secs_f64();
    let avg_lib_case_ms = if selected_case_count > 0 {
        lib_work_seconds * 1000.0 / selected_case_count as f64
    } else {
        0.0
    };
    println!(
        "lib_work_elapsed: {:.2}s ({}) avg_per_case: {:.2}ms",
        lib_work_seconds,
        format_duration_hms(lib_work_elapsed),
        avg_lib_case_ms
    );

    match args.output_format {
        OutputFormat::Json => {
            let mut sentences = sentence_reports;
            let aggregates = aggregate_reports(&sentences);
            attach_outlier_traces(
                &mut sentences,
                &predicted_by_id,
                &references_by_id,
                OUTLIER_TRACE_TOP_N,
            );

            let report = Report {
                schema_version: 1,
                meta: Meta {
                    generated_at: Utc::now().to_rfc3339(),
                    model_path: model_dir.to_string_lossy().into_owned(),
                    device: args.device,
                    frame_stride_ms: frame_stride_ms as f32,
                    case_count: sentences.len(),
                },
                sentences,
                aggregates,
            };

            let out_path = out_path.ok_or_else(|| {
                "internal error: missing output path for JSON report format".to_string()
            })?;
            json_report_formatter::write_report(&out_path, &report)?;
            println!("{}", out_path.display());
        }
        OutputFormat::TextGrid => {
            if args.textgrid_suffix.is_empty() {
                println!(
                    "Wrote {written_textgrids} TextGrid file(s) alongside LibriSpeech .flac files."
                );
            } else {
                println!(
                    "Wrote {written_textgrids} TextGrid file(s) with suffix '{}' alongside LibriSpeech .flac files.",
                    args.textgrid_suffix
                );
            }
        }
        OutputFormat::Perf => {}
    }

    #[cfg(feature = "alignment-profiling")]
    if let Some(perf_path) = perf_out_path.as_ref() {
        let aggregate = perf_report_formatter::PerfAggregateStats {
            utterance_count: perf_total_samples.len(),
            forward_ms: summarize_metric(&perf_forward_samples),
            post_ms: summarize_metric(&perf_post_samples),
            dp_ms: summarize_metric(&perf_dp_samples),
            group_ms: summarize_metric(&perf_group_samples),
            conf_ms: summarize_metric(&perf_conf_samples),
            align_ms: summarize_metric(&perf_align_samples),
            align_ms_per_ts: summarize_metric(&perf_align_per_ts_samples),
            align_ms_per_t: summarize_metric(&perf_align_per_t_samples),
            total_ms: summarize_metric(&perf_total_samples),
            memory: perf_gpu_total.map(|gpu_total| perf_report_formatter::PerfAggregateMemory {
                forward_gpu_used: summarize_metric(&perf_forward_gpu_used_samples),
                dp_gpu_used: summarize_metric(&perf_dp_gpu_used_samples),
                gpu_total,
            }),
        };
        if args.perf_append {
            if let Some(appender) = perf_jsonl_appender.take() {
                appender.finish()?;
            }
            let summary_path = perf_report_formatter::summary_path_for(perf_path);
            perf_report_formatter::write_summary_report(&summary_path, &perf_config, &aggregate)?;
            println!("{}", perf_path.display());
            println!("{}", summary_path.display());
        } else {
            perf_report_formatter::write_json_report(
                perf_path,
                &perf_config,
                &perf_records,
                &aggregate,
            )?;
            println!("{}", perf_path.display());
        }

        if args.perf_scaling_report {
            print_scaling_report(&scaling_samples);
        }
    }
    Ok(())
}

fn build_aligner(
    model_dir: &Path,
    device: &str,
    runtime: RuntimeChoice,
) -> Result<ForcedAligner, String> {
    if runtime == RuntimeChoice::Onnx && !cfg!(feature = "onnx") {
        return Err(
            "runtime=onnx requested but this binary was built without ONNX support. Rebuild with `--features \"report-cli,onnx\"`."
                .to_string(),
        );
    }
    let model_filename = runtime.model_filename();
    require_path_exists(
        &model_dir.join(model_filename),
        &format!(
            "Missing model weights ({model_filename}) for runtime='{}'.",
            runtime.as_str()
        ),
    )?;
    require_path_exists(
        &model_dir.join("config.json"),
        "Missing model config (config.json).",
    )?;
    require_path_exists(
        &model_dir.join("vocab.json"),
        "Missing model vocabulary (vocab.json).",
    )?;

    let config = Wav2Vec2Config {
        model_path: model_dir
            .join(model_filename)
            .to_string_lossy()
            .into_owned(),
        config_path: model_dir.join("config.json").to_string_lossy().into_owned(),
        vocab_path: model_dir.join("vocab.json").to_string_lossy().into_owned(),
        device: device.to_string(),
        expected_sample_rate_hz: Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ,
    };
    ForcedAlignerBuilder::new(config)
        .with_runtime_kind(runtime.runtime_kind())
        .build()
        .map_err(|err| format!("Failed to build ForcedAligner: {err}"))
}

fn resolve_out_path(repo_root: &Path, out: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = out {
        return resolve_path(repo_root, path);
    }

    let run_id = Utc::now().format("%Y%m%dT%H%M%SZ");
    repo_root
        .join("target")
        .join("alignment_reports")
        .join(format!("alignment-report-{run_id}.json"))
}

fn load_all_cases(dataset_root: &Path) -> Result<Vec<Case>, String> {
    let librispeech_dir = dataset_root.join("LibriSpeech");
    let mut all_cases = Vec::new();
    for subset in LIBRISPEECH_SUBSETS {
        all_cases.extend(load_cases_from_subset(
            &librispeech_dir.join(subset),
            dataset_root,
        )?);
    }
    Ok(all_cases)
}

fn load_all_cases_from_transcripts(dataset_root: &Path) -> Result<Vec<Case>, String> {
    let librispeech_dir = dataset_root.join("LibriSpeech");
    let mut all_cases = Vec::new();
    for subset in LIBRISPEECH_SUBSETS {
        all_cases.extend(load_cases_from_subset_transcripts(
            &librispeech_dir.join(subset),
            dataset_root,
        )?);
    }
    Ok(all_cases)
}

fn load_cases_from_subset(subset_dir: &Path, dataset_root: &Path) -> Result<Vec<Case>, String> {
    require_path_exists(
        subset_dir,
        "Missing LibriSpeech subset directory under dataset root.",
    )?;

    let mut textgrids = Vec::new();
    collect_textgrid_files(subset_dir, &mut textgrids)?;
    textgrids.sort();
    if textgrids.is_empty() {
        return Err(format!(
            "No TextGrid files found in '{}'.",
            subset_dir.display()
        ));
    }

    textgrids
        .iter()
        .map(|path| parse_textgrid_case(path, dataset_root))
        .collect()
}

fn load_cases_from_subset_transcripts(
    subset_dir: &Path,
    dataset_root: &Path,
) -> Result<Vec<Case>, String> {
    require_path_exists(
        subset_dir,
        "Missing LibriSpeech subset directory under dataset root.",
    )?;

    let mut transcriptions = Vec::new();
    collect_transcription_files(subset_dir, &mut transcriptions)?;
    transcriptions.sort();
    if transcriptions.is_empty() {
        return Err(format!(
            "No *.trans.txt files found in '{}'.",
            subset_dir.display()
        ));
    }

    let mut cases = Vec::new();
    for transcript_path in transcriptions {
        let transcript_contents = fs::read_to_string(&transcript_path).map_err(|err| {
            format!(
                "Failed to read transcript file '{}': {err}",
                transcript_path.display()
            )
        })?;
        for (line_no, raw_line) in transcript_contents.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(id) = parts.next() else {
                continue;
            };
            let transcript = line[id.len()..].trim();
            if transcript.is_empty() {
                continue;
            }

            let audio_abs_path = transcript_path
                .parent()
                .unwrap_or(subset_dir)
                .join(format!("{id}.flac"));
            require_path_exists(
                &audio_abs_path,
                &format!(
                    "Missing sibling .flac for transcript entry '{}' at line {} in '{}'.",
                    id,
                    line_no + 1,
                    transcript_path.display()
                ),
            )?;
            let audio_path = audio_abs_path
                .strip_prefix(dataset_root)
                .map_err(|err| {
                    format!(
                        "Failed to make audio path '{}' relative to '{}': {err}",
                        audio_abs_path.display(),
                        dataset_root.display()
                    )
                })?
                .to_string_lossy()
                .replace('\\', "/");

            cases.push(Case {
                id: id.to_string(),
                audio_path,
                transcript: transcript.to_string(),
                reference_words: Vec::new(),
            });
        }
    }

    Ok(cases)
}

fn collect_textgrid_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|err| format!("Failed to read directory '{}': {err}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "Failed to read directory entry in '{}': {err}",
                dir.display()
            )
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_textgrid_files(&path, out)?;
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("TextGrid"))
        {
            out.push(path);
        }
    }
    Ok(())
}

fn collect_transcription_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|err| format!("Failed to read directory '{}': {err}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "Failed to read directory entry in '{}': {err}",
                dir.display()
            )
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_transcription_files(&path, out)?;
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".trans.txt"))
        {
            out.push(path);
        }
    }
    Ok(())
}

fn parse_textgrid_case(textgrid_path: &Path, dataset_root: &Path) -> Result<Case, String> {
    let reference_words = match parse_words_with_textgrid_crate(textgrid_path) {
        Ok(words) => words,
        Err(crate_err) => parse_words_tier_fallback(textgrid_path).map_err(|fallback_err| {
            format!(
                "Failed to parse TextGrid '{}' with textgrid crate ({crate_err}) and fallback parser ({fallback_err})",
                textgrid_path.display()
            )
        })?,
    };

    let transcript = reference_words
        .iter()
        .map(|word| word.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let id = textgrid_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| {
            format!(
                "Failed to derive utterance id from TextGrid path '{}'.",
                textgrid_path.display()
            )
        })?
        .to_string();

    let audio_path = textgrid_path.with_extension("flac");
    require_path_exists(&audio_path, "Missing sibling .flac for TextGrid reference.")?;
    let audio_path = audio_path
        .strip_prefix(dataset_root)
        .map_err(|err| {
            format!(
                "Failed to make audio path '{}' relative to '{}': {err}",
                audio_path.display(),
                dataset_root.display()
            )
        })?
        .to_string_lossy()
        .replace('\\', "/");

    Ok(Case {
        id,
        audio_path,
        transcript,
        reference_words,
    })
}

fn parse_words_with_textgrid_crate(path: &Path) -> Result<Vec<ReferenceWord>, String> {
    let textgrid =
        TextGrid::from_file(path).map_err(|err| format!("textgrid crate parse failed: {err}"))?;

    let words_tier = textgrid
        .tiers
        .iter()
        .find(|tier| {
            tier.tier_type == TierType::IntervalTier && tier.name.eq_ignore_ascii_case("words")
        })
        .ok_or_else(|| "missing IntervalTier named 'words'".to_string())?;

    let mut words = Vec::new();
    for interval in &words_tier.intervals {
        let word = interval.text.trim();
        if word.is_empty() {
            continue;
        }
        words.push(ReferenceWord {
            word: word.to_string(),
            start_ms: seconds_to_ms(interval.xmin, path)?,
            end_ms: seconds_to_ms(interval.xmax, path)?,
        });
    }
    Ok(words)
}

fn parse_words_tier_fallback(path: &Path) -> Result<Vec<ReferenceWord>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|err| format!("failed to read TextGrid '{}': {err}", path.display()))?;

    let mut in_item = false;
    let mut item_is_interval_tier = false;
    let mut item_is_words_tier = false;
    let mut in_words_tier = false;

    let mut cur_xmin: Option<f64> = None;
    let mut cur_xmax: Option<f64> = None;
    let mut words = Vec::new();

    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.starts_with("item [") {
            in_item = true;
            item_is_interval_tier = false;
            item_is_words_tier = false;
            in_words_tier = false;
            cur_xmin = None;
            cur_xmax = None;
            continue;
        }

        if !in_item {
            continue;
        }

        if let Some(value) = parse_assignment_value(line, "class") {
            item_is_interval_tier = value.eq_ignore_ascii_case("\"IntervalTier\"");
            in_words_tier = item_is_interval_tier && item_is_words_tier;
            continue;
        }

        if let Some(value) = parse_assignment_value(line, "name") {
            item_is_words_tier = value.eq_ignore_ascii_case("\"words\"");
            in_words_tier = item_is_interval_tier && item_is_words_tier;
            continue;
        }

        if !in_words_tier {
            continue;
        }

        if let Some(value) = parse_assignment_value(line, "xmin") {
            cur_xmin = Some(parse_number(value, path, "xmin")?);
            continue;
        }

        if let Some(value) = parse_assignment_value(line, "xmax") {
            cur_xmax = Some(parse_number(value, path, "xmax")?);
            continue;
        }

        if let Some(value) = parse_assignment_value(line, "text") {
            let word = strip_quotes(value).trim().to_string();
            if !word.is_empty() {
                let xmin = cur_xmin
                    .ok_or_else(|| format!("missing xmin before text in '{}'", path.display()))?;
                let xmax = cur_xmax
                    .ok_or_else(|| format!("missing xmax before text in '{}'", path.display()))?;
                words.push(ReferenceWord {
                    word,
                    start_ms: seconds_to_ms(xmin, path)?,
                    end_ms: seconds_to_ms(xmax, path)?,
                });
            }
            cur_xmin = None;
            cur_xmax = None;
        }
    }

    if !words.is_empty() {
        return Ok(words);
    }
    Err("no words found in fallback parse".to_string())
}

fn seconds_to_ms(seconds: f64, source_path: &Path) -> Result<u64, String> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(format!(
            "Invalid TextGrid timestamp in '{}': {seconds}",
            source_path.display()
        ));
    }
    let millis = (seconds * 1000.0).round();
    if !(0.0..=u64::MAX as f64).contains(&millis) {
        return Err(format!(
            "TextGrid timestamp out of range in '{}': {seconds}",
            source_path.display()
        ));
    }
    Ok(millis as u64)
}

fn parse_assignment_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (lhs, rhs) = line.split_once('=')?;
    if lhs.trim() == key {
        Some(rhs.trim())
    } else {
        None
    }
}

fn strip_quotes(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(value)
}

fn parse_number(value: &str, source_path: &Path, field: &str) -> Result<f64, String> {
    value.trim().parse::<f64>().map_err(|err| {
        format!(
            "failed to parse {field}='{value}' in '{}': {err}",
            source_path.display()
        )
    })
}

fn read_flac_mono(path: &Path) -> Result<(u32, Vec<f32>), String> {
    let mut reader = FlacReader::open(path)
        .map_err(|err| format!("Failed to decode FLAC '{}': {err}", path.display()))?;
    let streaminfo = reader.streaminfo();
    let channels = streaminfo.channels as usize;
    let bits_per_sample = streaminfo.bits_per_sample as i32;
    let scale = if bits_per_sample > 1 {
        ((1_i64 << (bits_per_sample - 1)) - 1) as f32
    } else {
        1.0
    };
    let sample_rate_hz = streaminfo.sample_rate;

    if channels == 0 {
        return Err(format!("FLAC has zero channels: {}", path.display()));
    }

    if channels == 1 {
        let mut mono = Vec::new();
        for sample in reader.samples() {
            let sample = sample
                .map_err(|err| format!("Failed reading sample from '{}': {err}", path.display()))?;
            mono.push(sample as f32 / scale);
        }
        return Ok((sample_rate_hz, mono));
    }

    let mut mono = Vec::new();
    let mut frame = Vec::with_capacity(channels);
    for sample in reader.samples() {
        let sample = sample
            .map_err(|err| format!("Failed reading sample from '{}': {err}", path.display()))?;
        frame.push(sample as f32 / scale);
        if frame.len() == channels {
            let avg = frame.iter().sum::<f32>() / channels as f32;
            mono.push(avg);
            frame.clear();
        }
    }
    Ok((sample_rate_hz, mono))
}

fn load_case_filter(
    cases_file: Option<&PathBuf>,
    repo_root: &Path,
) -> Result<Option<HashSet<String>>, String> {
    let Some(path) = cases_file else {
        return Ok(None);
    };
    let file_path = resolve_path(repo_root, path);
    require_path_exists(&file_path, "Missing --cases-file path.")?;

    let contents = fs::read_to_string(&file_path)
        .map_err(|err| format!("Failed to read cases file '{}': {err}", file_path.display()))?;

    let mut ids = HashSet::new();
    for raw_line in contents.lines() {
        let line = strip_line_number_prefix(raw_line.trim());
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for id in extract_audio_ids_from_line(line) {
            ids.insert(id);
        }
    }

    if ids.is_empty() {
        return Err(format!(
            "No valid case IDs were parsed from '{}'.",
            file_path.display()
        ));
    }
    Ok(Some(ids))
}

fn strip_line_number_prefix(line: &str) -> &str {
    let Some(rest) = line.strip_prefix('L') else {
        return line;
    };
    let Some((digits, suffix)) = rest.split_once(':') else {
        return line;
    };
    if !digits.is_empty() && digits.chars().all(|ch| ch.is_ascii_digit()) {
        suffix.trim()
    } else {
        line
    }
}

fn extract_audio_ids_from_line(line: &str) -> Vec<String> {
    let suffix_or_whole = line
        .rsplit_once("::audio::")
        .map_or(line, |(_, suffix)| suffix);
    if looks_like_audio_id(suffix_or_whole) {
        return vec![suffix_or_whole.to_string()];
    }

    line.split(|ch: char| !(ch.is_ascii_digit() || ch == '-'))
        .filter(|token| looks_like_audio_id(token))
        .map(ToString::to_string)
        .collect()
}

fn looks_like_audio_id(value: &str) -> bool {
    let parts: Vec<&str> = value.split('-').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
}

fn resolve_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

#[cfg(feature = "alignment-profiling")]
fn aggregate_measurements(values: &[f64], mode: PerfAggregate) -> f64 {
    match mode {
        PerfAggregate::Median => median(values),
        PerfAggregate::Mean => mean(values),
    }
}

#[cfg(feature = "alignment-profiling")]
fn summarize_metric(values: &[f64]) -> perf_report_formatter::PerfMetricStats {
    if values.is_empty() {
        return perf_report_formatter::PerfMetricStats {
            mean: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }
    let min = values
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or_default();
    let max = values
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or_default();
    perf_report_formatter::PerfMetricStats {
        mean: mean(values),
        median: median(values),
        min,
        max,
    }
}

#[cfg(feature = "alignment-profiling")]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "alignment-profiling")]
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(feature = "alignment-profiling")]
fn print_scaling_report(samples: &[ScalingSample]) {
    println!("perf_scaling_report:");
    if samples.is_empty() {
        println!("scaling_summary: utterances=0 corr_dp_ms_vs_ts=0.000000");
        return;
    }

    for sample in samples {
        println!(
            "scaling_case utterance_id={} T={} S={} TS={} dp_ms={:.6} group_ms={:.6} conf_ms={:.6}",
            sample.utterance_id,
            sample.num_frames_t,
            sample.state_len,
            sample.ts_product,
            sample.dp_ms,
            sample.group_ms,
            sample.conf_ms
        );
    }

    let xs = samples
        .iter()
        .map(|sample| sample.ts_product as f64)
        .collect::<Vec<_>>();
    let ys = samples
        .iter()
        .map(|sample| sample.dp_ms)
        .collect::<Vec<_>>();
    let corr = pearson_correlation(&xs, &ys);
    println!(
        "scaling_summary: utterances={} corr_dp_ms_vs_ts={:.6}",
        samples.len(),
        corr
    );

    let mut outliers = samples
        .iter()
        .map(|sample| {
            let dp_ms_per_ts = if sample.ts_product > 0 {
                sample.dp_ms / sample.ts_product as f64
            } else {
                0.0
            };
            (sample, dp_ms_per_ts)
        })
        .collect::<Vec<_>>();
    outliers.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (rank, (sample, dp_ms_per_ts)) in outliers.into_iter().take(5).enumerate() {
        println!(
            "scaling_outlier rank={} utterance_id={} dp_ms_per_ts={:.9} dp_ms={:.6} TS={}",
            rank + 1,
            sample.utterance_id,
            dp_ms_per_ts,
            sample.dp_ms,
            sample.ts_product
        );
    }
}

#[cfg(feature = "alignment-profiling")]
fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return 0.0;
    }
    let x_mean = mean(xs);
    let y_mean = mean(ys);
    let mut cov = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let xd = x - x_mean;
        let yd = y - y_mean;
        cov += xd * yd;
        x_var += xd * xd;
        y_var += yd * yd;
    }
    if x_var <= f64::EPSILON || y_var <= f64::EPSILON {
        return 0.0;
    }
    cov / (x_var.sqrt() * y_var.sqrt())
}

fn format_duration_hms(duration: Duration) -> String {
    let total_ms = duration.as_millis();
    let hours = total_ms / 3_600_000;
    let rem_after_hours = total_ms % 3_600_000;
    let minutes = rem_after_hours / 60_000;
    let rem_after_minutes = rem_after_hours % 60_000;
    let seconds = rem_after_minutes / 1_000;
    let millis = rem_after_minutes % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

fn require_path_exists(path: &Path, message: &str) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    Err(format!("{message} Missing path: {}", path.display()))
}
