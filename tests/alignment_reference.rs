use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use claxon::FlacReader;
use libtest_mimic::{Arguments, Failed, Trial};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;
use wav2vec2_rs::{AlignmentInput, ForcedAligner, ForcedAlignerBuilder, Wav2Vec2Config};

const DEFAULT_SAMPLE_SIZE: usize = 50;
const DEFAULT_DELTA_MS: f64 = 20.0;
const DEFAULT_SAMPLE_SEED: u64 = 42;
const SUITE_NAME: &str = "pytorch_alignment_reference_matches_within_delta";

#[derive(Debug, Deserialize)]
struct ReferenceUtterance {
    id: String,
    audio_path: String,
    transcript: String,
    words: Vec<ReferenceWordTiming>,
}

#[derive(Debug, Deserialize)]
struct ReferenceWordTiming {
    word: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug)]
struct RuntimeContext {
    repo_root: PathBuf,
    model_dir: PathBuf,
    device: String,
}

thread_local! {
    static THREAD_ALIGNER: RefCell<Option<ForcedAligner>> = RefCell::new(None);
}

fn main() {
    let mut args = Arguments::from_args();
    // The aligner is heavy and GPU backends are often not thread-safe in CI.
    if args.test_threads.is_none() {
        args.test_threads = Some(1);
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let full_mode = env_flag("WAV2VEC2_IT_FULL");
    let sample_seed = env_u64("WAV2VEC2_IT_SEED", DEFAULT_SAMPLE_SEED);
    let delta_ms = env_f64("WAV2VEC2_IT_DELTA_MS", DEFAULT_DELTA_MS);

    let all_rows = match load_all_rows(&repo_root) {
        Ok(rows) => rows,
        Err(err) => {
            run_setup_failure(&args, err);
            return;
        }
    };
    if all_rows.is_empty() {
        run_setup_failure(
            &args,
            "No reference rows found under test-data/alignments.".to_string(),
        );
        return;
    }

    let sampled_ids = select_row_ids(&all_rows, full_mode, sample_seed);
    let mut tests = Vec::with_capacity(all_rows.len());
    for row in all_rows {
        let test_name = format!("{SUITE_NAME}::audio::{}", row.id);
        let run_by_default = sampled_ids.contains(&row.id);
        let should_ignore = !run_by_default && !should_run_ignored_case(&args, &test_name);

        tests.push(
            Trial::test(test_name, move || run_reference_case(&row, delta_ms).map_err(Failed::from))
                .with_ignored_flag(should_ignore),
        );
    }

    libtest_mimic::run(&args, tests).exit();
}

fn run_setup_failure(args: &Arguments, message: String) {
    let test = Trial::test(format!("{SUITE_NAME}::setup"), move || {
        Err(Failed::from(message))
    });
    libtest_mimic::run(args, vec![test]).exit();
}

fn run_reference_case(row: &ReferenceUtterance, delta_ms: f64) -> Result<(), String> {
    let ctx = runtime_context()?;
    let audio_path = ctx.repo_root.join("test-data").join(&row.audio_path);
    require_path_exists(
        &audio_path,
        "Missing audio file referenced by alignment fixture. Rebuild test-data or adjust fixture paths.",
    )?;

    let (sample_rate_hz, samples) = read_flac_mono(&audio_path)?;
    let input = AlignmentInput {
        sample_rate_hz,
        samples,
        transcript: row.transcript.clone(),
    };
    let output = with_thread_aligner(|aligner| {
        aligner
            .align(&input)
            .map_err(|err| format!("{}: align() failed: {}", row.id, err))
    })?;

    compare_alignment(row, &output.words, delta_ms)
}

fn compare_alignment(
    row: &ReferenceUtterance,
    observed_words: &[wav2vec2_rs::WordTiming],
    delta_ms: f64,
) -> Result<(), String> {
    if observed_words.len() != row.words.len() {
        return Err(format!(
            "{}: word count mismatch (expected {}, got {})",
            row.id,
            row.words.len(),
            observed_words.len()
        ));
    }

    if row.words.is_empty() {
        return Ok(());
    }

    let mut sum_abs_ms = 0.0f64;
    let mut worst = Vec::with_capacity(row.words.len());
    for (idx, (expected, observed)) in row.words.iter().zip(observed_words.iter()).enumerate() {
        if !observed.word.eq_ignore_ascii_case(&expected.word) {
            return Err(format!(
                "{}: word mismatch at index {} (expected '{}', got '{}')",
                row.id, idx, expected.word, observed.word
            ));
        }

        let start_diff = observed.start_ms.abs_diff(expected.start_ms);
        let end_diff = observed.end_ms.abs_diff(expected.end_ms);
        sum_abs_ms += (start_diff + end_diff) as f64;
        worst.push((idx, expected.word.as_str(), start_diff, end_diff));
    }

    let avg_ms = sum_abs_ms / (2.0 * row.words.len() as f64);
    if avg_ms > delta_ms {
        worst.sort_by_key(|(_, _, start_diff, end_diff)| Reverse(start_diff + end_diff));
        let top = worst
            .iter()
            .take(3)
            .map(|(idx, word, start_diff, end_diff)| {
                format!(
                    "#{idx} '{word}' (start_abs={}ms, end_abs={}ms)",
                    start_diff, end_diff
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{}: avg_abs_ms={avg_ms:.2} exceeds delta_ms={delta_ms:.2}; worst={}",
            row.id, top
        ));
    }

    Ok(())
}

fn with_thread_aligner<T>(
    f: impl FnOnce(&ForcedAligner) -> Result<T, String>,
) -> Result<T, String> {
    let ctx = runtime_context()?;

    THREAD_ALIGNER.with(|cell| {
        let needs_init = cell.borrow().is_none();
        if needs_init {
            let config = Wav2Vec2Config {
                model_path: ctx
                    .model_dir
                    .join("model.safetensors")
                    .to_string_lossy()
                    .into_owned(),
                config_path: ctx.model_dir.join("config.json").to_string_lossy().into_owned(),
                vocab_path: ctx.model_dir.join("vocab.json").to_string_lossy().into_owned(),
                device: ctx.device.clone(),
                expected_sample_rate_hz: Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ,
            };
            let aligner = ForcedAlignerBuilder::new(config).build().map_err(|err| {
                format!(
                    "Failed to build ForcedAligner with model dir '{}': {err}",
                    ctx.model_dir.display()
                )
            })?;
            *cell.borrow_mut() = Some(aligner);
        }

        let borrow = cell.borrow();
        let aligner = borrow
            .as_ref()
            .ok_or_else(|| "aligner init failed unexpectedly".to_string())?;
        f(aligner)
    })
}

fn runtime_context() -> Result<&'static RuntimeContext, String> {
    static CONTEXT: OnceLock<Result<RuntimeContext, String>> = OnceLock::new();
    CONTEXT
        .get_or_init(build_runtime_context)
        .as_ref()
        .map_err(|err| err.clone())
}

fn build_runtime_context() -> Result<RuntimeContext, String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_dir = resolve_model_dir(&repo_root);

    require_path_exists(
        &model_dir.join("model.safetensors"),
        "Missing model weights. Fetch Git LFS model files or set WAV2VEC2_IT_MODEL_DIR to a valid model directory.",
    )?;
    require_path_exists(
        &model_dir.join("config.json"),
        "Missing model config. Ensure config.json exists in the model directory.",
    )?;
    require_path_exists(
        &model_dir.join("vocab.json"),
        "Missing vocab file. Ensure vocab.json exists in the model directory.",
    )?;

    Ok(RuntimeContext {
        repo_root,
        model_dir,
        device: env::var("WAV2VEC2_IT_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
    })
}

fn load_all_rows(repo_root: &Path) -> Result<Vec<ReferenceUtterance>, String> {
    let mut all_rows = Vec::new();
    all_rows.extend(load_reference_subset(
        &repo_root.join("test-data/alignments/test-clean.json"),
    )?);
    all_rows.extend(load_reference_subset(
        &repo_root.join("test-data/alignments/test-other.json"),
    )?);
    Ok(all_rows)
}

fn load_reference_subset(path: &Path) -> Result<Vec<ReferenceUtterance>, String> {
    require_path_exists(
        path,
        "Missing alignment fixture JSON. Generate fixtures via scripts/pytorch_aligner.py before running integration tests.",
    )?;
    let file = File::open(path)
        .map_err(|err| format!("Failed to open fixture '{}': {err}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|err| format!("Failed to parse fixture '{}': {err}", path.display()))
}

fn select_row_ids(rows: &[ReferenceUtterance], full_mode: bool, seed: u64) -> HashSet<String> {
    if full_mode || rows.len() <= DEFAULT_SAMPLE_SIZE {
        return rows.iter().map(|row| row.id.clone()).collect();
    }

    let mut indices: Vec<usize> = (0..rows.len()).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices
        .into_iter()
        .take(DEFAULT_SAMPLE_SIZE)
        .map(|idx| rows[idx].id.clone())
        .collect()
}

fn matches_filter(args: &Arguments, test_name: &str) -> bool {
    match args.filter.as_deref() {
        None => false,
        Some(filter) if args.exact => test_name == filter,
        Some(filter) => test_name.contains(filter),
    }
}

fn should_run_ignored_case(args: &Arguments, test_name: &str) -> bool {
    if !matches_filter(args, test_name) {
        return false;
    }

    let Some(filter) = args.filter.as_deref() else {
        return false;
    };
    if args.exact {
        return true;
    }

    filter.contains("::audio::")
        || filter.starts_with("audio::")
        || looks_like_audio_id(filter)
}

fn looks_like_audio_id(filter: &str) -> bool {
    let parts: Vec<&str> = filter.split('-').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
}

fn resolve_model_dir(repo_root: &Path) -> PathBuf {
    let model_dir = env::var("WAV2VEC2_IT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/wav2vec2-base-960h"));
    if model_dir.is_absolute() {
        model_dir
    } else {
        repo_root.join(model_dir)
    }
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
        let sample =
            sample.map_err(|err| format!("Failed reading sample from '{}': {err}", path.display()))?;
        frame.push(sample as f32 / scale);
        if frame.len() == channels {
            let avg = frame.iter().sum::<f32>() / channels as f32;
            mono.push(avg);
            frame.clear();
        }
    }

    Ok((sample_rate_hz, mono))
}

fn env_flag(name: &str) -> bool {
    match env::var(name) {
        Ok(value) => matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

fn env_f64(name: &str, default: f64) -> f64 {
    match env::var(name) {
        Ok(value) => value.trim().parse::<f64>().unwrap_or_else(|err| {
            panic!(
                "Invalid value for {}='{}' (expected f64): {}",
                name, value, err
            )
        }),
        Err(_) => default,
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    match env::var(name) {
        Ok(value) => value.trim().parse::<u64>().unwrap_or_else(|err| {
            panic!(
                "Invalid value for {}='{}' (expected u64): {}",
                name, value, err
            )
        }),
        Err(_) => default,
    }
}

fn require_path_exists(path: &Path, message: &str) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    Err(format!("{} Missing path: {}", message, path.display()))
}
