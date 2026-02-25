use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use clap::Parser;
use claxon::FlacReader;
use indicatif::{ProgressBar, ProgressStyle};
use textgrid::{TextGrid, TierType};
use wav2vec2_rs::{
    aggregate_reports, attach_outlier_traces, compute_sentence_report, infer_split, AlignmentInput,
    ForcedAligner, ForcedAlignerBuilder, Meta, ReferenceWord, Report, SentenceReport,
    Wav2Vec2Config, WordTiming,
};

const LIBRISPEECH_SUBSETS: [&str; 2] = ["test-clean", "test-other"];
const OUTLIER_TRACE_TOP_N: usize = 20;

#[derive(Debug, Parser)]
#[command(name = "alignment_report")]
#[command(about = "Generate deterministic forced-alignment quality reports")]
struct Args {
    #[arg(
        long,
        env = "WAV2VEC2_REPORT_MODEL_DIR",
        default_value = "models/wav2vec2-base-960h"
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
}

#[derive(Debug, Clone)]
struct Case {
    id: String,
    audio_path: String,
    transcript: String,
    reference_words: Vec<ReferenceWord>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("alignment_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let model_dir = resolve_path(&repo_root, &args.model_dir);
    let dataset_root = resolve_path(&repo_root, &args.dataset_root);
    let out_path = resolve_out_path(&repo_root, args.out.as_ref());

    let include_ids = load_case_filter(args.cases_file.as_ref(), &repo_root)?;
    let mut cases = load_all_cases(&dataset_root)?;

    if let Some(ids) = include_ids.as_ref() {
        let known: HashSet<&str> = cases.iter().map(|case| case.id.as_str()).collect();
        let mut missing = ids
            .iter()
            .filter(|id| !known.contains(id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        missing.sort();
        if !missing.is_empty() {
            eprintln!(
                "warning: {} case id(s) from --cases-file were not found in the dataset (showing up to 10): {}",
                missing.len(),
                missing.into_iter().take(10).collect::<Vec<_>>().join(", ")
            );
        }

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

    let aligner = build_aligner(&model_dir, &args.device)?;
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

        let output = aligner
            .align(&AlignmentInput {
                sample_rate_hz,
                samples,
                transcript: case.transcript.clone(),
            })
            .map_err(|err| format!("{}: align() failed: {err}", case.id))?;

        let split = infer_split(&case.audio_path);
        let sentence = compute_sentence_report(
            &case.id,
            split,
            &output.words,
            Some(&case.reference_words),
            duration_ms,
        )
        .map_err(|err| format!("{}: metric computation failed: {err}", case.id))?;

        predicted_by_id.insert(case.id.clone(), output.words);
        references_by_id.insert(case.id.clone(), case.reference_words.clone());
        sentence_reports.push(sentence);
        progress.inc(1);
    }
    progress.finish_with_message("alignment pass complete");

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

    write_report(&out_path, &report)?;
    println!("{}", out_path.display());
    Ok(())
}

fn build_aligner(model_dir: &Path, device: &str) -> Result<ForcedAligner, String> {
    require_path_exists(
        &model_dir.join("model.safetensors"),
        "Missing model weights (model.safetensors).",
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
            .join("model.safetensors")
            .to_string_lossy()
            .into_owned(),
        config_path: model_dir.join("config.json").to_string_lossy().into_owned(),
        vocab_path: model_dir.join("vocab.json").to_string_lossy().into_owned(),
        device: device.to_string(),
        expected_sample_rate_hz: Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ,
    };
    ForcedAlignerBuilder::new(config)
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

fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create report output directory '{}': {err}",
                parent.display()
            )
        })?;
    }

    let mut file = File::create(path)
        .map_err(|err| format!("Failed to create report file '{}': {err}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, report).map_err(|err| {
        format!(
            "Failed to serialize report JSON '{}': {err}",
            path.display()
        )
    })?;
    file.write_all(b"\n")
        .map_err(|err| format!("Failed to finalize report file '{}': {err}", path.display()))?;
    Ok(())
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

fn require_path_exists(path: &Path, message: &str) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    Err(format!("{message} Missing path: {}", path.display()))
}
