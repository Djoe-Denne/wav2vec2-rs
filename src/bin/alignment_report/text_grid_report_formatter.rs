use std::fs;
use std::path::{Path, PathBuf};

use textgrid::{Interval, TextGrid, Tier, TierType};
use wav2vec2_rs::WordTiming;

pub fn write_textgrid(
    dataset_root: &Path,
    relative_audio_path: &str,
    transcript: &str,
    words: &[WordTiming],
    duration_ms: u64,
    suffix: &str,
) -> Result<PathBuf, String> {
    let audio_path = dataset_root.join(relative_audio_path);
    let out_path = build_textgrid_output_path(&audio_path, suffix)?;
    let max_word_end_ms = words.iter().map(|word| word.end_ms).max().unwrap_or(0);
    let total_duration_ms = duration_ms.max(max_word_end_ms).max(1);
    let xmax = millis_to_seconds(total_duration_ms);

    let mut textgrid = TextGrid::new(0.0, xmax).map_err(|err| {
        format!(
            "Failed to build TextGrid structure '{}': {err}",
            out_path.display()
        )
    })?;

    let mut sorted_words = words.iter().collect::<Vec<_>>();
    sorted_words.sort_by_key(|word| (word.start_ms, word.end_ms));

    let mut words_intervals = Vec::with_capacity(sorted_words.len());
    let mut confidence_intervals = Vec::with_capacity(sorted_words.len());
    let mut last_end_ms = 0u64;
    for word in sorted_words {
        let start_ms = word.start_ms.min(total_duration_ms).max(last_end_ms);
        let end_ms = word.end_ms.min(total_duration_ms);
        if end_ms <= start_ms {
            continue;
        }
        let xmin = millis_to_seconds(start_ms);
        let xmax = millis_to_seconds(end_ms);
        words_intervals.push(Interval {
            xmin,
            xmax,
            text: word.word.clone(),
        });
        confidence_intervals.push(Interval {
            xmin,
            xmax,
            text: word
                .confidence
                .map(|value| format!("{value:.2}"))
                .unwrap_or_default(),
        });
        last_end_ms = end_ms;
    }

    let words_tier = Tier {
        name: "words".to_string(),
        tier_type: TierType::IntervalTier,
        xmin: 0.0,
        xmax,
        intervals: words_intervals,
        points: Vec::new(),
    };
    let confidence_tier = Tier {
        name: "words-confidence".to_string(),
        tier_type: TierType::IntervalTier,
        xmin: 0.0,
        xmax,
        intervals: confidence_intervals,
        points: Vec::new(),
    };

    textgrid.add_tier(words_tier).map_err(|err| {
        format!(
            "Failed to add words tier for '{}': {err}",
            out_path.display()
        )
    })?;
    textgrid.add_tier(confidence_tier).map_err(|err| {
        format!(
            "Failed to add confidence tier for '{}': {err}",
            out_path.display()
        )
    })?;

    let transcript_text = transcript.trim();
    if !transcript_text.is_empty() {
        let transcript_tier = Tier {
            name: "transcript".to_string(),
            tier_type: TierType::IntervalTier,
            xmin: 0.0,
            xmax,
            intervals: vec![Interval {
                xmin: 0.0,
                xmax,
                text: transcript_text.to_string(),
            }],
            points: Vec::new(),
        };
        textgrid.add_tier(transcript_tier).map_err(|err| {
            format!(
                "Failed to add transcript tier for '{}': {err}",
                out_path.display()
            )
        })?;
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create TextGrid output directory '{}': {err}",
                parent.display()
            )
        })?;
    }
    textgrid
        .to_file(&out_path, false)
        .map_err(|err| format!("Failed to write TextGrid '{}': {err}", out_path.display()))?;

    Ok(out_path)
}

fn millis_to_seconds(ms: u64) -> f64 {
    ms as f64 / 1000.0
}

fn build_textgrid_output_path(audio_path: &Path, suffix: &str) -> Result<PathBuf, String> {
    let stem = audio_path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            format!(
                "Failed to derive file stem for audio path '{}'.",
                audio_path.display()
            )
        })?;
    let file_name = format!("{stem}{suffix}.TextGrid");
    Ok(audio_path.with_file_name(file_name))
}
