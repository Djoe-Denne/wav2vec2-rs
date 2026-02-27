use crate::types::{WordConfidenceStats, WordTiming};
use std::time::{Duration, Instant};

mod blank_expansion;
mod candidate_selector;
mod path_to_words;
#[cfg(test)]
mod tests;

/// Word with frame-level boundaries before blank expansion.
#[derive(Clone, Debug)]
pub(crate) struct RawWord {
    word: String,
    start_frame: usize,
    end_frame: usize,
    confidence: Option<f32>,
    confidence_stats: WordConfidenceStats,
}

pub struct ProfiledWordGroupingOutput {
    pub words: Vec<WordTiming>,
    pub conf_ms: f64,
    /// Time spent walking the Viterbi path and collecting raw word boundaries.
    pub collect_ms: f64,
    /// Time spent cloning candidates, expanding blanks, and selecting the best policy.
    pub expand_select_ms: f64,
}

pub fn group_into_words(
    path: &[(usize, usize)],
    tokens: &[usize],
    chars: &[Option<char>],
    expected_words: &[String],
    log_probs: &[Vec<f32>],
    blank_id: usize,
    word_sep_id: usize,
    stride_ms: f64,
) -> Vec<WordTiming> {
    group_into_words_profiled(
        path,
        tokens,
        chars,
        expected_words,
        log_probs,
        blank_id,
        word_sep_id,
        stride_ms,
    )
    .words
}

pub fn group_into_words_profiled(
    path: &[(usize, usize)],
    tokens: &[usize],
    chars: &[Option<char>],
    expected_words: &[String],
    log_probs: &[Vec<f32>],
    blank_id: usize,
    word_sep_id: usize,
    stride_ms: f64,
) -> ProfiledWordGroupingOutput {
    // --- Block 1: collect raw words from Viterbi path ---
    let collect_started = Instant::now();
    let profiled_raw = path_to_words::collect_profiled(
        path,
        tokens,
        chars,
        expected_words,
        log_probs,
        blank_id,
        word_sep_id,
        stride_ms,
    );
    let collect_ms = duration_to_ms(collect_started.elapsed());

    let raw = profiled_raw.words;
    if raw.is_empty() {
        return ProfiledWordGroupingOutput {
            words: Vec::new(),
            conf_ms: 0.0,
            collect_ms,
            expand_select_ms: 0.0,
        };
    }

    // --- Block 2: expand + select best candidate ---
    let expand_select_started = Instant::now();
    let first_frame = path.first().map(|&(_, f)| f).unwrap_or(0);
    let last_frame = path.last().map(|&(_, f)| f).unwrap_or(0);
    let mut candidates = Vec::with_capacity(blank_expansion::ExpansionPolicy::ALL.len());
    for policy in blank_expansion::ExpansionPolicy::ALL {
        candidates.push((
            policy,
            blank_expansion::expand_with_policy(raw.clone(), first_frame, last_frame, policy),
        ));
    }

    let (selected_policy, expanded, selected_score) =
        match candidate_selector::select_best(&raw, candidates, log_probs, blank_id) {
            Some(chosen) => (chosen.policy, chosen.words, Some(chosen.score)),
            None => (
                blank_expansion::ExpansionPolicy::Balanced,
                blank_expansion::expand(raw, first_frame, last_frame),
                None,
            ),
        };
    let expand_select_ms = duration_to_ms(expand_select_started.elapsed());

    if let Some(score) = selected_score {
        tracing::debug!(
            selected_policy = selected_policy.as_str(),
            score_total = format!("{:.3}", score.total_score),
            score_blank_boundary = format!("{:.3}", score.boundary_confidence_term),
            score_boundary_shift = format!("{:.3}", score.boundary_shift_penalty),
            score_pause = format!("{:.3}", score.pause_penalty),
            "grouping: selected expansion policy"
        );
    }

    // --- Block 3: confidence scoring ---
    let conf_started = Instant::now();
    let words = expanded
        .into_iter()
        .map(|mut w| {
            // Timing contract: [start_ms, end_ms), start inclusive and end exclusive.
            let start_ms = (w.start_frame as f64 * stride_ms) as u64;
            let end_ms = ((w.end_frame + 1) as f64 * stride_ms) as u64;
            let quality_confidence = quality_confidence_score(&w.confidence_stats);
            let calibrated_confidence = quality_confidence.map(calibrate_quality_confidence);
            w.confidence_stats.quality_confidence = quality_confidence;
            w.confidence_stats.calibrated_confidence = calibrated_confidence;
            tracing::debug!(
                word = w.word.as_str(),
                start_frame = w.start_frame,
                end_frame = w.end_frame,
                start_ms,
                end_ms,
                quality_confidence = quality_confidence,
                calibrated_confidence = calibrated_confidence,
                "grouping: final word boundary (after blank expansion)"
            );
            WordTiming {
                word: w.word,
                start_ms,
                end_ms,
                confidence: calibrated_confidence,
                confidence_stats: w.confidence_stats,
            }
        })
        .collect();
    let conf_ms = duration_to_ms(conf_started.elapsed());

    ProfiledWordGroupingOutput {
        words,
        conf_ms,
        collect_ms,
        expand_select_ms,
    }
}

fn quality_confidence_score(stats: &WordConfidenceStats) -> Option<f32> {
    let geo = stats.geo_mean_prob? as f64;

    // Deterministic confidence score: blend raw support with separability and boundary evidence.
    let mut weighted_sum = 0.0f64;
    let mut total_weight = 0.0f64;

    weighted_sum += 0.40 * geo;
    total_weight += 0.40;

    if let Some(margin) = stats.mean_margin {
        let margin_score = sigmoid((margin as f64 - 1.0) / 1.5);
        weighted_sum += 0.30 * margin_score;
        total_weight += 0.30;
    }

    if let Some(p10_logp) = stats.p10_logp {
        let p10_prob = (p10_logp as f64).exp().clamp(0.0, 1.0);
        weighted_sum += 0.20 * p10_prob;
        total_weight += 0.20;
    }

    let boundary_score = stats.boundary_confidence.map(|v| v as f64).unwrap_or(0.5);
    weighted_sum += 0.10 * boundary_score.clamp(0.0, 1.0);
    total_weight += 0.10;

    if total_weight <= 0.0 {
        None
    } else {
        Some((weighted_sum / total_weight).clamp(0.0, 1.0) as f32)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn calibrate_quality_confidence(score: f32) -> f32 {
    const KNOTS: &[(f64, f64)] = &[
        (0.00, 0.02),
        (0.20, 0.12),
        (0.35, 0.28),
        (0.50, 0.50),
        (0.65, 0.72),
        (0.80, 0.88),
        (0.95, 0.97),
        (1.00, 0.99),
    ];

    let x = (score as f64).clamp(0.0, 1.0);
    for window in KNOTS.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];
        if x <= x1 {
            let t = if (x1 - x0).abs() < f64::EPSILON {
                0.0
            } else {
                (x - x0) / (x1 - x0)
            };
            return (y0 + t * (y1 - y0)).clamp(0.0, 1.0) as f32;
        }
    }
    0.99
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}