use super::RawWord;
use crate::types::WordConfidenceStats;

/// Word completion check used before flushing a boundary.
///
/// The comparison is case-insensitive because different models/vocabs can
/// represent casing differently (for example, uppercase-only vocabularies).
fn matches_expected_word(cur_word: &str, expected_words: &[String], produced_words: usize) -> bool {
    expected_words
        .get(produced_words)
        .map(|expected| cur_word.eq_ignore_ascii_case(expected))
        // If we cannot infer an expected word (unexpected tokenizer shape),
        // fall back to permissive behavior and allow flush.
        .unwrap_or(true)
}

fn flush_word(
    cur_word: &mut String,
    start_frame: &mut Option<usize>,
    end_frame: usize,
    emission_lp_accum: &mut Vec<f32>,
    emission_margin_accum: &mut Vec<f32>,
    coverage_frame_count: &mut usize,
    out: &mut Vec<RawWord>,
) {
    if cur_word.is_empty() {
        return;
    }
    let confidence_stats = build_confidence_stats(
        emission_lp_accum,
        emission_margin_accum,
        *coverage_frame_count,
    );
    // Raw acoustic support retained for boundary scoring; final word confidence
    // score is derived later from full confidence_stats.
    let confidence = confidence_stats.geo_mean_prob;
    if confidence.is_none() {
        tracing::warn!(
            word = cur_word.as_str(),
            start_frame = start_frame,
            end_frame,
            "grouping: invalid word confidence (no covered frames)"
        );
    }
    out.push(RawWord {
        word: cur_word.clone(),
        start_frame: start_frame.unwrap_or(end_frame),
        end_frame,
        confidence,
        confidence_stats,
    });
    cur_word.clear();
    *start_frame = None;
    emission_lp_accum.clear();
    emission_margin_accum.clear();
    *coverage_frame_count = 0;
}

/// Phase 1: Walk the Viterbi path and group character frames into words.
/// Boundaries are tight — only character-emitting frames set start/end.
pub(super) fn collect(
    path: &[(usize, usize)],
    tokens: &[usize],
    chars: &[Option<char>],
    expected_words: &[String],
    log_probs: &[Vec<f32>],
    blank_id: usize,
    word_sep_id: usize,
    stride_ms: f64,
) -> Vec<RawWord> {
    let mut words = Vec::new();
    let mut cur_word = String::new();
    let mut start_frame: Option<usize> = None;
    let mut end_frame: usize = 0;
    let mut emission_lp_accum = Vec::new();
    let mut emission_margin_accum = Vec::new();
    let mut coverage_frame_count = 0usize;
    let mut prev_state: Option<usize> = None;

    let words_from_chars = reconstruct_words_from_chars(chars);
    if words_from_chars != expected_words {
        tracing::warn!(
            expected = ?expected_words,
            from_chars = ?words_from_chars,
            "grouping: normalized transcript words differ from char stream words"
        );
    }

    for &(s, frame) in path {
        let tid = tokens[s];
        let frame_ms = frame as f64 * stride_ms;

        if tid == blank_id {
            tracing::debug!(
                frame,
                frame_ms = format!("{:.0}", frame_ms),
                state = s,
                kind = "blank",
                cur_word = cur_word.as_str(),
                "grouping: blank frame"
            );
            prev_state = Some(s);
            continue;
        }

        if tid == word_sep_id {
            tracing::debug!(
                frame,
                frame_ms = format!("{:.0}", frame_ms),
                state = s,
                kind = "sep",
                cur_word = cur_word.as_str(),
                "grouping: separator frame"
            );
            if !cur_word.is_empty()
                && !matches_expected_word(&cur_word, &expected_words, words.len())
            {
                prev_state = Some(s);
                continue;
            }
            flush_word(
                &mut cur_word,
                &mut start_frame,
                end_frame,
                &mut emission_lp_accum,
                &mut emission_margin_accum,
                &mut coverage_frame_count,
                &mut words,
            );
            prev_state = Some(s);
            continue;
        }
        if let Some(c) = chars[s] {
            let is_new_state = prev_state != Some(s);
            if start_frame.is_none() {
                start_frame = Some(frame);
            }
            end_frame = frame;
            coverage_frame_count += 1;
            if is_new_state {
                // Confidence uses emission events (state changes), not repeated holds.
                emission_lp_accum.push(log_probs[frame][tid]);
                emission_margin_accum.push(top2_margin_logp(&log_probs[frame]));
                cur_word.push(c);
            }
            tracing::debug!(
                frame,
                frame_ms = format!("{:.0}", frame_ms),
                state = s,
                kind = "char",
                char = %c,
                new_state = is_new_state,
                cur_word = cur_word.as_str(),
                start_frame = start_frame,
                end_frame = end_frame,
                "grouping: char frame"
            );
        }
        prev_state = Some(s);
    }

    flush_word(
        &mut cur_word,
        &mut start_frame,
        end_frame,
        &mut emission_lp_accum,
        &mut emission_margin_accum,
        &mut coverage_frame_count,
        &mut words,
    );
    words
}

fn reconstruct_words_from_chars(chars: &[Option<char>]) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for c in chars.iter().copied().flatten() {
        if c == '|' {
            if !cur.is_empty() {
                words.push(cur.clone());
                cur.clear();
            }
            continue;
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

fn top2_margin_logp(row: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &value in row {
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    if best.is_finite() && second.is_finite() {
        best - second
    } else {
        0.0
    }
}

fn build_confidence_stats(
    emission_lp_accum: &[f32],
    emission_margin_accum: &[f32],
    coverage_frame_count: usize,
) -> WordConfidenceStats {
    if emission_lp_accum.is_empty() {
        return WordConfidenceStats {
            coverage_frame_count: coverage_frame_count as u32,
            ..WordConfidenceStats::default()
        };
    }

    let mut sorted = emission_lp_accum.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean_logp = emission_lp_accum.iter().sum::<f32>() / emission_lp_accum.len() as f32;
    let min_logp = sorted[0];
    let p10_logp = percentile_sorted(&sorted, 0.10);
    let mean_margin = if emission_margin_accum.is_empty() {
        None
    } else {
        Some(emission_margin_accum.iter().sum::<f32>() / emission_margin_accum.len() as f32)
    };
    let geo_mean_prob = Some(((mean_logp as f64).exp().max(f32::MIN_POSITIVE as f64)) as f32);

    WordConfidenceStats {
        mean_logp: Some(mean_logp),
        geo_mean_prob,
        quality_confidence: None,
        calibrated_confidence: None,
        min_logp: Some(min_logp),
        p10_logp: Some(p10_logp),
        mean_margin,
        coverage_frame_count: coverage_frame_count as u32,
        boundary_confidence: None,
    }
}

fn percentile_sorted(sorted_values: &[f32], percentile: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }

    let clamped = percentile.clamp(0.0, 1.0);
    let max_index = (sorted_values.len() - 1) as f32;
    let rank = clamped * max_index;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = rank - lower as f32;
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    }
}
