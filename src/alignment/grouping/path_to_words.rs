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

/// Input for one path step (read-only).
struct PathStepInput<'a> {
    s: usize,
    frame: usize,
    tid: usize,
    stride_ms: f64,
    blank_id: usize,
    word_sep_id: usize,
    expected_words: &'a [String],
    chars: &'a [Option<char>],
    log_probs: &'a [Vec<f32>],
}

/// Mutable state updated by process_path_step.
struct PathStepState<'a> {
    cur_word: &'a mut String,
    start_frame: &'a mut Option<usize>,
    end_frame: &'a mut usize,
    emission_lp_accum: &'a mut Vec<f32>,
    emission_margin_accum: &'a mut Vec<f32>,
    coverage_frame_count: &'a mut usize,
    out: &'a mut Vec<RawWord>,
    prev_state: &'a mut Option<usize>,
}

/// Returns true if the rest of the loop iteration should be skipped (caller will set prev_state).
fn process_path_step(input: &PathStepInput<'_>, state: &mut PathStepState<'_>) -> bool {
    let frame_ms = input.frame as f64 * input.stride_ms;

    if input.tid == input.blank_id {
        tracing::debug!(
            frame = input.frame,
            frame_ms = format!("{:.0}", frame_ms),
            state = input.s,
            kind = "blank",
            cur_word = state.cur_word.as_str(),
            "grouping: blank frame"
        );
        *state.prev_state = Some(input.s);
        return true;
    }

    if input.tid == input.word_sep_id {
        tracing::debug!(
            frame = input.frame,
            frame_ms = format!("{:.0}", frame_ms),
            state = input.s,
            kind = "sep",
            cur_word = state.cur_word.as_str(),
            "grouping: separator frame"
        );
        if !state.cur_word.is_empty()
            && !matches_expected_word(state.cur_word, input.expected_words, state.out.len())
        {
            *state.prev_state = Some(input.s);
            return true;
        }
        flush_word(
            state.cur_word,
            state.start_frame,
            *state.end_frame,
            state.emission_lp_accum,
            state.emission_margin_accum,
            state.coverage_frame_count,
            state.out,
        );
        *state.prev_state = Some(input.s);
        return true;
    }

    if let Some(c) = input.chars[input.s] {
        let is_new_state = *state.prev_state != Some(input.s);
        if state.start_frame.is_none() {
            *state.start_frame = Some(input.frame);
        }
        *state.end_frame = input.frame;
        *state.coverage_frame_count += 1;
        if is_new_state {
            state
                .emission_lp_accum
                .push(input.log_probs[input.frame][input.tid]);
            state
                .emission_margin_accum
                .push(top2_margin_logp(&input.log_probs[input.frame]));
            state.cur_word.push(c);
        }
        tracing::debug!(
            frame = input.frame,
            frame_ms = format!("{:.0}", frame_ms),
            state = input.s,
            kind = "char",
            char = %c,
            new_state = is_new_state,
            cur_word = state.cur_word.as_str(),
            start_frame = state.start_frame,
            end_frame = state.end_frame,
            "grouping: char frame"
        );
    }
    false
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

pub(super) struct ProfiledRawWords {
    pub(super) words: Vec<RawWord>,
    #[allow(dead_code)]
    pub(super) conf_ms: f64,
}

/// Phase 1: Walk the Viterbi path and group character frames into words.
/// Boundaries are tight — only character-emitting frames set start/end.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_profiled(
    path: &[(usize, usize)],
    tokens: &[usize],
    chars: &[Option<char>],
    expected_words: &[String],
    log_probs: &[Vec<f32>],
    blank_id: usize,
    word_sep_id: usize,
    stride_ms: f64,
) -> ProfiledRawWords {
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

    // Confidence-related work (top2_margin, build_confidence_stats) is timed
    // as a single batch at the caller level rather than per-item to avoid
    // Instant::now() syscall overhead in the hot loop.
    for &(s, frame) in path {
        let tid = tokens[s];
        let step_input = PathStepInput {
            s,
            frame,
            tid,
            stride_ms,
            blank_id,
            word_sep_id,
            expected_words,
            chars,
            log_probs,
        };
        let mut step_state = PathStepState {
            cur_word: &mut cur_word,
            start_frame: &mut start_frame,
            end_frame: &mut end_frame,
            emission_lp_accum: &mut emission_lp_accum,
            emission_margin_accum: &mut emission_margin_accum,
            coverage_frame_count: &mut coverage_frame_count,
            out: &mut words,
            prev_state: &mut prev_state,
        };
        if process_path_step(&step_input, &mut step_state) {
            continue;
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
    // conf_ms is no longer tracked per-item; the caller batches the measurement.
    ProfiledRawWords {
        words,
        conf_ms: 0.0,
    }
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
