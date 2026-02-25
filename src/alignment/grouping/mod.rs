use crate::types::{WordConfidenceStats, WordTiming};

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
    let raw = path_to_words::collect(
        path,
        tokens,
        chars,
        expected_words,
        log_probs,
        blank_id,
        word_sep_id,
        stride_ms,
    );

    if raw.is_empty() {
        return Vec::new();
    }

    let first_frame = path.first().map(|&(_, f)| f).unwrap_or(0);
    let last_frame = path.last().map(|&(_, f)| f).unwrap_or(0);
    let candidates = blank_expansion::ExpansionPolicy::ALL
        .into_iter()
        .map(|policy| {
            (
                policy,
                blank_expansion::expand_with_policy(raw.clone(), first_frame, last_frame, policy),
            )
        })
        .collect::<Vec<_>>();

    let selected = candidate_selector::select_best(&raw, candidates, log_probs, blank_id);
    let (selected_policy, score, expanded) = if let Some(chosen) = selected {
        (chosen.policy, Some(chosen.score), chosen.words)
    } else {
        (
            blank_expansion::ExpansionPolicy::Balanced,
            None,
            blank_expansion::expand(raw, first_frame, last_frame),
        )
    };

    if let Some(score) = score {
        tracing::debug!(
            selected_policy = selected_policy.as_str(),
            score_total = format!("{:.3}", score.total_score),
            score_confidence = format!("{:.3}", score.confidence_term),
            score_boundary_confidence = format!("{:.3}", score.boundary_confidence_term),
            score_boundary_non_blank = format!("{:.3}", score.boundary_non_blank_term),
            score_boundary_margin = format!("{:.3}", score.boundary_margin_term),
            score_boundary_penalty = format!("{:.3}", score.boundary_shift_penalty),
            score_pause_penalty = format!("{:.3}", score.pause_penalty),
            "grouping: selected expansion policy"
        );
    }

    expanded
        .into_iter()
        .map(|w| {
            // Timing contract: [start_ms, end_ms), start inclusive and end exclusive.
            let start_ms = (w.start_frame as f64 * stride_ms) as u64;
            let end_ms = ((w.end_frame + 1) as f64 * stride_ms) as u64;
            tracing::debug!(
                word = w.word.as_str(),
                start_frame = w.start_frame,
                end_frame = w.end_frame,
                start_ms,
                end_ms,
                "grouping: final word boundary (after blank expansion)"
            );
            WordTiming {
                word: w.word,
                start_ms,
                end_ms,
                confidence: w.confidence,
                confidence_stats: w.confidence_stats,
            }
        })
        .collect()
}
