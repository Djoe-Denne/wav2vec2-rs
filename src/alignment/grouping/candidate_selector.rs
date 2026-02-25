use super::blank_expansion::ExpansionPolicy;
use super::RawWord;

const WEIGHT_CONFIDENCE_TERM: f64 = 0.75;
const WEIGHT_BOUNDARY_CONFIDENCE: f64 = 3.4;
const WEIGHT_BOUNDARY_MARGIN: f64 = 0.8;
const WEIGHT_BOUNDARY_NON_BLANK: f64 = 2.8;
const WEIGHT_BOUNDARY_SHIFT: f64 = 0.6;
const WEIGHT_PAUSE_PLAUSIBILITY: f64 = 1.3;
const LARGE_GAP_THRESHOLD_FRAMES: isize = 8;
const OVERLAP_PENALTY_PER_FRAME: f64 = 12.0;
const NEAR_COLLAPSE_PENALTY: f64 = 4.0;

#[derive(Debug, Clone, Copy)]
pub(super) struct ScoreBreakdown {
    pub(super) confidence_term: f64,
    pub(super) boundary_confidence_term: f64,
    pub(super) boundary_non_blank_term: f64,
    pub(super) boundary_margin_term: f64,
    pub(super) boundary_shift_penalty: f64,
    pub(super) pause_penalty: f64,
    pub(super) total_score: f64,
}

#[derive(Debug)]
pub(super) struct SelectedCandidate {
    pub(super) policy: ExpansionPolicy,
    pub(super) words: Vec<RawWord>,
    pub(super) score: ScoreBreakdown,
}

pub(super) fn select_best(
    raw_words: &[RawWord],
    candidates: Vec<(ExpansionPolicy, Vec<RawWord>)>,
    log_probs: &[Vec<f32>],
    blank_id: usize,
) -> Option<SelectedCandidate> {
    let mut best: Option<SelectedCandidate> = None;

    for (policy, mut words) in candidates {
        let (score, per_word_boundary_confidence) =
            score_candidate(raw_words, &words, log_probs, blank_id);
        for (word, boundary_confidence) in words
            .iter_mut()
            .zip(per_word_boundary_confidence.into_iter())
        {
            word.confidence_stats.boundary_confidence = boundary_confidence;
        }
        let should_replace = match &best {
            None => true,
            Some(current) if score.total_score > current.score.total_score + 1e-6 => true,
            Some(current)
                if (score.total_score - current.score.total_score).abs() <= 1e-6
                    && policy == ExpansionPolicy::Balanced
                    && current.policy != ExpansionPolicy::Balanced =>
            {
                true
            }
            _ => false,
        };

        if should_replace {
            best = Some(SelectedCandidate {
                policy,
                words,
                score,
            });
        }
    }

    best
}

fn score_candidate(
    raw_words: &[RawWord],
    candidate_words: &[RawWord],
    log_probs: &[Vec<f32>],
    blank_id: usize,
) -> (ScoreBreakdown, Vec<Option<f32>>) {
    if raw_words.is_empty() || raw_words.len() != candidate_words.len() {
        return (
            ScoreBreakdown {
                confidence_term: 0.0,
                boundary_confidence_term: 0.0,
                boundary_non_blank_term: 0.0,
                boundary_margin_term: 0.0,
                boundary_shift_penalty: 1_000_000.0,
                pause_penalty: 1_000_000.0,
                total_score: -2_000_000.0,
            },
            Vec::new(),
        );
    }

    let n = raw_words.len() as f64;
    let confidence_term = {
        let mut sum = 0.0;
        let mut count = 0usize;
        for word in candidate_words {
            if let Some(conf) = word.confidence {
                sum += conf as f64;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    };

    let boundary_evidence =
        compute_boundary_evidence(raw_words, candidate_words, log_probs, blank_id);

    let boundary_shift_penalty = raw_words
        .iter()
        .zip(candidate_words.iter())
        .map(|(raw, cand)| {
            let start_shift = cand.start_frame.abs_diff(raw.start_frame) as f64;
            let end_shift = cand.end_frame.abs_diff(raw.end_frame) as f64;
            let conf_weight = 0.75 + raw.confidence.unwrap_or(0.0) as f64;
            conf_weight * (start_shift + end_shift)
        })
        .sum::<f64>()
        / n;

    let mut pause_penalty = 0.0;
    let mut gap_count = 0usize;
    for i in 0..raw_words.len().saturating_sub(1) {
        let raw_gap = raw_words[i + 1].start_frame as isize - raw_words[i].end_frame as isize - 1;
        let cand_gap =
            candidate_words[i + 1].start_frame as isize - candidate_words[i].end_frame as isize - 1;
        gap_count += 1;

        if cand_gap < 0 {
            pause_penalty += (-cand_gap) as f64 * OVERLAP_PENALTY_PER_FRAME;
        }

        if raw_gap >= LARGE_GAP_THRESHOLD_FRAMES {
            let collapsed = (raw_gap - cand_gap).max(0) as f64;
            pause_penalty += collapsed;
            if cand_gap <= 1 {
                pause_penalty += NEAR_COLLAPSE_PENALTY;
            }
        }
    }

    if gap_count > 0 {
        pause_penalty /= gap_count as f64;
    }

    let total_score = WEIGHT_CONFIDENCE_TERM * confidence_term
        + WEIGHT_BOUNDARY_CONFIDENCE * boundary_evidence.mean_blank_prob
        + WEIGHT_BOUNDARY_MARGIN * boundary_evidence.mean_margin
        - WEIGHT_BOUNDARY_NON_BLANK * boundary_evidence.mean_non_blank_prob
        - WEIGHT_BOUNDARY_SHIFT * boundary_shift_penalty
        - WEIGHT_PAUSE_PLAUSIBILITY * pause_penalty;

    (
        ScoreBreakdown {
            confidence_term,
            boundary_confidence_term: boundary_evidence.mean_blank_prob,
            boundary_non_blank_term: boundary_evidence.mean_non_blank_prob,
            boundary_margin_term: boundary_evidence.mean_margin,
            boundary_shift_penalty,
            pause_penalty,
            total_score,
        },
        boundary_evidence.per_word_blank_prob,
    )
}

#[derive(Default)]
struct BoundaryEvidence {
    mean_blank_prob: f64,
    mean_non_blank_prob: f64,
    mean_margin: f64,
    per_word_blank_prob: Vec<Option<f32>>,
}

fn compute_boundary_evidence(
    raw_words: &[RawWord],
    candidate_words: &[RawWord],
    log_probs: &[Vec<f32>],
    blank_id: usize,
) -> BoundaryEvidence {
    if candidate_words.is_empty() {
        return BoundaryEvidence::default();
    }

    let mut blank_sum = 0.0f64;
    let mut non_blank_sum = 0.0f64;
    let mut margin_sum = 0.0f64;
    let mut count = 0usize;

    let mut per_word_sum = vec![0.0f64; candidate_words.len()];
    let mut per_word_count = vec![0usize; candidate_words.len()];

    for (idx, (raw, cand)) in raw_words.iter().zip(candidate_words.iter()).enumerate() {
        if cand.start_frame < raw.start_frame {
            for frame in cand.start_frame..raw.start_frame {
                if let Some((blank_prob, non_blank_prob, margin)) =
                    frame_boundary_stats(log_probs, frame, blank_id)
                {
                    blank_sum += blank_prob;
                    non_blank_sum += non_blank_prob;
                    margin_sum += margin;
                    count += 1;
                    per_word_sum[idx] += blank_prob;
                    per_word_count[idx] += 1;
                }
            }
        }
        if cand.end_frame > raw.end_frame {
            for frame in (raw.end_frame + 1)..=cand.end_frame {
                if let Some((blank_prob, non_blank_prob, margin)) =
                    frame_boundary_stats(log_probs, frame, blank_id)
                {
                    blank_sum += blank_prob;
                    non_blank_sum += non_blank_prob;
                    margin_sum += margin;
                    count += 1;
                    per_word_sum[idx] += blank_prob;
                    per_word_count[idx] += 1;
                }
            }
        }
    }

    let per_word_blank_prob = per_word_sum
        .into_iter()
        .zip(per_word_count)
        .map(|(sum, c)| if c == 0 { None } else { Some((sum / c as f64) as f32) })
        .collect::<Vec<_>>();

    if count == 0 {
        return BoundaryEvidence {
            mean_blank_prob: 0.0,
            mean_non_blank_prob: 0.0,
            mean_margin: 0.0,
            per_word_blank_prob,
        };
    }

    BoundaryEvidence {
        mean_blank_prob: blank_sum / count as f64,
        mean_non_blank_prob: non_blank_sum / count as f64,
        mean_margin: margin_sum / count as f64,
        per_word_blank_prob,
    }
}

fn frame_boundary_stats(
    log_probs: &[Vec<f32>],
    frame: usize,
    blank_id: usize,
) -> Option<(f64, f64, f64)> {
    let row = log_probs.get(frame)?;
    let &blank_logp = row.get(blank_id)?;

    let mut best_non_blank = f32::NEG_INFINITY;
    for (tid, &value) in row.iter().enumerate() {
        if tid == blank_id {
            continue;
        }
        if value > best_non_blank {
            best_non_blank = value;
        }
    }

    let blank_prob = blank_logp.exp() as f64;
    let non_blank_prob = if best_non_blank.is_finite() {
        best_non_blank.exp() as f64
    } else {
        0.0
    };
    let local_margin = if best_non_blank.is_finite() {
        (blank_logp - best_non_blank) as f64
    } else {
        0.0
    };
    Some((blank_prob, non_blank_prob, local_margin))
}
