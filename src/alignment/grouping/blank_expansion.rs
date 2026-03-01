use super::RawWord;

/// Maximum number of blank frames to absorb per word boundary during
/// expansion. Gaps shorter than 2× this value are split at the midpoint
/// (phonetic transitions). Longer gaps are capped so that the interior
/// silence stays unattributed — matching MFA's explicit silence intervals.
#[allow(dead_code)]
pub(super) const MAX_EXPANSION_FRAMES: usize = 10;
#[allow(dead_code)]
const SHORT_GAP_SPLIT_MAX_FRAMES: usize = 2 * MAX_EXPANSION_FRAMES;
const BALANCED_MAX_LEFT_EXPANSION_FRAMES: usize = 12;
const BALANCED_MAX_RIGHT_PULLBACK_FRAMES: usize = 6;
const BALANCED_MIN_INTERIOR_SILENCE_FRAMES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExpansionPolicy {
    Balanced,
    ConservativeStart,
    AggressiveTail,
}

impl ExpansionPolicy {
    pub(super) const ALL: [ExpansionPolicy; 3] = [
        ExpansionPolicy::Balanced,
        ExpansionPolicy::ConservativeStart,
        ExpansionPolicy::AggressiveTail,
    ];

    pub(super) fn as_str(self) -> &'static str {
        match self {
            ExpansionPolicy::Balanced => "balanced",
            ExpansionPolicy::ConservativeStart => "conservative_start",
            ExpansionPolicy::AggressiveTail => "aggressive_tail",
        }
    }
}

#[derive(Clone, Copy)]
struct PolicyConfig {
    max_left_expansion_frames: usize,
    max_right_pullback_frames: usize,
    min_interior_silence_frames: usize,
}

fn config_for(policy: ExpansionPolicy) -> PolicyConfig {
    match policy {
        ExpansionPolicy::Balanced => PolicyConfig {
            max_left_expansion_frames: BALANCED_MAX_LEFT_EXPANSION_FRAMES,
            max_right_pullback_frames: BALANCED_MAX_RIGHT_PULLBACK_FRAMES,
            min_interior_silence_frames: BALANCED_MIN_INTERIOR_SILENCE_FRAMES,
        },
        ExpansionPolicy::ConservativeStart => PolicyConfig {
            max_left_expansion_frames: 10,
            max_right_pullback_frames: 2,
            min_interior_silence_frames: 6,
        },
        ExpansionPolicy::AggressiveTail => PolicyConfig {
            max_left_expansion_frames: 16,
            max_right_pullback_frames: 4,
            min_interior_silence_frames: 2,
        },
    }
}

/// Phase 2: Expand word boundaries to include adjacent blank/separator frames.
///
/// CTC Viterbi assigns many frames to the blank token even when acoustic
/// energy is present, because blank often has the highest log-probability
/// during transitions. MFA-style aligners attribute those frames to the
/// neighboring word instead.
///
/// Short gaps (≤ 2 × `MAX_EXPANSION_FRAMES`) are phonetic transitions and
/// are split at the midpoint.
///
/// Longer gaps contain genuine silence and are handled asymmetrically:
/// - previous word can absorb more (recover release tails),
/// - next word is pulled back less aggressively (avoid starting too early
///   after long pauses),
/// - and a minimum interior silence region is preserved.
///
/// Leading and trailing silence are NOT attributed to words. The first
/// word's start is left for onset detection to handle.
pub(super) fn expand(words: Vec<RawWord>, _first_frame: usize, _last_frame: usize) -> Vec<RawWord> {
    expand_with_policy(words, _first_frame, _last_frame, ExpansionPolicy::Balanced)
}

#[allow(clippy::needless_range_loop)]
pub(super) fn expand_with_policy(
    mut words: Vec<RawWord>,
    _first_frame: usize,
    _last_frame: usize,
    policy: ExpansionPolicy,
) -> Vec<RawWord> {
    if words.is_empty() {
        return words;
    }
    let config = config_for(policy);

    for i in 0..words.len().saturating_sub(1) {
        let prev_end = words[i].end_frame;
        let next_start = words[i + 1].start_frame;
        if next_start <= prev_end + 1 {
            continue;
        }
        let gap = next_start - prev_end - 1;
        let min_silence = config.min_interior_silence_frames.min(gap);
        let absorb_budget = gap.saturating_sub(min_silence);
        let left_take = absorb_budget.min(config.max_left_expansion_frames);
        let right_take = absorb_budget
            .saturating_sub(left_take)
            .min(config.max_right_pullback_frames);
        words[i].end_frame = prev_end + left_take;
        words[i + 1].start_frame = next_start - right_take;
    }

    words
}