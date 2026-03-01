use super::RawWord;

/// Maximum number of blank frames to absorb per word boundary during
/// expansion. Gaps shorter than 2× this value are split at the midpoint
/// (phonetic transitions). Longer gaps are capped so that the interior
/// silence stays unattributed — matching MFA's explicit silence intervals.
#[allow(dead_code)]
pub(super) const MAX_EXPANSION_FRAMES: usize = 10;
const BALANCED_MAX_LEFT_EXPANSION_FRAMES: usize = 12;
const BALANCED_MAX_RIGHT_PULLBACK_FRAMES: usize = 6;
const BALANCED_MIN_INTERIOR_SILENCE_FRAMES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // variants used via ALL and in config_for; parent module uses them
pub(super) enum ExpansionPolicy {
    Balanced,
    ConservativeStart,
    AggressiveTail,
}

/// ALL and as_str are used by the parent module (grouping::mod) and tests.
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::{expand, expand_with_policy, ExpansionPolicy};
    use crate::alignment::grouping::RawWord;
    use crate::types::WordConfidenceStats;

    fn make_raw(word: &str, start: usize, end: usize) -> RawWord {
        RawWord {
            word: word.to_string(),
            start_frame: start,
            end_frame: end,
            confidence: Some(1.0),
            confidence_stats: WordConfidenceStats {
                geo_mean_prob: Some(1.0),
                ..WordConfidenceStats::default()
            },
        }
    }

    #[test]
    fn expansion_policy_as_str() {
        assert_eq!(ExpansionPolicy::Balanced.as_str(), "balanced");
        assert_eq!(
            ExpansionPolicy::ConservativeStart.as_str(),
            "conservative_start"
        );
        assert_eq!(ExpansionPolicy::AggressiveTail.as_str(), "aggressive_tail");
    }

    #[test]
    fn expansion_policy_all_has_three_variants() {
        assert_eq!(ExpansionPolicy::ALL.len(), 3);
        assert!(ExpansionPolicy::ALL.iter().all(|p| match p {
            ExpansionPolicy::Balanced
            | ExpansionPolicy::ConservativeStart
            | ExpansionPolicy::AggressiveTail => true,
        }));
    }

    #[test]
    fn expand_empty_returns_empty() {
        let result = expand(Vec::new(), 0, 100);
        assert!(result.is_empty());
    }

    #[test]
    fn expand_with_policy_empty_returns_empty() {
        let result = expand_with_policy(Vec::new(), 0, 100, ExpansionPolicy::Balanced);
        assert!(result.is_empty());
    }

    #[test]
    fn expand_with_policy_adjacent_words_no_gap_unchanged() {
        let words = vec![make_raw("A", 5, 10), make_raw("B", 11, 15)];
        for policy in ExpansionPolicy::ALL {
            let result = expand_with_policy(words.clone(), 0, 20, policy);
            assert_eq!(result[0].start_frame, 5);
            assert_eq!(result[0].end_frame, 10);
            assert_eq!(result[1].start_frame, 11);
            assert_eq!(result[1].end_frame, 15);
        }
    }

    #[test]
    fn expand_with_policy_gap_one_unchanged() {
        // gap = 1: min_silence = min(4, 1) = 1, absorb_budget = 0
        let words = vec![make_raw("A", 5, 10), make_raw("B", 12, 17)];
        let result = expand_with_policy(words, 0, 20, ExpansionPolicy::Balanced);
        assert_eq!(result[0].end_frame, 10);
        assert_eq!(result[1].start_frame, 12);
    }

    #[test]
    fn expand_with_policy_balanced_two_words_splits_gap() {
        let words = vec![make_raw("A", 10, 20), make_raw("B", 30, 40)];
        let result = expand_with_policy(words, 0, 50, ExpansionPolicy::Balanced);
        assert_eq!(result[0].end_frame, 25);
        assert_eq!(result[1].start_frame, 30);
    }

    #[test]
    fn expand_with_policy_conservative_start_caps_left_more_than_right() {
        let words = vec![make_raw("A", 5, 10), make_raw("B", 36, 41)];
        let result = expand_with_policy(words, 0, 50, ExpansionPolicy::ConservativeStart);
        assert_eq!(result[0].end_frame, 20);
        assert_eq!(result[1].start_frame, 34);
    }

    #[test]
    fn expand_with_policy_aggressive_tail_takes_more_left() {
        let words = vec![make_raw("A", 5, 10), make_raw("B", 36, 41)];
        let result = expand_with_policy(words, 0, 50, ExpansionPolicy::AggressiveTail);
        assert_eq!(result[0].end_frame, 26);
        assert_eq!(result[1].start_frame, 32);
    }

    #[test]
    fn expand_single_word_unchanged() {
        let words = vec![make_raw("X", 10, 20)];
        let result = expand_with_policy(words, 0, 30, ExpansionPolicy::Balanced);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_frame, 10);
        assert_eq!(result[0].end_frame, 20);
    }
}
