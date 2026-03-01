use super::blank_expansion::{expand, expand_with_policy, ExpansionPolicy};
use super::candidate_selector::select_best;
use super::{group_into_words, RawWord};
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

fn make_uniform_log_probs(frame_count: usize, vocab_size: usize) -> Vec<Vec<f32>> {
    (0..frame_count)
        .map(|_| {
            let mut row = vec![-3.0; vocab_size];
            row[0] = -0.1;
            row
        })
        .collect()
}

#[test]
fn expand_single_word_unchanged() {
    let words = vec![make_raw("HELLO", 10, 20)];
    let result = expand(words, 0, 30);
    assert_eq!(result[0].start_frame, 10);
    assert_eq!(result[0].end_frame, 20);
}

#[test]
fn expand_two_words_splits_gap_at_midpoint() {
    let words = vec![make_raw("A", 10, 20), make_raw("B", 30, 40)];
    let result = expand(words, 0, 50);

    assert_eq!(result[0].start_frame, 10); // not extended to first_frame
    assert_eq!(result[0].end_frame, 25); // midpoint of 20..30
    assert_eq!(result[1].start_frame, 30);
    assert_eq!(result[1].end_frame, 40); // not extended to last_frame
}

#[test]
fn expand_adjacent_words_no_gap() {
    let words = vec![make_raw("A", 5, 10), make_raw("B", 11, 15)];
    let result = expand(words, 0, 20);

    assert_eq!(result[0].start_frame, 5);
    assert_eq!(result[0].end_frame, 10);
    assert_eq!(result[1].start_frame, 11);
    assert_eq!(result[1].end_frame, 15);
}

#[test]
fn expand_three_words() {
    let words = vec![
        make_raw("A", 10, 15),
        make_raw("B", 25, 30),
        make_raw("C", 40, 45),
    ];
    let result = expand(words, 0, 50);

    assert_eq!(result[0].start_frame, 10);
    assert_eq!(result[0].end_frame, 20); // mid(15,25) = 20
    assert_eq!(result[1].start_frame, 25);
    assert_eq!(result[1].end_frame, 35); // mid(30,40) = 35
    assert_eq!(result[2].start_frame, 40);
    assert_eq!(result[2].end_frame, 45);
}

#[test]
fn expand_large_gap_capped() {
    // Gap of 28 frames (10..40) — much larger than 2*MAX_EXPANSION_FRAMES (20).
    // Long gaps use asymmetric expansion: previous word gets more frames,
    // next word gets less pullback to avoid early starts after silence.
    let words = vec![make_raw("A", 5, 10), make_raw("B", 40, 45)];
    let result = expand(words, 0, 50);

    assert_eq!(result[0].start_frame, 5);
    assert_eq!(result[0].end_frame, 22); // 10 + left_take(12)
    assert_eq!(result[1].start_frame, 34); // 40 - right_take(6)
    assert_eq!(result[1].end_frame, 45);
}

#[test]
fn expand_empty_returns_empty() {
    let result = expand(Vec::new(), 0, 50);
    assert!(result.is_empty());
}

#[test]
fn expand_large_gap_preserves_interior_silence() {
    let words = vec![make_raw("A", 5, 10), make_raw("B", 80, 85)];
    let result = expand(words, 0, 100);

    let interior_silence = result[1].start_frame - result[0].end_frame - 1;
    assert!(interior_silence >= 4);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn all_policies_preserve_non_overlapping_boundaries() {
    let words = vec![
        make_raw("A", 5, 10),
        make_raw("B", 40, 45),
        make_raw("C", 80, 90),
    ];

    for policy in ExpansionPolicy::ALL {
        let result = expand_with_policy(words.clone(), 0, 120, policy);
        for i in 0..result.len().saturating_sub(1) {
            assert!(
                result[i].end_frame < result[i + 1].start_frame,
                "policy={} produced overlap at index {}",
                policy.as_str(),
                i
            );
        }
    }
}

#[test]
fn selector_prefers_balanced_when_scores_tie() {
    let raw = vec![make_raw("A", 10, 20), make_raw("B", 21, 30)];
    let candidates = ExpansionPolicy::ALL
        .into_iter()
        .map(|policy| (policy, expand_with_policy(raw.clone(), 0, 40, policy)))
        .collect::<Vec<_>>();
    let log_probs = make_uniform_log_probs(50, 4);

    let selected =
        select_best(&raw, candidates, &log_probs, 0).expect("selector should return a candidate");
    assert_eq!(selected.policy, ExpansionPolicy::Balanced);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn selector_uses_blank_boundary_evidence() {
    let raw = vec![make_raw("A", 5, 10), make_raw("B", 20, 25)];
    let candidate_blank_heavy = vec![make_raw("A", 5, 13), make_raw("B", 20, 25)];
    let candidate_nonblank_heavy = vec![make_raw("A", 5, 10), make_raw("B", 17, 25)];
    let candidates = vec![
        (ExpansionPolicy::Balanced, candidate_blank_heavy),
        (ExpansionPolicy::ConservativeStart, candidate_nonblank_heavy),
    ];

    let mut log_probs = vec![vec![-1.0; 4]; 32];
    for frame in 11..=13 {
        log_probs[frame][0] = -0.05; // high blank probability
        log_probs[frame][1] = -4.0;
        log_probs[frame][2] = -4.0;
        log_probs[frame][3] = -4.0;
    }
    for frame in 17..=19 {
        log_probs[frame][0] = -4.0; // strong non-blank peak
        log_probs[frame][1] = -0.05;
        log_probs[frame][2] = -3.0;
        log_probs[frame][3] = -3.0;
    }

    let selected =
        select_best(&raw, candidates, &log_probs, 0).expect("selector should return a candidate");
    assert_eq!(selected.policy, ExpansionPolicy::Balanced);
    assert!(
        selected.words[0]
            .confidence_stats
            .boundary_confidence
            .is_some(),
        "selected candidate should carry per-word boundary confidence"
    );
}

#[test]
fn group_into_words_basic() {
    // Tokens: blank(0), A(1), blank(0), sep(2), blank(0), B(3), blank(0)
    // chars:  None,     Some('A'), None, Some('|'), None, Some('B'), None
    let tokens = vec![0, 1, 0, 2, 0, 3, 0];
    let chars: Vec<Option<char>> = vec![None, Some('A'), None, Some('|'), None, Some('B'), None];
    let blank_id = 0;
    let word_sep_id = 2;
    let stride_ms = 20.0;

    // Path: 10 frames, each assigned to a state
    // Frames 0-2: blank, 3-4: A, 5: blank, 6: sep, 7: blank, 8-9: B
    let path = vec![
        (0, 0),  // blank
        (0, 1),  // blank
        (0, 2),  // blank
        (1, 3),  // A
        (1, 4),  // A (same state, not new)
        (0, 5),  // blank
        (0, 6),  // blank
        (3, 7),  // sep (token id 2) -- wait, state 3 has token 2 (sep)
        (0, 8),  // blank
        (0, 9),  // blank
        (5, 10), // B
        (5, 11), // B (same state, not new)
    ];

    let num_frames = 12;
    let vocab_size = 4; // blank, A, sep, B
    let log_probs: Vec<Vec<f32>> = (0..num_frames).map(|_| vec![-1.0; vocab_size]).collect();

    let words = group_into_words(
        &path,
        &tokens,
        &chars,
        &["A".to_string(), "B".to_string()],
        &log_probs,
        blank_id,
        word_sep_id,
        stride_ms,
    );

    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word, "A");
    assert_eq!(words[1].word, "B");
    // Char-only: A=[3,4], B=[8,9], gap frames 5,6,7
    // mid(4, 8) = 6, so A=[3,6], B=[10,12]
    // Leading blanks (0-2) and trailing frames stay unused (genuine silence).
    assert_eq!(words[0].start_ms, 3 * 20); // 60
    assert_eq!(words[0].end_ms, (5 + 1) as u64 * 20); // 120
    assert_eq!(words[1].start_ms, 10 * 20); // 200
    assert_eq!(words[1].end_ms, (11 + 1) as u64 * 20); // 220
    assert!(words[0].confidence.is_some());
    assert!(words[0].confidence_stats.geo_mean_prob.is_some());
    assert_eq!(words[0].confidence_stats.coverage_frame_count, 2);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn confidence_is_stable_across_repeated_state_holds() {
    let tokens = vec![0, 1, 0];
    let chars: Vec<Option<char>> = vec![None, Some('A'), None];
    let expected_words = ["A".to_string()];
    let blank_id = 0;
    let word_sep_id = 2;
    let stride_ms = 20.0;

    let short_path = vec![(0, 0), (1, 1), (0, 2)];
    let long_path = vec![(0, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (0, 6)];

    let mut short_log_probs = vec![vec![-5.0; 3]; 3];
    short_log_probs[1][1] = -0.1;
    short_log_probs[1][0] = -3.0;

    let mut long_log_probs = vec![vec![-5.0; 3]; 7];
    long_log_probs[1][1] = -0.1; // emission frame
    long_log_probs[1][0] = -3.0;
    for frame in 2..=5 {
        long_log_probs[frame][1] = -4.5; // repeated hold frames should not drag confidence down
        long_log_probs[frame][0] = -0.2;
    }

    let short_words = group_into_words(
        &short_path,
        &tokens,
        &chars,
        &expected_words,
        &short_log_probs,
        blank_id,
        word_sep_id,
        stride_ms,
    );
    let long_words = group_into_words(
        &long_path,
        &tokens,
        &chars,
        &expected_words,
        &long_log_probs,
        blank_id,
        word_sep_id,
        stride_ms,
    );

    let short_conf = short_words[0]
        .confidence
        .expect("short path should have confidence");
    let long_conf = long_words[0]
        .confidence
        .expect("long path should have confidence");

    assert!(
        (short_conf - long_conf).abs() < 1e-6,
        "confidence should be emission-based and stable across repeated holds"
    );
    assert_eq!(short_words[0].confidence_stats.coverage_frame_count, 1);
    assert_eq!(long_words[0].confidence_stats.coverage_frame_count, 5);
}

#[test]
fn selector_returns_none_for_empty_candidates() {
    let raw = vec![make_raw("A", 10, 20)];
    let log_probs = make_uniform_log_probs(30, 4);
    let selected = select_best(&raw, vec![], &log_probs, 0);
    assert!(selected.is_none());
}

#[test]
fn selector_returns_single_candidate() {
    let raw = vec![make_raw("A", 5, 10), make_raw("B", 25, 30)];
    let candidate = expand_with_policy(raw.clone(), 0, 40, ExpansionPolicy::AggressiveTail);
    let candidates = vec![(ExpansionPolicy::AggressiveTail, candidate)];
    let log_probs = make_uniform_log_probs(50, 4);
    let selected = select_best(&raw, candidates, &log_probs, 0).expect("one candidate");
    assert_eq!(selected.policy, ExpansionPolicy::AggressiveTail);
}

#[test]
fn expand_conservative_start_large_gap_frame_bounds() {
    // ConservativeStart: max_left=10, max_right=2, min_interior_silence=6.
    // Gap 25 (11..35): absorb_budget=19, left_take=10, right_take=2.
    let words = vec![make_raw("A", 5, 10), make_raw("B", 36, 41)];
    let result = expand_with_policy(words, 0, 50, ExpansionPolicy::ConservativeStart);
    assert_eq!(result[0].start_frame, 5);
    assert_eq!(result[0].end_frame, 20); // 10 + 10
    assert_eq!(result[1].start_frame, 34); // 36 - 2
    assert_eq!(result[1].end_frame, 41);
}

#[test]
fn expand_aggressive_tail_large_gap_frame_bounds() {
    // AggressiveTail: max_left=16, max_right=4, min_interior_silence=2.
    // Gap 25 (11..35): absorb_budget=23, left_take=16, right_take=4.
    let words = vec![make_raw("A", 5, 10), make_raw("B", 36, 41)];
    let result = expand_with_policy(words, 0, 50, ExpansionPolicy::AggressiveTail);
    assert_eq!(result[0].start_frame, 5);
    assert_eq!(result[0].end_frame, 26); // 10 + 16
    assert_eq!(result[1].start_frame, 32); // 36 - 4
    assert_eq!(result[1].end_frame, 41);
}
