use crate::types::WordTiming;

/// Reconstruct the expected word list directly from token metadata.
///
/// Why do this here instead of trusting separator states only?
/// - In some difficult utterances the path can touch a separator earlier than
///   we would like acoustically.
/// - Keeping a word-level expectation allows us to gate "flush" on word
///   completion rather than "separator seen".
/// - This makes grouping behavior more robust for models with different
///   calibration/temporal dynamics.
fn expected_words_from_chars(chars: &[Option<char>]) -> Vec<String> {
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

pub fn group_into_words(
    path: &[(usize, usize)],
    tokens: &[usize],
    chars: &[Option<char>],
    log_probs: &[Vec<f32>],
    blank_id: usize,
    word_sep_id: usize,
    stride_ms: f64,
) -> Vec<WordTiming> {
    let mut words = Vec::new();
    let mut cur_word = String::new();
    let mut start_frame: Option<usize> = None;
    let mut end_frame: usize = 0;
    let mut lp_accum = Vec::new();
    let mut prev_state: Option<usize> = None;
    let expected_words = expected_words_from_chars(chars);

    let flush = |w: &mut String,
                 sf: &mut Option<usize>,
                 ef: usize,
                 lps: &mut Vec<f32>,
                 out: &mut Vec<WordTiming>,
                 ms: f64| {
        if w.is_empty() {
            return;
        }
        let conf = if lps.is_empty() {
            0.0
        } else {
            (lps.iter().sum::<f32>() / lps.len() as f32).exp()
        };
        out.push(WordTiming {
            word: w.clone(),
            start_ms: (sf.unwrap_or(0) as f64 * ms) as u64,
            end_ms: ((ef + 1) as f64 * ms) as u64,
            confidence: conf,
        });
        w.clear();
        *sf = None;
        lps.clear();
    };

    for &(s, frame) in path {
        let tid = tokens[s];

        // Blank states carry timing but do not emit lexical content.
        // We still update `prev_state` because dedup logic relies on state
        // transitions, including transitions through blank states.
        if tid == blank_id {
            prev_state = Some(s);
            continue;
        }

        // Separator tokens are natural word boundaries in CTC token streams,
        // but we now guard the flush by checking that the currently accumulated
        // word is actually complete relative to expected tokenized words.
        //
        // This avoids splitting too early when a separator state appears before
        // we've reliably built the intended word text.
        if tid == word_sep_id {
            if !cur_word.is_empty() && !matches_expected_word(&cur_word, &expected_words, words.len()) {
                prev_state = Some(s);
                continue;
            }
            flush(
                &mut cur_word,
                &mut start_frame,
                end_frame,
                &mut lp_accum,
                &mut words,
                stride_ms,
            );
            prev_state = Some(s);
            continue;
        }
        if let Some(c) = chars[s] {
            if start_frame.is_none() {
                start_frame = Some(frame);
            }
            // Even when we don't append a new character (because we stayed in
            // the same token state), we keep extending end_frame so elongated
            // pronunciations can stretch the timing window.
            end_frame = frame;
            if prev_state != Some(s) {
                // CTC can emit the same token over many frames; append only
                // on state entry to avoid duplicated letters.
                cur_word.push(c);
                lp_accum.push(log_probs[frame][tid]);
            }
        }
        prev_state = Some(s);
    }

    // End-of-path fallback:
    // if a trailing word exists, emit it. This keeps behavior predictable
    // for paths that end without an explicit separator token.
    flush(
        &mut cur_word,
        &mut start_frame,
        end_frame,
        &mut lp_accum,
        &mut words,
        stride_ms,
    );
    words
}
