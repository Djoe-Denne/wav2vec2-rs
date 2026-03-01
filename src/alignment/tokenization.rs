use std::collections::HashMap;

use crate::types::TokenSequence;

pub fn build_token_sequence_case_aware(
    transcript: &str,
    vocab: &HashMap<char, usize>,
    blank_id: usize,
    word_sep_id: usize,
) -> TokenSequence {
    let mut has_upper = false;
    let mut has_lower = false;
    for c in vocab.keys().copied().filter(|c| c.is_alphabetic()) {
        if c.is_uppercase() {
            has_upper = true;
        }
        if c.is_lowercase() {
            has_lower = true;
        }
    }
    let cleaned = if has_upper && !has_lower {
        transcript.to_uppercase()
    } else {
        transcript.to_lowercase()
    };

    let mut tokens = vec![blank_id];
    let mut chars: Vec<Option<char>> = vec![None];
    let mut normalized_words = Vec::new();

    for word in cleaned.split_whitespace() {
        let mut emitted: Vec<(char, usize)> = Vec::new();
        let mut normalized_word = String::new();
        for c in word.chars() {
            if let Some(&id) = vocab.get(&c) {
                emitted.push((c, id));
                normalized_word.push(c);
            }
        }

        if emitted.is_empty() {
            continue;
        }

        if !normalized_words.is_empty() {
            tokens.push(word_sep_id);
            chars.push(Some('|'));
            tokens.push(blank_id);
            chars.push(None);
        }

        for (c, id) in emitted {
            tokens.push(id);
            chars.push(Some(c));
            tokens.push(blank_id);
            chars.push(None);
        }
        normalized_words.push(normalized_word);
    }

    debug_assert_eq!(
        normalized_words,
        rebuild_words_from_chars(&chars),
        "tokenization normalization contract violated"
    );

    TokenSequence {
        tokens,
        chars,
        normalized_words,
    }
}

fn rebuild_words_from_chars(chars: &[Option<char>]) -> Vec<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const BLANK_ID: usize = 0;
    const WORD_SEP_ID: usize = 99;

    fn vocab_lower() -> HashMap<char, usize> {
        let mut m = HashMap::new();
        m.insert('a', 1);
        m.insert('b', 2);
        m.insert('c', 3);
        m
    }

    fn vocab_upper() -> HashMap<char, usize> {
        let mut m = HashMap::new();
        m.insert('A', 1);
        m.insert('B', 2);
        m.insert('C', 3);
        m
    }

    fn vocab_mixed() -> HashMap<char, usize> {
        let mut m = HashMap::new();
        m.insert('a', 1);
        m.insert('B', 2);
        m.insert('c', 3);
        m
    }

    #[test]
    fn empty_transcript_produces_single_blank() {
        let seq = build_token_sequence_case_aware("", &vocab_lower(), BLANK_ID, WORD_SEP_ID);
        assert_eq!(seq.tokens, vec![BLANK_ID]);
        assert_eq!(seq.chars, vec![None]);
        assert!(seq.normalized_words.is_empty());
    }

    #[test]
    fn single_word_lowercase_vocab() {
        let seq = build_token_sequence_case_aware("AB", &vocab_lower(), BLANK_ID, WORD_SEP_ID);
        assert_eq!(seq.normalized_words, ["ab"]);
        assert_eq!(seq.tokens[0], BLANK_ID);
        assert_eq!(seq.chars[0], None);
        // pattern: blank, A, blank, B, blank
        assert_eq!(seq.tokens.len(), 5);
        assert_eq!(seq.tokens[1], 1);
        assert_eq!(seq.tokens[3], 2);
    }

    #[test]
    fn uppercase_only_vocab_uppercases_transcript() {
        let seq = build_token_sequence_case_aware("a b", &vocab_upper(), BLANK_ID, WORD_SEP_ID);
        assert_eq!(seq.normalized_words, ["A", "B"]);
        assert!(seq.tokens.contains(&WORD_SEP_ID));
    }

    #[test]
    fn lowercase_only_vocab_lowercases_transcript() {
        let seq = build_token_sequence_case_aware("A B", &vocab_lower(), BLANK_ID, WORD_SEP_ID);
        assert_eq!(seq.normalized_words, ["a", "b"]);
    }

    #[test]
    fn mixed_vocab_lowercases_transcript() {
        // Vocab has 'a', 'B', 'c' (mixed case). Transcript is lowercased to "a b c";
        // 'b' is not in vocab so the middle word is skipped.
        let seq = build_token_sequence_case_aware("A B c", &vocab_mixed(), BLANK_ID, WORD_SEP_ID);
        assert_eq!(seq.normalized_words, ["a", "c"]);
    }

    #[test]
    fn multiple_words_have_sep_and_blanks() {
        let seq = build_token_sequence_case_aware("a b c", &vocab_lower(), BLANK_ID, WORD_SEP_ID);
        assert_eq!(seq.normalized_words, ["a", "b", "c"]);
        // First token is blank, then a, blank, sep, blank, b, blank, sep, blank, c, blank
        let sep_count = seq.tokens.iter().filter(|&&t| t == WORD_SEP_ID).count();
        assert_eq!(sep_count, 2);
    }

    #[test]
    fn unknown_chars_skipped() {
        let seq = build_token_sequence_case_aware("aXb", &vocab_lower(), BLANK_ID, WORD_SEP_ID);
        assert_eq!(seq.normalized_words, ["ab"]);
    }

    #[test]
    fn normalized_words_match_chars_contract() {
        let seq = build_token_sequence_case_aware("a b c", &vocab_lower(), BLANK_ID, WORD_SEP_ID);
        let mut words = Vec::new();
        let mut cur = String::new();
        for c in seq.chars.iter().copied().flatten() {
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
        assert_eq!(seq.normalized_words, words);
    }
}
