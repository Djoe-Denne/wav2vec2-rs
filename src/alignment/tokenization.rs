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
