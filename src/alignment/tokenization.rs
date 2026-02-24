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
    for (wi, word) in cleaned.split_whitespace().enumerate() {
        if wi > 0 {
            tokens.push(word_sep_id);
            chars.push(Some('|'));
            tokens.push(blank_id);
            chars.push(None);
        }
        for c in word.chars() {
            if let Some(&id) = vocab.get(&c) {
                tokens.push(id);
                chars.push(Some(c));
                tokens.push(blank_id);
                chars.push(None);
            }
        }
    }
    TokenSequence { tokens, chars }
}
