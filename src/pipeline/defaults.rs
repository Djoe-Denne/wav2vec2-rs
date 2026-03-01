use std::collections::HashMap;

use crate::alignment::grouping::{group_into_words, group_into_words_profiled};
use crate::alignment::tokenization::build_token_sequence_case_aware;
use crate::alignment::viterbi::forced_align_viterbi;
use crate::error::AlignmentError;
use crate::pipeline::traits::{ProfiledWordGrouping, SequenceAligner, Tokenizer, WordGrouper};
use crate::types::{TokenSequence, WordTiming};

pub struct CaseAwareTokenizer;

impl Tokenizer for CaseAwareTokenizer {
    fn tokenize(
        &self,
        transcript: &str,
        vocab: &HashMap<char, usize>,
        blank_id: usize,
        word_sep_id: usize,
    ) -> TokenSequence {
        build_token_sequence_case_aware(transcript, vocab, blank_id, word_sep_id)
    }
}

pub struct ViterbiSequenceAligner;

impl SequenceAligner for ViterbiSequenceAligner {
    fn align_path(
        &self,
        log_probs: &[Vec<f32>],
        tokens: &[usize],
    ) -> Result<Vec<(usize, usize)>, AlignmentError> {
        Ok(forced_align_viterbi(log_probs, tokens))
    }
}

pub struct DefaultWordGrouper;

impl WordGrouper for DefaultWordGrouper {
    fn group_words(
        &self,
        path: &[(usize, usize)],
        token_sequence: &TokenSequence,
        log_probs: &[Vec<f32>],
        blank_id: usize,
        word_sep_id: usize,
        stride_ms: f64,
    ) -> Vec<WordTiming> {
        group_into_words(
            path,
            &token_sequence.tokens,
            &token_sequence.chars,
            &token_sequence.normalized_words,
            log_probs,
            blank_id,
            word_sep_id,
            stride_ms,
        )
    }

    fn group_words_profiled(
        &self,
        path: &[(usize, usize)],
        token_sequence: &TokenSequence,
        log_probs: &[Vec<f32>],
        blank_id: usize,
        word_sep_id: usize,
        stride_ms: f64,
    ) -> ProfiledWordGrouping {
        let profiled = group_into_words_profiled(
            path,
            &token_sequence.tokens,
            &token_sequence.chars,
            &token_sequence.normalized_words,
            log_probs,
            blank_id,
            word_sep_id,
            stride_ms,
        );
        ProfiledWordGrouping {
            words: profiled.words,
            conf_ms: profiled.conf_ms,
            collect_ms: profiled.collect_ms,
            expand_select_ms: profiled.expand_select_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::alignment::grouping::group_into_words;
    use crate::alignment::viterbi::forced_align_viterbi;

    use super::*;

    #[test]
    fn case_aware_tokenizer_tokenize() {
        let mut vocab = HashMap::new();
        vocab.insert('h', 1);
        vocab.insert('e', 2);
        vocab.insert('l', 3);
        vocab.insert('o', 4);
        vocab.insert('|', 5);
        let tokenizer = CaseAwareTokenizer;
        let seq = tokenizer.tokenize("hello", &vocab, 0, 5);
        assert!(!seq.tokens.is_empty());
        assert_eq!(seq.normalized_words.len(), 1);
        assert!(seq.normalized_words[0].eq_ignore_ascii_case("hello"));
    }

    #[test]
    fn viterbi_sequence_aligner_align_path() {
        let aligner = ViterbiSequenceAligner;
        let log_probs = vec![vec![0.0f32, -10.0], vec![-10.0, 0.0f32]];
        let tokens = vec![0, 1];
        let path = aligner.align_path(&log_probs, &tokens).unwrap();
        let expected = forced_align_viterbi(&log_probs, &tokens);
        assert_eq!(path, expected);
    }

    #[test]
    fn default_word_grouper_group_words() {
        let grouper = DefaultWordGrouper;
        let path = vec![(0, 0), (1, 1), (0, 2)];
        let token_sequence = TokenSequence {
            tokens: vec![0, 1, 0],
            chars: vec![None, Some('A'), None],
            normalized_words: vec!["A".to_string()],
        };
        let log_probs = vec![vec![-1.0f32; 4]; 3];
        let words = grouper.group_words(&path, &token_sequence, &log_probs, 0, 2, 20.0);
        let expected = group_into_words(
            &path,
            &token_sequence.tokens,
            &token_sequence.chars,
            &token_sequence.normalized_words,
            &log_probs,
            0,
            2,
            20.0,
        );
        assert_eq!(words.len(), expected.len());
        if !words.is_empty() {
            assert_eq!(words[0].word, expected[0].word);
        }
    }

    #[test]
    fn default_word_grouper_group_words_profiled() {
        let grouper = DefaultWordGrouper;
        let path = vec![(0, 0), (1, 1), (0, 2)];
        let token_sequence = TokenSequence {
            tokens: vec![0, 1, 0],
            chars: vec![None, Some('A'), None],
            normalized_words: vec!["A".to_string()],
        };
        let log_probs = vec![vec![-1.0f32; 4]; 3];
        let profiled = grouper.group_words_profiled(&path, &token_sequence, &log_probs, 0, 2, 20.0);
        let expected_words = group_into_words(
            &path,
            &token_sequence.tokens,
            &token_sequence.chars,
            &token_sequence.normalized_words,
            &log_probs,
            0,
            2,
            20.0,
        );
        assert_eq!(profiled.words.len(), expected_words.len());
    }
}
