use std::collections::HashMap;

use crate::alignment::grouping::group_into_words;
use crate::alignment::tokenization::build_token_sequence_case_aware;
use crate::alignment::viterbi::forced_align_viterbi;
use crate::error::AlignmentError;
use crate::pipeline::traits::{SequenceAligner, Tokenizer, WordGrouper};
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
}
