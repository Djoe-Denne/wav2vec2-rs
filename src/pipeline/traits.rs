use std::collections::HashMap;

use crate::error::AlignmentError;
use crate::types::{TokenSequence, WordTiming};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeKind {
    #[default]
    Candle,
    Onnx,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeInferenceOutput {
    pub log_probs: Vec<Vec<f32>>,
    pub num_frames_t: usize,
    pub vocab_size: usize,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfiledRuntimeInferenceOutput {
    pub output: RuntimeInferenceOutput,
    pub forward_ms: f64,
    pub post_ms: f64,
}

pub trait RuntimeBackend: Send + Sync {
    fn infer(&self, normalized_audio: &[f32]) -> Result<RuntimeInferenceOutput, AlignmentError>;

    fn infer_profiled(
        &self,
        normalized_audio: &[f32],
    ) -> Result<ProfiledRuntimeInferenceOutput, AlignmentError> {
        let output = self.infer(normalized_audio)?;
        Ok(ProfiledRuntimeInferenceOutput {
            output,
            forward_ms: 0.0,
            post_ms: 0.0,
        })
    }

    fn synchronize(&self, _context: &'static str) -> Result<(), AlignmentError> {
        Ok(())
    }

    fn device_label(&self) -> String;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfiledWordGrouping {
    pub words: Vec<WordTiming>,
    pub conf_ms: f64,
}

pub trait Tokenizer: Send + Sync {
    fn tokenize(
        &self,
        transcript: &str,
        vocab: &HashMap<char, usize>,
        blank_id: usize,
        word_sep_id: usize,
    ) -> TokenSequence;
}

pub trait SequenceAligner: Send + Sync {
    fn align_path(
        &self,
        log_probs: &[Vec<f32>],
        tokens: &[usize],
    ) -> Result<Vec<(usize, usize)>, AlignmentError>;
}

pub trait WordGrouper: Send + Sync {
    fn group_words(
        &self,
        path: &[(usize, usize)],
        token_sequence: &TokenSequence,
        log_probs: &[Vec<f32>],
        blank_id: usize,
        word_sep_id: usize,
        stride_ms: f64,
    ) -> Vec<WordTiming>;

    fn group_words_profiled(
        &self,
        path: &[(usize, usize)],
        token_sequence: &TokenSequence,
        log_probs: &[Vec<f32>],
        blank_id: usize,
        word_sep_id: usize,
        stride_ms: f64,
    ) -> ProfiledWordGrouping {
        ProfiledWordGrouping {
            words: self.group_words(
                path,
                token_sequence,
                log_probs,
                blank_id,
                word_sep_id,
                stride_ms,
            ),
            conf_ms: 0.0,
        }
    }
}
