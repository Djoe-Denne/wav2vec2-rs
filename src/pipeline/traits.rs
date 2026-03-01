use std::collections::HashMap;

use crate::error::AlignmentError;
use crate::types::{TokenSequence, WordTiming};

/// Output from the forward pass — either on host or still on GPU.
///
/// Used to support zero-copy CUDA Viterbi: when log_probs stay on device,
/// we avoid the post_ms GPU→CPU copy and run Viterbi directly on GPU.
#[derive(Debug)]
pub enum ForwardOutput {
    /// Log probs on host: [T][V] — used by CPU/wgpu Viterbi.
    Host(RuntimeInferenceOutput),

    /// Log probs still on CUDA device — used by cuda-dp zero-copy.
    #[cfg(feature = "cuda-dp")]
    CudaDevice(crate::pipeline::cuda_forward::CudaLogProbsBuffer),
}

impl ForwardOutput {
    /// Returns (num_frames_t, vocab_size, dtype) for validation and profiling.
    pub fn metadata(&self) -> (usize, usize, String) {
        match self {
            ForwardOutput::Host(o) => (o.num_frames_t, o.vocab_size, o.dtype.clone()),
            #[cfg(feature = "cuda-dp")]
            ForwardOutput::CudaDevice(buf) => (buf.t_len, buf.v_len, "f32".to_string()),
        }
    }

    /// Obtain host log_probs for grouping. For Host, returns immediately.
    /// For CudaDevice, copies from GPU and applies log_softmax if needed.
    pub fn into_runtime_inference_output(self) -> Result<RuntimeInferenceOutput, AlignmentError> {
        match self {
            ForwardOutput::Host(o) => Ok(o),
            #[cfg(feature = "cuda-dp")]
            ForwardOutput::CudaDevice(buf) => buf.into_runtime_inference_output(),
        }
    }
}

/// Profiled forward output with timing.
#[derive(Debug)]
pub struct ProfiledForwardOutput {
    pub forward_output: ForwardOutput,
    pub forward_ms: f64,
    pub post_ms: f64,
}

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
    fn infer(&self, normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError>;

    fn infer_profiled(
        &self,
        normalized_audio: &[f32],
    ) -> Result<ProfiledForwardOutput, AlignmentError> {
        let forward_output = self.infer(normalized_audio)?;
        Ok(ProfiledForwardOutput {
            forward_output,
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
    /// Time spent walking the Viterbi path and collecting raw word boundaries.
    pub collect_ms: f64,
    /// Time spent cloning candidates, expanding blanks, and selecting the best policy.
    pub expand_select_ms: f64,
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
            collect_ms: 0.0,
            expand_select_ms: 0.0,
        }
    }
}
