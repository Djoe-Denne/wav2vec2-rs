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

#[cfg(test)]
mod tests {
    use crate::types::TokenSequence;

    use super::*;

    #[test]
    fn forward_output_host_metadata() {
        let out = RuntimeInferenceOutput {
            log_probs: vec![vec![0.0f32; 32]; 10],
            num_frames_t: 10,
            vocab_size: 32,
            dtype: "f32".to_string(),
        };
        let host = ForwardOutput::Host(out);
        let (t, v, dtype) = host.metadata();
        assert_eq!(t, 10);
        assert_eq!(v, 32);
        assert_eq!(dtype, "f32");
    }

    #[test]
    fn forward_output_host_into_runtime_inference_output() {
        let out = RuntimeInferenceOutput {
            log_probs: vec![vec![0.0f32; 8]; 5],
            num_frames_t: 5,
            vocab_size: 8,
            dtype: "f32".to_string(),
        };
        let host = ForwardOutput::Host(out.clone());
        let result = host.into_runtime_inference_output().unwrap();
        assert_eq!(result.log_probs, out.log_probs);
        assert_eq!(result.num_frames_t, out.num_frames_t);
        assert_eq!(result.vocab_size, out.vocab_size);
    }

    struct DummyWordGrouper;

    impl WordGrouper for DummyWordGrouper {
        fn group_words(
            &self,
            _path: &[(usize, usize)],
            _token_sequence: &TokenSequence,
            _log_probs: &[Vec<f32>],
            _blank_id: usize,
            _word_sep_id: usize,
            _stride_ms: f64,
        ) -> Vec<WordTiming> {
            vec![]
        }
    }

    #[test]
    fn word_grouper_default_group_words_profiled() {
        let grouper = DummyWordGrouper;
        let path: Vec<(usize, usize)> = vec![];
        let token_sequence = TokenSequence {
            tokens: vec![],
            chars: vec![],
            normalized_words: vec![],
        };
        let log_probs: Vec<Vec<f32>> = vec![];
        let profiled = grouper.group_words_profiled(&path, &token_sequence, &log_probs, 0, 0, 20.0);
        assert!(profiled.words.is_empty());
        assert_eq!(profiled.conf_ms, 0.0);
        assert_eq!(profiled.collect_ms, 0.0);
        assert_eq!(profiled.expand_select_ms, 0.0);
    }

    struct MockRuntimeBackend;

    impl RuntimeBackend for MockRuntimeBackend {
        fn infer(&self, _normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError> {
            Ok(ForwardOutput::Host(RuntimeInferenceOutput {
                log_probs: vec![vec![0.0f32; 4]; 2],
                num_frames_t: 2,
                vocab_size: 4,
                dtype: "f32".to_string(),
            }))
        }

        fn device_label(&self) -> String {
            "mock".to_string()
        }
    }

    #[test]
    fn runtime_backend_default_infer_profiled() {
        let backend = MockRuntimeBackend;
        let result = backend.infer_profiled(&[0.0f32; 100]).unwrap();
        assert_eq!(result.forward_ms, 0.0);
        assert_eq!(result.post_ms, 0.0);
        let (t, v, _) = result.forward_output.metadata();
        assert_eq!(t, 2);
        assert_eq!(v, 4);
    }

    #[test]
    fn runtime_backend_default_synchronize() {
        let backend = MockRuntimeBackend;
        let result = backend.synchronize("test");
        assert!(result.is_ok());
    }
}
