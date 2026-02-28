use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::AlignmentError;
use crate::pipeline::traits::{
    ForwardOutput, RuntimeBackend, SequenceAligner, Tokenizer, WordGrouper,
};
use crate::types::{AlignmentInput, AlignmentOutput};

pub struct ForcedAligner {
    runtime_backend: Box<dyn RuntimeBackend>,
    vocab: HashMap<char, usize>,
    blank_id: usize,
    word_sep_id: usize,
    frame_stride_ms: f64,
    expected_sample_rate_hz: u32,
    tokenizer: Box<dyn Tokenizer>,
    sequence_aligner: Box<dyn SequenceAligner>,
    word_grouper: Box<dyn WordGrouper>,
}

pub(crate) struct ForcedAlignerParts {
    pub runtime_backend: Box<dyn RuntimeBackend>,
    pub vocab: HashMap<char, usize>,
    pub blank_id: usize,
    pub word_sep_id: usize,
    pub frame_stride_ms: f64,
    pub expected_sample_rate_hz: u32,
    pub tokenizer: Box<dyn Tokenizer>,
    pub sequence_aligner: Box<dyn SequenceAligner>,
    pub word_grouper: Box<dyn WordGrouper>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignmentStageTimings {
    pub forward_ms: f64,
    pub post_ms: f64,
    pub dp_ms: f64,
    pub group_ms: f64,
    pub conf_ms: f64,
    pub align_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfiledAlignmentOutput {
    pub output: AlignmentOutput,
    pub timings: AlignmentStageTimings,
    pub num_frames_t: usize,
    pub state_len: usize,
    pub ts_product: u64,
    pub vocab_size: usize,
    pub dtype: String,
    pub device: String,
    pub frame_stride_ms: f64,
}

impl ForcedAligner {
    pub(crate) fn from_parts(parts: ForcedAlignerParts) -> Self {
        Self {
            runtime_backend: parts.runtime_backend,
            vocab: parts.vocab,
            blank_id: parts.blank_id,
            word_sep_id: parts.word_sep_id,
            frame_stride_ms: parts.frame_stride_ms,
            expected_sample_rate_hz: parts.expected_sample_rate_hz,
            tokenizer: parts.tokenizer,
            sequence_aligner: parts.sequence_aligner,
            word_grouper: parts.word_grouper,
        }
    }

    pub fn align(&self, input: &AlignmentInput) -> Result<AlignmentOutput, AlignmentError> {
        if input.samples.is_empty() || input.transcript.trim().is_empty() {
            return Ok(AlignmentOutput { words: Vec::new() });
        }

        if input.sample_rate_hz != self.expected_sample_rate_hz {
            tracing::warn!(
                expected_rate_hz = self.expected_sample_rate_hz,
                actual_rate_hz = input.sample_rate_hz,
                "wav2vec2 aligner expects a specific sample rate; quality may degrade"
            );
        }

        let normalized = normalize_audio(&input.samples);
        let forward_output = self.runtime_backend.infer(&normalized)?;

        let token_sequence = self.tokenizer.tokenize(
            &input.transcript,
            &self.vocab,
            self.blank_id,
            self.word_sep_id,
        );

        if token_sequence.tokens.is_empty() {
            return Ok(AlignmentOutput { words: Vec::new() });
        }

        let (t_len, _, _) = forward_output.metadata();
        let min_frames = (token_sequence.tokens.len() + 1) / 2;
        if t_len < min_frames {
            return Err(AlignmentError::invalid_input(format!(
                "audio too short for transcript: {t_len} frames < {min_frames} required"
            )));
        }

        let (path, log_probs) = dispatch_viterbi(
            self.sequence_aligner.as_ref(),
            forward_output,
            &token_sequence.tokens,
        )?;
        let words = self.word_grouper.group_words(
            &path,
            &token_sequence,
            &log_probs,
            self.blank_id,
            self.word_sep_id,
            self.frame_stride_ms,
        );

        Ok(AlignmentOutput { words })
    }

    pub fn align_profiled(
        &self,
        input: &AlignmentInput,
    ) -> Result<ProfiledAlignmentOutput, AlignmentError> {
        if input.samples.is_empty() || input.transcript.trim().is_empty() {
            return Ok(ProfiledAlignmentOutput {
                output: AlignmentOutput { words: Vec::new() },
                timings: AlignmentStageTimings {
                    forward_ms: 0.0,
                    post_ms: 0.0,
                    dp_ms: 0.0,
                    group_ms: 0.0,
                    conf_ms: 0.0,
                    align_ms: 0.0,
                    total_ms: 0.0,
                },
                num_frames_t: 0,
                state_len: 0,
                ts_product: 0,
                vocab_size: self.vocab.len(),
                dtype: "f32".to_string(),
                device: self.runtime_backend.device_label(),
                frame_stride_ms: self.frame_stride_ms,
            });
        }

        if input.sample_rate_hz != self.expected_sample_rate_hz {
            tracing::warn!(
                expected_rate_hz = self.expected_sample_rate_hz,
                actual_rate_hz = input.sample_rate_hz,
                "wav2vec2 aligner expects a specific sample rate; quality may degrade"
            );
        }

        self.runtime_backend
            .synchronize("runtime synchronize before total timing")?;
        let total_started = Instant::now();

        let normalized = normalize_audio(&input.samples);
        let profiled_runtime = self.runtime_backend.infer_profiled(&normalized)?;
        let forward_ms = profiled_runtime.forward_ms;
        let post_ms = profiled_runtime.post_ms;
        let forward_output = profiled_runtime.forward_output;
        let (num_frames_t, vocab_size, dtype) = forward_output.metadata();

        self.runtime_backend
            .synchronize("runtime synchronize before align timing")?;
        let align_started = Instant::now();
        let tokenize_started = Instant::now();
        let token_sequence = self.tokenizer.tokenize(
            &input.transcript,
            &self.vocab,
            self.blank_id,
            self.word_sep_id,
        );
        let tokenization_ms = duration_to_ms(tokenize_started.elapsed());
        let state_len = token_sequence.tokens.len();
        let ts_product = (num_frames_t as u64).saturating_mul(state_len as u64);
        if token_sequence.tokens.is_empty() {
            self.runtime_backend
                .synchronize("runtime synchronize after align timing")?;
            let align_ms = duration_to_ms(align_started.elapsed());
            self.runtime_backend
                .synchronize("runtime synchronize after total timing")?;
            let total_ms = duration_to_ms(total_started.elapsed());
            return Ok(ProfiledAlignmentOutput {
                output: AlignmentOutput { words: Vec::new() },
                timings: AlignmentStageTimings {
                    forward_ms,
                    post_ms,
                    dp_ms: 0.0,
                    // Empty-token path has no DP/confidence work.
                    // Keep stage fields consistent with align_ms.
                    group_ms: align_ms,
                    conf_ms: 0.0,
                    align_ms,
                    total_ms,
                },
                num_frames_t,
                state_len,
                ts_product,
                vocab_size,
                dtype,
                device: self.runtime_backend.device_label(),
                frame_stride_ms: self.frame_stride_ms,
            });
        }

        let t_len = num_frames_t;
        let min_frames = (token_sequence.tokens.len() + 1) / 2;
        if t_len < min_frames {
            return Err(AlignmentError::invalid_input(format!(
                "audio too short for transcript: {t_len} frames < {min_frames} required"
            )));
        }

        let dp_started = Instant::now();
        let (path, log_probs) = dispatch_viterbi(
            self.sequence_aligner.as_ref(),
            forward_output,
            &token_sequence.tokens,
        )?;
        let dp_ms = duration_to_ms(dp_started.elapsed());

        let group_started = Instant::now();
        let profiled_grouping = self.word_grouper.group_words_profiled(
            &path,
            &token_sequence,
            &log_probs,
            self.blank_id,
            self.word_sep_id,
            self.frame_stride_ms,
        );
        let group_total_ms = duration_to_ms(group_started.elapsed());
        let conf_ms = profiled_grouping.conf_ms.max(0.0);
        let words = profiled_grouping.words;
        let group_words_only_ms = (group_total_ms - conf_ms).max(0.0);

        self.runtime_backend
            .synchronize("runtime synchronize after align timing")?;
        let align_elapsed_ms = duration_to_ms(align_started.elapsed());
        let mut group_ms = tokenization_ms + group_words_only_ms;
        let residual = align_elapsed_ms - (dp_ms + conf_ms + group_ms);
        if residual.abs() > 1e-9 {
            group_ms = (group_ms + residual).max(0.0);
        }
        let align_ms = dp_ms + conf_ms + group_ms;
        self.runtime_backend
            .synchronize("runtime synchronize after total timing")?;
        let total_ms = duration_to_ms(total_started.elapsed());

        Ok(ProfiledAlignmentOutput {
            output: AlignmentOutput { words },
            timings: AlignmentStageTimings {
                forward_ms,
                post_ms,
                dp_ms,
                group_ms,
                conf_ms,
                align_ms,
                total_ms,
            },
            num_frames_t,
            state_len,
            ts_product,
            vocab_size,
            dtype,
            device: self.runtime_backend.device_label(),
            frame_stride_ms: self.frame_stride_ms,
        })
    }

    pub fn frame_stride_ms(&self) -> f64 {
        self.frame_stride_ms
    }
}

fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = var.sqrt().max(1e-7);
    samples
        .iter()
        .map(|&x| ((x as f64 - mean) / std) as f32)
        .collect()
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn dispatch_viterbi(
    sequence_aligner: &dyn SequenceAligner,
    forward_output: ForwardOutput,
    tokens: &[usize],
) -> Result<(Vec<(usize, usize)>, Vec<Vec<f32>>), AlignmentError> {
    match forward_output {
        #[cfg(feature = "cuda-dp")]
        ForwardOutput::CudaDevice(buf) => {
            let path = buf
                .run_viterbi(tokens)
                .unwrap_or_default();
            if path.is_empty() {
                return Err(AlignmentError::runtime(
                    "cuda viterbi zerocopy",
                    "CUDA Viterbi failed; zero-copy path unavailable",
                ));
            }
            let runtime_output = buf.to_runtime_inference_output()?;
            Ok((path, runtime_output.log_probs))
        }
        ForwardOutput::Host(o) => {
            let path = sequence_aligner.align_path(&o.log_probs, tokens)?;
            Ok((path, o.log_probs))
        }
    }
}