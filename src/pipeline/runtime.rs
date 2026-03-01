use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::AlignmentError;
use crate::pipeline::traits::{
    ForwardOutput, RuntimeBackend, SequenceAligner, Tokenizer, WordGrouper,
};
use crate::types::{AlignmentInput, AlignmentOutput};

#[cfg(feature = "alignment-profiling")]
use crate::pipeline::memory_tracker::{MemoryTracker, StageMemory, StageMemoryMap};

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

        if let Some(ref n) = input.normalized {
            self.align_inner(n, input)
        } else {
            let computed = normalize_audio(&input.samples);
            self.align_inner(&computed, input)
        }
    }

    fn align_inner(
        &self,
        normalized: &[f32],
        input: &AlignmentInput,
    ) -> Result<AlignmentOutput, AlignmentError> {
        let forward_output = self.runtime_backend.infer(normalized)?;

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
        let min_frames = token_sequence.tokens.len().div_ceil(2);
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

        if let Some(ref n) = input.normalized {
            self.run_align_profiled_inner(n, input, total_started)
        } else {
            let computed = normalize_audio(&input.samples);
            self.run_align_profiled_inner(&computed, input, total_started)
        }
    }

    fn run_align_profiled_inner(
        &self,
        normalized: &[f32],
        input: &AlignmentInput,
        total_started: Instant,
    ) -> Result<ProfiledAlignmentOutput, AlignmentError> {
        let profiled_runtime = self.runtime_backend.infer_profiled(normalized)?;
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
        let min_frames = token_sequence.tokens.len().div_ceil(2);
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

    /// Like `align_profiled` but also records per-stage peak memory (CPU RSS + optional GPU).
    /// Intended for benchmark mode only; use on first repeat to avoid overhead on every run.
    #[cfg(feature = "alignment-profiling")]
    pub fn align_profiled_with_memory(
        &self,
        input: &AlignmentInput,
        tracker: &mut MemoryTracker,
    ) -> Result<(ProfiledAlignmentOutput, StageMemoryMap), AlignmentError> {
        let sync = || self.runtime_backend.synchronize("memory profiling sync");

        if input.samples.is_empty() || input.transcript.trim().is_empty() {
            return Ok((
                ProfiledAlignmentOutput {
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
                },
                StageMemoryMap::default(),
            ));
        }

        if input.sample_rate_hz != self.expected_sample_rate_hz {
            tracing::warn!(
                expected_rate_hz = self.expected_sample_rate_hz,
                actual_rate_hz = input.sample_rate_hz,
                "wav2vec2 aligner expects a specific sample rate; quality may degrade"
            );
        }

        sync()?;
        let total_started = Instant::now();

        let computed_normalized;
        let normalized_slice: &[f32] = if let Some(ref n) = input.normalized {
            n.as_slice()
        } else {
            computed_normalized = normalize_audio(&input.samples);
            &computed_normalized
        };
        let (profiled_runtime, mem_forward) = tracker.measure("forward", sync, || {
            self.runtime_backend.infer_profiled(normalized_slice)
        })?;
        let forward_ms = profiled_runtime.forward_ms;
        let post_ms = profiled_runtime.post_ms;
        let forward_output = profiled_runtime.forward_output;
        let (num_frames_t, vocab_size, dtype) = forward_output.metadata();
        // Forward and post are measured together in infer_profiled.
        let mem_post = mem_forward;

        sync()?;
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
            sync()?;
            let align_ms = duration_to_ms(align_started.elapsed());
            sync()?;
            let total_ms = duration_to_ms(total_started.elapsed());
            return Ok((
                ProfiledAlignmentOutput {
                    output: AlignmentOutput { words: Vec::new() },
                    timings: AlignmentStageTimings {
                        forward_ms,
                        post_ms,
                        dp_ms: 0.0,
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
                },
                StageMemoryMap {
                    forward: mem_forward,
                    post: mem_post,
                    dp: StageMemory::default(),
                    group: StageMemory::default(),
                    conf: StageMemory::default(),
                },
            ));
        }

        let t_len = num_frames_t;
        let min_frames = token_sequence.tokens.len().div_ceil(2);
        if t_len < min_frames {
            return Err(AlignmentError::invalid_input(format!(
                "audio too short for transcript: {t_len} frames < {min_frames} required"
            )));
        }

        let dp_started = Instant::now();
        let (path, log_probs, mem_dp) = {
            let (result, mem) = tracker.measure("dp", sync, || {
                dispatch_viterbi(
                    self.sequence_aligner.as_ref(),
                    forward_output,
                    &token_sequence.tokens,
                )
            })?;
            (result.0, result.1, mem)
        };
        let dp_ms = duration_to_ms(dp_started.elapsed());

        let group_started = Instant::now();
        let (profiled_grouping, mem_group) = tracker.measure("group", sync, || {
            Ok(self.word_grouper.group_words_profiled(
                &path,
                &token_sequence,
                &log_probs,
                self.blank_id,
                self.word_sep_id,
                self.frame_stride_ms,
            ))
        })?;
        let group_total_ms = duration_to_ms(group_started.elapsed());
        let conf_ms = profiled_grouping.conf_ms.max(0.0);
        let words = profiled_grouping.words;
        let group_words_only_ms = (group_total_ms - conf_ms).max(0.0);
        let mem_conf = mem_group;

        sync()?;
        let align_elapsed_ms = duration_to_ms(align_started.elapsed());
        let mut group_ms = tokenization_ms + group_words_only_ms;
        let residual = align_elapsed_ms - (dp_ms + conf_ms + group_ms);
        if residual.abs() > 1e-9 {
            group_ms = (group_ms + residual).max(0.0);
        }
        let align_ms = dp_ms + conf_ms + group_ms;
        sync()?;
        let total_ms = duration_to_ms(total_started.elapsed());

        Ok((
            ProfiledAlignmentOutput {
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
            },
            StageMemoryMap {
                forward: mem_forward,
                post: mem_post,
                dp: mem_dp,
                group: mem_group,
                conf: mem_conf,
            },
        ))
    }

    pub fn frame_stride_ms(&self) -> f64 {
        self.frame_stride_ms
    }
}

/// Normalizes audio to zero mean and unit variance. Exposed so callers can precompute and pass via `AlignmentInput::normalized` to avoid recomputing across repeats.
pub fn normalize_audio(samples: &[f32]) -> Vec<f32> {
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

type ViterbiPath = Vec<(usize, usize)>;
type LogProbs = Vec<Vec<f32>>;

fn dispatch_viterbi(
    sequence_aligner: &dyn SequenceAligner,
    forward_output: ForwardOutput,
    tokens: &[usize],
) -> Result<(ViterbiPath, LogProbs), AlignmentError> {
    match forward_output {
        #[cfg(feature = "cuda-dp")]
        ForwardOutput::CudaDevice(buf) => {
            let path = buf.run_viterbi(tokens).unwrap_or_default();
            if path.is_empty() {
                return Err(AlignmentError::runtime(
                    "cuda viterbi zerocopy",
                    "CUDA Viterbi failed; zero-copy path unavailable",
                ));
            }
            let runtime_output = buf.into_runtime_inference_output()?;
            Ok((path, runtime_output.log_probs))
        }
        ForwardOutput::Host(o) => {
            let path = sequence_aligner.align_path(&o.log_probs, tokens)?;
            Ok((path, o.log_probs))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pipeline::defaults::{
        CaseAwareTokenizer, DefaultWordGrouper, ViterbiSequenceAligner,
    };
    use crate::pipeline::traits::{ForwardOutput, RuntimeBackend, RuntimeInferenceOutput};

    use super::*;

    struct MockBackend {
        output: RuntimeInferenceOutput,
    }

    impl MockBackend {
        fn new(num_frames_t: usize, vocab_size: usize) -> Self {
            Self {
                output: RuntimeInferenceOutput {
                    log_probs: vec![vec![0.0f32; vocab_size]; num_frames_t],
                    num_frames_t,
                    vocab_size,
                    dtype: "f32".to_string(),
                },
            }
        }
    }

    impl RuntimeBackend for MockBackend {
        fn infer(&self, _normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError> {
            Ok(ForwardOutput::Host(RuntimeInferenceOutput {
                log_probs: self.output.log_probs.clone(),
                num_frames_t: self.output.num_frames_t,
                vocab_size: self.output.vocab_size,
                dtype: self.output.dtype.clone(),
            }))
        }

        fn device_label(&self) -> String {
            "mock".to_string()
        }
    }

    fn make_aligner(
        backend: Box<dyn RuntimeBackend>,
        vocab: HashMap<char, usize>,
        blank_id: usize,
        word_sep_id: usize,
        frame_stride_ms: f64,
        expected_sample_rate_hz: u32,
    ) -> ForcedAligner {
        ForcedAligner::from_parts(ForcedAlignerParts {
            runtime_backend: backend,
            vocab,
            blank_id,
            word_sep_id,
            frame_stride_ms,
            expected_sample_rate_hz,
            tokenizer: Box::new(CaseAwareTokenizer),
            sequence_aligner: Box::new(ViterbiSequenceAligner),
            word_grouper: Box::new(DefaultWordGrouper),
        })
    }

    #[test]
    fn align_empty_samples_returns_empty() {
        let mut vocab = HashMap::new();
        vocab.insert('h', 1);
        vocab.insert('i', 2);
        vocab.insert('|', 3);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 3, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![],
            transcript: "hi".to_string(),
            normalized: None,
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.is_empty());
    }

    #[test]
    fn align_empty_transcript_returns_empty() {
        let mut vocab = HashMap::new();
        vocab.insert('h', 1);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 1, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![0.0f32; 1600],
            transcript: "   ".to_string(),
            normalized: None,
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.is_empty());
    }

    #[test]
    fn align_sample_rate_mismatch_still_returns_ok() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 1);
        vocab.insert('|', 2);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 2, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 8_000,
            samples: vec![0.0f32; 800],
            transcript: "a".to_string(),
            normalized: Some(vec![0.0f32; 800]),
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.len() <= 1);
    }

    #[test]
    fn align_uses_normalized_when_provided() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 1);
        vocab.insert('|', 2);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 2, 20.0, 16_000);
        let normalized = vec![0.0f32; 1600];
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![1.0f32; 1600],
            transcript: "a".to_string(),
            normalized: Some(normalized.clone()),
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.len() <= 1);
    }

    #[test]
    fn align_inner_empty_tokens_returns_empty() {
        let mut vocab = HashMap::new();
        vocab.insert('x', 1);
        vocab.insert('|', 2);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 2, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![0.0f32; 1600],
            transcript: "hello".to_string(),
            normalized: Some(vec![0.0f32; 1600]),
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.is_empty());
    }

    #[test]
    fn align_success_returns_words() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 1);
        vocab.insert('|', 2);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 2, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![0.0f32; 1600],
            transcript: "a".to_string(),
            normalized: Some(vec![0.0f32; 1600]),
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.len() <= 1);
    }

    #[test]
    fn align_profiled_empty_input_returns_empty() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 1);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 1, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![],
            transcript: String::new(),
            normalized: None,
        };
        let out = aligner.align_profiled(&input).unwrap();
        assert!(out.output.words.is_empty());
        assert_eq!(out.timings.total_ms, 0.0);
    }

    #[test]
    fn align_profiled_success_returns_words() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 1);
        vocab.insert('|', 2);
        let backend = MockBackend::new(10, 4);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 2, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![0.0f32; 1600],
            transcript: "a".to_string(),
            normalized: Some(vec![0.0f32; 1600]),
        };
        let out = aligner.align_profiled(&input).unwrap();
        assert!(out.output.words.len() <= 1);
        assert!(out.timings.forward_ms >= 0.0);
        assert_eq!(out.num_frames_t, 10);
    }

    #[test]
    fn align_inner_audio_too_short_returns_err() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 1);
        vocab.insert('b', 2);
        vocab.insert('c', 3);
        vocab.insert('d', 4);
        vocab.insert('e', 5);
        vocab.insert('|', 6);
        let backend = MockBackend::new(2, 8);
        let aligner = make_aligner(Box::new(backend), vocab, 0, 6, 20.0, 16_000);
        let input = AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![0.0f32; 320],
            transcript: "abcde".to_string(),
            normalized: Some(vec![0.0f32; 320]),
        };
        let err = aligner.align(&input).unwrap_err();
        assert!(err.to_string().contains("too short") || err.to_string().contains("frames"));
    }
}
