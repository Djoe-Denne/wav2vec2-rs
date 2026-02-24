use std::collections::HashMap;

use candle_core::{Device, Tensor, D};

use crate::error::AlignmentError;
use crate::model::ctc_model::Wav2Vec2ForCTC;
use crate::pipeline::traits::{SequenceAligner, Tokenizer, WordGrouper};
use crate::types::{AlignmentInput, AlignmentOutput};

const ONSET_MIN_CONSEC_FRAMES: usize = 3;

pub struct ForcedAligner {
    model: Wav2Vec2ForCTC,
    vocab: HashMap<char, usize>,
    blank_id: usize,
    word_sep_id: usize,
    frame_stride_ms: f64,
    device: Device,
    expected_sample_rate_hz: u32,
    tokenizer: Box<dyn Tokenizer>,
    sequence_aligner: Box<dyn SequenceAligner>,
    word_grouper: Box<dyn WordGrouper>,
}

pub(crate) struct ForcedAlignerParts {
    pub model: Wav2Vec2ForCTC,
    pub vocab: HashMap<char, usize>,
    pub blank_id: usize,
    pub word_sep_id: usize,
    pub frame_stride_ms: f64,
    pub device: Device,
    pub expected_sample_rate_hz: u32,
    pub tokenizer: Box<dyn Tokenizer>,
    pub sequence_aligner: Box<dyn SequenceAligner>,
    pub word_grouper: Box<dyn WordGrouper>,
}

impl ForcedAligner {
    pub(crate) fn from_parts(parts: ForcedAlignerParts) -> Self {
        Self {
            model: parts.model,
            vocab: parts.vocab,
            blank_id: parts.blank_id,
            word_sep_id: parts.word_sep_id,
            frame_stride_ms: parts.frame_stride_ms,
            device: parts.device,
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
        let audio_tensor = Tensor::from_vec(normalized, (1, input.samples.len()), &self.device)
            .map_err(|e| AlignmentError::runtime("tensor creation", e))?;

        let logits = self
            .model
            .forward(&audio_tensor)
            .map_err(|e| AlignmentError::runtime("forward pass", e))?;

        let log_probs_t = candle_nn::ops::log_softmax(&logits, D::Minus1)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| AlignmentError::runtime("log_softmax", e))?;

        let log_probs: Vec<Vec<f32>> = log_probs_t
            .to_vec2()
            .map_err(|e| AlignmentError::runtime("to_vec2", e))?;

        let token_sequence =
            self.tokenizer
                .tokenize(&input.transcript, &self.vocab, self.blank_id, self.word_sep_id);

        if token_sequence.tokens.is_empty() {
            return Ok(AlignmentOutput { words: Vec::new() });
        }

        let t_len = log_probs.len();
        let min_frames = (token_sequence.tokens.len() + 1) / 2;
        if t_len < min_frames {
            return Err(AlignmentError::invalid_input(format!(
                "audio too short for transcript: {t_len} frames < {min_frames} required"
            )));
        }

        let path = self
            .sequence_aligner
            .align_path(&log_probs, &token_sequence.tokens)?;
        let mut words = self.word_grouper.group_words(
            &path,
            &token_sequence,
            &log_probs,
            self.blank_id,
            self.word_sep_id,
            self.frame_stride_ms,
        );

        if let Some(onset_frame) =
            detect_audio_onset_frame(&input.samples, input.sample_rate_hz, self.frame_stride_ms)
        {
            if let Some(first_word) = words.first() {
                let onset_ms = (onset_frame as f64 * self.frame_stride_ms) as u64;
                let shift_ms = onset_ms.saturating_sub(first_word.start_ms);
                if shift_ms > 0 {
                    for w in &mut words {
                        w.start_ms += shift_ms;
                        w.end_ms += shift_ms;
                    }
                }
            }
        }
        Ok(AlignmentOutput { words })
    }
}

fn detect_audio_onset_frame(samples: &[f32], sample_rate_hz: u32, frame_stride_ms: f64) -> Option<usize> {
    if samples.is_empty() || sample_rate_hz == 0 {
        return None;
    }
    let frame_len = ((sample_rate_hz as f64 * frame_stride_ms) / 1000.0).round() as usize;
    let frame_len = frame_len.max(1);

    let mut frame_rms = Vec::new();
    for chunk in samples.chunks(frame_len) {
        let mean_sq = chunk.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / chunk.len() as f64;
        frame_rms.push(mean_sq.sqrt() as f32);
    }
    if frame_rms.is_empty() {
        return None;
    }

    let baseline_frames = frame_rms.len().min(10);
    let noise_floor =
        frame_rms.iter().take(baseline_frames).copied().sum::<f32>() / baseline_frames as f32;
    let threshold = (noise_floor * 4.0).max(0.01);

    let mut run_start = 0usize;
    let mut run_len = 0usize;
    for (frame_idx, rms) in frame_rms.iter().copied().enumerate() {
        if rms >= threshold {
            if run_len == 0 {
                run_start = frame_idx;
            }
            run_len += 1;
            if run_len >= ONSET_MIN_CONSEC_FRAMES {
                return Some(run_start);
            }
            continue;
        }
        run_len = 0;
    }

    None
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
