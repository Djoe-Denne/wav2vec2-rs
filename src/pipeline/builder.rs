use std::collections::HashMap;
use std::path::Path;

use crate::config::{Wav2Vec2Config, Wav2Vec2ModelConfig};
use crate::error::AlignmentError;
use crate::pipeline::defaults::{CaseAwareTokenizer, DefaultWordGrouper, ViterbiSequenceAligner};
use crate::pipeline::model_runtime::build_runtime_backend;
use crate::pipeline::runtime::{ForcedAligner, ForcedAlignerParts};
use crate::pipeline::traits::{
    RuntimeBackend, RuntimeKind, SequenceAligner, Tokenizer, WordGrouper,
};

pub struct ForcedAlignerBuilder {
    config: Wav2Vec2Config,
    runtime_kind: RuntimeKind,
    runtime_backend: Option<Box<dyn RuntimeBackend>>,
    tokenizer: Option<Box<dyn Tokenizer>>,
    sequence_aligner: Option<Box<dyn SequenceAligner>>,
    word_grouper: Option<Box<dyn WordGrouper>>,
}

impl ForcedAlignerBuilder {
    pub fn new(config: Wav2Vec2Config) -> Self {
        Self {
            config,
            runtime_kind: RuntimeKind::Candle,
            runtime_backend: None,
            tokenizer: None,
            sequence_aligner: None,
            word_grouper: None,
        }
    }

    pub fn with_runtime_kind(mut self, runtime_kind: RuntimeKind) -> Self {
        self.runtime_kind = runtime_kind;
        self
    }

    pub fn with_runtime_backend(mut self, runtime_backend: Box<dyn RuntimeBackend>) -> Self {
        self.runtime_backend = Some(runtime_backend);
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn with_sequence_aligner(mut self, sequence_aligner: Box<dyn SequenceAligner>) -> Self {
        self.sequence_aligner = Some(sequence_aligner);
        self
    }

    pub fn with_word_grouper(mut self, word_grouper: Box<dyn WordGrouper>) -> Self {
        self.word_grouper = Some(word_grouper);
        self
    }

    pub fn build(self) -> Result<ForcedAligner, AlignmentError> {
        let model_cfg = Wav2Vec2ModelConfig::load(Path::new(&self.config.config_path))?;
        let expected_sample_rate_hz = if self.config.expected_sample_rate_hz == 0 {
            Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ
        } else {
            self.config.expected_sample_rate_hz
        };
        let frame_stride_ms = model_cfg.frame_stride_ms(expected_sample_rate_hz);
        let blank_id = model_cfg.pad_token_id;

        let vocab = load_vocab(Path::new(&self.config.vocab_path))?;
        let word_sep_id = vocab.get(&'|').copied().unwrap_or(0);

        let runtime_backend = if let Some(runtime_backend) = self.runtime_backend {
            runtime_backend
        } else {
            build_runtime_backend(self.runtime_kind, &self.config, &model_cfg)?
        };

        Ok(ForcedAligner::from_parts(ForcedAlignerParts {
            runtime_backend,
            vocab,
            blank_id,
            word_sep_id,
            frame_stride_ms,
            expected_sample_rate_hz,
            tokenizer: self
                .tokenizer
                .unwrap_or_else(|| Box::new(CaseAwareTokenizer)),
            sequence_aligner: self
                .sequence_aligner
                .unwrap_or_else(|| Box::new(ViterbiSequenceAligner)),
            word_grouper: self
                .word_grouper
                .unwrap_or_else(|| Box::new(DefaultWordGrouper)),
        }))
    }
}

fn load_vocab(path: &Path) -> Result<HashMap<char, usize>, AlignmentError> {
    let data =
        std::fs::read_to_string(path).map_err(|e| AlignmentError::io("read vocab.json", e))?;
    let raw: HashMap<String, usize> =
        serde_json::from_str(&data).map_err(|e| AlignmentError::json("parse vocab.json", e))?;

    Ok(raw
        .into_iter()
        .filter_map(|(k, v)| {
            let mut it = k.chars();
            let c = it.next()?;
            if it.next().is_some() {
                return None;
            }
            Some((c, v))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_to_candle_runtime() {
        let builder = ForcedAlignerBuilder::new(Wav2Vec2Config::default());
        assert_eq!(builder.runtime_kind, RuntimeKind::Candle);
        assert!(builder.runtime_backend.is_none());
    }

    #[test]
    fn builder_runtime_kind_can_be_overridden() {
        let builder = ForcedAlignerBuilder::new(Wav2Vec2Config::default())
            .with_runtime_kind(RuntimeKind::Onnx);
        assert_eq!(builder.runtime_kind, RuntimeKind::Onnx);
    }
}
