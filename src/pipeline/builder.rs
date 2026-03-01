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
    use crate::pipeline::traits::{ForwardOutput, RuntimeBackend, RuntimeInferenceOutput};

    use super::*;

    struct MockBackend;

    impl RuntimeBackend for MockBackend {
        fn infer(&self, _normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError> {
            Ok(ForwardOutput::Host(RuntimeInferenceOutput {
                log_probs: vec![vec![0.0f32; 32]; 10],
                num_frames_t: 10,
                vocab_size: 32,
                dtype: "f32".to_string(),
            }))
        }

        fn device_label(&self) -> String {
            "mock".to_string()
        }
    }

    const MINIMAL_CONFIG_JSON: &str = r#"{
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "conv_dim": [512],
        "conv_kernel": [10],
        "conv_stride": [2, 2, 2, 2, 2],
        "num_conv_pos_embeddings": 128,
        "num_conv_pos_embedding_groups": 16,
        "pad_token_id": 0,
        "vocab_size": 32
    }"#;

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

    #[test]
    fn build_success_with_mock_backend_and_temp_files() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("wav2vec2_rs_builder_config.json");
        let vocab_path = temp_dir.join("wav2vec2_rs_builder_vocab.json");
        std::fs::write(&config_path, MINIMAL_CONFIG_JSON).expect("write config");
        let vocab_json = r#"{"a": 1, "b": 2, "|": 3}"#;
        std::fs::write(&vocab_path, vocab_json).expect("write vocab");

        let config = Wav2Vec2Config {
            model_path: String::new(),
            config_path: config_path.to_string_lossy().to_string(),
            vocab_path: vocab_path.to_string_lossy().to_string(),
            device: "cpu".to_string(),
            expected_sample_rate_hz: 16_000,
        };
        let aligner = ForcedAlignerBuilder::new(config)
            .with_runtime_backend(Box::new(MockBackend))
            .build()
            .expect("build should succeed");
        let input = crate::types::AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![0.0f32; 1600],
            transcript: "a".to_string(),
            normalized: Some(vec![0.0f32; 1600]),
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.len() <= 1);

        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&vocab_path);
    }

    #[test]
    fn build_fails_on_invalid_config_path() {
        let temp_dir = std::env::temp_dir();
        let vocab_path = temp_dir.join("wav2vec2_rs_builder_vocab_missing.json");
        std::fs::write(&vocab_path, r#"{"a":1}"#).expect("write vocab");
        let config = Wav2Vec2Config {
            model_path: String::new(),
            config_path: "/nonexistent/config.json".to_string(),
            vocab_path: vocab_path.to_string_lossy().to_string(),
            device: "cpu".to_string(),
            expected_sample_rate_hz: 16_000,
        };
        let result = ForcedAlignerBuilder::new(config)
            .with_runtime_backend(Box::new(MockBackend))
            .build();
        assert!(result.is_err());
        let _ = std::fs::remove_file(&vocab_path);
    }

    #[test]
    fn build_fails_on_invalid_vocab_path() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("wav2vec2_rs_builder_config_missing.json");
        std::fs::write(&config_path, MINIMAL_CONFIG_JSON).expect("write config");
        let config = Wav2Vec2Config {
            model_path: String::new(),
            config_path: config_path.to_string_lossy().to_string(),
            vocab_path: "/nonexistent/vocab.json".to_string(),
            device: "cpu".to_string(),
            expected_sample_rate_hz: 16_000,
        };
        let result = ForcedAlignerBuilder::new(config)
            .with_runtime_backend(Box::new(MockBackend))
            .build();
        assert!(result.is_err());
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn build_vocab_filters_multi_char_keys() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("wav2vec2_rs_builder_vocab_filter_config.json");
        let vocab_path = temp_dir.join("wav2vec2_rs_builder_vocab_filter.json");
        std::fs::write(&config_path, MINIMAL_CONFIG_JSON).expect("write config");
        let vocab_json = r#"{"a": 1, "b": 2, "ab": 3, "|": 4}"#;
        std::fs::write(&vocab_path, vocab_json).expect("write vocab");

        let config = Wav2Vec2Config {
            model_path: String::new(),
            config_path: config_path.to_string_lossy().to_string(),
            vocab_path: vocab_path.to_string_lossy().to_string(),
            device: "cpu".to_string(),
            expected_sample_rate_hz: 16_000,
        };
        let aligner = ForcedAlignerBuilder::new(config)
            .with_runtime_backend(Box::new(MockBackend))
            .build()
            .expect("build should succeed");
        let input = crate::types::AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![],
            transcript: "a b".to_string(),
            normalized: None,
        };
        let out = aligner.align(&input).unwrap();
        assert!(out.words.is_empty());
        let input2 = crate::types::AlignmentInput {
            sample_rate_hz: 16_000,
            samples: vec![0.0f32; 1600],
            transcript: "a".to_string(),
            normalized: Some(vec![0.0f32; 1600]),
        };
        let out2 = aligner.align(&input2).unwrap();
        assert!(out2.words.len() <= 1);
        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&vocab_path);
    }
}
