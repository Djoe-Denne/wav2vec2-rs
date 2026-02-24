use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::config::{Wav2Vec2Config, Wav2Vec2ModelConfig};
use crate::error::AlignmentError;
use crate::model::ctc_model::Wav2Vec2ForCTC;
use crate::pipeline::defaults::{CaseAwareTokenizer, DefaultWordGrouper, ViterbiSequenceAligner};
use crate::pipeline::runtime::{ForcedAligner, ForcedAlignerParts};
use crate::pipeline::traits::{SequenceAligner, Tokenizer, WordGrouper};

pub struct ForcedAlignerBuilder {
    config: Wav2Vec2Config,
    tokenizer: Option<Box<dyn Tokenizer>>,
    sequence_aligner: Option<Box<dyn SequenceAligner>>,
    word_grouper: Option<Box<dyn WordGrouper>>,
}

impl ForcedAlignerBuilder {
    pub fn new(config: Wav2Vec2Config) -> Self {
        Self {
            config,
            tokenizer: None,
            sequence_aligner: None,
            word_grouper: None,
        }
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
        let device = match self.config.device.as_str() {
            "cuda" => Device::new_cuda(0).map_err(|e| AlignmentError::runtime("CUDA init", e))?,
            _ => Device::Cpu,
        };

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

        let model_data = std::fs::read(&self.config.model_path)
            .map_err(|e| AlignmentError::io("read safetensors", e))?;
        let vb = VarBuilder::from_buffered_safetensors(model_data, DType::F32, &device)
            .map_err(|e| AlignmentError::runtime("load safetensors", e))?;
        let model =
            Wav2Vec2ForCTC::load(&model_cfg, vb).map_err(|e| AlignmentError::runtime("build model", e))?;

        tracing::info!(
            hidden_size = model_cfg.hidden_size,
            layers = model_cfg.num_hidden_layers,
            vocab = model_cfg.vocab_size,
            blank_id,
            frame_stride_ms,
            ?device,
            "wav2vec2 model loaded"
        );

        Ok(ForcedAligner::from_parts(ForcedAlignerParts {
            model,
            vocab,
            blank_id,
            word_sep_id,
            frame_stride_ms,
            device,
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
    let data = std::fs::read_to_string(path).map_err(|e| AlignmentError::io("read vocab.json", e))?;
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
