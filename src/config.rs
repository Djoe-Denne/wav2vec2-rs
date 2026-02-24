use std::path::Path;

use crate::error::AlignmentError;

#[derive(Debug, Clone)]
pub struct Wav2Vec2Config {
    pub model_path: String,
    pub config_path: String,
    pub vocab_path: String,
    pub device: String,
    pub expected_sample_rate_hz: u32,
}

impl Wav2Vec2Config {
    pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
}

impl Default for Wav2Vec2Config {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            config_path: String::new(),
            vocab_path: String::new(),
            device: "cpu".to_string(),
            expected_sample_rate_hz: Self::DEFAULT_SAMPLE_RATE_HZ,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Wav2Vec2ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    #[serde(default)]
    pub do_stable_layer_norm: bool,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub vocab_size: usize,
    #[serde(default = "default_feat_norm")]
    pub feat_extract_norm: String,
    #[serde(default = "default_conv_bias")]
    pub conv_bias: bool,
}

fn default_eps() -> f64 {
    1e-5
}
fn default_feat_norm() -> String {
    "layer".to_string()
}
fn default_conv_bias() -> bool {
    true
}

impl Wav2Vec2ModelConfig {
    pub(crate) fn load(path: &Path) -> Result<Self, AlignmentError> {
        let data =
            std::fs::read_to_string(path).map_err(|e| AlignmentError::io("read config.json", e))?;
        serde_json::from_str(&data).map_err(|e| AlignmentError::json("parse config.json", e))
    }

    pub(crate) fn frame_stride_ms(&self, sample_rate: u32) -> f64 {
        let stride_samples: usize = self.conv_stride.iter().product();
        stride_samples as f64 / sample_rate as f64 * 1000.0
    }
}
