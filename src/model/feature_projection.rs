use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::Wav2Vec2ModelConfig;
use crate::model::layers::{layer_norm, LayerNorm};

pub(crate) struct FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
}

impl FeatureProjection {
    pub(crate) fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dim = *cfg.conv_dim.last().unwrap_or(&cfg.hidden_size);
        Ok(Self {
            layer_norm: layer_norm(dim, cfg.layer_norm_eps, vb.pp("layer_norm"))?,
            projection: candle_nn::linear(dim, cfg.hidden_size, vb.pp("projection"))?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.projection.forward(&self.layer_norm.forward(xs)?)
    }
}
