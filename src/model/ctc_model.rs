use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::Wav2Vec2ModelConfig;
use crate::model::encoder::Encoder;
use crate::model::feature_extractor::FeatureExtractor;
use crate::model::feature_projection::FeatureProjection;

pub(crate) struct Wav2Vec2ForCTC {
    feat_extract: FeatureExtractor,
    feat_proj: FeatureProjection,
    encoder: Encoder,
    lm_head: Linear,
}

impl Wav2Vec2ForCTC {
    pub(crate) fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let w = vb.pp("wav2vec2");
        Ok(Self {
            feat_extract: FeatureExtractor::load(cfg, w.pp("feature_extractor"))?,
            feat_proj: FeatureProjection::load(cfg, w.pp("feature_projection"))?,
            encoder: Encoder::load(cfg, w.pp("encoder"))?,
            lm_head: candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?,
        })
    }

    pub(crate) fn forward(&self, audio: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.feat_extract.forward(&audio.unsqueeze(1)?)?;
        let h = self.feat_proj.forward(&h.transpose(1, 2)?.contiguous()?)?;
        let h = self.encoder.forward(&h)?;
        self.lm_head.forward(&h)
    }
}
