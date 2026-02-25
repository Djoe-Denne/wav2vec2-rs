use candle_core::{Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

use crate::config::Wav2Vec2ModelConfig;
use crate::model::feature_extractor::load_weight_norm_conv;
use crate::model::layers::{layer_norm, LayerNorm};

struct PosConvEmbed {
    conv: Conv1d,
}

impl PosConvEmbed {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: cfg.num_conv_pos_embeddings / 2,
            groups: cfg.num_conv_pos_embedding_groups,
            ..Default::default()
        };
        Ok(Self {
            conv: load_weight_norm_conv(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.num_conv_pos_embeddings,
                conv_cfg,
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let seq_len = xs.dim(1)?;
        let h = self.conv.forward(&xs.transpose(1, 2)?.contiguous()?)?;
        h.narrow(2, 0, seq_len)?
            .gelu()?
            .transpose(1, 2)?
            .contiguous()
    }
}

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SelfAttention {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let hd = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            k: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?,
            v: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?,
            out: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let reshape = |x: Tensor| {
            x.reshape((b, t, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };

        let q = reshape((self.q.forward(xs)? * self.scale)?)?;
        let k = reshape(self.k.forward(xs)?)?;
        let v = reshape(self.v.forward(xs)?)?;

        let attn =
            candle_nn::ops::softmax(&q.matmul(&k.transpose(2, 3)?.contiguous()?)?, D::Minus1)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t,
            self.num_heads * self.head_dim,
        ))?;
        self.out.forward(&out)
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            up: candle_nn::linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("intermediate_dense"),
            )?,
            down: candle_nn::linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("output_dense"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.down.forward(&self.up.forward(xs)?.gelu()?)
    }
}

struct EncoderLayer {
    attn: SelfAttention,
    ln1: LayerNorm,
    ff: FeedForward,
    ln2: LayerNorm,
    stable_pre_norm: bool,
}

impl EncoderLayer {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(cfg, vb.pp("attention"))?,
            ln1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm"))?,
            ff: FeedForward::load(cfg, vb.pp("feed_forward"))?,
            ln2: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
            stable_pre_norm: cfg.do_stable_layer_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        if self.stable_pre_norm {
            // Stable pre-norm variant:
            //   h = x + attn(ln1(x))
            //   y = h + ff(ln2(h))
            let h = (xs + self.attn.forward(&self.ln1.forward(xs)?)?)?;
            &h + self.ff.forward(&self.ln2.forward(&h)?)?
        } else {
            // Standard post-norm variant used by wav2vec2-base:
            //   h = ln1(x + attn(x))
            //   y = ln2(h + ff(h))
            let h = self.ln1.forward(&(xs + self.attn.forward(xs)?)?)?;
            self.ln2.forward(&(&h + self.ff.forward(&h)?)?)
        }
    }
}

pub(crate) struct Encoder {
    pos_conv: PosConvEmbed,
    layer_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub(crate) fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(cfg, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            pos_conv: PosConvEmbed::load(cfg, vb.pp("pos_conv_embed"))?,
            layer_norm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm"))?,
            layers,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = (xs + self.pos_conv.forward(xs)?)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        self.layer_norm.forward(&h)
    }
}
