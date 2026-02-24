use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

use crate::config::Wav2Vec2ModelConfig;
use crate::model::layers::{group_norm_1d, layer_norm, GroupNorm1d, LayerNorm};

struct ConvLayer {
    conv: Conv1d,
    layer_norm: Option<LayerNorm>,
    group_norm: Option<GroupNorm1d>,
}

impl ConvLayer {
    fn load(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        use_bias: bool,
        use_ln: bool,
        use_gn: bool,
        eps: f64,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv1dConfig {
            stride,
            ..Default::default()
        };
        let conv = if use_bias {
            candle_nn::conv1d(in_c, out_c, kernel, cfg, vb.pp("conv"))?
        } else {
            candle_nn::conv1d_no_bias(in_c, out_c, kernel, cfg, vb.pp("conv"))?
        };
        let layer_norm = if use_ln {
            Some(layer_norm(out_c, eps, vb.pp("layer_norm"))?)
        } else {
            None
        };
        // HF wav2vec2 "group" mode uses GroupNorm on the first conv layer.
        let group_norm = if use_gn {
            Some(group_norm_1d(out_c, out_c, eps, vb.pp("layer_norm"))?)
        } else {
            None
        };
        Ok(Self {
            conv,
            layer_norm,
            group_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = if let Some(gn) = &self.group_norm {
            // GroupNorm expects (batch, channels, time).
            gn.forward(&xs)?
        } else if let Some(ln) = &self.layer_norm {
            // LayerNorm path normalizes over channels for each time step.
            ln.forward(&xs.transpose(1, 2)?)?.transpose(1, 2)?.contiguous()?
        } else {
            xs
        };
        xs.gelu()
    }
}

pub(crate) struct FeatureExtractor {
    layers: Vec<ConvLayer>,
}

impl FeatureExtractor {
    pub(crate) fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(cfg.conv_dim.len());
        for i in 0..cfg.conv_dim.len() {
            let in_c = if i == 0 { 1 } else { cfg.conv_dim[i - 1] };
            let use_ln = cfg.feat_extract_norm == "layer";
            let use_gn = cfg.feat_extract_norm == "group" && i == 0;
            layers.push(ConvLayer::load(
                in_c,
                cfg.conv_dim[i],
                cfg.conv_kernel[i],
                cfg.conv_stride[i],
                cfg.conv_bias,
                use_ln,
                use_gn,
                cfg.layer_norm_eps,
                vb.pp(format!("conv_layers.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = xs.clone();
        for l in &self.layers {
            h = l.forward(&h)?;
        }
        Ok(h)
    }
}

pub(crate) fn load_weight_norm_conv(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> candle_core::Result<Conv1d> {
    let dim_per_group = in_c / cfg.groups;
    let wv = vb.get((out_c, dim_per_group, kernel), "weight_v");

    let weight = if let Ok(wv) = wv {
        let wg = vb
            .get((1, 1, kernel), "weight_g")
            .or_else(|_| vb.get((out_c, 1, 1), "weight_g"))?;

        let wg_dims = wg.dims3()?;
        if wg_dims == (1, 1, kernel) {
            let norm = wv.sqr()?.sum_keepdim(0)?.sum_keepdim(1)?.sqrt()?;
            wv.broadcast_div(&norm)?.broadcast_mul(&wg)?
        } else {
            let (o, ig, k) = wv.dims3()?;
            let norm = wv
                .reshape((o, ig * k))?
                .sqr()?
                .sum_keepdim(1)?
                .sqrt()?
                .unsqueeze(2)?;
            wv.broadcast_div(&norm)?.broadcast_mul(&wg)?
        }
    } else {
        vb.get((out_c, dim_per_group, kernel), "weight")?
    };

    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}
