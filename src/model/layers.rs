use candle_core::{Tensor, D};
use candle_nn::VarBuilder;

pub(crate) struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub(crate) fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.weight.dim(0)? as f64;
        let mean = (x.sum_keepdim(D::Minus1)? / hidden)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
        let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

pub(crate) fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    LayerNorm::load(size, eps, vb)
}

pub(crate) struct GroupNorm1d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    num_groups: usize,
    num_channels: usize,
}

impl GroupNorm1d {
    pub(crate) fn load(
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        let bias = vb.get(num_channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps,
            num_groups,
            num_channels,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        if c != self.num_channels || self.num_groups == 0 || c % self.num_groups != 0 {
            return Err(candle_core::Error::Msg(format!(
                "invalid GroupNorm1d shape/groups: channels={c}, configured_channels={}, groups={}",
                self.num_channels, self.num_groups
            )));
        }

        let channels_per_group = c / self.num_groups;
        let denom = (channels_per_group * t) as f64;

        // Mirror PyTorch GroupNorm over (channels_per_group, time) axes.
        let grouped = x.reshape((b, self.num_groups, channels_per_group, t))?;
        let mean = (grouped.sum_keepdim(D::Minus1)?.sum_keepdim(D::Minus2)? / denom)?;
        let centered = grouped.broadcast_sub(&mean)?;
        let var = (centered.sqr()?.sum_keepdim(D::Minus1)?.sum_keepdim(D::Minus2)? / denom)?;
        let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let normed = normed.reshape((b, c, t))?;

        let weight = self.weight.reshape((1, c, 1))?;
        let bias = self.bias.reshape((1, c, 1))?;
        normed.broadcast_mul(&weight)?.broadcast_add(&bias)
    }
}

pub(crate) fn group_norm_1d(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<GroupNorm1d> {
    GroupNorm1d::load(num_groups, num_channels, eps, vb)
}
