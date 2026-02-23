use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};
use thiserror::Error;

// ── LayerNorm (manual, CUDA-safe) ───────────────────────────────────────────
// candle's native layer_norm op lacks a CUDA kernel, so we implement it with
// basic tensor operations that all have CUDA support.

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.weight.dim(0)? as f64;
        let mean = (x.sum_keepdim(D::Minus1)? / hidden)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
        let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    LayerNorm::load(size, eps, vb)
}

// ── Model Configuration ─────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct Wav2Vec2ModelConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    conv_dim: Vec<usize>,
    conv_kernel: Vec<usize>,
    conv_stride: Vec<usize>,
    num_conv_pos_embeddings: usize,
    num_conv_pos_embedding_groups: usize,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
    pad_token_id: usize,
    vocab_size: usize,
    #[serde(default = "default_feat_norm")]
    feat_extract_norm: String,
    #[serde(default = "default_conv_bias")]
    conv_bias: bool,
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
    fn load(path: &Path) -> Result<Self, AlignmentError> {
        let data =
            std::fs::read_to_string(path).map_err(|e| AlignmentError::io("read config.json", e))?;
        serde_json::from_str(&data).map_err(|e| AlignmentError::json("parse config.json", e))
    }

    fn frame_stride_ms(&self, sample_rate: u32) -> f64 {
        let stride_samples: usize = self.conv_stride.iter().product();
        stride_samples as f64 / sample_rate as f64 * 1000.0
    }
}

// ── Feature Extractor (7 conv layers) ────────────────────────────────────────

struct ConvLayer {
    conv: Conv1d,
    layer_norm: Option<LayerNorm>,
}

impl ConvLayer {
    fn load(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        use_bias: bool,
        use_ln: bool,
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
        Ok(Self { conv, layer_norm })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = match &self.layer_norm {
            Some(ln) => ln.forward(&xs.transpose(1, 2)?)?.transpose(1, 2)?.contiguous()?,
            None => xs,
        };
        xs.gelu()
    }
}

struct FeatureExtractor {
    layers: Vec<ConvLayer>,
}

impl FeatureExtractor {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(cfg.conv_dim.len());
        for i in 0..cfg.conv_dim.len() {
            let in_c = if i == 0 { 1 } else { cfg.conv_dim[i - 1] };
            let use_ln =
                cfg.feat_extract_norm == "layer" || (cfg.feat_extract_norm == "group" && i == 0);
            layers.push(ConvLayer::load(
                in_c,
                cfg.conv_dim[i],
                cfg.conv_kernel[i],
                cfg.conv_stride[i],
                cfg.conv_bias,
                use_ln,
                cfg.layer_norm_eps,
                vb.pp(format!("conv_layers.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = xs.clone();
        for l in &self.layers {
            h = l.forward(&h)?;
        }
        Ok(h)
    }
}

// ── Feature Projection ──────────────────────────────────────────────────────

struct FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
}

impl FeatureProjection {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dim = *cfg.conv_dim.last().unwrap_or(&cfg.hidden_size);
        Ok(Self {
            layer_norm: layer_norm(dim, cfg.layer_norm_eps, vb.pp("layer_norm"))?,
            projection: candle_nn::linear(dim, cfg.hidden_size, vb.pp("projection"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.projection.forward(&self.layer_norm.forward(xs)?)
    }
}

// ── Positional Conv Embedding (with weight-norm support) ─────────────────────

fn load_weight_norm_conv(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> candle_core::Result<Conv1d> {
    let dim_per_group = in_c / cfg.groups;
    let wv = vb.get((out_c, dim_per_group, kernel), "weight_v");

    let weight = if let Ok(wv) = wv {
        // HF Wav2Vec2 uses dim=2 weight norm: weight_g shape (1, 1, kernel).
        // Fall back to dim=0 (weight_g shape (out_c, 1, 1)) for other models.
        let wg = vb
            .get((1, 1, kernel), "weight_g")
            .or_else(|_| vb.get((out_c, 1, 1), "weight_g"))?;

        let wg_dims = wg.dims3()?;
        if wg_dims == (1, 1, kernel) {
            // dim=2: norm over output and group-input channels
            let norm = wv.sqr()?.sum_keepdim(0)?.sum_keepdim(1)?.sqrt()?;
            wv.broadcast_div(&norm)?.broadcast_mul(&wg)?
        } else {
            // dim=0: norm over group-input and kernel dims
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
        h.narrow(2, 0, seq_len)?.gelu()?.transpose(1, 2)?.contiguous()
    }
}

// ── Self-Attention ───────────────────────────────────────────────────────────

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

        let attn = candle_nn::ops::softmax(
            &q.matmul(&k.transpose(2, 3)?.contiguous()?)?,
            D::Minus1,
        )?;
        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.out.forward(&out)
    }
}

// ── Feed-Forward ─────────────────────────────────────────────────────────────

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

// ── Encoder Layer (stable pre-norm) ──────────────────────────────────────────

struct EncoderLayer {
    attn: SelfAttention,
    ln1: LayerNorm,
    ff: FeedForward,
    ln2: LayerNorm,
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
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let h = (xs + self.attn.forward(&self.ln1.forward(xs)?)?)?;
        &h + self.ff.forward(&self.ln2.forward(&h)?)?
    }
}

// ── Transformer Encoder ──────────────────────────────────────────────────────

struct Encoder {
    pos_conv: PosConvEmbed,
    layer_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(cfg, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            pos_conv: PosConvEmbed::load(cfg, vb.pp("pos_conv_embed"))?,
            layer_norm: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("layer_norm"),
            )?,
            layers,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = (xs + self.pos_conv.forward(xs)?)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        self.layer_norm.forward(&h)
    }
}

// ── Wav2Vec2ForCTC (full model) ──────────────────────────────────────────────

struct Wav2Vec2ForCTC {
    feat_extract: FeatureExtractor,
    feat_proj: FeatureProjection,
    encoder: Encoder,
    lm_head: Linear,
}

impl Wav2Vec2ForCTC {
    fn load(cfg: &Wav2Vec2ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let w = vb.pp("wav2vec2");
        Ok(Self {
            feat_extract: FeatureExtractor::load(cfg, w.pp("feature_extractor"))?,
            feat_proj: FeatureProjection::load(cfg, w.pp("feature_projection"))?,
            encoder: Encoder::load(cfg, w.pp("encoder"))?,
            lm_head: candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?,
        })
    }

    fn forward(&self, audio: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.feat_extract.forward(&audio.unsqueeze(1)?)?;
        let h = self.feat_proj.forward(&h.transpose(1, 2)?.contiguous()?)?;
        let h = self.encoder.forward(&h)?;
        self.lm_head.forward(&h)
    }
}

// ── CTC Forced Alignment (Viterbi) ──────────────────────────────────────────

fn build_token_sequence(
    transcript: &str,
    vocab: &HashMap<char, usize>,
    blank_id: usize,
    word_sep_id: usize,
) -> (Vec<usize>, Vec<Option<char>>) {
    let cleaned = transcript.to_lowercase();
    let mut tokens = vec![blank_id];
    let mut chars: Vec<Option<char>> = vec![None];

    for (wi, word) in cleaned.split_whitespace().enumerate() {
        if wi > 0 {
            tokens.push(word_sep_id);
            chars.push(Some('|'));
            tokens.push(blank_id);
            chars.push(None);
        }
        for c in word.chars() {
            if let Some(&id) = vocab.get(&c) {
                tokens.push(id);
                chars.push(Some(c));
                tokens.push(blank_id);
                chars.push(None);
            }
        }
    }

    (tokens, chars)
}

fn forced_align_viterbi(log_probs: &[Vec<f32>], tokens: &[usize]) -> Vec<(usize, usize)> {
    let t_len = log_probs.len();
    let s_len = tokens.len();
    if t_len == 0 || s_len == 0 {
        return Vec::new();
    }

    let mut dp = vec![vec![f32::NEG_INFINITY; s_len]; t_len];
    let mut bp = vec![vec![0usize; s_len]; t_len];

    dp[0][0] = log_probs[0][tokens[0]];
    if s_len > 1 {
        dp[0][1] = log_probs[0][tokens[1]];
    }

    for t in 1..t_len {
        for s in 0..s_len {
            let emit = log_probs[t][tokens[s]];
            let mut best = dp[t - 1][s];
            let mut from = s;

            if s >= 1 && dp[t - 1][s - 1] > best {
                best = dp[t - 1][s - 1];
                from = s - 1;
            }
            if s >= 2 && tokens[s] != tokens[s - 2] && dp[t - 1][s - 2] > best {
                best = dp[t - 1][s - 2];
                from = s - 2;
            }

            dp[t][s] = best + emit;
            bp[t][s] = from;
        }
    }

    let mut s = s_len - 1;
    if s_len >= 2 && dp[t_len - 1][s_len - 2] > dp[t_len - 1][s_len - 1] {
        s = s_len - 2;
    }

    let mut path = vec![(s, t_len - 1)];
    for t in (1..t_len).rev() {
        s = bp[t][s];
        path.push((s, t - 1));
    }
    path.reverse();
    path
}

fn group_into_words(
    path: &[(usize, usize)],
    tokens: &[usize],
    chars: &[Option<char>],
    log_probs: &[Vec<f32>],
    blank_id: usize,
    word_sep_id: usize,
    stride_ms: f64,
) -> Vec<WordTiming> {
    let mut words = Vec::new();
    let mut cur_word = String::new();
    let mut start_frame: Option<usize> = None;
    let mut end_frame: usize = 0;
    let mut lp_accum = Vec::new();

    let flush = |w: &mut String,
                 sf: &mut Option<usize>,
                 ef: usize,
                 lps: &mut Vec<f32>,
                 out: &mut Vec<WordTiming>,
                 ms: f64| {
        if w.is_empty() {
            return;
        }
        let conf = if lps.is_empty() {
            0.0
        } else {
            (lps.iter().sum::<f32>() / lps.len() as f32).exp()
        };
        out.push(WordTiming {
            word: w.clone(),
            start_ms: (sf.unwrap_or(0) as f64 * ms) as u64,
            end_ms: ((ef + 1) as f64 * ms) as u64,
            confidence: conf,
        });
        w.clear();
        *sf = None;
        lps.clear();
    };

    for &(s, frame) in path {
        let tid = tokens[s];
        if tid == blank_id {
            continue;
        }
        if tid == word_sep_id {
            flush(
                &mut cur_word,
                &mut start_frame,
                end_frame,
                &mut lp_accum,
                &mut words,
                stride_ms,
            );
            continue;
        }
        if let Some(c) = chars[s] {
            if start_frame.is_none() {
                start_frame = Some(frame);
            }
            end_frame = frame;
            cur_word.push(c);
            lp_accum.push(log_probs[frame][tid]);
        }
    }

    flush(
        &mut cur_word,
        &mut start_frame,
        end_frame,
        &mut lp_accum,
        &mut words,
        stride_ms,
    );
    words
}

#[derive(Debug, Error)]
pub enum AlignmentError {
    #[error("I/O error while {context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON parse error while {context}: {source}")]
    Json {
        context: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("{context}: {message}")]
    Runtime {
        context: &'static str,
        message: String,
    },
    #[error("invalid input: {message}")]
    InvalidInput { message: String },
}

impl AlignmentError {
    fn io(context: &'static str, source: std::io::Error) -> Self {
        Self::Io { context, source }
    }

    fn json(context: &'static str, source: serde_json::Error) -> Self {
        Self::Json { context, source }
    }

    fn runtime(context: &'static str, err: impl std::fmt::Display) -> Self {
        Self::Runtime {
            context,
            message: err.to_string(),
        }
    }

    fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }
}

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

#[derive(Debug, Clone)]
pub struct AlignmentInput {
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
    pub transcript: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentOutput {
    pub words: Vec<WordTiming>,
}

pub struct Wav2Vec2Aligner {
    model: Wav2Vec2ForCTC,
    vocab: HashMap<char, usize>,
    blank_id: usize,
    word_sep_id: usize,
    frame_stride_ms: f64,
    device: Device,
    expected_sample_rate_hz: u32,
}

impl Wav2Vec2Aligner {
    pub fn load(config: &Wav2Vec2Config) -> Result<Self, AlignmentError> {
        let device = match config.device.as_str() {
            "cuda" => Device::new_cuda(0).map_err(|e| AlignmentError::runtime("CUDA init", e))?,
            _ => Device::Cpu,
        };

        let model_cfg = Wav2Vec2ModelConfig::load(Path::new(&config.config_path))?;
        let expected_sample_rate_hz =
            if config.expected_sample_rate_hz == 0 {
                Wav2Vec2Config::DEFAULT_SAMPLE_RATE_HZ
            } else {
                config.expected_sample_rate_hz
            };
        let frame_stride_ms = model_cfg.frame_stride_ms(expected_sample_rate_hz);
        let blank_id = model_cfg.pad_token_id;

        let vocab = load_vocab(Path::new(&config.vocab_path))?;
        let word_sep_id = vocab.get(&'|').copied().unwrap_or(0);

        let model_data =
            std::fs::read(&config.model_path).map_err(|e| AlignmentError::io("read safetensors", e))?;
        let vb = VarBuilder::from_buffered_safetensors(model_data, DType::F32, &device)
            .map_err(|e| AlignmentError::runtime("load safetensors", e))?;

        let model = Wav2Vec2ForCTC::load(&model_cfg, vb)
            .map_err(|e| AlignmentError::runtime("build model", e))?;

        tracing::info!(
            hidden_size = model_cfg.hidden_size,
            layers = model_cfg.num_hidden_layers,
            vocab = model_cfg.vocab_size,
            blank_id,
            frame_stride_ms,
            ?device,
            "wav2vec2 model loaded"
        );

        Ok(Self {
            model,
            vocab,
            blank_id,
            word_sep_id,
            frame_stride_ms,
            device,
            expected_sample_rate_hz,
        })
    }

    pub fn align(&self, input: &AlignmentInput) -> Result<AlignmentOutput, AlignmentError> {
        let words = self.align_words(&input.samples, input.sample_rate_hz, &input.transcript)?;
        Ok(AlignmentOutput { words })
    }

    fn align_words(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        transcript: &str,
    ) -> Result<Vec<WordTiming>, AlignmentError> {
        if samples.is_empty() || transcript.trim().is_empty() {
            return Ok(Vec::new());
        }

        if sample_rate_hz != self.expected_sample_rate_hz {
            tracing::warn!(
                expected_rate_hz = self.expected_sample_rate_hz,
                actual_rate_hz = sample_rate_hz,
                "wav2vec2 aligner expects a specific sample rate; quality may degrade"
            );
        }

        let normalized = normalize_audio(samples);

        let audio_tensor = Tensor::from_vec(normalized, (1, samples.len()), &self.device)
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

        let (tokens, chars) =
            build_token_sequence(transcript, &self.vocab, self.blank_id, self.word_sep_id);

        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let t_len = log_probs.len();
        let min_frames = (tokens.len() + 1) / 2;
        if t_len < min_frames {
            return Err(AlignmentError::invalid_input(format!(
                "audio too short for transcript: {t_len} frames < {min_frames} required"
            )));
        }

        let path = forced_align_viterbi(&log_probs, &tokens);

        Ok(group_into_words(
            &path,
            &tokens,
            &chars,
            &log_probs,
            self.blank_id,
            self.word_sep_id,
            self.frame_stride_ms,
        ))
    }
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
