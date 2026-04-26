#[cfg(feature = "onnx")]
use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::{Wav2Vec2Config, Wav2Vec2ModelConfig};
use crate::error::AlignmentError;
use crate::model::ctc_model::Wav2Vec2ForCTC;
use crate::pipeline::traits::{
    ForwardOutput, ProfiledForwardOutput, RuntimeBackend, RuntimeInferenceOutput, RuntimeKind,
};

pub(crate) fn build_runtime_backend(
    runtime_kind: RuntimeKind,
    config: &Wav2Vec2Config,
    model_cfg: &Wav2Vec2ModelConfig,
) -> Result<Box<dyn RuntimeBackend>, AlignmentError> {
    match runtime_kind {
        RuntimeKind::Candle => Ok(Box::new(CandleRuntimeBackend::load(config, model_cfg)?)),
        RuntimeKind::Onnx => build_onnx_runtime_backend(config, model_cfg),
    }
}

fn build_onnx_runtime_backend(
    config: &Wav2Vec2Config,
    model_cfg: &Wav2Vec2ModelConfig,
) -> Result<Box<dyn RuntimeBackend>, AlignmentError> {
    #[cfg(feature = "onnx")]
    {
        Ok(Box::new(OnnxRuntimeBackend::load(config, model_cfg)?))
    }

    #[cfg(not(feature = "onnx"))]
    {
        let _ = config;
        let _ = model_cfg;
        Err(AlignmentError::runtime(
            "build runtime backend",
            "ONNX runtime support is disabled; enable the `onnx` cargo feature",
        ))
    }
}

struct CandleRuntimeBackend {
    model: Wav2Vec2ForCTC,
    device: Device,
    dtype: DType,
}

impl CandleRuntimeBackend {
    fn load(
        config: &Wav2Vec2Config,
        model_cfg: &Wav2Vec2ModelConfig,
    ) -> Result<Self, AlignmentError> {
        let device = match config.device.as_str() {
            "cuda" => Device::new_cuda(0).map_err(|e| AlignmentError::runtime("CUDA init", e))?,
            _ => Device::Cpu,
        };
        let dtype = candle_dtype(model_cfg)?;

        let model_data = std::fs::read(&config.model_path)
            .map_err(|e| AlignmentError::io("read safetensors", e))?;
        let vb = VarBuilder::from_buffered_safetensors(model_data, dtype, &device)
            .map_err(|e| AlignmentError::runtime("load safetensors", e))?;
        let model = Wav2Vec2ForCTC::load(model_cfg, vb)
            .map_err(|e| AlignmentError::runtime("build model", e))?;

        tracing::info!(
            hidden_size = model_cfg.hidden_size,
            layers = model_cfg.num_hidden_layers,
            vocab = model_cfg.vocab_size,
            ?device,
            ?dtype,
            "wav2vec2 Candle runtime loaded"
        );

        Ok(Self {
            model,
            device,
            dtype,
        })
    }

    fn build_audio_tensor(&self, normalized_audio: &[f32]) -> Result<Tensor, AlignmentError> {
        let tensor = Tensor::from_vec(
            normalized_audio.to_vec(),
            (1, normalized_audio.len()),
            &self.device,
        )
        .map_err(|e| AlignmentError::runtime("tensor creation", e))?;
        tensor
            .to_dtype(self.dtype)
            .map_err(|e| AlignmentError::runtime("tensor dtype conversion", e))
    }

    fn log_probs_to_host_output(
        &self,
        log_probs_t: Tensor,
    ) -> Result<RuntimeInferenceOutput, AlignmentError> {
        let (num_frames_t, vocab_size) = log_probs_t
            .dims2()
            .map_err(|e| AlignmentError::runtime("log_probs dims2", e))?;
        let dtype = format!("{:?}", log_probs_t.dtype()).to_ascii_lowercase();
        let log_probs = log_probs_t
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec2())
            .map_err(|e| AlignmentError::runtime("to_vec2", e))?;

        Ok(RuntimeInferenceOutput {
            log_probs,
            num_frames_t,
            vocab_size,
            dtype,
        })
    }
}

fn candle_dtype(model_cfg: &Wav2Vec2ModelConfig) -> Result<DType, AlignmentError> {
    match model_cfg.dtype.as_deref().map(str::to_ascii_lowercase) {
        Some(dtype) if dtype == "float16" || dtype == "f16" => Ok(DType::F16),
        Some(dtype) if dtype == "float32" || dtype == "f32" => Ok(DType::F32),
        Some(dtype) => Err(AlignmentError::invalid_input(format!(
            "unsupported Candle model dtype '{dtype}', expected float32 or float16"
        ))),
        None => Ok(DType::F32),
    }
}

impl RuntimeBackend for CandleRuntimeBackend {
    fn infer(&self, normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError> {
        let audio_tensor = self.build_audio_tensor(normalized_audio)?;
        let logits = self
            .model
            .forward(&audio_tensor)
            .map_err(|e| AlignmentError::runtime("forward pass", e))?;

        let log_probs_t = candle_nn::ops::log_softmax(&logits, D::Minus1)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| AlignmentError::runtime("log_softmax", e))?;
        Ok(ForwardOutput::Host(
            self.log_probs_to_host_output(log_probs_t)?,
        ))
    }

    fn infer_profiled(
        &self,
        normalized_audio: &[f32],
    ) -> Result<ProfiledForwardOutput, AlignmentError> {
        let audio_tensor = self.build_audio_tensor(normalized_audio)?;

        self.synchronize("cuda synchronize before forward timing")?;
        let forward_started = Instant::now();
        let logits = self
            .model
            .forward(&audio_tensor)
            .map_err(|e| AlignmentError::runtime("forward pass", e))?;
        self.synchronize("cuda synchronize after forward timing")?;
        let forward_ms = duration_to_ms(forward_started.elapsed());

        self.synchronize("cuda synchronize before post timing")?;
        let post_started = Instant::now();
        let log_probs_t = candle_nn::ops::log_softmax(&logits, D::Minus1)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| AlignmentError::runtime("log_softmax", e))?;
        let runtime_output = self.log_probs_to_host_output(log_probs_t)?;
        self.synchronize("cuda synchronize after post timing")?;
        let post_ms = duration_to_ms(post_started.elapsed());

        Ok(ProfiledForwardOutput {
            forward_output: ForwardOutput::Host(runtime_output),
            forward_ms,
            post_ms,
        })
    }

    fn synchronize(&self, context: &'static str) -> Result<(), AlignmentError> {
        if self.device.is_cuda() {
            self.device
                .synchronize()
                .map_err(|err| AlignmentError::runtime(context, err))?;
        }
        Ok(())
    }

    fn device_label(&self) -> String {
        if self.device.is_cuda() {
            "cuda".to_string()
        } else if self.device.is_metal() {
            "metal".to_string()
        } else {
            "cpu".to_string()
        }
    }
}

#[cfg(feature = "onnx")]
struct OnnxRuntimeBackend {
    session: std::sync::Mutex<ort::session::Session>,
    device_label: String,
    configured_precision: Option<OnnxTensorPrecision>,
}

#[cfg(feature = "onnx")]
impl OnnxRuntimeBackend {
    fn load(
        config: &Wav2Vec2Config,
        model_cfg: &Wav2Vec2ModelConfig,
    ) -> Result<Self, AlignmentError> {
        let execution_providers = onnx_execution_providers(config.device.as_str())?;
        let configured_precision = OnnxTensorPrecision::from_config(model_cfg.dtype.as_deref())?;
        let session = ort::session::Session::builder()
            .map_err(|e| AlignmentError::runtime("onnx session builder", e))?
            .with_execution_providers(execution_providers)
            .map_err(|e| AlignmentError::runtime("onnx execution providers", e))?
            .commit_from_file(Path::new(&config.model_path))
            .map_err(|e| AlignmentError::runtime("onnx model load", e))?;

        tracing::info!(
            inputs = session.inputs().len(),
            outputs = session.outputs().len(),
            model_path = %config.model_path,
            device = %config.device,
            configured_precision = configured_precision.map(|p| p.label()).unwrap_or("unspecified"),
            "wav2vec2 ONNX runtime loaded"
        );

        let device_label = parse_onnx_device(config.device.as_str())?;
        Ok(Self {
            session: std::sync::Mutex::new(session),
            device_label: device_label.to_string(),
            configured_precision,
        })
    }

    #[allow(dead_code)]
    fn run_raw_logits(&self, normalized_audio: &[f32]) -> Result<OnnxRawLogits, AlignmentError> {
        let input = ort::value::TensorRef::from_array_view((
            [1usize, normalized_audio.len()],
            normalized_audio,
        ))
        .map_err(|e| AlignmentError::runtime("onnx input tensor", e))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| AlignmentError::runtime("onnx session lock", "session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| AlignmentError::runtime("onnx forward pass", e))?;
        if outputs.len() == 0 {
            return Err(AlignmentError::runtime(
                "onnx forward pass",
                "model produced no outputs",
            ));
        }
        let output = &outputs[0];
        extract_onnx_raw_logits(output)
    }

    fn run_forward(&self, normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError> {
        let input = ort::value::TensorRef::from_array_view((
            [1usize, normalized_audio.len()],
            normalized_audio,
        ))
        .map_err(|e| AlignmentError::runtime("onnx input tensor", e))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| AlignmentError::runtime("onnx session lock", "session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| self.map_forward_error(e))?;
        if outputs.len() == 0 {
            return Err(AlignmentError::runtime(
                "onnx forward pass",
                "model produced no outputs",
            ));
        }
        let output = &outputs[0];

        #[cfg(all(feature = "onnx", feature = "cuda-dp"))]
        if self.device_label == "cuda" {
            if let Some(cuda_output) = try_cuda_forward_output(output) {
                return Ok(ForwardOutput::CudaDevice(cuda_output));
            }
        }

        let raw = extract_onnx_raw_logits(output)?;
        let runtime_output = onnx_raw_logits_to_log_probs(raw)?;
        Ok(ForwardOutput::Host(runtime_output))
    }

    fn map_forward_error(&self, err: impl std::fmt::Display) -> AlignmentError {
        let message = err.to_string();
        if self.device_label == "cuda" && is_known_low_precision_cuda_conv_failure(&message) {
            let precision = self
                .configured_precision
                .map(|p| p.label())
                .unwrap_or("low precision");
            return AlignmentError::runtime(
                "onnx forward pass",
                format!(
                    "{message}. ONNX Runtime CUDA failed while executing a {precision} wav2vec2 positional convolution; export a CUDA-safe mixed-precision ONNX artifact or use fp32 for CUDA inference."
                ),
            );
        }

        AlignmentError::runtime("onnx forward pass", message)
    }
}

#[cfg(feature = "onnx")]
impl RuntimeBackend for OnnxRuntimeBackend {
    fn infer(&self, normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError> {
        self.run_forward(normalized_audio)
    }

    fn infer_profiled(
        &self,
        normalized_audio: &[f32],
    ) -> Result<ProfiledForwardOutput, AlignmentError> {
        let forward_started = Instant::now();
        let forward_output = self.run_forward(normalized_audio)?;
        let forward_ms = duration_to_ms(forward_started.elapsed());
        // post_ms = 0 for CudaDevice (zero-copy); for Host the extraction is included in forward
        let post_ms = 0.0;

        Ok(ProfiledForwardOutput {
            forward_output,
            forward_ms,
            post_ms,
        })
    }

    fn device_label(&self) -> String {
        self.device_label.clone()
    }
}

#[cfg(feature = "onnx")]
struct OnnxRawLogits {
    dims: Vec<i64>,
    logits: Vec<f32>,
    dtype: String,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnnxTensorPrecision {
    F16,
    BF16,
    F32,
    F64,
}

#[cfg(feature = "onnx")]
impl OnnxTensorPrecision {
    fn from_config(dtype: Option<&str>) -> Result<Option<Self>, AlignmentError> {
        let Some(dtype) = dtype else {
            return Ok(None);
        };
        let normalized = normalize_precision_label(dtype);
        let precision = match normalized.as_str() {
            "float16" | "f16" | "fp16" => Self::F16,
            "bfloat16" | "bf16" => Self::BF16,
            "float32" | "f32" | "fp32" => Self::F32,
            "float64" | "f64" | "fp64" => Self::F64,
            _ => {
                return Err(AlignmentError::invalid_input(format!(
                    "unsupported ONNX model dtype '{dtype}', expected f32, f16, bf16, or f64"
                )))
            }
        };
        Ok(Some(precision))
    }

    fn from_tensor_element_type(
        ty: ort::tensor::TensorElementType,
    ) -> Result<Self, AlignmentError> {
        match ty {
            ort::tensor::TensorElementType::Float16 => Ok(Self::F16),
            ort::tensor::TensorElementType::Bfloat16 => Ok(Self::BF16),
            ort::tensor::TensorElementType::Float32 => Ok(Self::F32),
            ort::tensor::TensorElementType::Float64 => Ok(Self::F64),
            _ => Err(AlignmentError::invalid_input(format!(
                "unsupported ONNX logits dtype '{ty}', expected f32, f16, bf16, or f64"
            ))),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

#[cfg(feature = "onnx")]
fn normalize_precision_label(dtype: &str) -> String {
    dtype.trim().to_ascii_lowercase().replace(['-', '_'], "")
}

#[cfg(feature = "onnx")]
fn extract_onnx_raw_logits(output: &ort::value::DynValue) -> Result<OnnxRawLogits, AlignmentError> {
    use ort::value::DynTensorValueType;

    let tensor_ref = output.downcast_ref::<DynTensorValueType>().map_err(|e| {
        AlignmentError::runtime(
            "onnx extract logits",
            format!("model output is not a tensor: {e}"),
        )
    })?;
    let precision = OnnxTensorPrecision::from_tensor_element_type(*tensor_ref.data_type())?;
    if !tensor_ref.memory_info().is_cpu_accessible() {
        return Err(AlignmentError::runtime(
            "onnx extract logits",
            format!(
                "ONNX logits are {} on {:?}; host fallback requires CPU-accessible output. The CUDA zero-copy path currently accepts f32 output only; export logits as f32 or use a CUDA-safe mixed-precision artifact.",
                precision.label(),
                tensor_ref.memory_info().allocation_device()
            ),
        ));
    }

    let dtype = precision.label().to_string();
    match precision {
        OnnxTensorPrecision::F16 => {
            let (shape, logits) = output
                .try_extract_tensor::<half::f16>()
                .map_err(|e| AlignmentError::runtime("onnx extract f16 logits", e))?;
            Ok(OnnxRawLogits {
                dims: shape.iter().copied().collect(),
                logits: f16_logits_to_f32(logits),
                dtype,
            })
        }
        OnnxTensorPrecision::BF16 => {
            let (shape, logits) = output
                .try_extract_tensor::<half::bf16>()
                .map_err(|e| AlignmentError::runtime("onnx extract bf16 logits", e))?;
            Ok(OnnxRawLogits {
                dims: shape.iter().copied().collect(),
                logits: bf16_logits_to_f32(logits),
                dtype,
            })
        }
        OnnxTensorPrecision::F32 => {
            let (shape, logits) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| AlignmentError::runtime("onnx extract f32 logits", e))?;
            Ok(OnnxRawLogits {
                dims: shape.iter().copied().collect(),
                logits: logits.to_vec(),
                dtype,
            })
        }
        OnnxTensorPrecision::F64 => {
            let (shape, logits) = output
                .try_extract_tensor::<f64>()
                .map_err(|e| AlignmentError::runtime("onnx extract f64 logits", e))?;
            Ok(OnnxRawLogits {
                dims: shape.iter().copied().collect(),
                logits: f64_logits_to_f32(logits),
                dtype,
            })
        }
    }
}

#[cfg(feature = "onnx")]
fn f16_logits_to_f32(logits: &[half::f16]) -> Vec<f32> {
    logits.iter().map(|v| v.to_f32()).collect()
}

#[cfg(feature = "onnx")]
fn bf16_logits_to_f32(logits: &[half::bf16]) -> Vec<f32> {
    logits.iter().map(|v| v.to_f32()).collect()
}

#[cfg(feature = "onnx")]
fn f64_logits_to_f32(logits: &[f64]) -> Vec<f32> {
    logits.iter().map(|v| *v as f32).collect()
}

#[cfg(feature = "onnx")]
fn onnx_execution_providers(
    device: &str,
) -> Result<Vec<ort::ep::ExecutionProviderDispatch>, AlignmentError> {
    match parse_onnx_device(device)? {
        "cpu" => Ok(vec![ort::ep::CPU::default().build()]),
        "cuda" => Ok(vec![
            ort::ep::CUDA::default()
                .with_device_id(0)
                .build()
                .error_on_failure(),
            ort::ep::CPU::default().build(),
        ]),
        _ => Err(AlignmentError::invalid_input(format!(
            "unsupported ONNX device '{device}', expected 'cpu' or 'cuda'"
        ))),
    }
}

#[cfg(all(feature = "onnx", feature = "cuda-dp"))]
fn try_cuda_forward_output(
    output: &ort::value::DynValue,
) -> Option<crate::pipeline::cuda_forward::CudaLogProbsBuffer> {
    use crate::alignment::viterbi::cuda::log_softmax_logits_to_device;
    use crate::pipeline::cuda_forward::CudaLogProbsBuffer;
    use ort::memory::AllocationDevice;
    use ort::value::DynTensorValueType;

    let tensor_ref = output.downcast_ref::<DynTensorValueType>().ok()?;
    let mem_info = tensor_ref.memory_info();
    if mem_info.allocation_device() != AllocationDevice::CUDA {
        return None;
    }
    if *tensor_ref.data_type() != ort::tensor::TensorElementType::Float32 {
        return None;
    }

    let shape = tensor_ref.shape();
    let dims: &[i64] = shape.as_ref();
    let (t_len, v_len) = match dims {
        [batch, t, v] if *batch == 1 && *t > 0 && *v > 0 => (*t as usize, *v as usize),
        [t, v] if *t > 0 && *v > 0 => (*t as usize, *v as usize),
        _ => return None,
    };

    let ptr = tensor_ref.data_ptr() as *const f32;
    if ptr.is_null() {
        return None;
    }

    let (ctx, slice) = unsafe { log_softmax_logits_to_device(ptr, t_len, v_len)? };
    Some(CudaLogProbsBuffer::new(ctx, slice, t_len, v_len))
}

#[cfg(feature = "onnx")]
fn parse_onnx_device(device: &str) -> Result<&'static str, AlignmentError> {
    if device.eq_ignore_ascii_case("cpu") {
        Ok("cpu")
    } else if device.eq_ignore_ascii_case("cuda") {
        Ok("cuda")
    } else {
        Err(AlignmentError::invalid_input(format!(
            "unsupported ONNX device '{device}', expected 'cpu' or 'cuda'"
        )))
    }
}

#[cfg(feature = "onnx")]
fn is_known_low_precision_cuda_conv_failure(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    let mentions_pos_conv = lower.contains("pos_conv_embed") || lower.contains("convfwd");
    let mentions_engine_failure = lower.contains("cudnn_fe")
        || lower.contains("heuristic_query_failed")
        || lower.contains("no valid engine");
    mentions_pos_conv && mentions_engine_failure
}

#[cfg(feature = "onnx")]
fn onnx_raw_logits_to_log_probs(
    raw: OnnxRawLogits,
) -> Result<RuntimeInferenceOutput, AlignmentError> {
    let (num_frames_t, vocab_size) = parse_onnx_output_shape(&raw.dims, raw.logits.len())?;
    let mut log_probs = Vec::with_capacity(num_frames_t);
    for t in 0..num_frames_t {
        let start = t * vocab_size;
        let end = start + vocab_size;
        log_probs.push(log_softmax_row(&raw.logits[start..end]));
    }
    Ok(RuntimeInferenceOutput {
        log_probs,
        num_frames_t,
        vocab_size,
        dtype: raw.dtype,
    })
}

#[cfg(feature = "onnx")]
fn parse_onnx_output_shape(
    dims: &[i64],
    logits_len: usize,
) -> Result<(usize, usize), AlignmentError> {
    let (num_frames_t, vocab_size) = match dims {
        [batch, t, v] => {
            let batch = non_negative_dim(*batch, "batch")?;
            if batch != 1 {
                return Err(AlignmentError::invalid_input(format!(
                    "ONNX logits batch size must be 1, got {batch}"
                )));
            }
            (positive_dim(*t, "time")?, positive_dim(*v, "vocab")?)
        }
        [t, v] => (positive_dim(*t, "time")?, positive_dim(*v, "vocab")?),
        _ => {
            return Err(AlignmentError::invalid_input(format!(
                "unsupported ONNX logits rank {}; expected [1, T, V] or [T, V]",
                dims.len()
            )));
        }
    };

    let expected_len = num_frames_t
        .checked_mul(vocab_size)
        .ok_or_else(|| AlignmentError::invalid_input("ONNX logits shape is too large"))?;
    if expected_len != logits_len {
        return Err(AlignmentError::invalid_input(format!(
            "ONNX logits shape/data mismatch: shape implies {expected_len} values, got {logits_len}"
        )));
    }
    Ok((num_frames_t, vocab_size))
}

#[cfg(feature = "onnx")]
fn non_negative_dim(value: i64, name: &'static str) -> Result<usize, AlignmentError> {
    if value < 0 {
        return Err(AlignmentError::invalid_input(format!(
            "ONNX output {name} dimension must be >= 0, got {value}"
        )));
    }
    Ok(value as usize)
}

#[cfg(feature = "onnx")]
fn positive_dim(value: i64, name: &'static str) -> Result<usize, AlignmentError> {
    if value <= 0 {
        return Err(AlignmentError::invalid_input(format!(
            "ONNX output {name} dimension must be > 0, got {value}"
        )));
    }
    Ok(value as usize)
}

#[cfg(feature = "onnx")]
fn log_softmax_row(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&x| ((x - max_logit) as f64).exp()).sum();
    let log_denom = if sum_exp > 0.0 && sum_exp.is_finite() {
        max_logit + sum_exp.ln() as f32
    } else {
        f32::INFINITY
    };

    logits.iter().map(|&x| x - log_denom).collect()
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(all(test, feature = "onnx"))]
mod onnx_tests {
    use super::*;

    #[test]
    fn parse_shape_accepts_batched_logits() {
        let (t, v) = parse_onnx_output_shape(&[1, 7, 32], 7 * 32).expect("shape should be valid");
        assert_eq!((t, v), (7, 32));
    }

    #[test]
    fn parse_shape_rejects_non_unit_batch() {
        let err = parse_onnx_output_shape(&[2, 7, 32], 2 * 7 * 32)
            .expect_err("non-unit batch must be rejected");
        assert!(err.to_string().contains("batch size must be 1"));
    }

    #[test]
    fn parse_shape_rejects_len_mismatch() {
        let err = parse_onnx_output_shape(&[7, 32], 7 * 32 - 1)
            .expect_err("shape/data mismatch must be rejected");
        assert!(err.to_string().contains("shape/data mismatch"));
    }

    #[test]
    fn precision_config_accepts_supported_aliases() {
        assert_eq!(
            OnnxTensorPrecision::from_config(Some("float16")).unwrap(),
            Some(OnnxTensorPrecision::F16)
        );
        assert_eq!(
            OnnxTensorPrecision::from_config(Some("bf16")).unwrap(),
            Some(OnnxTensorPrecision::BF16)
        );
        assert_eq!(
            OnnxTensorPrecision::from_config(Some("fp32")).unwrap(),
            Some(OnnxTensorPrecision::F32)
        );
        assert_eq!(
            OnnxTensorPrecision::from_config(Some("float64")).unwrap(),
            Some(OnnxTensorPrecision::F64)
        );
        assert_eq!(OnnxTensorPrecision::from_config(None).unwrap(), None);
    }

    #[test]
    fn precision_config_rejects_unknown_dtype() {
        let err = OnnxTensorPrecision::from_config(Some("int8"))
            .expect_err("unsupported ONNX precision should fail");
        assert!(err.to_string().contains("unsupported ONNX model dtype"));
    }

    #[test]
    fn lower_precision_logits_convert_to_f32() {
        let f16 = [half::f16::from_f32(1.5), half::f16::from_f32(-2.0)];
        assert_eq!(f16_logits_to_f32(&f16), vec![1.5, -2.0]);

        let bf16 = [half::bf16::from_f32(3.0), half::bf16::from_f32(-4.0)];
        assert_eq!(bf16_logits_to_f32(&bf16), vec![3.0, -4.0]);

        assert_eq!(f64_logits_to_f32(&[5.0, -6.0]), vec![5.0, -6.0]);
    }

    #[test]
    fn raw_logits_to_log_probs_preserves_original_dtype_label() {
        let raw = OnnxRawLogits {
            dims: vec![1, 2, 2],
            logits: vec![1.0, 2.0, 3.0, 4.0],
            dtype: "f16".to_string(),
        };
        let out = onnx_raw_logits_to_log_probs(raw).expect("log-softmax should succeed");
        assert_eq!(out.dtype, "f16");
        assert_eq!(out.num_frames_t, 2);
        assert_eq!(out.vocab_size, 2);
    }

    #[test]
    fn known_cuda_conv_failure_is_detected() {
        let message =
            "CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED No valid engine configs for ConvFwd_ /wav2vec2/encoder/pos_conv_embed/conv/Conv";
        assert!(is_known_low_precision_cuda_conv_failure(message));
        assert!(!is_known_low_precision_cuda_conv_failure(
            "unrelated CUDA error"
        ));
    }
}
