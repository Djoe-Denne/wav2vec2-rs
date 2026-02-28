//! CUDA forward output buffer for zero-copy Viterbi.
//!
//! Holds log_probs on device; provides device pointer for Viterbi and
//! host copy for grouping.
//!
//! Feature-gated: `cuda-dp`

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use std::sync::Arc;

use crate::error::AlignmentError;
use crate::pipeline::traits::RuntimeInferenceOutput;

/// Owned CUDA buffer of log_probs [T, V]. Keeps device memory alive and
/// provides device pointer for zero-copy Viterbi plus host copy for grouping.
#[derive(Debug)]
pub struct CudaLogProbsBuffer {
    ctx: Arc<CudaContext>,
    slice: CudaSlice<f32>,
    pub t_len: usize,
    pub v_len: usize,
}

impl CudaLogProbsBuffer {
    /// Create from an existing device buffer (e.g. after log_softmax kernel).
    pub fn new(ctx: Arc<CudaContext>, slice: CudaSlice<f32>, t_len: usize, v_len: usize) -> Self {
        Self {
            ctx,
            slice,
            t_len,
            v_len,
        }
    }

    /// Run zero-copy Viterbi using this buffer's device pointer.
    pub fn run_viterbi(&self, tokens: &[usize]) -> Option<Vec<(usize, usize)>> {
        let stream = self.ctx.default_stream();
        let (ptr, _sync) = self.slice.device_ptr(&stream);
        unsafe {
            crate::alignment::viterbi::cuda::forced_align_viterbi_cuda_zerocopy(
                ptr as *const f32,
                self.t_len,
                self.v_len,
                tokens,
            )
        }
    }

    /// Copy to host and return RuntimeInferenceOutput for grouping.
    pub fn to_runtime_inference_output(self) -> Result<RuntimeInferenceOutput, AlignmentError> {
        let stream = self.ctx.default_stream();
        let flat = stream
            .clone_dtoh(&self.slice)
            .map_err(|e| AlignmentError::runtime("cuda log_probs copy to host", e))?;

        let mut log_probs = Vec::with_capacity(self.t_len);
        for t in 0..self.t_len {
            let start = t * self.v_len;
            let end = start + self.v_len;
            log_probs.push(flat[start..end].to_vec());
        }

        Ok(RuntimeInferenceOutput {
            log_probs,
            num_frames_t: self.t_len,
            vocab_size: self.v_len,
            dtype: "f32".to_string(),
        })
    }
}
