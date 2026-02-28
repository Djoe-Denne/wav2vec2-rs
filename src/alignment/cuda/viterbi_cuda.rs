//! CUDA-native Viterbi kernel via cudarc.
//!
//! Zero-copy: reads log_probs directly from ORT's CUDA output pointer.
//! Only the backpointer array (T×S × i32) and 2 floats are copied to host.
//!
//! Feature-gated: `cuda-dp`

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::{Arc, OnceLock};

const KERNEL_SRC: &str = include_str!("viterbi.cu");
const BLOCK_SIZE: u32 = 256;

struct CudaViterbiCtx {
    ctx: Arc<CudaContext>,
    module: Arc<cudarc::driver::CudaModule>,
}

static CTX: OnceLock<Option<CudaViterbiCtx>> = OnceLock::new();

fn get_ctx() -> Option<&'static CudaViterbiCtx> {
    CTX.get_or_init(|| {
        let ctx = CudaContext::new(0).ok()?;
        let ptx = compile_ptx(KERNEL_SRC).ok()?;
        let module = ctx.load_module(ptx).ok()?;
        Some(CudaViterbiCtx { ctx, module })
    })
    .as_ref()
}

/// Synchronize the CUDA device so that all prior GPU work (e.g. ORT inference)
/// is complete. Used by OnnxRuntimeBackend so that forward_ms and dp_ms are
/// attributed correctly in profiling.
pub fn synchronize_cuda_device(context: &'static str) -> Result<(), crate::error::AlignmentError> {
    let ctx = get_ctx()
        .ok_or_else(|| crate::error::AlignmentError::runtime(context, "CUDA context not available"))?;
    ctx.ctx
        .set_blocking_synchronize()
        .map_err(|e| crate::error::AlignmentError::runtime(context, e))?;
    ctx.ctx
        .synchronize()
        .map_err(|e| crate::error::AlignmentError::runtime(context, e))?;
    Ok(())
}

/// Run log_softmax over rows: logits [T,V] -> log_probs [T,V] on device.
/// Returns (context, log_probs_slice) for use in CudaLogProbsBuffer.
///
/// # Safety
/// `logits_ptr` must be a valid CUDA device pointer to t_len * v_len f32 values.
#[allow(dead_code)]
pub(crate) unsafe fn log_softmax_logits_to_device(
    logits_ptr: *const f32,
    t_len: usize,
    v_len: usize,
) -> Option<(Arc<CudaContext>, CudaSlice<f32>)> {
    let ctx = get_ctx()?;
    let stream = ctx.ctx.default_stream();
    if t_len == 0 || v_len == 0 {
        return Some((ctx.ctx.clone(), stream.alloc_zeros(0).ok()?));
    }

    let func = ctx.module.load_function("log_softmax_rows").ok()?;
    let mut out_slice: CudaSlice<f32> = stream.alloc_zeros(t_len * v_len).ok()?;

    let cfg = LaunchConfig {
        block_dim: (256.min(v_len as u32), 1, 1),
        grid_dim: (t_len as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let logits_cu = logits_ptr as cudarc::driver::sys::CUdeviceptr;
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&logits_cu)
            .arg(&mut out_slice)
            .arg(&(t_len as i32))
            .arg(&(v_len as i32))
            .launch(cfg)
            .ok()?;
    }

    Some((ctx.ctx.clone(), out_slice))
}

/// Run Viterbi on GPU, reading log_probs directly from ORT's CUDA output.
///
/// # Arguments
/// * `log_probs_dev_ptr` - Raw CUDA device pointer to ORT's f32 output [T, V]
/// * `t_len` - Number of time frames (T)
/// * `v_len` - Vocab size (V)
/// * `tokens` - Token sequence on host (S elements)
///
/// # Returns
/// `Some(path)` on success, `None` if CUDA unavailable.
///
/// # Safety
/// `log_probs_dev_ptr` must be a valid CUDA device pointer to at least
/// `t_len * v_len` f32 values, and must remain valid until this function returns.
pub unsafe fn forced_align_viterbi_cuda_zerocopy(
    log_probs_dev_ptr: *const f32,
    t_len: usize,
    v_len: usize,
    tokens: &[usize],
) -> Option<Vec<(usize, usize)>> {
    let ctx = get_ctx()?;
    let stream = ctx.ctx.default_stream();
    let s_len = tokens.len();

    if t_len == 0 || s_len == 0 {
        return Some(Vec::new());
    }

    let func: CudaFunction = ctx.module.load_function("viterbi_forward").ok()?;

    // Upload tokens (small: S × 4 bytes, typically < 2KB)
    let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_dev: CudaSlice<i32> = stream.clone_htod(&tokens_i32).ok()?;

    // Allocate device buffers for outputs
    let bp_len = t_len * s_len;
    let mut bp_dev: CudaSlice<i32> = stream.alloc_zeros(bp_len).ok()?;
    let mut out_dev: CudaSlice<f32> = stream.alloc_zeros(2).ok()?;

    let final_floor_state = s_len.saturating_sub(2) as i32;
    let shared_mem = (2 * s_len * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    let log_probs_cu = log_probs_dev_ptr as cudarc::driver::sys::CUdeviceptr;
    // Launch — log_probs_dev_ptr is read directly, zero copy!
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&log_probs_cu)
            .arg(&tokens_dev)
            .arg(&mut bp_dev)
            .arg(&mut out_dev)
            .arg(&(t_len as i32))
            .arg(&(s_len as i32))
            .arg(&(v_len as i32))
            .arg(&final_floor_state)
            .launch(cfg)
            .ok()?;
    }

    // Readback only bp + 2 scores (NOT log_probs — that stays on GPU)
    let bp_host: Vec<i32> = stream.clone_dtoh(&bp_dev).ok()?;
    let out_host: Vec<f32> = stream.clone_dtoh(&out_dev).ok()?;

    // --- Backtrack on CPU ---
    let score_last = out_host[0];
    let score_prev = out_host[1];

    let mut s = s_len - 1;
    if s_len >= 2 && score_prev > score_last {
        s = s_len - 2;
    }

    let mut path = Vec::with_capacity(t_len);
    path.push((s, t_len - 1));
    for t in (1..t_len).rev() {
        let step = bp_host[t * s_len + s];
        s = match step {
            0 => s,
            1 => s - 1,
            2 => s - 2,
            _ => s,
        };
        path.push((s, t - 1));
    }
    path.reverse();

    Some(path)
}

/// Non-zero-copy variant: accepts host-side log_probs, uploads them.
/// Use this when ORT runs on CPU or when the device pointer isn't accessible.
pub fn forced_align_viterbi_cuda(
    log_probs: &[Vec<f32>],
    tokens: &[usize],
) -> Option<Vec<(usize, usize)>> {
    let ctx = get_ctx()?;
    let stream = ctx.ctx.default_stream();
    let t_len = log_probs.len();
    if t_len == 0 || tokens.is_empty() {
        return Some(Vec::new());
    }
    let v_len = log_probs[0].len();

    // Flatten and upload
    let flat: Vec<f32> = log_probs.iter().flat_map(|r| r.iter().copied()).collect();
    let log_probs_dev: CudaSlice<f32> = stream.clone_htod(&flat).ok()?;

    let (ptr, _sync) = log_probs_dev.device_ptr(&stream);
    // Safety: we own log_probs_dev, ptr is valid for the duration of this call
    unsafe {
        forced_align_viterbi_cuda_zerocopy(ptr as *const f32, t_len, v_len, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_matches_cpu() {
        let log_probs = vec![
            vec![-1.0f32, -2.0, -3.0, -0.5],
            vec![-0.5, -1.5, -2.5, -1.0],
            vec![-2.0, -0.3, -1.5, -2.0],
            vec![-1.5, -1.0, -0.5, -2.0],
            vec![-0.8, -1.2, -1.0, -0.3],
        ];
        let tokens = vec![0usize, 3, 1];

        let cpu_path =
            crate::alignment::viterbi::forced_align_viterbi_cpu(&log_probs, &tokens);

        if let Some(cuda_path) = forced_align_viterbi_cuda(&log_probs, &tokens) {
            assert_eq!(cpu_path, cuda_path, "CUDA path must match CPU");
        } else {
            eprintln!("CUDA unavailable, skipping");
        }
    }
}
