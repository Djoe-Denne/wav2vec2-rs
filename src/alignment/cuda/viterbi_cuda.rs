//! CUDA-native Viterbi kernel via cudarc.
//!
//! Zero-copy: reads log_probs directly from ORT's CUDA output pointer.
//! Only the backpointer array (T×S × i32) and 2 floats are copied to host.
//!
//! Feature-gated: `cuda-dp`

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::{Arc, OnceLock};

const KERNEL_SRC: &str = include_str!("viterbi.cu");
const BLOCK_SIZE: u32 = 256;

struct CudaViterbiCtx {
    dev: Arc<CudaDevice>,
}

static CTX: OnceLock<Option<CudaViterbiCtx>> = OnceLock::new();

fn get_ctx() -> Option<&'static CudaViterbiCtx> {
    CTX.get_or_init(|| {
        let dev = CudaDevice::new(0).ok()?;
        let ptx = compile_ptx(KERNEL_SRC).ok()?;
        dev.load_ptx(ptx, "viterbi", &["viterbi_forward"]).ok()?;
        Some(CudaViterbiCtx { dev })
    })
    .as_ref()
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
    let dev = &ctx.dev;
    let s_len = tokens.len();

    if t_len == 0 || s_len == 0 {
        return Some(Vec::new());
    }

    let func: CudaFunction = dev.get_func("viterbi", "viterbi_forward").ok()?;

    // Upload tokens (small: S × 4 bytes, typically < 2KB)
    let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_dev: CudaSlice<i32> = dev.htod_copy(tokens_i32).ok()?;

    // Allocate device buffers for outputs
    let bp_len = t_len * s_len;
    let mut bp_dev: CudaSlice<i32> = dev.alloc_zeros(bp_len).ok()?;
    let mut out_dev: CudaSlice<f32> = dev.alloc_zeros(2).ok()?;

    let final_floor_state = s_len.saturating_sub(2) as i32;
    let shared_mem = (2 * s_len * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    // Launch — log_probs_dev_ptr is read directly, zero copy!
    unsafe {
        func.launch(
            cfg,
            (
                log_probs_dev_ptr,          // ORT's device pointer, no copy
                &tokens_dev,
                &mut bp_dev,
                &mut out_dev,
                t_len as i32,
                s_len as i32,
                v_len as i32,
                final_floor_state,
            ),
        )
    }
    .ok()?;

    // Readback only bp + 2 scores (NOT log_probs — that stays on GPU)
    let bp_host: Vec<i32> = dev.dtoh_sync_copy(&bp_dev).ok()?;
    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out_dev).ok()?;

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
    let t_len = log_probs.len();
    if t_len == 0 || tokens.is_empty() {
        return Some(Vec::new());
    }
    let v_len = log_probs[0].len();

    // Flatten and upload
    let flat: Vec<f32> = log_probs.iter().flat_map(|r| r.iter().copied()).collect();
    let log_probs_dev: CudaSlice<f32> = ctx.dev.htod_copy(flat).ok()?;

    // Safety: we own log_probs_dev, it's valid for the duration of this call
    unsafe {
        forced_align_viterbi_cuda_zerocopy(
            *log_probs_dev.device_ptr() as *const f32,
            t_len,
            v_len,
            tokens,
        )
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
