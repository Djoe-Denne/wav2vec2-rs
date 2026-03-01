#[cfg(feature = "cuda-dp")]
#[path = "cuda/viterbi_cuda.rs"]
pub mod cuda;

#[cfg(feature = "gpu-dp")]
#[path = "gpu/viterbi_gpu.rs"]
pub mod gpu;

/// GPU DP threshold: below this T×S, CPU is faster than GPU launch overhead.
const GPU_DP_THRESHOLD: usize = 40_000;

/// CTC Viterbi forced alignment.
///
/// Dispatch priority:
/// 1. `cuda-dp` zero-copy (reads ORT output directly on GPU — no transfer)
/// 2. `gpu-dp` wgpu (Vulkan/DX12/Metal — needs host log_probs)
/// 3. CPU fallback (always available)
pub fn forced_align_viterbi(log_probs: &[Vec<f32>], tokens: &[usize]) -> Vec<(usize, usize)> {
    let ts_product = log_probs.len() * tokens.len();

    if ts_product >= GPU_DP_THRESHOLD {
        #[cfg(feature = "gpu-dp")]
        {
            if let Some(path) = gpu::forced_align_viterbi_gpu(log_probs, tokens) {
                return path;
            }
            tracing::debug!("wgpu Viterbi unavailable, falling back to CPU");
        }
        #[cfg(feature = "cuda-dp")]	
        {
            if let Some(path) = cuda::forced_align_viterbi_cuda(log_probs, tokens) {
                return path;
            }
            tracing::debug!("cuda Viterbi unavailable, falling back to CPU");
        }
    }

    forced_align_viterbi_cpu(log_probs, tokens)
}


/// CPU-only CTC Viterbi (always available).
pub fn forced_align_viterbi_cpu(log_probs: &[Vec<f32>], tokens: &[usize]) -> Vec<(usize, usize)> {
    let t_len = log_probs.len();
    let s_len = tokens.len();
    if t_len == 0 || s_len == 0 {
        return Vec::new();
    }

    let mut prev = vec![f32::NEG_INFINITY; s_len];
    let mut curr = vec![f32::NEG_INFINITY; s_len];
    let mut bp = vec![0u8; t_len * s_len];

    prev[0] = log_probs[0][tokens[0]];
    if s_len > 1 {
        prev[1] = log_probs[0][tokens[1]];
    }

    let mut prev_start = 0usize;
    let mut prev_end = if s_len > 1 { 1 } else { 0 };
    let final_floor_state = s_len.saturating_sub(2);

    for t in 1..t_len {
        let row = &log_probs[t];
        let remaining = t_len - 1 - t;
        let curr_start = final_floor_state.saturating_sub(2 * remaining);
        let curr_end = (2 * t + 1).min(s_len - 1);

        let bp_offset = t * s_len;
        for s in curr_start..=curr_end {
            let emit = row[tokens[s]];
            let (best, step) = best_transition(&prev, s, prev_start, prev_end, tokens);
            curr[s] = best + emit;
            bp[bp_offset + s] = step;
        }

        std::mem::swap(&mut prev, &mut curr);
        prev_start = curr_start;
        prev_end = curr_end;
    }

    let mut s = s_len - 1;
    if s_len >= 2 && prev[s_len - 2] > prev[s_len - 1] {
        s = s_len - 2;
    }

    let mut path = Vec::with_capacity(t_len);
    path.push((s, t_len - 1));
    for t in (1..t_len).rev() {
        s = match bp[t * s_len + s] {
            0 => s,
            1 => { debug_assert!(s >= 1); s - 1 }
            2 => { debug_assert!(s >= 2); s - 2 }
            _ => s,
        };
        path.push((s, t - 1));
    }
    path.reverse();
    path
}

#[inline(always)]
fn best_transition(
    prev: &[f32],
    s: usize,
    prev_start: usize,
    prev_end: usize,
    tokens: &[usize],
) -> (f32, u8) {
    let mut best = f32::NEG_INFINITY;
    let mut step = 0u8;

    if s >= prev_start && s <= prev_end {
        best = prev[s];
    }

    if s >= 1 {
        let p = s - 1;
        if p >= prev_start && p <= prev_end {
            let cand = prev[p];
            if cand > best {
                best = cand;
                step = 1;
            }
        }
    }

    if s >= 2 && tokens[s] != tokens[s - 2] {
        let p = s - 2;
        if p >= prev_start && p <= prev_end {
            let cand = prev[p];
            if cand > best {
                best = cand;
                step = 2;
            }
        }
    }

    (best, step)
}