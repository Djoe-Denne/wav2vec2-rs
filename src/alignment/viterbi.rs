#[cfg(feature = "cuda-dp")]
#[path = "cuda/viterbi_cuda.rs"]
pub mod cuda;

#[cfg(feature = "wgpu-dp")]
#[path = "gpu/viterbi_gpu.rs"]
pub mod gpu;

/// GPU DP threshold: below this T×S, CPU is faster than GPU launch overhead.
const GPU_DP_THRESHOLD: usize = 40_000;

/// Attempt GPU Viterbi when available (wgpu then cuda). Returns None to fall back to CPU.
#[cfg(any(feature = "wgpu-dp", feature = "cuda-dp"))]
fn try_gpu_viterbi(log_probs: &[Vec<f32>], tokens: &[usize]) -> Option<Vec<(usize, usize)>> {
    #[cfg(feature = "wgpu-dp")]
    {
        if let Some(path) = gpu::forced_align_viterbi_gpu(log_probs, tokens) {
            return Some(path);
        }
        tracing::debug!("wgpu Viterbi unavailable, falling back to CPU");
    }
    #[cfg(feature = "cuda-dp")]
    {
        if let Some(path) = cuda::forced_align_viterbi_cuda(log_probs, tokens) {
            return Some(path);
        }
        tracing::debug!("cuda Viterbi unavailable, falling back to CPU");
    }
    None
}

/// CTC Viterbi forced alignment.
///
/// Dispatch priority:
/// 1. `cuda-dp` zero-copy (reads ORT output directly on GPU — no transfer)
/// 2. `wgpu-dp` wgpu (Vulkan/DX12/Metal — needs host log_probs)
/// 3. CPU fallback (always available)
pub fn forced_align_viterbi(log_probs: &[Vec<f32>], tokens: &[usize]) -> Vec<(usize, usize)> {
    let ts_product = log_probs.len() * tokens.len();
    if ts_product >= GPU_DP_THRESHOLD {
        #[cfg(any(feature = "wgpu-dp", feature = "cuda-dp"))]
        if let Some(path) = try_gpu_viterbi(log_probs, tokens) {
            return path;
        }
    }
    forced_align_viterbi_cpu(log_probs, tokens)
}

/// CPU-only CTC Viterbi (always available).
#[allow(clippy::needless_range_loop)]
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
            1 => {
                debug_assert!(s >= 1);
                s - 1
            }
            2 => {
                debug_assert!(s >= 2);
                s - 2
            }
            _ => s,
        };
        path.push((s, t - 1));
    }
    path.reverse();
    path
}

/// Updates (best, step) if prev[p] is in range and better than best.
#[inline(always)]
fn consider_transition(
    prev: &[f32],
    prev_start: usize,
    prev_end: usize,
    best: f32,
    step: u8,
    p: usize,
    new_step: u8,
) -> (f32, u8) {
    if p >= prev_start && p <= prev_end {
        let cand = prev[p];
        if cand > best {
            return (cand, new_step);
        }
    }
    (best, step)
}

#[inline(always)]
fn best_transition(
    prev: &[f32],
    s: usize,
    prev_start: usize,
    prev_end: usize,
    tokens: &[usize],
) -> (f32, u8) {
    let (best, step) = consider_transition(prev, prev_start, prev_end, f32::NEG_INFINITY, 0, s, 0);
    let (best, step) = if s >= 1 {
        consider_transition(prev, prev_start, prev_end, best, step, s - 1, 1)
    } else {
        (best, step)
    };
    let (best, step) = if s >= 2 && tokens[s] != tokens[s - 2] {
        consider_transition(prev, prev_start, prev_end, best, step, s - 2, 2)
    } else {
        (best, step)
    };
    (best, step)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log_probs(
        t_len: usize,
        vocab_size: usize,
        path: &[(usize, usize)],
        tokens: &[usize],
    ) -> Vec<Vec<f32>> {
        let low: f32 = -10.0;
        let high: f32 = 0.0;
        let mut log_probs = vec![vec![low; vocab_size]; t_len];
        for (s, t) in path {
            if *t < t_len && *s < tokens.len() {
                let tid = tokens[*s];
                if tid < vocab_size {
                    log_probs[*t][tid] = high;
                }
            }
        }
        log_probs
    }

    #[test]
    fn empty_log_probs_returns_empty_path() {
        let path = forced_align_viterbi_cpu(&[], &[0]);
        assert!(path.is_empty());
    }

    #[test]
    fn empty_tokens_returns_empty_path() {
        let path = forced_align_viterbi_cpu(&[vec![0.0f32; 4]], &[]);
        assert!(path.is_empty());
    }

    #[test]
    fn single_frame_single_token() {
        let log_probs = vec![vec![0.0f32, -10.0, -10.0]];
        let tokens = vec![0];
        let path = forced_align_viterbi_cpu(&log_probs, &tokens);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], (0, 0));
    }

    #[test]
    fn two_frames_two_tokens_straight_path() {
        let log_probs = vec![vec![0.0f32, -10.0, -10.0], vec![-10.0, 0.0f32, -10.0]];
        let tokens = vec![0, 1];
        let path = forced_align_viterbi_cpu(&log_probs, &tokens);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], (0, 0));
        assert_eq!(path[1], (1, 1));
        assert!(path[0].1 <= path[1].1);
    }

    #[test]
    fn s_len_one() {
        let t_len = 4;
        let vocab_size = 4;
        let tokens = vec![0];
        let path_states = vec![(0, 0), (0, 1), (0, 2), (0, 3)];
        let log_probs = make_log_probs(t_len, vocab_size, &path_states, &tokens);
        let path = forced_align_viterbi_cpu(&log_probs, &tokens);
        assert_eq!(path.len(), t_len);
        for (i, &(s, t)) in path.iter().enumerate() {
            assert_eq!(s, 0);
            assert_eq!(t, i);
        }
    }

    #[test]
    fn backtrack_step_one() {
        let tokens = vec![0, 1];
        let log_probs = vec![vec![0.0f32, -10.0], vec![-10.0, 0.0f32]];
        let path = forced_align_viterbi_cpu(&log_probs, &tokens);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].1, 0);
        assert_eq!(path[1].1, 1);
    }

    #[test]
    fn backtrack_step_two() {
        let tokens = vec![0, 1, 2];
        let log_probs = vec![
            vec![0.0f32, -10.0, -10.0],
            vec![0.0f32, -10.0, -10.0],
            vec![-10.0, -10.0, 0.0f32],
        ];
        let path = forced_align_viterbi_cpu(&log_probs, &tokens);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], (0, 0));
        assert_eq!(path[1], (0, 1));
        assert_eq!(path[2], (2, 2));
    }

    #[test]
    fn final_state_prefer_s_len_minus_2() {
        let tokens = vec![0, 1, 2];
        let log_probs = vec![
            vec![0.0f32, -10.0, -10.0],
            vec![-10.0, 0.0f32, -10.0],
            vec![-10.0, 0.0f32, -10.0],
            vec![-100.0, 0.0f32, -100.0],
        ];
        let path = forced_align_viterbi_cpu(&log_probs, &tokens);
        assert_eq!(path.len(), 4);
        assert_eq!(
            path[3],
            (1, 3),
            "last state should prefer s_len-2 when prev[1] > prev[2]"
        );
    }

    #[test]
    fn forced_align_viterbi_cpu_path_equals_public() {
        let log_probs = vec![vec![0.0f32, -10.0], vec![-10.0, 0.0f32]];
        let tokens = vec![0, 1];
        let cpu_path = forced_align_viterbi_cpu(&log_probs, &tokens);
        let pub_path = forced_align_viterbi(&log_probs, &tokens);
        assert_eq!(cpu_path, pub_path);
    }
}
