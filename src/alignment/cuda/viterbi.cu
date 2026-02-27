// CTC Viterbi forced alignment — CUDA single-block wavefront.
//
// Reads log_probs directly from ORT's CUDA output buffer (zero copy).
// Single block loops over T time steps with __syncthreads().
// Only bp[] and 2 final scores are copied back to host.

extern "C" __global__ void viterbi_forward(
    const float* __restrict__ log_probs,   // [T, V] row-major — ORT device ptr
    const int*   __restrict__ tokens,      // [S]
    int*         __restrict__ bp,          // [T, S] output backpointers
    float*       __restrict__ out_scores,  // [2] final scores for backtrack start
    int t_len,
    int s_len,
    int v_len,
    int final_floor_state
) {
    // Shared memory ping-pong score rows: 2 × S floats.
    // Declared dynamically, sized at launch: 2 * s_len * sizeof(float).
    extern __shared__ float smem[];

    float* prev = smem;
    float* curr = smem + s_len;

    const int lid = threadIdx.x;
    const int stride = blockDim.x;
    const float NEG_INF = -1.0e30f;

    // --- Initialize both rows to -inf ---
    for (int s = lid; s < s_len; s += stride) {
        prev[s] = NEG_INF;
        curr[s] = NEG_INF;
    }
    __syncthreads();

    // --- t=0: seed first 1-2 states ---
    if (lid == 0) {
        prev[0] = log_probs[tokens[0]];
        if (s_len > 1) {
            prev[1] = log_probs[tokens[1]];
        }
    }
    __syncthreads();

    // --- Main DP: t = 1..T-1 ---
    for (int t = 1; t < t_len; t++) {
        const float* row = log_probs + t * v_len;
        const int remaining = t_len - 1 - t;

        // Reachability band (mirrors CPU/WGSL logic)
        int curr_start = final_floor_state - 2 * remaining;
        if (curr_start < 0) curr_start = 0;
        int curr_end = 2 * t + 1;
        if (curr_end >= s_len) curr_end = s_len - 1;

        const int bp_base = t * s_len;

        for (int s = lid + curr_start; s <= curr_end; s += stride) {
            const float emit = row[tokens[s]];
            float best = NEG_INF;
            int step = 0;

            // Stay: s ← s
            float cand = prev[s];
            if (cand > best) {
                best = cand;
                step = 0;
            }

            // From s-1
            if (s >= 1) {
                cand = prev[s - 1];
                if (cand > best) {
                    best = cand;
                    step = 1;
                }
            }

            // From s-2 (only if tokens differ — CTC skip constraint)
            if (s >= 2 && tokens[s] != tokens[s - 2]) {
                cand = prev[s - 2];
                if (cand > best) {
                    best = cand;
                    step = 2;
                }
            }

            curr[s] = best + emit;
            bp[bp_base + s] = step;
        }

        __syncthreads();

        // Ping-pong swap via pointer swap
        float* tmp = prev;
        prev = curr;
        curr = tmp;

        __syncthreads();
    }

    // --- Write final two scores ---
    if (lid == 0) {
        out_scores[0] = prev[s_len - 1];
        out_scores[1] = (s_len >= 2) ? prev[s_len - 2] : NEG_INF;
    }
}
