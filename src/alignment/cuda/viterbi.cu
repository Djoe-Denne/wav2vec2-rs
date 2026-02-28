// CTC Viterbi forced alignment — CUDA single-block wavefront.
//
// Reads log_probs directly from ORT's CUDA output buffer (zero copy).
// Single block loops over T time steps with __syncthreads().
// Only bp[] and 2 final scores are copied back to host.

// Log-softmax over rows: logits [T,V] -> log_probs [T,V].
// One block per row. Used when ORT outputs logits and we need log_probs on GPU.
extern "C" __global__ void log_softmax_rows(
    const float* __restrict__ logits,
    float* __restrict__ log_probs,
    int t_len,
    int v_len
) {
    int t = blockIdx.x;
    if (t >= t_len) return;
    const float* row_in = logits + t * v_len;
    float* row_out = log_probs + t * v_len;

    // Step 1: find max (block reduction)
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < v_len; i += blockDim.x) {
        float v = row_in[i];
        if (v > max_val) max_val = v;
    }
    __shared__ float smem_max[256];
    smem_max[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s && smem_max[threadIdx.x + s] > smem_max[threadIdx.x]) {
            smem_max[threadIdx.x] = smem_max[threadIdx.x + s];
        }
        __syncthreads();
    }
    max_val = smem_max[0];

    // Step 2: sum exp(x - max)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < v_len; i += blockDim.x) {
        sum += expf(row_in[i] - max_val);
    }
    __shared__ float smem_sum[256];
    smem_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) smem_sum[threadIdx.x] += smem_sum[threadIdx.x + s];
        __syncthreads();
    }
    float total = smem_sum[0];
    float log_denom = max_val + logf(total);

    // Step 3: write output
    for (int i = threadIdx.x; i < v_len; i += blockDim.x) {
        row_out[i] = row_in[i] - log_denom;
    }
}

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

// Backtrack on GPU: single thread, O(T), reads bp + scores, writes path_out[T].
// Caller downloads only path_out (T × 4 bytes) instead of bp (T×S × 4 bytes).
extern "C" __global__ void viterbi_backtrace(
    const int* __restrict__ bp,         // [T, S]
    const float* __restrict__ scores,   // [2] final scores
    int* __restrict__ path_out,         // [T] output: state index per frame
    int t_len,
    int s_len
) {
    if (threadIdx.x != 0) return;

    int s = s_len - 1;
    if (s_len >= 2 && scores[1] > scores[0]) {
        s = s_len - 2;
    }

    path_out[t_len - 1] = s;
    for (int t = t_len - 1; t >= 1; t--) {
        int step = bp[t * s_len + s];
        s -= step;
        path_out[t - 1] = s;
    }
}
