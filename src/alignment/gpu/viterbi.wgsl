// CTC Viterbi forced alignment — single-workgroup wavefront.
//
// One workgroup loops over T time steps. Each thread handles one or more
// states (strided). workgroupBarrier() synchronizes between steps.
// Backpointers are written to a storage buffer; final scores are written
// to a small output buffer so the CPU can pick the backtrack start.

struct Params {
    t_len: u32,
    s_len: u32,
    v_len: u32,
    final_floor_state: u32,
}

@group(0) @binding(0) var<storage, read>       log_probs : array<f32>;  // T × V, row-major
@group(0) @binding(1) var<storage, read>       tokens    : array<u32>;  // S
@group(0) @binding(2) var<uniform>             params    : Params;
@group(0) @binding(3) var<storage, read_write> bp        : array<u32>;  // T × S
@group(0) @binding(4) var<storage, read_write> scores    : array<f32>;  // 2 × S (ping-pong)
@group(0) @binding(5) var<storage, read_write> out       : array<f32>;  // [score_last, score_prev]
@group(0) @binding(6) var<storage, read_write> path      : array<u32>;  // [T] state index per frame

const NEG_INF: f32 = -1.0e30;
const WG_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn viterbi_main(@builtin(local_invocation_index) lid: u32) {
    let t_len = params.t_len;
    let s_len = params.s_len;
    let v_len = params.v_len;
    let final_floor = params.final_floor_state;

    // --- Initialize both score rows to -inf ---
    for (var s = lid; s < s_len; s += WG_SIZE) {
        scores[s] = NEG_INF;
        scores[s_len + s] = NEG_INF;
    }
    workgroupBarrier();

    // --- t=0: seed first 1–2 states ---
    if lid == 0u {
        scores[0] = log_probs[tokens[0]];
        if s_len > 1u {
            scores[1] = log_probs[tokens[1]];
        }
    }
    workgroupBarrier();

    var prev_off = 0u;
    var curr_off = s_len;

    // --- Main DP loop: t = 1..T-1 ---
    for (var t = 1u; t < t_len; t++) {
        let remaining = t_len - 1u - t;
        let row_off = t * v_len;

        // Reachability band (mirrors CPU logic).
        var curr_start = 0u;
        if final_floor > 2u * remaining {
            curr_start = final_floor - 2u * remaining;
        }
        let curr_end = min(2u * t + 1u, s_len - 1u);

        // Each thread processes states in stride.
        for (var s = lid + curr_start; s <= curr_end; s += WG_SIZE) {
            let emit = log_probs[row_off + tokens[s]];
            var best = NEG_INF;
            var step = 0u;

            // Stay: s ← s
            let stay = scores[prev_off + s];
            if stay > best {
                best = stay;
                step = 0u;
            }

            // Step from s-1
            if s >= 1u {
                let cand = scores[prev_off + s - 1u];
                if cand > best {
                    best = cand;
                    step = 1u;
                }
            }

            // Step from s-2 (only if tokens differ — CTC constraint)
            if s >= 2u && tokens[s] != tokens[s - 2u] {
                let cand = scores[prev_off + s - 2u];
                if cand > best {
                    best = cand;
                    step = 2u;
                }
            }

            scores[curr_off + s] = best + emit;
            bp[t * s_len + s] = step;
        }

        workgroupBarrier();

        // Ping-pong swap.
        let tmp = prev_off;
        prev_off = curr_off;
        curr_off = tmp;
    }

    // --- Write final two scores and inline backtrack (single thread) ---
    if lid == 0u {
        out[0] = scores[prev_off + s_len - 1u];
        if s_len >= 2u {
            out[1] = scores[prev_off + s_len - 2u];
        } else {
            out[1] = NEG_INF;
        }

        // Inline backtrack: write path[t] = state at time t (only T×4 bytes readback)
        var s = s_len - 1u;
        if s_len >= 2u && out[1] > out[0] {
            s = s_len - 2u;
        }
        path[t_len - 1u] = s;
        for (var t = t_len - 1u; t >= 1u; t--) {
            s -= bp[t * s_len + s];
            path[t - 1u] = s;
        }
    }
}
