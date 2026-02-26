pub fn forced_align_viterbi(log_probs: &[Vec<f32>], tokens: &[usize]) -> Vec<(usize, usize)> {
    const STEP_STAY: u8 = 0;
    const STEP_PREV_1: u8 = 1;
    const STEP_PREV_2: u8 = 2;

    let t_len = log_probs.len();
    let s_len = tokens.len();
    if t_len == 0 || s_len == 0 {
        return Vec::new();
    }

    // Two rolling score rows keep hot data cache-friendly.
    let mut prev = vec![f32::NEG_INFINITY; s_len];
    let mut curr = vec![f32::NEG_INFINITY; s_len];
    // Compact transition-steps are enough to reconstruct the full path:
    // 0=stay, 1=from s-1, 2=from s-2.
    let mut bp = vec![STEP_STAY; t_len * s_len];

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

        // Reachability band:
        // - forward: from start, at most +2 states per frame (+1 offset at t=0)
        // - backward: enough frames left to still reach one of final states
        let curr_start = final_floor_state.saturating_sub(2 * remaining);
        let curr_end = (2 * t + 1).min(s_len - 1);

        for s in curr_start..=curr_end {
            let emit = row[tokens[s]];
            let mut best = f32::NEG_INFINITY;
            let mut step = STEP_STAY;

            if s >= prev_start && s <= prev_end {
                best = prev[s];
                step = STEP_STAY;
            }

            if s >= 1 {
                let p = s - 1;
                if p >= prev_start && p <= prev_end {
                    let cand = prev[p];
                    if cand > best {
                        best = cand;
                        step = STEP_PREV_1;
                    }
                }
            }

            if s >= 2 && tokens[s] != tokens[s - 2] {
                let p = s - 2;
                if p >= prev_start && p <= prev_end {
                    let cand = prev[p];
                    if cand > best {
                        best = cand;
                        step = STEP_PREV_2;
                    }
                }
            }

            curr[s] = best + emit;
            bp[t * s_len + s] = step;
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
            STEP_STAY => s,
            STEP_PREV_1 => {
                debug_assert!(s >= 1);
                s - 1
            }
            STEP_PREV_2 => {
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
