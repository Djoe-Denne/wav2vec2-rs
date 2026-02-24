pub fn forced_align_viterbi(log_probs: &[Vec<f32>], tokens: &[usize]) -> Vec<(usize, usize)> {
    let t_len = log_probs.len();
    let s_len = tokens.len();
    if t_len == 0 || s_len == 0 {
        return Vec::new();
    }

    let mut dp = vec![vec![f32::NEG_INFINITY; s_len]; t_len];
    let mut bp = vec![vec![0usize; s_len]; t_len];

    dp[0][0] = log_probs[0][tokens[0]];
    if s_len > 1 {
        dp[0][1] = log_probs[0][tokens[1]];
    }

    for t in 1..t_len {
        for s in 0..s_len {
            let emit = log_probs[t][tokens[s]];
            let mut best = dp[t - 1][s];
            let mut from = s;

            if s >= 1 && dp[t - 1][s - 1] > best {
                best = dp[t - 1][s - 1];
                from = s - 1;
            }
            if s >= 2 && tokens[s] != tokens[s - 2] && dp[t - 1][s - 2] > best {
                best = dp[t - 1][s - 2];
                from = s - 2;
            }

            dp[t][s] = best + emit;
            bp[t][s] = from;
        }
    }

    let mut s = s_len - 1;
    if s_len >= 2 && dp[t_len - 1][s_len - 2] > dp[t_len - 1][s_len - 1] {
        s = s_len - 2;
    }

    let mut path = vec![(s, t_len - 1)];
    for t in (1..t_len).rev() {
        s = bp[t][s];
        path.push((s, t - 1));
    }
    path.reverse();
    path
}
