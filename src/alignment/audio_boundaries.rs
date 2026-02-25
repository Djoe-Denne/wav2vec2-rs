const BASELINE_FRAMES: usize = 10;
const STRICT_MIN_CONSEC_FRAMES: usize = 3;
const LENIENT_MIN_CONSEC_FRAMES: usize = 2;
const STRICT_THRESHOLD_MULTIPLIER: f32 = 4.0;
const LENIENT_THRESHOLD_MULTIPLIER: f32 = 3.0;
const STRICT_MIN_THRESHOLD: f32 = 0.01;
const LENIENT_MIN_THRESHOLD: f32 = 0.0075;
const ONSET_STRICT_EARLY_WINDOW_FRAMES: usize = 25;
const OFFSET_STRICT_LATE_WINDOW_FRAMES: usize = 25;

pub(crate) fn detect_audio_onset_frame(
    samples: &[f32],
    sample_rate_hz: u32,
    frame_stride_ms: f64,
) -> Option<usize> {
    let frame_rms = compute_frame_rms(samples, sample_rate_hz, frame_stride_ms)?;

    let noise_floor = front_noise_floor(&frame_rms);
    let strict_threshold = (noise_floor * STRICT_THRESHOLD_MULTIPLIER).max(STRICT_MIN_THRESHOLD);
    let strict_onset =
        first_run_above_threshold(&frame_rms, strict_threshold, STRICT_MIN_CONSEC_FRAMES);
    if let Some(strict) = strict_onset {
        if strict <= ONSET_STRICT_EARLY_WINDOW_FRAMES {
            return Some(strict);
        }
    }

    let lenient_threshold = (noise_floor * LENIENT_THRESHOLD_MULTIPLIER).max(LENIENT_MIN_THRESHOLD);
    let lenient_onset =
        first_run_above_threshold(&frame_rms, lenient_threshold, LENIENT_MIN_CONSEC_FRAMES);

    match (strict_onset, lenient_onset) {
        (None, fallback) => fallback,
        (Some(strict), Some(fallback))
            if strict > ONSET_STRICT_EARLY_WINDOW_FRAMES && fallback < strict =>
        {
            Some(fallback)
        }
        (Some(strict), _) => Some(strict),
    }
}

pub(crate) fn detect_audio_offset_frame(
    samples: &[f32],
    sample_rate_hz: u32,
    frame_stride_ms: f64,
) -> Option<usize> {
    let frame_rms = compute_frame_rms(samples, sample_rate_hz, frame_stride_ms)?;

    let noise_floor = back_noise_floor(&frame_rms);
    let strict_threshold = (noise_floor * STRICT_THRESHOLD_MULTIPLIER).max(STRICT_MIN_THRESHOLD);
    let strict_offset =
        last_run_above_threshold(&frame_rms, strict_threshold, STRICT_MIN_CONSEC_FRAMES);
    if let Some(strict) = strict_offset {
        if is_late_enough_for_strict_offset(strict, frame_rms.len()) {
            return Some(strict);
        }
    }

    let lenient_threshold = (noise_floor * LENIENT_THRESHOLD_MULTIPLIER).max(LENIENT_MIN_THRESHOLD);
    let lenient_offset =
        last_run_above_threshold(&frame_rms, lenient_threshold, LENIENT_MIN_CONSEC_FRAMES);

    match (strict_offset, lenient_offset) {
        (None, fallback) => fallback,
        (Some(strict), Some(fallback))
            if !is_late_enough_for_strict_offset(strict, frame_rms.len()) && fallback > strict =>
        {
            Some(fallback)
        }
        (Some(strict), _) => Some(strict),
    }
}

fn front_noise_floor(frame_rms: &[f32]) -> f32 {
    let baseline_frames = frame_rms.len().min(BASELINE_FRAMES);
    frame_rms.iter().take(baseline_frames).copied().sum::<f32>() / baseline_frames as f32
}

fn back_noise_floor(frame_rms: &[f32]) -> f32 {
    let baseline_frames = frame_rms.len().min(BASELINE_FRAMES);
    frame_rms
        .iter()
        .rev()
        .take(baseline_frames)
        .copied()
        .sum::<f32>()
        / baseline_frames as f32
}

fn first_run_above_threshold(
    frame_rms: &[f32],
    threshold: f32,
    min_consec_frames: usize,
) -> Option<usize> {
    let mut run_start = 0usize;
    let mut run_len = 0usize;
    for (frame_idx, rms) in frame_rms.iter().copied().enumerate() {
        if rms >= threshold {
            if run_len == 0 {
                run_start = frame_idx;
            }
            run_len += 1;
            if run_len >= min_consec_frames {
                return Some(run_start);
            }
            continue;
        }
        run_len = 0;
    }
    None
}

fn last_run_above_threshold(
    frame_rms: &[f32],
    threshold: f32,
    min_consec_frames: usize,
) -> Option<usize> {
    let mut run_end = 0usize;
    let mut run_len = 0usize;
    for (frame_idx, rms) in frame_rms.iter().copied().enumerate().rev() {
        if rms >= threshold {
            if run_len == 0 {
                run_end = frame_idx;
            }
            run_len += 1;
            if run_len >= min_consec_frames {
                return Some(run_end);
            }
            continue;
        }
        run_len = 0;
    }
    None
}

fn is_late_enough_for_strict_offset(offset_frame: usize, total_frames: usize) -> bool {
    let min_late_offset = total_frames.saturating_sub(1 + OFFSET_STRICT_LATE_WINDOW_FRAMES);
    offset_frame >= min_late_offset
}

fn compute_frame_rms(
    samples: &[f32],
    sample_rate_hz: u32,
    frame_stride_ms: f64,
) -> Option<Vec<f32>> {
    if samples.is_empty() || sample_rate_hz == 0 {
        return None;
    }
    let frame_len = ((sample_rate_hz as f64 * frame_stride_ms) / 1000.0).round() as usize;
    let frame_len = frame_len.max(1);

    let mut frame_rms = Vec::new();
    for chunk in samples.chunks(frame_len) {
        let mean_sq =
            chunk.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / chunk.len() as f64;
        frame_rms.push(mean_sq.sqrt() as f32);
    }
    if frame_rms.is_empty() {
        return None;
    }
    Some(frame_rms)
}
