use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::error::AlignmentError;
use crate::types::WordTiming;

const OUTLIER_TOP_N: usize = 20;
const EPS_DURATION_SEC: f64 = 0.001;
const BASE_LOW_CONF_THRESHOLD: f64 = 0.50;
const MIN_LOW_CONF_THRESHOLD: f64 = 0.40;
const MAX_LOW_CONF_THRESHOLD: f64 = 0.60;
const DRIFT_OUTLIER_MIN_DURATION_MS: u64 = 3_000;
const DRIFT_OUTLIER_MIN_WORD_COUNT: u32 = 5;
const PASS_RATE_50_MS: f64 = 50.0;
const PASS_RATE_100_MS: f64 = 100.0;
const PASS_RATE_150_MS: f64 = 150.0;

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub meta: Meta,
    pub sentences: Vec<SentenceReport>,
    pub aggregates: AggregateReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct Meta {
    pub generated_at: String,
    pub model_path: String,
    pub device: String,
    pub frame_stride_ms: f32,
    pub case_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Clean,
    Other,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct SentenceReport {
    pub id: String,
    pub split: Split,
    pub has_reference: bool,
    pub duration_ms: u64,
    pub word_count_pred: u32,
    pub word_count_ref: u32,
    pub structural: StructuralMetrics,
    pub confidence: Option<ConfidenceMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_word: Option<Vec<PerWordTrace>>,
    pub notes: Vec<String>,
    #[serde(skip)]
    pub word_abs_errors_ms: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StructuralMetrics {
    pub negative_duration_word_count: u32,
    pub overlap_word_count: u32,
    pub non_monotonic_word_count: u32,
    pub invalid_confidence_word_count: u32,
    pub gap_ratio: f32,
    pub overlap_ratio: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceMetrics {
    pub word_conf_mean: f32,
    pub word_conf_min: f32,
    pub low_conf_threshold_used: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_word_margin: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_boundary_confidence: Option<f32>,
    pub low_conf_word_ratio: f32,
    pub blank_frame_ratio: Option<f32>,
    pub token_entropy_mean: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimingMetrics {
    pub start: EndpointMetrics,
    pub end: EndpointMetrics,
    pub abs_err_ms_median: f32,
    pub abs_err_ms_p90: f32,
    pub trimmed_mean_abs_err_ms: f32,
    pub offset_ms: f32,
    pub drift_ms_per_sec: f32,
    pub drift_delta_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct EndpointMetrics {
    pub mean_signed_ms: f32,
    pub median_abs_ms: f32,
    pub p90_abs_ms: f32,
    pub max_abs_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerWordTrace {
    pub word: String,
    pub ref_start_ms: u64,
    pub ref_end_ms: u64,
    pub pred_start_ms: u64,
    pub pred_end_ms: u64,
    pub start_err_ms: f32,
    pub end_err_ms: f32,
    pub conf: Option<f32>,
    pub quality_confidence: Option<f32>,
    pub calibrated_confidence: Option<f32>,
    pub mean_logp: Option<f32>,
    pub geo_mean_prob: Option<f32>,
    pub min_logp: Option<f32>,
    pub p10_logp: Option<f32>,
    pub mean_margin: Option<f32>,
    pub coverage_frame_count: u32,
    pub boundary_confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceWord {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateReport {
    pub counts: AggregateCounts,
    pub global: AggregateMetrics,
    pub by_split: AggregateBySplit,
    pub outliers: OutlierReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateCounts {
    pub total: u32,
    pub with_reference: u32,
    pub without_reference: u32,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct AggregateMetrics {
    pub abs_err_ms_median: Option<MetricDistribution>,
    pub abs_err_ms_p90: Option<MetricDistribution>,
    pub drift_ms_per_sec: Option<MetricDistribution>,
    pub drift_delta_ms: Option<MetricDistribution>,
    pub low_conf_word_ratio: Option<MetricDistribution>,
    pub avg_word_margin: Option<MetricDistribution>,
    pub avg_boundary_confidence: Option<MetricDistribution>,
    pub blank_frame_ratio: Option<MetricDistribution>,
    pub abs_err_ms_p90_pass_rate: Option<ThresholdPassRates>,
    pub word_abs_err_ms: Option<MetricDistribution>,
    pub word_abs_err_pass_rate: Option<ThresholdPassRates>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct AggregateBySplit {
    pub clean: AggregateMetrics,
    pub other: AggregateMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unknown: Option<AggregateMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricDistribution {
    pub mean: f32,
    pub p50: f32,
    pub p90: f32,
    pub p95: f32,
    pub p99: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThresholdPassRates {
    pub le_50_ms: f32,
    pub le_100_ms: f32,
    pub le_150_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutlierReport {
    pub worst_abs_err_ms_p90: Vec<OutlierEntry>,
    pub worst_drift_ms_per_sec: Vec<OutlierEntry>,
    pub worst_low_conf_word_ratio: Option<Vec<OutlierEntry>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutlierEntry {
    pub id: String,
    pub split: Split,
    pub value: f32,
}

pub fn infer_split(path_or_id: &str) -> Split {
    let normalized = path_or_id.to_ascii_lowercase();
    if normalized.contains("test-clean") {
        Split::Clean
    } else if normalized.contains("test-other") {
        Split::Other
    } else {
        Split::Unknown
    }
}

pub fn compute_sentence_report(
    id: &str,
    split: Split,
    predicted: &[WordTiming],
    reference: Option<&[ReferenceWord]>,
    duration_ms: u64,
) -> Result<SentenceReport, AlignmentError> {
    let mut notes = Vec::new();
    let reference_words = reference.unwrap_or(&[]);
    let has_reference = reference.is_some();

    if !has_reference {
        notes.push("reference_missing".to_string());
    }
    if predicted.is_empty() {
        notes.push("no_predicted_words".to_string());
    }
    if has_reference && reference_words.is_empty() {
        notes.push("empty_reference_words".to_string());
    }

    let structural = compute_structural_metrics(predicted, duration_ms)?;
    if structural.invalid_confidence_word_count > 0 {
        notes.push(format!(
            "invalid_confidence_words={}",
            structural.invalid_confidence_word_count
        ));
    }
    let confidence = Some(compute_confidence_metrics(predicted)?);

    let (timing, word_abs_errors_ms) = if has_reference {
        let timing = compute_timing_metrics(predicted, reference_words, duration_ms, &mut notes)?;
        (Some(timing.metrics), timing.word_abs_errors_ms)
    } else {
        (None, Vec::new())
    };

    if has_reference {
        if predicted.len() != reference_words.len() {
            notes.push(format!(
                "word_count_mismatch:pred={} ref={}",
                predicted.len(),
                reference_words.len()
            ));
        }
        let mismatches = predicted
            .iter()
            .zip(reference_words.iter())
            .filter(|(pred, reference_word)| {
                normalize_word_for_comparison(&pred.word)
                    != normalize_word_for_comparison(&reference_word.word)
            })
            .count();
        if mismatches > 0 {
            notes.push(format!("word_label_mismatches={mismatches}"));
        }
    }

    Ok(SentenceReport {
        id: id.to_string(),
        split,
        has_reference,
        duration_ms,
        word_count_pred: to_u32(predicted.len()),
        word_count_ref: to_u32(reference_words.len()),
        structural,
        confidence,
        timing,
        per_word: None,
        notes,
        word_abs_errors_ms,
    })
}

pub fn aggregate_reports(sentences: &[SentenceReport]) -> AggregateReport {
    let with_reference: Vec<&SentenceReport> = sentences
        .iter()
        .filter(|sentence| sentence.has_reference && sentence.timing.is_some())
        .collect();
    let without_reference_count = sentences.len().saturating_sub(with_reference.len());

    let global = aggregate_metrics_from_sentences(&with_reference);
    let clean = aggregate_metrics_for_split(&with_reference, Split::Clean);
    let other = aggregate_metrics_for_split(&with_reference, Split::Other);
    let unknown = aggregate_metrics_for_split(&with_reference, Split::Unknown);
    let outliers = build_outliers(&with_reference, OUTLIER_TOP_N);

    AggregateReport {
        counts: AggregateCounts {
            total: to_u32(sentences.len()),
            with_reference: to_u32(with_reference.len()),
            without_reference: to_u32(without_reference_count),
        },
        global,
        by_split: AggregateBySplit {
            clean,
            other,
            unknown: (!split_is_empty(&with_reference, Split::Unknown)).then_some(unknown),
        },
        outliers,
    }
}

pub fn attach_outlier_traces(
    sentences: &mut [SentenceReport],
    predicted_by_id: &HashMap<String, Vec<WordTiming>>,
    references_by_id: &HashMap<String, Vec<ReferenceWord>>,
    top_n: usize,
) {
    let mut ranked: Vec<(String, f32)> = sentences
        .iter()
        .filter_map(|sentence| {
            sentence
                .timing
                .as_ref()
                .map(|timing| (sentence.id.clone(), timing.abs_err_ms_p90))
        })
        .collect();

    ranked.sort_by(|(id_a, value_a), (id_b, value_b)| {
        value_b
            .partial_cmp(value_a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| id_a.cmp(id_b))
    });

    let outlier_ids: HashSet<String> = ranked.into_iter().take(top_n).map(|(id, _)| id).collect();
    for sentence in sentences.iter_mut() {
        if !outlier_ids.contains(&sentence.id) {
            continue;
        }
        let Some(predicted) = predicted_by_id.get(&sentence.id) else {
            continue;
        };
        let Some(reference) = references_by_id.get(&sentence.id) else {
            continue;
        };
        let paired_len = predicted.len().min(reference.len());
        let mut traces = Vec::with_capacity(paired_len);
        for (pred, reference_word) in predicted.iter().zip(reference.iter()) {
            traces.push(PerWordTrace {
                word: reference_word.word.clone(),
                ref_start_ms: reference_word.start_ms,
                ref_end_ms: reference_word.end_ms,
                pred_start_ms: pred.start_ms,
                pred_end_ms: pred.end_ms,
                start_err_ms: (pred.start_ms as f64 - reference_word.start_ms as f64) as f32,
                end_err_ms: (pred.end_ms as f64 - reference_word.end_ms as f64) as f32,
                conf: pred.confidence,
                quality_confidence: pred.confidence_stats.quality_confidence,
                calibrated_confidence: pred
                    .confidence_stats
                    .calibrated_confidence
                    .or(pred.confidence),
                mean_logp: pred.confidence_stats.mean_logp,
                geo_mean_prob: pred.confidence_stats.geo_mean_prob,
                min_logp: pred.confidence_stats.min_logp,
                p10_logp: pred.confidence_stats.p10_logp,
                mean_margin: pred.confidence_stats.mean_margin,
                coverage_frame_count: pred.confidence_stats.coverage_frame_count,
                boundary_confidence: pred.confidence_stats.boundary_confidence,
            });
        }
        if !traces.is_empty() {
            sentence.per_word = Some(traces);
        }
    }
}

fn aggregate_metrics_for_split(sentences: &[&SentenceReport], split: Split) -> AggregateMetrics {
    let filtered: Vec<&SentenceReport> = sentences
        .iter()
        .copied()
        .filter(|sentence| sentence.split == split)
        .collect();
    aggregate_metrics_from_sentences(&filtered)
}

fn split_is_empty(sentences: &[&SentenceReport], split: Split) -> bool {
    !sentences.iter().any(|sentence| sentence.split == split)
}

fn aggregate_metrics_from_sentences(sentences: &[&SentenceReport]) -> AggregateMetrics {
    let mut abs_err_ms_median = Vec::new();
    let mut abs_err_ms_p90 = Vec::new();
    let mut drift_ms_per_sec = Vec::new();
    let mut drift_delta_ms = Vec::new();
    let mut low_conf_word_ratio = Vec::new();
    let mut avg_word_margin = Vec::new();
    let mut avg_boundary_confidence = Vec::new();
    let mut blank_frame_ratio = Vec::new();
    let mut word_abs_err_ms = Vec::new();

    for sentence in sentences {
        if let Some(timing) = sentence.timing.as_ref() {
            abs_err_ms_median.push(timing.abs_err_ms_median as f64);
            abs_err_ms_p90.push(timing.abs_err_ms_p90 as f64);
            drift_ms_per_sec.push(timing.drift_ms_per_sec as f64);
            drift_delta_ms.push(timing.drift_delta_ms as f64);
            word_abs_err_ms.extend(
                sentence
                    .word_abs_errors_ms
                    .iter()
                    .map(|value| *value as f64),
            );
        }

        if let Some(confidence) = sentence.confidence.as_ref() {
            low_conf_word_ratio.push(confidence.low_conf_word_ratio as f64);
            if let Some(value) = confidence.avg_word_margin {
                avg_word_margin.push(value as f64);
            }
            if let Some(value) = confidence.avg_boundary_confidence {
                avg_boundary_confidence.push(value as f64);
            }
            if let Some(value) = confidence.blank_frame_ratio {
                blank_frame_ratio.push(value as f64);
            }
        }
    }

    AggregateMetrics {
        abs_err_ms_median: distribution_or_none(&abs_err_ms_median),
        abs_err_ms_p90: distribution_or_none(&abs_err_ms_p90),
        drift_ms_per_sec: distribution_or_none(&drift_ms_per_sec),
        drift_delta_ms: distribution_or_none(&drift_delta_ms),
        low_conf_word_ratio: distribution_or_none(&low_conf_word_ratio),
        avg_word_margin: distribution_or_none(&avg_word_margin),
        avg_boundary_confidence: distribution_or_none(&avg_boundary_confidence),
        blank_frame_ratio: distribution_or_none(&blank_frame_ratio),
        abs_err_ms_p90_pass_rate: pass_rates_or_none(
            &abs_err_ms_p90,
            "aggregate.abs_err_ms_p90_pass_rate",
        ),
        word_abs_err_ms: distribution_or_none(&word_abs_err_ms),
        word_abs_err_pass_rate: pass_rates_or_none(
            &word_abs_err_ms,
            "aggregate.word_abs_err_pass_rate",
        ),
    }
}

fn build_outliers(sentences: &[&SentenceReport], top_n: usize) -> OutlierReport {
    let worst_abs_err_ms_p90 = ranked_outliers(sentences, top_n, |sentence| {
        sentence
            .timing
            .as_ref()
            .map(|timing| timing.abs_err_ms_p90 as f64)
    });
    let drift_candidates = robust_drift_outlier_candidates(sentences);
    let worst_drift_ms_per_sec = ranked_outliers_by(
        &drift_candidates,
        top_n,
        |sentence| {
            sentence
                .timing
                .as_ref()
                .map(|timing| timing.drift_ms_per_sec as f64)
        },
        |value, _sentence| value.abs(),
        abs_err_ms_p90_tiebreak,
    );
    let worst_low_conf_word_ratio = {
        let values = ranked_outliers_by(
            sentences,
            top_n,
            |sentence| {
                sentence
                    .confidence
                    .as_ref()
                    .map(|confidence| confidence.low_conf_word_ratio as f64)
            },
            |value, _sentence| value,
            abs_err_ms_p90_tiebreak,
        );
        (!values.is_empty()).then_some(values)
    };

    OutlierReport {
        worst_abs_err_ms_p90,
        worst_drift_ms_per_sec,
        worst_low_conf_word_ratio,
    }
}

fn ranked_outliers(
    sentences: &[&SentenceReport],
    top_n: usize,
    metric: impl Fn(&SentenceReport) -> Option<f64>,
) -> Vec<OutlierEntry> {
    ranked_outliers_by(
        sentences,
        top_n,
        metric,
        |value, _sentence| value,
        |_sentence| 0.0,
    )
}

fn ranked_outliers_by(
    sentences: &[&SentenceReport],
    top_n: usize,
    metric: impl Fn(&SentenceReport) -> Option<f64>,
    sort_score: impl Fn(f64, &SentenceReport) -> f64,
    secondary_score: impl Fn(&SentenceReport) -> f64,
) -> Vec<OutlierEntry> {
    struct RankedOutlier {
        entry: OutlierEntry,
        sort_score: f64,
        secondary_score: f64,
    }

    let mut entries: Vec<RankedOutlier> = sentences
        .iter()
        .filter_map(|sentence| {
            let value = metric(sentence)?;
            let sort_value = sort_score(value, sentence);
            let tie_break = secondary_score(sentence);
            if !value.is_finite() || !sort_value.is_finite() || !tie_break.is_finite() {
                return None;
            }
            Some(RankedOutlier {
                entry: OutlierEntry {
                    id: sentence.id.clone(),
                    split: sentence.split,
                    value: value as f32,
                },
                sort_score: sort_value,
                secondary_score: tie_break,
            })
        })
        .collect();

    entries.sort_by(|a, b| {
        b.sort_score
            .partial_cmp(&a.sort_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.secondary_score
                    .partial_cmp(&a.secondary_score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| a.entry.id.cmp(&b.entry.id))
    });
    entries.truncate(top_n);
    entries.into_iter().map(|entry| entry.entry).collect()
}

fn robust_drift_outlier_candidates<'a>(
    sentences: &'a [&'a SentenceReport],
) -> Vec<&'a SentenceReport> {
    let filtered: Vec<&SentenceReport> = sentences
        .iter()
        .copied()
        .filter(|sentence| {
            sentence.duration_ms >= DRIFT_OUTLIER_MIN_DURATION_MS
                && sentence.word_count_ref >= DRIFT_OUTLIER_MIN_WORD_COUNT
        })
        .collect();
    if filtered.is_empty() {
        sentences.to_vec()
    } else {
        filtered
    }
}

fn abs_err_ms_p90_tiebreak(sentence: &SentenceReport) -> f64 {
    sentence
        .timing
        .as_ref()
        .map(|timing| timing.abs_err_ms_p90 as f64)
        .unwrap_or(0.0)
}

fn distribution_or_none(values: &[f64]) -> Option<MetricDistribution> {
    if values.is_empty() {
        return None;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mean_value = mean(&sorted);
    let p50_value = percentile_sorted(&sorted, 0.5);
    let p90_value = percentile_sorted(&sorted, 0.9);
    let p95_value = percentile_sorted(&sorted, 0.95);
    let p99_value = percentile_sorted(&sorted, 0.99);

    Some(MetricDistribution {
        mean: checked_f32(mean_value, "aggregate.mean").ok()?,
        p50: checked_f32(p50_value, "aggregate.p50").ok()?,
        p90: checked_f32(p90_value, "aggregate.p90").ok()?,
        p95: checked_f32(p95_value, "aggregate.p95").ok()?,
        p99: checked_f32(p99_value, "aggregate.p99").ok()?,
    })
}

fn pass_rates_or_none(values: &[f64], metric_prefix: &str) -> Option<ThresholdPassRates> {
    if values.is_empty() {
        return None;
    }

    let count = values.len() as f64;
    let le_50_ms = values
        .iter()
        .filter(|value| **value <= PASS_RATE_50_MS)
        .count() as f64
        / count;
    let le_100_ms = values
        .iter()
        .filter(|value| **value <= PASS_RATE_100_MS)
        .count() as f64
        / count;
    let le_150_ms = values
        .iter()
        .filter(|value| **value <= PASS_RATE_150_MS)
        .count() as f64
        / count;

    Some(ThresholdPassRates {
        le_50_ms: checked_f32(le_50_ms, &format!("{metric_prefix}.le_50_ms")).ok()?,
        le_100_ms: checked_f32(le_100_ms, &format!("{metric_prefix}.le_100_ms")).ok()?,
        le_150_ms: checked_f32(le_150_ms, &format!("{metric_prefix}.le_150_ms")).ok()?,
    })
}

fn compute_structural_metrics(
    predicted: &[WordTiming],
    duration_ms: u64,
) -> Result<StructuralMetrics, AlignmentError> {
    // We treat word timing as [start_ms, end_ms), so end must be strictly greater than start.
    let negative_duration_word_count = predicted
        .iter()
        .filter(|word| word.end_ms <= word.start_ms)
        .count();
    let invalid_confidence_word_count = predicted
        .iter()
        .filter(|word| {
            word.confidence.is_none()
                || word.confidence_stats.geo_mean_prob.is_none()
                || word.confidence_stats.coverage_frame_count == 0
        })
        .count();

    let mut overlap_word_count = 0usize;
    let mut non_monotonic_word_count = 0usize;
    let mut gap_ms = 0u64;
    let mut overlap_ms = 0u64;

    for pair in predicted.windows(2) {
        let current = &pair[0];
        let next = &pair[1];

        if current.end_ms > next.start_ms {
            overlap_word_count += 1;
            overlap_ms = overlap_ms.saturating_add(current.end_ms.saturating_sub(next.start_ms));
        } else {
            gap_ms = gap_ms.saturating_add(next.start_ms.saturating_sub(current.end_ms));
        }

        if current.start_ms > next.start_ms {
            non_monotonic_word_count += 1;
        }
    }

    let denom = duration_ms as f64;
    let gap_ratio = if denom > 0.0 {
        gap_ms as f64 / denom
    } else {
        0.0
    };
    let overlap_ratio = if denom > 0.0 {
        overlap_ms as f64 / denom
    } else {
        0.0
    };

    Ok(StructuralMetrics {
        negative_duration_word_count: to_u32(negative_duration_word_count),
        overlap_word_count: to_u32(overlap_word_count),
        non_monotonic_word_count: to_u32(non_monotonic_word_count),
        invalid_confidence_word_count: to_u32(invalid_confidence_word_count),
        gap_ratio: checked_f32(gap_ratio, "structural.gap_ratio")?,
        overlap_ratio: checked_f32(overlap_ratio, "structural.overlap_ratio")?,
    })
}

fn compute_confidence_metrics(
    predicted: &[WordTiming],
) -> Result<ConfidenceMetrics, AlignmentError> {
    if predicted.is_empty() {
        return Ok(ConfidenceMetrics {
            word_conf_mean: 0.0,
            word_conf_min: 0.0,
            low_conf_threshold_used: BASE_LOW_CONF_THRESHOLD as f32,
            avg_word_margin: None,
            avg_boundary_confidence: None,
            low_conf_word_ratio: 0.0,
            blank_frame_ratio: None,
            token_entropy_mean: None,
        });
    }

    let mut conf_values = Vec::new();
    let mut margin_values = Vec::new();
    let mut boundary_values = Vec::new();
    let mut low_conf = 0usize;
    let low_conf_threshold = tuned_low_conf_threshold(predicted);

    for word in predicted {
        let confidence_score = word.confidence;
        let mean_margin = word.confidence_stats.mean_margin;
        let boundary_conf = word.confidence_stats.boundary_confidence;

        if let Some(conf) = confidence_score {
            conf_values.push(conf as f64);
        }
        if let Some(margin) = mean_margin {
            margin_values.push(margin as f64);
        }
        if let Some(boundary) = boundary_conf {
            boundary_values.push(boundary as f64);
        }

        let is_invalid_conf =
            confidence_score.is_none() || word.confidence_stats.coverage_frame_count == 0;
        let is_low_conf = is_invalid_conf
            || confidence_score.is_some_and(|value| (value as f64) < low_conf_threshold);
        if is_low_conf {
            low_conf += 1;
        }
    }

    let count = predicted.len() as f64;
    let mean_conf = if conf_values.is_empty() {
        0.0
    } else {
        mean(&conf_values)
    };
    let min_conf = conf_values
        .iter()
        .copied()
        .fold(f64::INFINITY, |cur, value| cur.min(value));
    let min_conf = if min_conf.is_finite() { min_conf } else { 0.0 };

    Ok(ConfidenceMetrics {
        word_conf_mean: checked_f32(mean_conf, "confidence.word_conf_mean")?,
        word_conf_min: checked_f32(min_conf, "confidence.word_conf_min")?,
        low_conf_threshold_used: checked_f32(
            low_conf_threshold,
            "confidence.low_conf_threshold_used",
        )?,
        avg_word_margin: if margin_values.is_empty() {
            None
        } else {
            Some(checked_f32(
                mean(&margin_values),
                "confidence.avg_word_margin",
            )?)
        },
        avg_boundary_confidence: if boundary_values.is_empty() {
            None
        } else {
            Some(checked_f32(
                mean(&boundary_values),
                "confidence.avg_boundary_confidence",
            )?)
        },
        low_conf_word_ratio: checked_f32(
            low_conf as f64 / count,
            "confidence.low_conf_word_ratio",
        )?,
        blank_frame_ratio: None,
        token_entropy_mean: None,
    })
}

fn tuned_low_conf_threshold(predicted: &[WordTiming]) -> f64 {
    let mut margins = Vec::new();
    let mut boundaries = Vec::new();
    for word in predicted {
        if let Some(margin) = word.confidence_stats.mean_margin {
            margins.push(margin as f64);
        }
        if let Some(boundary) = word.confidence_stats.boundary_confidence {
            boundaries.push(boundary as f64);
        }
    }

    let mut threshold = BASE_LOW_CONF_THRESHOLD;
    if !margins.is_empty() {
        let avg_margin = mean(&margins);
        let margin_score = confidence_sigmoid((avg_margin - 3.0) / 1.5);
        threshold += (0.5 - margin_score) * 0.12;
    }
    if !boundaries.is_empty() {
        let avg_boundary = mean(&boundaries).clamp(0.0, 1.0);
        threshold -= (avg_boundary - 0.5) * 0.06;
    }

    threshold.clamp(MIN_LOW_CONF_THRESHOLD, MAX_LOW_CONF_THRESHOLD)
}

fn confidence_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

struct TimingComputation {
    metrics: TimingMetrics,
    word_abs_errors_ms: Vec<f32>,
}

fn compute_timing_metrics(
    predicted: &[WordTiming],
    reference: &[ReferenceWord],
    duration_ms: u64,
    notes: &mut Vec<String>,
) -> Result<TimingComputation, AlignmentError> {
    let paired_len = predicted.len().min(reference.len());
    if paired_len == 0 {
        notes.push("no_aligned_word_pairs_for_timing".to_string());
        let zero_endpoint = EndpointMetrics {
            mean_signed_ms: 0.0,
            median_abs_ms: 0.0,
            p90_abs_ms: 0.0,
            max_abs_ms: 0.0,
        };
        return Ok(TimingComputation {
            metrics: TimingMetrics {
                start: zero_endpoint.clone(),
                end: zero_endpoint,
                abs_err_ms_median: 0.0,
                abs_err_ms_p90: 0.0,
                trimmed_mean_abs_err_ms: 0.0,
                offset_ms: 0.0,
                drift_ms_per_sec: 0.0,
                drift_delta_ms: 0.0,
            },
            word_abs_errors_ms: Vec::new(),
        });
    }

    let mut start_signed = Vec::with_capacity(paired_len);
    let mut end_signed = Vec::with_capacity(paired_len);
    let mut center_signed = Vec::with_capacity(paired_len);
    let mut abs_all = Vec::with_capacity(paired_len * 2);

    for (pred, reference_word) in predicted.iter().zip(reference.iter()) {
        let start_err = pred.start_ms as f64 - reference_word.start_ms as f64;
        let end_err = pred.end_ms as f64 - reference_word.end_ms as f64;
        let center_err = ((pred.start_ms + pred.end_ms) as f64
            - (reference_word.start_ms + reference_word.end_ms) as f64)
            / 2.0;

        start_signed.push(start_err);
        end_signed.push(end_err);
        center_signed.push(center_err);
        abs_all.push(start_err.abs());
        abs_all.push(end_err.abs());
    }

    let start = endpoint_metrics("timing.start", &start_signed)?;
    let end = endpoint_metrics("timing.end", &end_signed)?;
    let mut abs_sorted = abs_all.clone();
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let abs_err_ms_median = checked_f32(median_sorted(&abs_sorted), "timing.abs_err_ms_median")?;
    let abs_err_ms_p90 = checked_f32(percentile_sorted(&abs_sorted, 0.9), "timing.abs_err_ms_p90")?;
    let trimmed_mean_abs_err_ms = checked_f32(
        trimmed_mean_drop_top_fraction(&abs_all, 0.1),
        "timing.trimmed_mean_abs_err_ms",
    )?;
    let offset_ms = checked_f32(mean(&center_signed), "timing.offset_ms")?;
    let duration_sec = (duration_ms as f64 / 1000.0).max(EPS_DURATION_SEC);
    let drift_delta_ms = end.mean_signed_ms as f64 - start.mean_signed_ms as f64;
    let drift_ms_per_sec = checked_f32(drift_delta_ms / duration_sec, "timing.drift_ms_per_sec")?;
    let drift_delta_ms = checked_f32(drift_delta_ms, "timing.drift_delta_ms")?;
    let word_abs_errors_ms = abs_all
        .iter()
        .map(|value| checked_f32(*value, "timing.word_abs_errors_ms"))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(TimingComputation {
        metrics: TimingMetrics {
            start,
            end,
            abs_err_ms_median,
            abs_err_ms_p90,
            trimmed_mean_abs_err_ms,
            offset_ms,
            drift_ms_per_sec,
            drift_delta_ms,
        },
        word_abs_errors_ms,
    })
}

fn endpoint_metrics(
    metric_prefix: &str,
    signed_errors: &[f64],
) -> Result<EndpointMetrics, AlignmentError> {
    if signed_errors.is_empty() {
        return Ok(EndpointMetrics {
            mean_signed_ms: 0.0,
            median_abs_ms: 0.0,
            p90_abs_ms: 0.0,
            max_abs_ms: 0.0,
        });
    }

    let mut abs_values: Vec<f64> = signed_errors.iter().map(|value| value.abs()).collect();
    abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let max_abs = abs_values.last().copied().unwrap_or(0.0);

    Ok(EndpointMetrics {
        mean_signed_ms: checked_f32(
            mean(signed_errors),
            &format!("{metric_prefix}.mean_signed_ms"),
        )?,
        median_abs_ms: checked_f32(
            median_sorted(&abs_values),
            &format!("{metric_prefix}.median_abs_ms"),
        )?,
        p90_abs_ms: checked_f32(
            percentile_sorted(&abs_values, 0.9),
            &format!("{metric_prefix}.p90_abs_ms"),
        )?,
        max_abs_ms: checked_f32(max_abs, &format!("{metric_prefix}.max_abs_ms"))?,
    })
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median_sorted(sorted_values: &[f64]) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let mid = sorted_values.len() / 2;
    if sorted_values.len().is_multiple_of(2) {
        (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    } else {
        sorted_values[mid]
    }
}

fn percentile_sorted(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }

    let clamped = percentile.clamp(0.0, 1.0);
    let max_index = (sorted_values.len() - 1) as f64;
    let rank = clamped * max_index;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = rank - lower as f64;
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    }
}

fn trimmed_mean_drop_top_fraction(values: &[f64], top_fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let drop_count = ((sorted.len() as f64) * top_fraction.clamp(0.0, 1.0)).floor() as usize;
    let keep = sorted.len().saturating_sub(drop_count).max(1);
    mean(&sorted[..keep])
}

fn normalize_word_for_comparison(word: &str) -> String {
    let upper = word.trim().to_ascii_uppercase();
    if upper == "<UNK>" || upper == "UNK" {
        "UNK".to_string()
    } else {
        upper
    }
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn checked_f32(value: f64, metric_name: &str) -> Result<f32, AlignmentError> {
    if !value.is_finite() {
        return Err(AlignmentError::invalid_input(format!(
            "metric '{metric_name}' produced non-finite value: {value}"
        )));
    }
    if value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(AlignmentError::invalid_input(format!(
            "metric '{metric_name}' out of f32 range: {value}"
        )));
    }
    Ok(value as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn sample_sentence(
        id: &str,
        split: Split,
        duration_ms: u64,
        word_count_ref: u32,
        abs_err_ms_p90: f32,
        drift_ms_per_sec: f32,
        low_conf_word_ratio: f32,
        word_abs_errors_ms: Vec<f32>,
    ) -> SentenceReport {
        let duration_sec = (duration_ms as f64 / 1000.0).max(EPS_DURATION_SEC);
        let drift_delta_ms = (drift_ms_per_sec as f64 * duration_sec) as f32;
        SentenceReport {
            id: id.to_string(),
            split,
            has_reference: true,
            duration_ms,
            word_count_pred: word_count_ref,
            word_count_ref,
            structural: StructuralMetrics {
                negative_duration_word_count: 0,
                overlap_word_count: 0,
                non_monotonic_word_count: 0,
                invalid_confidence_word_count: 0,
                gap_ratio: 0.0,
                overlap_ratio: 0.0,
            },
            confidence: Some(ConfidenceMetrics {
                word_conf_mean: 0.8,
                word_conf_min: 0.8,
                low_conf_threshold_used: 0.5,
                avg_word_margin: Some(4.0),
                avg_boundary_confidence: Some(0.8),
                low_conf_word_ratio,
                blank_frame_ratio: None,
                token_entropy_mean: None,
            }),
            timing: Some(TimingMetrics {
                start: EndpointMetrics {
                    mean_signed_ms: 0.0,
                    median_abs_ms: abs_err_ms_p90 / 2.0,
                    p90_abs_ms: abs_err_ms_p90,
                    max_abs_ms: abs_err_ms_p90,
                },
                end: EndpointMetrics {
                    mean_signed_ms: drift_delta_ms,
                    median_abs_ms: abs_err_ms_p90 / 2.0,
                    p90_abs_ms: abs_err_ms_p90,
                    max_abs_ms: abs_err_ms_p90,
                },
                abs_err_ms_median: abs_err_ms_p90 / 2.0,
                abs_err_ms_p90,
                trimmed_mean_abs_err_ms: abs_err_ms_p90 / 2.0,
                offset_ms: 0.0,
                drift_ms_per_sec,
                drift_delta_ms,
            }),
            per_word: None,
            notes: Vec::new(),
            word_abs_errors_ms,
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "actual={actual} expected={expected}"
        );
    }

    #[test]
    fn percentile_sorted_interpolates_linearly() {
        let sorted = [10.0, 20.0, 30.0, 40.0];
        assert_close(percentile_sorted(&sorted, 0.0) as f32, 10.0);
        assert_close(percentile_sorted(&sorted, 0.25) as f32, 17.5);
        assert_close(percentile_sorted(&sorted, 0.5) as f32, 25.0);
        assert_close(percentile_sorted(&sorted, 0.9) as f32, 37.0);
        assert_close(percentile_sorted(&sorted, 1.0) as f32, 40.0);
    }

    #[test]
    fn drift_outliers_use_absolute_value_and_filter_tiny_utterances() {
        let sentences = vec![
            sample_sentence(
                "short-neg",
                Split::Other,
                1_000,
                1,
                90.0,
                -120.0,
                0.2,
                vec![30.0, 80.0],
            ),
            sample_sentence(
                "long-pos",
                Split::Other,
                6_000,
                8,
                95.0,
                30.0,
                0.2,
                vec![40.0, 70.0],
            ),
            sample_sentence(
                "long-neg",
                Split::Other,
                6_000,
                8,
                100.0,
                -40.0,
                0.2,
                vec![40.0, 70.0],
            ),
            sample_sentence(
                "long-small",
                Split::Other,
                7_000,
                10,
                80.0,
                10.0,
                0.2,
                vec![35.0, 50.0],
            ),
        ];

        let report = aggregate_reports(&sentences);
        let drift_outliers = &report.outliers.worst_drift_ms_per_sec;
        assert_eq!(drift_outliers[0].id, "long-neg");
        assert_eq!(drift_outliers[0].value, -40.0);
        assert_eq!(drift_outliers[1].id, "long-pos");
        assert!(!drift_outliers.iter().any(|entry| entry.id == "short-neg"));
    }

    #[test]
    fn low_conf_outliers_use_abs_err_tiebreak() {
        let sentences = vec![
            sample_sentence(
                "tie-low-err",
                Split::Clean,
                5_000,
                8,
                80.0,
                2.0,
                1.0,
                vec![40.0, 50.0],
            ),
            sample_sentence(
                "tie-high-err",
                Split::Clean,
                5_000,
                8,
                160.0,
                2.0,
                1.0,
                vec![80.0, 90.0],
            ),
            sample_sentence(
                "lower-ratio",
                Split::Clean,
                5_000,
                8,
                300.0,
                2.0,
                0.9,
                vec![110.0, 120.0],
            ),
        ];

        let report = aggregate_reports(&sentences);
        let low_conf_outliers = report
            .outliers
            .worst_low_conf_word_ratio
            .expect("low confidence outliers should be present");

        assert_eq!(low_conf_outliers[0].id, "tie-high-err");
        assert_eq!(low_conf_outliers[1].id, "tie-low-err");
    }

    #[test]
    fn aggregates_include_word_error_distribution_and_pass_rates() {
        let sentences = vec![
            sample_sentence(
                "a",
                Split::Clean,
                5_000,
                6,
                80.0,
                2.0,
                0.2,
                vec![30.0, 60.0, 110.0, 160.0],
            ),
            sample_sentence(
                "b",
                Split::Clean,
                6_000,
                6,
                120.0,
                -1.0,
                0.4,
                vec![40.0, 70.0],
            ),
        ];

        let report = aggregate_reports(&sentences);
        let global = report.global;

        let word_dist = global
            .word_abs_err_ms
            .expect("word-level distribution should be present");
        assert_close(word_dist.mean, 78.333336);
        assert_close(word_dist.p50, 65.0);
        assert_close(word_dist.p90, 135.0);

        let word_pass = global
            .word_abs_err_pass_rate
            .expect("word-level pass rates should be present");
        assert_close(word_pass.le_50_ms, 2.0 / 6.0);
        assert_close(word_pass.le_100_ms, 4.0 / 6.0);
        assert_close(word_pass.le_150_ms, 5.0 / 6.0);

        let sentence_pass = global
            .abs_err_ms_p90_pass_rate
            .expect("sentence-level pass rates should be present");
        assert_close(sentence_pass.le_50_ms, 0.0);
        assert_close(sentence_pass.le_100_ms, 0.5);
        assert_close(sentence_pass.le_150_ms, 1.0);

        let drift_delta = global
            .drift_delta_ms
            .expect("drift delta distribution should be present");
        assert_close(drift_delta.p50, 2.0);
    }
}
