#[derive(Debug, Clone)]
pub struct AlignmentInput {
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
    pub transcript: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    pub word: String,
    /// Millisecond interval is [start_ms, end_ms), i.e. start inclusive/end exclusive.
    pub start_ms: u64,
    /// Millisecond interval is [start_ms, end_ms), i.e. start inclusive/end exclusive.
    pub end_ms: u64,
    /// Deterministic word-level quality confidence score in [0, 1].
    /// This blends acoustic support (`geo_mean_prob`) with separability and
    /// boundary evidence. `None` means confidence could not be computed.
    pub confidence: Option<f32>,
    pub confidence_stats: WordConfidenceStats,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct WordConfidenceStats {
    pub mean_logp: Option<f32>,
    pub geo_mean_prob: Option<f32>,
    /// Deterministic composite quality score before calibration.
    pub quality_confidence: Option<f32>,
    /// Monotonic calibrated confidence score in [0, 1].
    pub calibrated_confidence: Option<f32>,
    pub min_logp: Option<f32>,
    pub p10_logp: Option<f32>,
    pub mean_margin: Option<f32>,
    pub coverage_frame_count: u32,
    /// Mean blank probability over frames absorbed by boundary expansion.
    pub boundary_confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentOutput {
    pub words: Vec<WordTiming>,
}

#[derive(Debug, Clone)]
pub struct TokenSequence {
    pub tokens: Vec<usize>,
    pub chars: Vec<Option<char>>,
    /// Transcript normalized with the same logic as emitted token chars.
    pub normalized_words: Vec<String>,
}
