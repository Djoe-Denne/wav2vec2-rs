#[derive(Debug, Clone)]
pub struct AlignmentInput {
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
    pub transcript: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentOutput {
    pub words: Vec<WordTiming>,
}

#[derive(Debug, Clone)]
pub struct TokenSequence {
    pub tokens: Vec<usize>,
    pub chars: Vec<Option<char>>,
}
