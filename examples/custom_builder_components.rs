//! Example: build a forced aligner with custom injected components (runtime backend,
//! tokenizer, word grouper). Uses a mock backend and minimal custom implementations
//! so the example runs without real model files.
//!
//! Run: cargo run --example custom_builder_components

use std::collections::HashMap;
use std::error::Error;

use wav2vec2_rs::pipeline::traits::{ForwardOutput, RuntimeInferenceOutput};
use wav2vec2_rs::{
    AlignmentError, AlignmentInput, ForcedAlignerBuilder, RuntimeBackend, TokenSequence, Tokenizer,
    Wav2Vec2Config, WordConfidenceStats, WordGrouper, WordTiming,
};

const MINIMAL_CONFIG_JSON: &str = r#"{
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "conv_dim": [512],
    "conv_kernel": [10],
    "conv_stride": [2, 2, 2, 2, 2],
    "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16,
    "pad_token_id": 0,
    "vocab_size": 32
}"#;

/// Mock runtime backend: returns fixed host log-probs so no real model is needed.
struct MockRuntimeBackend;

impl RuntimeBackend for MockRuntimeBackend {
    fn infer(&self, _normalized_audio: &[f32]) -> Result<ForwardOutput, AlignmentError> {
        Ok(ForwardOutput::Host(RuntimeInferenceOutput {
            log_probs: vec![vec![0.0f32; 32]; 10],
            num_frames_t: 10,
            vocab_size: 32,
            dtype: "f32".to_string(),
        }))
    }

    fn device_label(&self) -> String {
        "mock".to_string()
    }
}

/// Custom tokenizer: normalizes transcript to uppercase before building token sequence.
/// In real use you might add custom normalization or vocabulary handling.
struct UppercaseTokenizer;

impl Tokenizer for UppercaseTokenizer {
    fn tokenize(
        &self,
        transcript: &str,
        vocab: &HashMap<char, usize>,
        blank_id: usize,
        word_sep_id: usize,
    ) -> TokenSequence {
        let upper = transcript.to_uppercase();
        let mut tokens = vec![blank_id];
        let mut chars = vec![None];
        let mut normalized_words = Vec::new();
        for word in upper.split_whitespace() {
            normalized_words.push(word.to_string());
            for c in word.chars() {
                if let Some(&id) = vocab.get(&c) {
                    tokens.push(id);
                    chars.push(Some(c));
                }
                tokens.push(blank_id);
                chars.push(None);
            }
            tokens.push(word_sep_id);
            chars.push(None);
            tokens.push(blank_id);
            chars.push(None);
        }
        TokenSequence {
            tokens,
            chars,
            normalized_words,
        }
    }
}

/// Custom word grouper: returns one word per normalized word with simple frame spans.
/// For demonstration only; the default grouper does full blank expansion and confidence.
struct SimpleWordGrouper;

impl WordGrouper for SimpleWordGrouper {
    fn group_words(
        &self,
        path: &[(usize, usize)],
        token_sequence: &TokenSequence,
        _log_probs: &[Vec<f32>],
        _blank_id: usize,
        _word_sep_id: usize,
        stride_ms: f64,
    ) -> Vec<WordTiming> {
        if path.is_empty() || token_sequence.normalized_words.is_empty() {
            return vec![];
        }
        let total_frames = path.len();
        let n = token_sequence.normalized_words.len();
        let frames_per_word = total_frames / n;
        token_sequence
            .normalized_words
            .iter()
            .enumerate()
            .map(|(i, word)| {
                let start_frame = i * frames_per_word;
                let end_frame = ((i + 1) * frames_per_word).min(total_frames);
                WordTiming {
                    word: word.clone(),
                    start_ms: (start_frame as f64 * stride_ms) as u64,
                    end_ms: (end_frame as f64 * stride_ms) as u64,
                    confidence: Some(0.9),
                    confidence_stats: WordConfidenceStats::default(),
                }
            })
            .collect()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("wav2vec2_rs_example_config.json");
    let vocab_path = temp_dir.join("wav2vec2_rs_example_vocab.json");
    std::fs::write(&config_path, MINIMAL_CONFIG_JSON)?;
    std::fs::write(&vocab_path, r#"{"A": 1, "B": 2, "|": 3}"#)?;

    let config = Wav2Vec2Config {
        model_path: String::new(),
        config_path: config_path.to_string_lossy().to_string(),
        vocab_path: vocab_path.to_string_lossy().to_string(),
        device: "cpu".to_string(),
        expected_sample_rate_hz: 16_000,
    };

    let aligner = ForcedAlignerBuilder::new(config)
        .with_runtime_backend(Box::new(MockRuntimeBackend))
        .with_tokenizer(Box::new(UppercaseTokenizer))
        .with_word_grouper(Box::new(SimpleWordGrouper))
        .build()?;

    let input = AlignmentInput {
        sample_rate_hz: 16_000,
        samples: vec![0.0f32; 1600],
        transcript: "a b".to_string(),
        normalized: Some(vec![0.0f32; 1600]),
    };

    let output = aligner.align(&input)?;
    println!("Custom builder produced {} word(s):", output.words.len());
    for w in &output.words {
        println!("  {}: [{}, {}) ms", w.word, w.start_ms, w.end_ms);
    }

    let _ = std::fs::remove_file(&config_path);
    let _ = std::fs::remove_file(&vocab_path);
    Ok(())
}
