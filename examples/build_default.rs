//! Build a forced aligner with all defaults (Candle runtime, case-aware tokenizer,
//! Viterbi aligner, default word grouper).
//!
//! Run with real model files:
//!   cargo run --example build_default
//! Set paths below to your model directory (safetensors or ONNX), config.json, and vocab.json.
//! Audio must be 16 kHz mono f32.

use std::error::Error;

use wav2vec2_rs::{AlignmentInput, ForcedAlignerBuilder, Wav2Vec2Config};

fn main() -> Result<(), Box<dyn Error>> {
    let config = Wav2Vec2Config {
        model_path: "path/to/model.safetensors".into(), // or "path/to/model.onnx"
        config_path: "path/to/config.json".into(),
        vocab_path: "path/to/vocab.json".into(),
        device: "cpu".into(),
        expected_sample_rate_hz: 16_000,
    };

    let aligner = ForcedAlignerBuilder::new(config).build()?;

    // Placeholder: use your own 16 kHz mono f32 audio and transcript.
    let audio_16k_f32: Vec<f32> = vec![]; // e.g. from a WAV file resampled to 16 kHz
    let transcript = "hello world";

    let input = AlignmentInput {
        sample_rate_hz: 16_000,
        samples: audio_16k_f32.clone(),
        transcript: transcript.into(),
        normalized: None,
    };

    let output = aligner.align(&input)?;
    for word in &output.words {
        println!(
            "{}: [{}, {}) ms  conf={:.2}",
            word.word,
            word.start_ms,
            word.end_ms,
            word.confidence.unwrap_or(0.0)
        );
    }

    Ok(())
}
