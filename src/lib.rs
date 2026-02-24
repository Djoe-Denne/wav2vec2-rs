pub mod alignment;
pub mod config;
pub mod error;
mod model;
pub mod pipeline;
pub mod types;

pub use config::Wav2Vec2Config;
pub use error::AlignmentError;
pub use pipeline::builder::ForcedAlignerBuilder;
pub use pipeline::runtime::ForcedAligner;
pub use pipeline::traits::{SequenceAligner, Tokenizer, WordGrouper};
pub use types::{AlignmentInput, AlignmentOutput, TokenSequence, WordTiming};
