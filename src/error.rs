use thiserror::Error;

#[derive(Debug, Error)]
pub enum AlignmentError {
    #[error("I/O error while {context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON parse error while {context}: {source}")]
    Json {
        context: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("{context}: {message}")]
    Runtime {
        context: &'static str,
        message: String,
    },
    #[error("invalid input: {message}")]
    InvalidInput { message: String },
}

impl AlignmentError {
    pub(crate) fn io(context: &'static str, source: std::io::Error) -> Self {
        Self::Io { context, source }
    }

    pub(crate) fn json(context: &'static str, source: serde_json::Error) -> Self {
        Self::Json { context, source }
    }

    pub(crate) fn runtime(context: &'static str, err: impl std::fmt::Display) -> Self {
        Self::Runtime {
            context,
            message: err.to_string(),
        }
    }

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }
}
