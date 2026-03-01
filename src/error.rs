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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn io_error_constructor_and_display() {
        let source = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = AlignmentError::io("reading model", source);
        let display = err.to_string();
        assert!(display.contains("I/O error"));
        assert!(display.contains("reading model"));
        assert!(err.source().is_some());
    }

    #[test]
    fn json_error_constructor_and_display() {
        let source = serde_json::from_str::<()>("{").unwrap_err();
        let err = AlignmentError::json("parse config.json", source);
        let display = err.to_string();
        assert!(display.contains("JSON parse error"));
        assert!(display.contains("parse config.json"));
        assert!(err.source().is_some());
    }

    #[test]
    fn runtime_error_constructor_and_display() {
        let err = AlignmentError::runtime("alignment", "buffer too short");
        let display = err.to_string();
        assert!(display.contains("alignment"));
        assert!(display.contains("buffer too short"));
        assert!(err.source().is_none());
    }

    #[test]
    fn invalid_input_constructor_and_display() {
        let err = AlignmentError::invalid_input("sample rate must be 16000");
        let display = err.to_string();
        assert!(display.contains("invalid input"));
        assert!(display.contains("sample rate must be 16000"));
        assert!(err.source().is_none());
    }
}
