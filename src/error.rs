//! Project-wide error type.
//!
//! A single [`Error`] enum covers every failure mode in csv-query:
//! CSV parsing, IO, HTTP fetching, glob expansion, model loading,
//! tokenization, schema validation, and model lookup.  Each variant
//! wraps the underlying error from the responsible dependency, with
//! `From` impls enabling `?` propagation throughout the codebase.

use csv_cat::error::CsvError;

/// All errors in csv-query.
#[derive(Debug)]
pub enum Error {
    /// CSV processing error from csv-cat.
    Csv(CsvError),
    /// IO error.
    Io(std::io::Error),
    /// HTTP request error.
    Http(reqwest::Error),
    /// Glob pattern error.
    GlobPattern(glob::PatternError),
    /// Glob iteration error.
    Glob(glob::GlobError),
    /// Model inference error.
    Model(candle_core::Error),
    /// Tokenizer error.
    Tokenizer(String),
    /// Schema mismatch between CSV files.
    SchemaMismatch {
        expected: Vec<String>,
        found: Vec<String>,
    },
    /// No CSV files found from the provided sources.
    NoFiles,
    /// Model file not found after download.
    ModelNotFound(String),
    /// Deterministic aggregation failed (column missing, no numeric values, etc.).
    Aggregate(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv(e) => write!(f, "csv error: {e}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Http(e) => write!(f, "HTTP error: {e}"),
            Self::GlobPattern(e) => write!(f, "glob pattern error: {e}"),
            Self::Glob(e) => write!(f, "glob error: {e}"),
            Self::Model(e) => write!(f, "model error: {e}"),
            Self::Tokenizer(msg) => write!(f, "tokenizer error: {msg}"),
            Self::SchemaMismatch { expected, found } => write!(
                f,
                "schema mismatch: expected {expected:?}, found {found:?}"
            ),
            Self::NoFiles => write!(f, "no CSV files found from the provided sources"),
            Self::ModelNotFound(path) => write!(f, "model file not found: {path}"),
            Self::Aggregate(msg) => write!(f, "aggregation error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Csv(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Http(e) => Some(e),
            Self::GlobPattern(e) => Some(e),
            Self::Glob(e) => Some(e),
            Self::Model(e) => Some(e),
            Self::Tokenizer(_)
            | Self::SchemaMismatch { .. }
            | Self::NoFiles
            | Self::ModelNotFound(_)
            | Self::Aggregate(_) => None,
        }
    }
}

impl From<CsvError> for Error {
    fn from(e: CsvError) -> Self { Self::Csv(e) }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self { Self::Http(e) }
}

impl From<glob::PatternError> for Error {
    fn from(e: glob::PatternError) -> Self { Self::GlobPattern(e) }
}

impl From<glob::GlobError> for Error {
    fn from(e: glob::GlobError) -> Self { Self::Glob(e) }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self { Self::Model(e) }
}
