//! CSV source resolution: local globs and remote URLs.
//!
//! User-provided source strings are parsed into [`CsvSource`] values,
//! then resolved into [`ResolvedSource`] values that the collection
//! loader can consume.  Local globs expand to file paths; remote URLs
//! are fetched via HTTP and their response bodies are held as raw
//! string data.

use comp_cat_rs::effect::io::Io;

use crate::error::Error;

/// A glob pattern for matching local CSV files.
#[derive(Debug, Clone)]
pub struct GlobPattern(String);

impl GlobPattern {
    /// Create a new glob pattern.
    #[must_use]
    pub fn new(pattern: impl Into<String>) -> Self {
        Self(pattern.into())
    }

    /// The raw pattern string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A URL pointing to a remote CSV file.
#[derive(Debug, Clone)]
pub struct RemoteUrl(String);

impl RemoteUrl {
    /// Create a new remote URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self(url.into())
    }

    /// The raw URL string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A source of CSV data, either local files via glob or a remote URL.
#[derive(Debug, Clone)]
pub enum CsvSource {
    /// Local files matching a glob pattern.
    Local(GlobPattern),
    /// A remote CSV accessible via HTTP(S).
    Remote(RemoteUrl),
}

impl CsvSource {
    /// Parse a source string into a `CsvSource`.
    ///
    /// Strings starting with `http://` or `https://` are treated as
    /// remote URLs; everything else is treated as a glob pattern.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        if s.starts_with("http://") || s.starts_with("https://") {
            Self::Remote(RemoteUrl::new(s))
        } else {
            Self::Local(GlobPattern::new(s))
        }
    }
}

/// Resolved CSV content: either a file path or raw string data.
#[derive(Debug, Clone)]
pub enum ResolvedSource {
    /// A local file path to read with csv-cat.
    FilePath(String),
    /// Raw CSV string data fetched from a remote URL.
    RawData(String),
}

/// Resolve a list of sources into concrete file paths and raw data.
///
/// Local globs are expanded into file paths; remote URLs are fetched
/// via HTTP and their response bodies become raw string data.
///
/// # Errors
///
/// - `Error::NoFiles` if resolution produces zero entries.
/// - `Error::GlobPattern` or `Error::Glob` if a local pattern is
///   invalid or a matched path cannot be read.
/// - `Error::Http` if a remote URL cannot be fetched.
pub fn resolve(sources: Vec<CsvSource>) -> Io<Error, Vec<ResolvedSource>> {
    Io::suspend(move || {
        sources
            .into_iter()
            .try_fold(Vec::new(), |acc, source| {
                let resolved = resolve_one(source)?;
                Ok(acc.into_iter().chain(resolved).collect())
            })
            .and_then(|resolved| {
                if resolved.is_empty() {
                    Err(Error::NoFiles)
                } else {
                    Ok(resolved)
                }
            })
    })
}

/// Resolve a single source into one or more `ResolvedSource` entries.
fn resolve_one(source: CsvSource) -> Result<Vec<ResolvedSource>, Error> {
    match source {
        CsvSource::Local(pattern) => expand_glob(&pattern),
        CsvSource::Remote(url) => fetch_remote(&url).map(|data| vec![data]),
    }
}

/// Expand a glob pattern into file paths.
fn expand_glob(pattern: &GlobPattern) -> Result<Vec<ResolvedSource>, Error> {
    glob::glob(pattern.as_str())?
        .map(|entry| {
            entry
                .map_err(Error::from)
                .map(|path| ResolvedSource::FilePath(path.to_string_lossy().into_owned()))
        })
        .collect()
}

/// Fetch CSV data from a remote URL.
fn fetch_remote(url: &RemoteUrl) -> Result<ResolvedSource, Error> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            reqwest::get(url.as_str())
                .await
                .map_err(Error::from)?
                .text()
                .await
                .map_err(Error::from)
                .map(ResolvedSource::RawData)
        })
    })
}
