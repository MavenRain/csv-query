//! Command-line interface definition.
//!
//! The [`Cli`] struct uses `clap` derive to parse command-line
//! arguments.  All fields are private with public getter methods,
//! following the project convention of no public struct fields.

use clap::Parser;

/// Query CSV collections using a local small language model.
///
/// Provide one or more CSV sources (local glob patterns or URLs) and
/// a natural-language prompt.  The tool loads all files, validates that
/// they share the same schema, and uses an embedded SLM to answer your
/// question or produce a new CSV.
#[derive(Parser, Debug)]
#[command(name = "csv-query")]
#[command(version)]
pub struct Cli {
    /// CSV source: a glob pattern or URL (repeatable).
    ///
    /// Local examples: `data/*.csv`, `reports/2024*.csv`
    /// Remote examples: `https://example.com/data.csv`
    #[arg(short, long, required = true, num_args = 1)]
    source: Vec<String>,

    /// Model to use for inference.
    ///
    /// Supported: phi3 (default), smollm2
    #[arg(short, long, default_value = "phi3")]
    model: String,

    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<String>,

    /// Maximum tokens to generate.
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Sampling temperature (0.0 for greedy).
    #[arg(long, default_value = "0.7")]
    temperature: f64,

    /// The prompt or question to ask about the CSV data.
    prompt: String,
}

impl Cli {
    /// The raw source strings.
    #[must_use]
    pub fn sources(&self) -> &[String] {
        &self.source
    }

    /// The model name string.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Optional output file path.
    #[must_use]
    pub fn output_path(&self) -> Option<&str> {
        self.output.as_deref()
    }

    /// Maximum tokens to generate.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Sampling temperature.
    #[must_use]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// The user's prompt.
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }
}
