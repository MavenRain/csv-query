//! Deterministic column aggregation.
//!
//! Recognises min / max / sum / average / count intent in a natural
//! language question and computes the answer directly from the CSV
//! collection, bypassing the LLM.  Numeric questions like "what is
//! the smallest sold price" are answered exactly and instantly;
//! everything else returns `None` from [`parse_intent`] so the
//! caller can fall back to LLM generation.

use crate::collection::{CsvCollection, Schema};
use crate::error::Error;

/// A reducer over a numeric column (or row count).
#[derive(Debug, Clone, Copy)]
pub enum AggregationOp {
    /// Minimum value in the column.
    Min,
    /// Maximum value in the column.
    Max,
    /// Sum of values in the column.
    Sum,
    /// Arithmetic mean of values in the column.
    Avg,
    /// Count of rows (no column needed).
    Count,
}

impl AggregationOp {
    /// Human label used in the rendered answer.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Min => "smallest",
            Self::Max => "largest",
            Self::Sum => "total",
            Self::Avg => "average",
            Self::Count => "count",
        }
    }
}

/// A parsed aggregation request: an operation plus an optional
/// target column (omitted for bare `Count`).
#[derive(Debug, Clone)]
pub struct AggregationIntent {
    op: AggregationOp,
    column: Option<String>,
}

impl AggregationIntent {
    /// The reduction operation.
    #[must_use]
    pub fn op(&self) -> AggregationOp {
        self.op
    }

    /// The resolved column name, if any.
    #[must_use]
    pub fn column(&self) -> Option<&str> {
        self.column.as_deref()
    }
}

/// Parse a user question into an [`AggregationIntent`].
///
/// Returns `None` when the question has no recognisable aggregation
/// keyword, or when a reducer requires a column but none of the
/// schema columns appear in the question.  In those cases the caller
/// should fall back to the LLM path.
#[must_use]
pub fn parse_intent(question: &str, schema: &Schema) -> Option<AggregationIntent> {
    let lower = question.to_lowercase();
    let op = detect_op(&lower)?;
    let column = detect_column(&lower, schema);
    let has_enough = matches!(op, AggregationOp::Count) || column.is_some();
    has_enough.then_some(AggregationIntent { op, column })
}

/// Execute a parsed intent against a loaded collection.
///
/// # Errors
///
/// - [`Error::Aggregate`] when the named column is not in the schema,
///   or when the column has no parseable numeric values.
pub fn execute(intent: &AggregationIntent, collection: &CsvCollection) -> Result<String, Error> {
    match intent.op() {
        AggregationOp::Count => Ok(format!("{}\n", collection.row_count())),
        AggregationOp::Min
        | AggregationOp::Max
        | AggregationOp::Sum
        | AggregationOp::Avg => execute_numeric(intent, collection),
    }
}

/// Run a numeric reducer over the resolved column.
fn execute_numeric(intent: &AggregationIntent, collection: &CsvCollection) -> Result<String, Error> {
    let column = intent
        .column()
        .ok_or_else(|| Error::Aggregate("numeric aggregation requires a column".into()))?;
    let col_idx = find_column_index(collection.schema(), column)?;
    let values = extract_numeric(collection, col_idx);
    (!values.is_empty())
        .then_some(())
        .ok_or_else(|| Error::Aggregate(format!("no numeric values in column '{column}'")))?;
    let answer = reduce(intent.op(), &values);
    Ok(format!("{} {}: {}\n", intent.op().label(), column, format_number(answer)))
}

/// Locate the index of a column in the schema.
fn find_column_index(schema: &Schema, column: &str) -> Result<usize, Error> {
    schema
        .columns()
        .iter()
        .position(|c| c == column)
        .ok_or_else(|| Error::Aggregate(format!("column '{column}' not in schema")))
}

/// Pull every row's value at `col_idx`, parsed as `f64`, skipping
/// missing and non-numeric entries.
fn extract_numeric(collection: &CsvCollection, col_idx: usize) -> Vec<f64> {
    collection
        .rows()
        .iter()
        .filter_map(|row| row.get(col_idx).ok().and_then(parse_number))
        .collect()
}

/// Parse a numeric CSV cell, stripping `$`, commas, and surrounding
/// whitespace.  Returns `None` when the cleaned string is empty or
/// does not parse as `f64`.
fn parse_number(s: &str) -> Option<f64> {
    let cleaned: String = s
        .chars()
        .filter(|c| !matches!(*c, '$' | ',' | ' ' | '\t'))
        .collect();
    cleaned.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Apply a reducer to a non-empty slice of `f64` values.
///
/// `Count` returns the slice length; the outer `execute` short-circuits
/// row counting before reaching this function, so the `Count` arm is
/// only reachable via direct test calls.
fn reduce(op: AggregationOp, values: &[f64]) -> f64 {
    match op {
        AggregationOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        AggregationOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        AggregationOp::Sum => values.iter().copied().sum(),
        AggregationOp::Avg => {
            let sum: f64 = values.iter().copied().sum();
            sum / len_as_f64(values.len())
        }
        AggregationOp::Count => len_as_f64(values.len()),
    }
}

/// `usize` to `f64` with a documented precision-loss allowance.  Row
/// counts beyond 2^53 would lose precision, but the tool caps well
/// below that for any realistic CSV.
#[allow(clippy::cast_precision_loss)]
fn len_as_f64(n: usize) -> f64 {
    n as f64
}

/// Render an aggregation result with a sensible number of decimals.
fn format_number(v: f64) -> String {
    if v.fract() == 0.0 { format!("{v:.0}") } else { format!("{v:.2}") }
}

/// Detect the aggregation operator from the lowercased question.
fn detect_op(lower: &str) -> Option<AggregationOp> {
    const KEYWORDS: &[(AggregationOp, &[&str])] = &[
        (
            AggregationOp::Min,
            &["smallest", "minimum", "lowest", "cheapest", "least"],
        ),
        (
            AggregationOp::Max,
            &[
                "largest",
                "maximum",
                "highest",
                "biggest",
                "most expensive",
                "greatest",
            ],
        ),
        (AggregationOp::Sum, &["total", "sum of", "aggregate"]),
        (AggregationOp::Avg, &["average", "mean ", "avg "]),
        (AggregationOp::Count, &["how many", "count of", "number of"]),
    ];
    KEYWORDS
        .iter()
        .find_map(|(op, keys)| keys.iter().any(|k| lower.contains(k)).then_some(*op))
}

/// Find the longest schema column name that appears as a substring of
/// the lowercased question.  Returns `None` when no column matches.
fn detect_column(lower: &str, schema: &Schema) -> Option<String> {
    schema
        .columns()
        .iter()
        .filter(|col| {
            let needle = col.to_lowercase();
            !needle.is_empty() && lower.contains(&needle)
        })
        .max_by_key(|col| col.len())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_number_handles_currency() {
        assert_eq!(parse_number("$1,850,000"), Some(1_850_000.0));
        assert_eq!(parse_number(" 42.5 "), Some(42.5));
        assert_eq!(parse_number(""), None);
        assert_eq!(parse_number("N/A"), None);
    }

    #[test]
    fn detect_op_finds_min() {
        assert!(matches!(
            detect_op("what is the smallest sold price"),
            Some(AggregationOp::Min)
        ));
        assert!(matches!(
            detect_op("lowest price please"),
            Some(AggregationOp::Min)
        ));
    }

    #[test]
    fn detect_op_finds_max() {
        assert!(matches!(
            detect_op("give me the largest value"),
            Some(AggregationOp::Max)
        ));
    }

    #[test]
    fn detect_op_none_for_descriptive() {
        assert!(detect_op("describe this data").is_none());
    }

    #[test]
    fn reduce_min_max_sum_avg() {
        let v = &[1.0, 2.0, 3.0, 4.0];
        assert!((reduce(AggregationOp::Min, v) - 1.0).abs() < f64::EPSILON);
        assert!((reduce(AggregationOp::Max, v) - 4.0).abs() < f64::EPSILON);
        assert!((reduce(AggregationOp::Sum, v) - 10.0).abs() < f64::EPSILON);
        assert!((reduce(AggregationOp::Avg, v) - 2.5).abs() < f64::EPSILON);
    }
}
