//! Prompt construction from CSV data and user queries.
//!
//! Formats schema, data rows, and the user's question into a
//! chat-templated prompt suitable for the target model.  When the
//! full dataset exceeds the per-model character budget, a sample of
//! [`SAMPLE_ROWS`] rows is included along with the total row count,
//! keeping the prompt within the model's context window.

use crate::collection::CsvCollection;
use crate::model::ChatTemplate;

/// Maximum number of rows to include when the full dataset exceeds
/// the model's character budget.
const SAMPLE_ROWS: usize = 50;

/// Build a complete prompt from a CSV collection and user question.
///
/// `max_data_chars` caps the CSV section (header plus rows) so the
/// prompt fits inside the target model's context window.
#[must_use]
pub fn build(
    collection: &CsvCollection,
    question: &str,
    template: ChatTemplate,
    max_data_chars: usize,
) -> String {
    let data_section = format_data(collection, max_data_chars);
    let system = "You are a precise data analyst.  You answer questions about CSV data.  \
                  When asked to produce a CSV, output only valid CSV with a header row.  \
                  Be concise.";
    let user = format!(
        "Here is the CSV data:\n\n{data_section}\n\nQuestion: {question}"
    );

    apply_template(template, system, &user)
}

/// Format the CSV collection as a string for inclusion in the prompt.
///
/// If the full dataset is small enough, include everything.
/// Otherwise, include summary statistics and a sample of rows.
fn format_data(collection: &CsvCollection, max_data_chars: usize) -> String {
    let header_line = collection.schema().columns().join(",");
    let full_data = format_all_rows(collection, &header_line);

    if full_data.len() <= max_data_chars {
        full_data
    } else {
        format_sampled(collection, &header_line, max_data_chars)
    }
}

/// Format all rows as CSV text.
fn format_all_rows(collection: &CsvCollection, header_line: &str) -> String {
    let rows_text: String = collection
        .rows()
        .iter()
        .map(|row| row.to_vec().join(","))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{header_line}\n{rows_text}")
}

/// Format a sample of rows with summary statistics.
///
/// Rows are capped by `SAMPLE_ROWS` and by a character budget derived
/// from `max_data_chars` so that wide CSVs (many columns per row) cannot
/// blow past the model context window.
fn format_sampled(collection: &CsvCollection, header_line: &str, max_data_chars: usize) -> String {
    let total = collection.row_count();
    let overhead = header_line.len() + 64;
    let budget = max_data_chars.saturating_sub(overhead);

    let row_strings: Vec<String> = collection
        .rows()
        .iter()
        .take(SAMPLE_ROWS)
        .map(|row| row.to_vec().join(","))
        .collect();

    let shown = fit_count(&row_strings, budget);
    let sample_rows = row_strings
        .iter()
        .take(shown)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "{header_line}\n\
         (showing {shown} of {total} total rows)\n\
         {sample_rows}"
    )
}

/// Count the longest prefix of `rows` whose total character length
/// (including one newline per row) fits within `budget`.
fn fit_count(rows: &[String], budget: usize) -> usize {
    rows.iter()
        .try_fold((0usize, 0usize), |(count, used), row| {
            let next = used + row.len() + 1;
            if next > budget {
                Err(count)
            } else {
                Ok((count + 1, next))
            }
        })
        .map_or_else(|c| c, |(c, _)| c)
}

/// Apply the model-specific chat template.
fn apply_template(template: ChatTemplate, system: &str, user: &str) -> String {
    match template {
        ChatTemplate::Phi3 => format!(
            "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
        ),
        ChatTemplate::ChatMl => format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
    }
}
