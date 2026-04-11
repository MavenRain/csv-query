//! Prompt construction from CSV data and user queries.
//!
//! Formats schema, data rows, and the user's question into a
//! chat-templated prompt suitable for the target model.  When the
//! full dataset exceeds [`MAX_DATA_CHARS`] characters, a sample of
//! [`SAMPLE_ROWS`] rows is included along with the total row count,
//! keeping the prompt within the model's context window.

use crate::collection::CsvCollection;
use crate::model::ChatTemplate;

/// Maximum number of characters of CSV data to include in the prompt.
/// Keeps prompt within model context window for typical tokenizer ratios.
const MAX_DATA_CHARS: usize = 8_000;

/// Maximum number of rows to include when the full dataset exceeds
/// `MAX_DATA_CHARS`.
const SAMPLE_ROWS: usize = 50;

/// Build a complete prompt from a CSV collection and user question.
#[must_use]
pub fn build(
    collection: &CsvCollection,
    question: &str,
    template: ChatTemplate,
) -> String {
    let data_section = format_data(collection);
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
fn format_data(collection: &CsvCollection) -> String {
    let header_line = collection.schema().columns().join(",");
    let full_data = format_all_rows(collection, &header_line);

    if full_data.len() <= MAX_DATA_CHARS {
        full_data
    } else {
        format_sampled(collection, &header_line)
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
fn format_sampled(collection: &CsvCollection, header_line: &str) -> String {
    let total = collection.row_count();
    let sample_rows: String = collection
        .rows()
        .iter()
        .take(SAMPLE_ROWS)
        .map(|row| row.to_vec().join(","))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "{header_line}\n\
         (showing {SAMPLE_ROWS} of {total} total rows)\n\
         {sample_rows}"
    )
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
