//! csv-query: query CSV collections using an embedded small language model.
//!
//! This is the entry point for the `csv-query` binary.  It wires
//! together CLI parsing, source resolution, collection loading,
//! prompt construction, model inference, and output.
//!
//! The entire pipeline is built as a single `Io<Error, String>` value
//! via combinator chains.  Nothing executes until `run` is called once
//! here at the boundary, keeping effects explicit and composable.

mod cli;
mod collection;
mod error;
mod model;
mod prompt;
mod source;

use clap::Parser;
use comp_cat_rs::effect::io::Io;

use crate::cli::Cli;
use crate::error::Error;
use crate::source::CsvSource;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = tokio::task::block_in_place(|| build_pipeline(&cli).run());

    match result {
        Ok(output) => handle_output(&cli, &output),
        Err(e) => eprintln!("error: {e}"),
    }
}

/// Build the full Io pipeline from CLI arguments.
///
/// Nothing executes until `run` is called in `main`.
fn build_pipeline(cli: &Cli) -> Io<Error, String> {
    let sources: Vec<CsvSource> = cli
        .sources()
        .iter()
        .map(|s| CsvSource::parse(s))
        .collect();

    let model_name = cli.model_name().to_owned();
    let prompt_text = cli.prompt().to_owned();
    let max_tokens = cli.max_tokens();
    let temperature = cli.temperature();

    source::resolve(sources)
        .map_error(Error::from)
        .flat_map(|resolved| {
            collection::load(resolved)
                .map_error(Error::from)
        })
        .flat_map(move |coll| {
            let model_id = model::parse_model_id(&model_name);
            Io::suspend(move || model_id)
                .flat_map(move |id| {
                    let spec = model::spec_for(&id);
                    let template = spec.template();
                    let full_prompt = prompt::build(&coll, &prompt_text, template);
                    let inference_config = model::InferenceConfig::new()
                        .max_tokens(max_tokens)
                        .temperature(temperature);
                    model::generate(spec, inference_config, full_prompt)
                })
        })
}

/// Write output to a file or stdout.
fn handle_output(cli: &Cli, output: &str) {
    cli.output_path().map_or_else(
        || print!("{output}"),
        |path| {
            std::fs::write(path, output).unwrap_or_else(|e| {
                eprintln!("failed to write output to {path}: {e}");
            });
        },
    );
}
