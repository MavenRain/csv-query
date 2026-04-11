# csv-query

Query CSV collections using an embedded small language model.  No API keys, no cloud services; inference runs locally via quantized GGUF weights downloaded from Hugging Face Hub.

## Features

- **Local and remote sources.**  Point at local files with glob patterns (`data/*.csv`, `reports/2024*`) or fetch remote CSVs via URL.  Mix freely in a single invocation.
- **Schema validation.**  All sources must share the same header row.  Mismatches produce a clear error before any inference begins.
- **Embedded SLM inference.**  Ships with support for Phi-3-mini-4k-Instruct (default) and SmolLM2-1.7B-Instruct.  Models are auto-downloaded on first use and cached locally.
- **Metal acceleration.**  On Apple Silicon, `candle` uses the Metal backend for faster inference out of the box.
- **Configurable generation.**  Control model selection, sampling temperature, and maximum output length from the command line.
- **Functional effects pipeline.**  Built on [comp-cat-rs](https://crates.io/crates/comp-cat-rs) and [csv-cat](https://crates.io/crates/csv-cat).  The entire pipeline is composed as a lazy `Io<Error, String>` value; nothing executes until `run` is called once at the boundary.

## Installation

```sh
cargo install --path .
```

Or build from source:

```sh
cargo build --release
```

The binary lands at `target/release/csv-query`.

## Usage

```
csv-query [OPTIONS] --source <SOURCE> <PROMPT>
```

### Required arguments

| Argument | Description |
|---|---|
| `--source`, `-s` | CSV source: a glob pattern or URL (repeatable) |
| `<PROMPT>` | Natural-language question or instruction about the data |

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `phi3` | Model to use (`phi3`, `smollm2`) |
| `--output`, `-o` | stdout | Write output to a file instead of stdout |
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (`0.0` for greedy) |

### Examples

Ask a question about local CSV files:

```sh
csv-query -s "sales/*.csv" "What was the total revenue in Q4?"
```

Combine local and remote sources:

```sh
csv-query \
  -s "local_data/*.csv" \
  -s "https://example.com/remote_data.csv" \
  "List the top 5 products by unit volume"
```

Produce a new CSV from the collection:

```sh
csv-query -s "logs/*.csv" -o summary.csv \
  "Produce a CSV with columns date, error_count, warning_count summarizing each day"
```

Use a different model with greedy decoding:

```sh
csv-query -s "data.csv" -m smollm2 --temperature 0.0 \
  "How many unique customers appear in this dataset?"
```

## Architecture

The codebase is organized by domain context, not technical layer:

```
src/
  main.rs        Entry point.  Wires the pipeline and calls run once.
  cli.rs         Command-line interface (clap derive).
  source.rs      Source resolution: glob expansion and HTTP fetch.
  collection.rs  Schema validation and row merging across sources.
  model.rs       Model download, GGUF loading, and token generation.
  prompt.rs      Chat-template prompt construction from CSV data.
  error.rs       Project-wide error enum with From impls for every dependency.
```

### Pipeline

The pipeline is a chain of `Io` combinators that composes without executing:

1. **Parse sources.**  Each `--source` string becomes a `CsvSource::Local` (glob) or `CsvSource::Remote` (URL).
2. **Resolve.**  Globs expand to file paths; URLs are fetched via HTTP.  All results are `ResolvedSource` values.
3. **Load.**  Each resolved source is read with csv-cat.  Headers are validated to match the schema established by the first file.  Rows are merged into a single `CsvCollection`.
4. **Build prompt.**  The collection's schema and data (or a sample, if the data exceeds 8,000 characters) are formatted into the model's chat template alongside the user's question.
5. **Generate.**  The model weights are downloaded (if not cached), loaded from GGUF, and used for autoregressive token generation.

`run` is called exactly once in `main`, at the boundary.

## Supported models

| CLI name | Model | Architecture | Quantization | Size |
|---|---|---|---|---|
| `phi3` (default) | Phi-3-mini-4k-Instruct | Phi-3 | Q4 | ~2.2 GB |
| `smollm2` | SmolLM2-1.7B-Instruct | Llama | Q4_K_M | ~1.0 GB |

Weights are auto-downloaded from Hugging Face Hub on first run and cached in `~/.cache/huggingface/`.

## Dependencies

| Crate | Role |
|---|---|
| [comp-cat-rs](https://crates.io/crates/comp-cat-rs) | Functional effects framework (`Io`, `Stream`, `Resource`) |
| [csv-cat](https://crates.io/crates/csv-cat) | CSV reading/writing built on comp-cat-rs |
| [candle-core](https://crates.io/crates/candle-core) | Tensor operations and GGUF weight loading (Metal-accelerated) |
| [candle-transformers](https://crates.io/crates/candle-transformers) | Quantized Phi-3 and Llama model architectures |
| [hf-hub](https://crates.io/crates/hf-hub) | Hugging Face Hub model downloading |
| [tokenizers](https://crates.io/crates/tokenizers) | HuggingFace tokenizer for prompt encoding |
| [clap](https://crates.io/crates/clap) | Command-line argument parsing |
| [reqwest](https://crates.io/crates/reqwest) | HTTP client for remote CSV fetching |
| [tokio](https://crates.io/crates/tokio) | Async runtime (bridged into synchronous `Io` via `block_in_place`) |
| [glob](https://crates.io/crates/glob) | File pattern matching |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.
