//! SLM loading and inference via candle.
//!
//! Supports multiple quantized model architectures with GGUF weights
//! auto-downloaded from Hugging Face Hub.  The public API is
//! [`generate`], which wraps the full download-load-infer pipeline
//! in an `Io<Error, String>` so the caller composes without
//! executing effects.
//!
//! Currently supported models:
//!
//! - **Phi-3-mini-4k-Instruct** (default): Microsoft's 3.8B parameter
//!   model, quantized to Q4.
//! - **SmolLM2-1.7B-Instruct**: `HuggingFace`'s 1.7B parameter model,
//!   quantized to `Q4_K_M`.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama as llama;
use candle_transformers::models::quantized_phi3 as phi3;
use comp_cat_rs::effect::io::Io;

use crate::error::Error;

/// Conservative cap on prompt-plus-generation tokens.  Matches
/// Phi-3-mini-4k's 4096 context and stays safely below `SmolLM2`'s 8192.
const MAX_CONTEXT_TOKENS: usize = 4096;

/// Identifies a supported model.
#[derive(Debug, Clone)]
pub enum ModelId {
    /// Microsoft Phi-3-mini-4k-Instruct (default).
    Phi3,
    /// SmolLM2-1.7B-Instruct.
    SmolLm2,
}

/// Which candle architecture to use for loading weights.
#[derive(Debug, Clone, Copy)]
enum Architecture {
    Phi3,
    Llama,
}

/// Chat template format for prompt construction.
#[derive(Debug, Clone, Copy)]
pub enum ChatTemplate {
    /// Phi-3 format: `<|system|>...<|end|><|user|>...<|end|><|assistant|>`
    Phi3,
    /// `ChatML` format: `<|im_start|>system\n...<|im_end|>`
    ChatMl,
}

/// Specification for a model: where to download, which architecture, how to prompt.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    weights_repo: String,
    weights_file: String,
    tokenizer_repo: String,
    architecture: Architecture,
    template: ChatTemplate,
}

impl ModelSpec {
    /// The chat template for this model.
    #[must_use]
    pub fn template(&self) -> ChatTemplate {
        self.template
    }

}

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    max_tokens: usize,
    temperature: f64,
    seed: u64,
}

impl InferenceConfig {
    /// Default inference config: 512 max tokens, temperature 0.7.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            seed: 42,
        }
    }

    /// Set maximum tokens to generate.
    #[must_use]
    pub fn max_tokens(self, n: usize) -> Self {
        Self { max_tokens: n, ..self }
    }

    /// Set sampling temperature.
    #[must_use]
    pub fn temperature(self, t: f64) -> Self {
        Self { temperature: t, ..self }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self { Self::new() }
}

/// Parse a model name string into a `ModelId`.
///
/// # Errors
///
/// Returns `Error::ModelNotFound` if the name is not recognized.
pub fn parse_model_id(name: &str) -> Result<ModelId, Error> {
    match name {
        "phi3" | "phi-3" | "phi3-mini" => Ok(ModelId::Phi3),
        "smollm2" | "smollm" => Ok(ModelId::SmolLm2),
        other => Err(Error::ModelNotFound(format!(
            "unknown model '{other}'; supported: phi3, smollm2"
        ))),
    }
}

/// Get the model specification for a given model ID.
#[must_use]
pub fn spec_for(id: &ModelId) -> ModelSpec {
    match id {
        ModelId::Phi3 => ModelSpec {
            weights_repo: "microsoft/Phi-3-mini-4k-instruct-gguf".into(),
            weights_file: "Phi-3-mini-4k-instruct-q4.gguf".into(),
            tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct".into(),
            architecture: Architecture::Phi3,
            template: ChatTemplate::Phi3,
        },
        ModelId::SmolLm2 => ModelSpec {
            weights_repo: "bartowski/SmolLM2-1.7B-Instruct-GGUF".into(),
            weights_file: "SmolLM2-1.7B-Instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct".into(),
            architecture: Architecture::Llama,
            template: ChatTemplate::ChatMl,
        },
    }
}

/// Run inference: download model if needed, load, generate text.
///
/// This is the main public entry point.  Everything is wrapped in
/// `Io` so the caller composes without executing effects.
///
/// # Errors
///
/// - `Error::Io` if model files cannot be downloaded or opened.
/// - `Error::Model` if GGUF weight loading or forward pass fails.
/// - `Error::Tokenizer` if the prompt cannot be encoded or output
///   tokens cannot be decoded.
pub fn generate(
    spec: ModelSpec,
    config: InferenceConfig,
    prompt: String,
) -> Io<Error, String> {
    Io::suspend(move || generate_inner(&spec, &config, &prompt))
}

/// Internal inference pipeline, called inside `Io::suspend`.
#[allow(clippy::too_many_lines)]
fn generate_inner(
    spec: &ModelSpec,
    config: &InferenceConfig,
    prompt: &str,
) -> Result<String, Error> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);

    // Download model weights and tokenizer
    let (weights_path, tokenizer_path) = download_model(spec)?;

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

    // Load GGUF weights
    let weights_file = std::fs::File::open(&weights_path)?;
    #[allow(unused_mut)]
    let mut reader = std::io::BufReader::new(weights_file);
    let content = gguf_file::Content::read(&mut reader)
        .map_err(candle_core::Error::wrap)?;

    // Load model and run inference based on architecture
    match spec.architecture {
        Architecture::Phi3 => {
            #[allow(unused_mut)]
            let mut model = phi3::ModelWeights::from_gguf(false, content, &mut reader, &device)?;
            run_inference(&mut model, &tokenizer, prompt, config, &device)
        }
        Architecture::Llama => {
            #[allow(unused_mut)]
            let mut model = llama::ModelWeights::from_gguf(content, &mut reader, &device)?;
            run_inference(&mut model, &tokenizer, prompt, config, &device)
        }
    }
}

/// Download model files from Hugging Face Hub, returning local paths.
fn download_model(spec: &ModelSpec) -> Result<(String, String), Error> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            let api = hf_hub::api::tokio::Api::new()
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;

            let weights_repo = api.model(spec.weights_repo.clone());
            let weights_path = weights_repo
                .get(&spec.weights_file)
                .await
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;

            let tokenizer_repo = api.model(spec.tokenizer_repo.clone());
            let tokenizer_path = tokenizer_repo
                .get("tokenizer.json")
                .await
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;

            Ok((
                weights_path.to_string_lossy().into_owned(),
                tokenizer_path.to_string_lossy().into_owned(),
            ))
        })
    })
}

/// Trait for quantized models that support forward pass.
///
/// This allows `run_inference` to be generic over model architectures
/// without using `dyn Trait`.
trait QuantizedForward {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor>;
}

impl QuantizedForward for phi3::ModelWeights {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(x, index_pos)
    }
}

impl QuantizedForward for llama::ModelWeights {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(x, index_pos)
    }
}

/// Run the token generation loop.
#[allow(unused_mut)]
fn run_inference<M: QuantizedForward>(
    model: &mut M,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    config: &InferenceConfig,
    device: &Device,
) -> Result<String, Error> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let prompt_tokens = encoding.get_ids();
    let prompt_len = prompt_tokens.len();

    let context_budget = MAX_CONTEXT_TOKENS.saturating_sub(config.max_tokens);
    (prompt_len <= context_budget)
        .then_some(())
        .ok_or_else(|| Error::Model(candle_core::Error::Msg(format!(
            "prompt tokenizes to {prompt_len} tokens, exceeding the \
             {context_budget}-token budget ({MAX_CONTEXT_TOKENS} ctx minus \
             {} for generation); reduce the CSV size or --max-tokens",
            config.max_tokens
        ))))?;

    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let last_logits = logits
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?;

    let sampling = if config.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::All { temperature: config.temperature }
    };
    let mut logits_proc = LogitsProcessor::from_sampling(config.seed, sampling);

    let first_token = logits_proc.sample(&last_logits)?;

    let eos_token = tokenizer
        .token_to_id("<|end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .unwrap_or(2);

    // Generate tokens via try_fold.  Short-circuits on error;
    // stops generating once the done flag is set.
    let (generated_tokens, _, _) = (1..config.max_tokens).try_fold(
        (vec![first_token], prompt_len, first_token == eos_token),
        |(tokens, pos, done), _| -> Result<(Vec<u32>, usize, bool), Error> {
            if done {
                Ok((tokens, pos, true))
            } else {
                let last = tokens[tokens.len() - 1];
                let input = Tensor::new(&[last], device)?.unsqueeze(0)?;
                let logits = model.forward(&input, pos)?;
                let logits = logits
                    .squeeze(0)?
                    .to_dtype(candle_core::DType::F32)?;
                let next = logits_proc.sample(&logits)?;
                let is_done = next == eos_token;
                let new_tokens: Vec<u32> =
                    tokens.into_iter().chain(std::iter::once(next)).collect();
                Ok((new_tokens, pos + 1, is_done))
            }
        },
    )?;

    tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| Error::Tokenizer(e.to_string()))
}
