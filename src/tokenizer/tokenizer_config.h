#pragma once
#include <string>

// Standalone tokenizer config file (see EXECUTORCH_PROTOCOL.md §7): describes
// how a text prompt is encoded for a generative model, INDEPENDENT of the model
// .pte. The exporter emits it as "<model-stem>.tokenizer.config.json" beside the
// .pte; it is loaded by neural.tokenizer (Max) and the headless gen_cli host, so
// the tokenizer settings live in one place rather than inside the model metadata.
struct TokenizerConfig {
  std::string type = "tokenizers-cpp";
  std::string tokenizer_file;             // HF tokenizer.json, relative to config
  int max_length = 256;
  std::string padding = "max_length";
  std::string pad_token = "<pad>";
  std::string padding_side = "right";     // "right" | "left"
  std::string ids_key = "input_ids";       // dict key = model token_ids input name
  std::string mask_key = "attention_mask"; // dict key = model attention_mask name
};

// Parse `config_path` into `out` and resolve `tokenizer_file` (relative to the
// config's directory) into `tokenizer_json_abs`. Returns false on a read/parse
// error or if `tokenizer_file` is missing.
bool load_tokenizer_config(const std::string &config_path, TokenizerConfig &out,
                           std::string &tokenizer_json_abs);
