#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer; // from tokenizers-cpp (HF fast-tokenizer core)
}

// Thin wrapper over tokenizers-cpp that reproduces the HuggingFace call
//   tokenizer(prompt, truncation=True, max_length=N, padding="max_length")
// for the SA3 t5gemma tokenizer (no special tokens added; right-pad with <pad>).
// Produces the exact input_ids (int64) + attention_mask (int32) the .pte expects.
class NnTokenizer {
public:
  NnTokenizer();
  ~NnTokenizer();

  // Load a HF fast-tokenizer `tokenizer.json`. Returns false on failure.
  bool load(const std::string &tokenizer_json_path);
  bool loaded() const { return (bool)m_tok; }

  // Encode `prompt` into fixed-length `ids` (int64) + `mask` (int32), both of
  // length `max_length`. Truncates/pads on `padding_side` ("right"/"left") using
  // the id of `pad_token`. Returns false if the tokenizer is not loaded.
  bool encode(const std::string &prompt, int max_length,
              const std::string &pad_token, const std::string &padding_side,
              std::vector<int64_t> &ids, std::vector<int32_t> &mask);

private:
  std::unique_ptr<tokenizers::Tokenizer> m_tok;
};
