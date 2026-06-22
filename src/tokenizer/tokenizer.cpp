#include "tokenizer.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include <tokenizers_cpp.h>

NnTokenizer::NnTokenizer() = default;
NnTokenizer::~NnTokenizer() = default;

static std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool NnTokenizer::load(const std::string &tokenizer_json_path) {
  std::string blob = read_file(tokenizer_json_path);
  if (blob.empty()) {
    std::cerr << "neural: could not read tokenizer " << tokenizer_json_path
              << std::endl;
    return false;
  }
  m_tok = tokenizers::Tokenizer::FromBlobJSON(blob);
  return (bool)m_tok;
}

bool NnTokenizer::encode(const std::string &prompt, int max_length,
                         const std::string &pad_token,
                         const std::string &padding_side,
                         std::vector<int64_t> &ids, std::vector<int32_t> &mask) {
  if (!m_tok)
    return false;

  // Content tokens (no special tokens for this tokenizer: add_bos/add_eos=False).
  std::vector<int32_t> enc = m_tok->Encode(prompt);

  // truncation=True, max_length=N: keep the first N tokens.
  if ((int)enc.size() > max_length)
    enc.resize(max_length);

  const int32_t pad_id = m_tok->TokenToId(pad_token);
  const int content = (int)enc.size();

  ids.assign(max_length, (int64_t)pad_id);
  mask.assign(max_length, 0);

  // padding="max_length" on padding_side; mask = 1 for content, 0 for pad.
  const int off = (padding_side == "left") ? (max_length - content) : 0;
  for (int i = 0; i < content; i++) {
    ids[off + i] = (int64_t)enc[i];
    mask[off + i] = 1;
  }
  return true;
}
