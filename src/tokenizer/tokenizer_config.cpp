#include "tokenizer_config.h"
#include "nn_platform.h" // nn::fs_path / utf8_string (UTF-8 paths on Windows)

#include "../backend/json.hpp" // nlohmann/json single-header (vendored)

#include <fstream>
#include <sstream>

using json = nlohmann::json;

bool load_tokenizer_config(const std::string &config_path, TokenizerConfig &out,
                           std::string &tokenizer_json_abs) {
  std::ifstream f(nn::fs_path(config_path));
  if (!f)
    return false;
  std::stringstream ss;
  ss << f.rdbuf();

  json j;
  try {
    j = json::parse(ss.str());
  } catch (...) {
    return false;
  }

  out.type = j.value("type", out.type);
  out.tokenizer_file = j.value("tokenizer_file", out.tokenizer_file);
  out.max_length = j.value("max_length", out.max_length);
  out.padding = j.value("padding", out.padding);
  out.pad_token = j.value("pad_token", out.pad_token);
  out.padding_side = j.value("padding_side", out.padding_side);
  out.ids_key = j.value("ids_key", out.ids_key);
  out.mask_key = j.value("mask_key", out.mask_key);

  if (out.tokenizer_file.empty())
    return false;
  // `tokenizer_file` is relative to the config's folder.
  tokenizer_json_abs = nn::utf8_string(nn::fs_path(config_path).parent_path() /
                                       nn::fs_path(out.tokenizer_file));
  return true;
}
