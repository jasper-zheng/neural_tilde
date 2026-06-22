#include "tokenizer_config.h"

#include "../backend/json.hpp" // nlohmann/json single-header (vendored)

#include <fstream>
#include <sstream>

using json = nlohmann::json;

static std::string dir_of(const std::string &path) {
  auto slash = path.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

bool load_tokenizer_config(const std::string &config_path, TokenizerConfig &out,
                           std::string &tokenizer_json_abs) {
  std::ifstream f(config_path);
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
  tokenizer_json_abs = dir_of(config_path) + "/" + out.tokenizer_file;
  return true;
}
