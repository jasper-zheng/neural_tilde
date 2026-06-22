// Verify the C++ tokenizer matches the HF reference token-for-token.
//
//   ./tok_test <tokenizer.json> <golden.json> [max_length] [pad_token] [side]
//
// golden.json is a list of {prompt, input_ids[N], attention_mask[N]} produced by
// export/dump_tokenizer_golden.py (the SA3 HF tokenizer). Defaults: N=256,
// pad_token="<pad>", side="right" (the SA3 t5gemma contract).

#include "tokenizer.h"

#include "../backend/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(
        stderr,
        "usage: %s <tokenizer.json> <golden.json> [max_length] [pad] [side]\n",
        argv[0]);
    return 2;
  }
  std::string tok_path = argv[1];
  std::string golden_path = argv[2];
  int max_length = argc > 3 ? std::atoi(argv[3]) : 256;
  std::string pad_token = argc > 4 ? argv[4] : "<pad>";
  std::string padding_side = argc > 5 ? argv[5] : "right";

  NnTokenizer tok;
  if (!tok.load(tok_path)) {
    std::fprintf(stderr, "FAIL: could not load tokenizer %s\n",
                 tok_path.c_str());
    return 1;
  }

  std::ifstream f(golden_path);
  if (!f) {
    std::fprintf(stderr, "FAIL: could not open %s\n", golden_path.c_str());
    return 1;
  }
  json g;
  f >> g;

  int total = 0, passed = 0;
  for (const auto &item : g) {
    std::string prompt = item["prompt"].get<std::string>();
    auto exp_ids = item["input_ids"].get<std::vector<int64_t>>();
    auto exp_mask = item["attention_mask"].get<std::vector<int32_t>>();

    std::vector<int64_t> ids;
    std::vector<int32_t> mask;
    tok.encode(prompt, max_length, pad_token, padding_side, ids, mask);
    total++;

    if (ids == exp_ids && mask == exp_mask) {
      passed++;
      continue;
    }
    int mm = -1;
    for (size_t i = 0; i < exp_ids.size() && i < ids.size(); i++)
      if (ids[i] != exp_ids[i]) {
        mm = (int)i;
        break;
      }
    std::fprintf(stderr,
                 "  MISMATCH prompt=\"%.50s\": ids %s, mask %s; first id diff "
                 "at %d (got %lld, exp %lld)\n",
                 prompt.c_str(), ids == exp_ids ? "ok" : "DIFF",
                 mask == exp_mask ? "ok" : "DIFF", mm,
                 mm >= 0 ? (long long)ids[mm] : 0,
                 mm >= 0 ? (long long)exp_ids[mm] : 0);
  }

  std::printf("tokenizer parity: %d/%d prompts match token-for-token\n", passed,
              total);
  if (passed != total) {
    std::printf("TOKENIZER FAIL\n");
    return 1;
  }
  std::printf("TOKENIZER OK\n");
  return 0;
}
