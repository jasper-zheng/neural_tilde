// Headless text->audio generation CLI for generative (kind: "generative") models.
// Exercises the full host path neural.gen~ will use (tokenize -> seeded noise ->
// GenRunner -> WAV), without Max. This is the only place that combines the HF
// tokenizer with the Rust-free GenRunner (the Max split is neural.tokenizer +
// neural.gen~), so the prompt->audio orchestration lives here directly.
//
//   gen_cli <model.pte> [--prompt P] [--neg N] [--attr NAME VALUE]... [--seed N]
//                       [--init [NAME=]init.wav]... [--init-offset [NAME=]MS]...
//                       [--strength S] [--noise NAME FILE]... [--cond-json FILE]
//                       [--method forward] [--out out.wav] [--out-raw out.f32]
//
// --noise NAME FILE overrides a noise input by name with raw little-endian
// float32 from FILE (numel must match the input) instead of seeding it — the
// headless analogue of neural.gen~'s jit_matrix noise inlet.
//
// --init/--strength drive an audio-to-audio model (a buffer-role input):
// the init WAV is read into the input by its metadata (resampled, or copied 1:1
// with "resample": false; channel-mapped; cropped/padded with pad_value), and
// --strength is shorthand for --attr init_noise_level S. `--init NAME=FILE`
// targets the buffer input NAME (several per method); a bare `--init FILE` goes
// to the first one. `--init-offset [NAME=]MS` starts the read MS into the source
// (the analogue of neural.gen~'s `init [input] <buffer> [ms]`).
//
// --cond-json FILE supplies condition inputs from a JSON object, keys matched to
// condition names (other keys ignored), converted to each input's dtype like
// neural.gen~'s dictionary path; a scalar value is a 1-element list. When every
// condition comes from the JSON, no tokenizer is needed.
//
// --out-raw FILE also writes the output as raw channel-major float32 (exact; the
// WAV is PCM16 and clipped to [-1, 1]).
//
// --neg sets the negative prompt for a CFG model (extra condition inputs such as
// neg_input_ids/neg_attention_mask). Any condition the positive prompt doesn't
// fill is supplied from --neg, matched by dtype; an empty --neg is unconditional.

#include "../backend/json.hpp"
#include "../tokenizer/tokenizer.h"
#include "../tokenizer/tokenizer_config.h"
#include "gen_runner.h"
#include "wav.h"
// After gen_runner.h: needs executorch's ScalarType in scope.
#include "../frontend/maxmsp/shared/cond_bytes.h" // cond_atoms_to_bytes

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// Derive "<model-stem>.tokenizer.config.json" beside the .pte (the model metadata
// carries no tokenizer block; settings live in this standalone config).
static std::string config_path_for(const std::string &pte_path) {
  std::string stem = pte_path;
  if (stem.size() >= 4 && stem.compare(stem.size() - 4, 4, ".pte") == 0)
    stem = stem.substr(0, stem.size() - 4);
  return stem + ".tokenizer.config.json";
}

// Split "NAME=VALUE" into (NAME, VALUE); without '=' NAME is empty.
static std::pair<std::string, std::string> split_named(const std::string &a) {
  size_t eq = a.find('=');
  if (eq == std::string::npos)
    return {"", a};
  return {a.substr(0, eq), a.substr(eq + 1)};
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.pte> [--prompt P] [--neg N] "
                 "[--attr NAME VALUE]... [--seed N] [--init [NAME=]init.wav]... "
                 "[--init-offset [NAME=]MS]... [--strength S] "
                 "[--noise NAME FILE]... [--cond-json FILE] [--method M] "
                 "[--out out.wav] [--out-raw out.f32]\n",
                 argv[0]);
    return 2;
  }
  std::string pte = argv[1];
  std::string prompt = "techno";
  std::string neg_prompt; // negative prompt for CFG models (empty => unconditional)
  std::string method = "forward";
  std::string out = "out.wav";
  std::string out_raw; // optional exact float32 dump of the output
  // Buffer-input sources: input name ("" => the first buffer input) -> WAV path,
  // and per-input read offsets in ms (same keying).
  std::map<std::string, std::string> init_wavs;
  std::map<std::string, double> init_offsets_ms;
  std::string cond_json; // optional JSON object of condition values by name
  unsigned long long seed = 0;
  // Generic model-attribute overrides by name (e.g. seconds_total, cfg_scale);
  // GenRunner fills any unspecified attribute from its metadata default.
  std::map<std::string, double> attr_values;
  // Noise-input overrides by name -> raw float32 file (the headless analogue of
  // the jit_matrix noise inlet); loaded after the model so numel can be checked.
  std::map<std::string, std::string> noise_paths;

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--prompt")
      prompt = next("--prompt");
    else if (a == "--neg")
      neg_prompt = next("--neg");
    else if (a == "--method")
      method = next("--method");
    else if (a == "--out")
      out = next("--out");
    else if (a == "--attr") {
      std::string name = next("--attr name");
      attr_values[name] = std::atof(next("--attr value").c_str());
    } else if (a == "--noise") {
      std::string name = next("--noise name");
      noise_paths[name] = next("--noise file");
    } else if (a == "--init") {
      auto nv = split_named(next("--init"));
      init_wavs[nv.first] = nv.second;
    } else if (a == "--init-offset") {
      auto nv = split_named(next("--init-offset"));
      init_offsets_ms[nv.first] = std::atof(nv.second.c_str());
    } else if (a == "--cond-json")
      cond_json = next("--cond-json");
    else if (a == "--out-raw")
      out_raw = next("--out-raw");
    else if (a == "--strength")
      // Convenience alias for the audio-to-audio strength attribute.
      attr_values["init_noise_level"] = std::atof(next("--strength").c_str());
    else if (a == "--seed")
      seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
    else
      std::fprintf(stderr, "warning: ignoring unknown arg %s\n", a.c_str());
  }

  GenRunner runner;
  if (runner.load(pte)) {
    std::fprintf(stderr, "FAIL: could not load %s\n", pte.c_str());
    return 1;
  }
  if (!runner.has_method(method)) {
    std::fprintf(stderr, "FAIL: method \"%s\" not found\n", method.c_str());
    return 1;
  }

  // Load the standalone tokenizer config beside the .pte, if present.
  NnTokenizer tok;
  TokenizerConfig cfg;
  bool have_cfg = false;
  {
    std::string tok_json;
    if (load_tokenizer_config(config_path_for(pte), cfg, tok_json)) {
      if (!tok.load(tok_json)) {
        std::fprintf(stderr, "FAIL: could not load tokenizer %s\n",
                     tok_json.c_str());
        return 1;
      }
      have_cfg = true;
    }
  }

  std::printf("model:    %s\n", pte.c_str());
  std::printf("prompt:   \"%s\"\n", prompt.c_str());
  std::printf("seed: %llu   method: %s\n", seed, method.c_str());
  for (const auto &kv : attr_values)
    std::printf("attr:     %s = %g\n", kv.first.c_str(), kv.second);
  std::printf("generating ...\n");

  // Tokenize once if the model declares condition inputs, then key the tokenizer
  // output by the model's condition input names (ids_key/mask_key from the
  // tokenizer config) so bind_condition matches them by name. The cond byte
  // buffers must outlive runner.generate() (the supplied entries reference them).
  const auto &inputs = runner.gen_inputs(method);
  std::map<std::string, std::vector<char>> cond; // input name -> raw dtype bytes
  std::map<std::string, SuppliedInput> supplied;

  // Conditions from a JSON object (the headless analogue of neural.gen~'s
  // dictionary inlet): keys matched to condition names, other keys ignored.
  if (!cond_json.empty()) {
    nlohmann::json j;
    try {
      std::ifstream f(cond_json);
      f >> j;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "FAIL: could not parse %s: %s\n", cond_json.c_str(),
                   e.what());
      return 1;
    }
    if (!j.is_object()) {
      std::fprintf(stderr, "FAIL: %s is not a JSON object\n", cond_json.c_str());
      return 1;
    }
    for (const auto &in : inputs) {
      if (in.role != Role::Condition || !j.contains(in.name))
        continue;
      nlohmann::json v = j[in.name];
      if (!v.is_array())
        v = nlohmann::json::array({v}); // a scalar is a 1-element list
      cond[in.name] = cond_atoms_to_bytes(
          in.dtype, (long)v.size(),
          [&](long k) { return v[k].get<long long>(); },
          [&](long k) { return v[k].get<double>(); });
      std::printf("cond:     %s = %zu value(s) from %s\n", in.name.c_str(),
                  v.size(), cond_json.c_str());
    }
  }

  // The tokenizer is needed only for conditions the JSON did not supply.
  bool needs_tokens = false;
  for (const auto &in : inputs)
    if (in.role == Role::Condition && !cond.count(in.name))
      needs_tokens = true;

  std::vector<int64_t> ids;
  std::vector<int32_t> mask;
  std::vector<int64_t> neg_ids;
  std::vector<int32_t> neg_mask;
  if (needs_tokens) {
    if (!have_cfg || !tok.loaded()) {
      std::fprintf(stderr, "FAIL: model needs a tokenizer but no "
                           "<model>.tokenizer.config.json was found beside the "
                           ".pte\n");
      return 1;
    }
    int max_len = cfg.max_length > 0 ? cfg.max_length : 256;
    if (!tok.encode(prompt, max_len, cfg.pad_token, cfg.padding_side, ids,
                    mask)) {
      std::fprintf(stderr, "FAIL: tokenization failed\n");
      return 1;
    }
    // Never overwrite a condition the JSON already supplied.
    auto put = [&](const std::string &key, const void *p, size_t bytes) {
      if (!cond.count(key))
        cond[key].assign((const char *)p, (const char *)p + bytes);
    };
    put(cfg.ids_key, ids.data(), ids.size() * sizeof(int64_t));
    put(cfg.mask_key, mask.data(), mask.size() * sizeof(int32_t));

    // Negative-prompt conditions (CFG models, e.g. neg_input_ids /
    // neg_attention_mask): any condition the positive prompt didn't fill is
    // supplied from --neg, matched by dtype (int64 => neg ids, int32 => neg
    // mask). An empty --neg tokenizes to all-pad / zero-mask = unconditional.
    bool need_neg = false;
    for (const auto &in : inputs)
      if (in.role == Role::Condition && !cond.count(in.name))
        need_neg = true;
    if (need_neg) {
      if (!tok.encode(neg_prompt, max_len, cfg.pad_token, cfg.padding_side,
                      neg_ids, neg_mask)) {
        std::fprintf(stderr, "FAIL: negative-prompt tokenization failed\n");
        return 1;
      }
      std::printf("neg:      \"%s\"\n", neg_prompt.c_str());
      for (const auto &in : inputs) {
        if (in.role != Role::Condition || cond.count(in.name))
          continue;
        if (in.dtype == executorch::aten::ScalarType::Long)
          put(in.name, neg_ids.data(), neg_ids.size() * sizeof(int64_t));
        else if (in.dtype == executorch::aten::ScalarType::Int)
          put(in.name, neg_mask.data(), neg_mask.size() * sizeof(int32_t));
        else
          std::fprintf(stderr,
                       "warning: condition '%s' left unfilled (unexpected "
                       "dtype for a negative-prompt input)\n",
                       in.name.c_str());
      }
    }

  }
  GenRunner::bind_condition(inputs, cond, supplied);

  // Buffer inputs: each --init WAV is read into its buffer input by the input's
  // metadata (GenRunner::prepare_buffer_input: resample or 1:1, channel map,
  // crop/pad). The buffers must outlive runner.generate() (same lifetime rule as
  // ids/mask). A buffer input without --init is filled with its pad_value.
  std::map<std::string, std::vector<float>> buffers;
  {
    std::vector<const InputSpec *> buf_inputs;
    for (const auto &in : inputs)
      if (in.role == Role::Buffer)
        buf_inputs.push_back(&in);
    const int fallback_sr = runner.output_spec(method).sample_rate;
    for (const auto &kv : init_wavs) {
      const InputSpec *bi = nullptr;
      if (kv.first.empty()) {
        if (!buf_inputs.empty())
          bi = buf_inputs.front();
        if (buf_inputs.size() > 1)
          std::fprintf(stderr, "warning: model has %zu buffer inputs, bare --init "
                               "goes to the first (\"%s\"); use --init NAME=FILE\n",
                       buf_inputs.size(), bi->name.c_str());
      } else {
        for (const auto *b : buf_inputs)
          if (b->name == kv.first)
            bi = b;
      }
      if (!bi) {
        std::fprintf(stderr, "warning: --init %s%s ignored: no such buffer input\n",
                     kv.first.empty() ? "" : (kv.first + "=").c_str(),
                     kv.second.c_str());
        continue;
      }
      std::vector<float> src;
      int sc = 0, sf = 0, ssr = 0;
      if (!read_wav(kv.second, src, sc, sf, ssr)) {
        std::fprintf(stderr, "FAIL: could not read init WAV %s\n",
                     kv.second.c_str());
        return 1;
      }
      double off_ms = 0.0;
      auto ot = init_offsets_ms.find(kv.first);
      if (ot == init_offsets_ms.end())
        ot = init_offsets_ms.find(bi->name);
      if (ot != init_offsets_ms.end())
        off_ms = ot->second;
      const auto g = GenRunner::buffer_geometry(*bi, fallback_sr);
      std::printf("init:     %s <- %s  (%d ch x %d @ %d Hz -> %d ch x %d @ %d Hz%s, "
                  "offset %g ms)\n",
                  bi->name.c_str(), kv.second.c_str(), sc, sf, ssr, g.ch, g.len,
                  g.sr, bi->resample ? "" : ", 1:1", off_ms);
      if (!bi->resample && ssr != g.sr)
        std::fprintf(stderr, "warning: \"%s\" is read 1:1; the WAV's sample rate "
                             "%d Hz is ignored (declared %d Hz)\n",
                     bi->name.c_str(), ssr, g.sr);
      GenRunner::prepare_buffer_input(*bi, fallback_sr, src, sc, (size_t)sf,
                                      (double)ssr, off_ms, buffers[bi->name]);
    }
    for (const auto *b : buf_inputs)
      if (!buffers.count(b->name))
        std::fprintf(stderr, "warning: buffer input \"%s\" not supplied, filling "
                             "with %g\n",
                     b->name.c_str(), b->pad_value);
    GenRunner::bind_buffers(inputs, buffers, supplied);
  }

  // Noise overrides: read each --noise FILE as raw float32 and bind it by name
  // (the headless analogue of neural.gen~'s jit_matrix inlet). noise_inputs must
  // outlive runner.generate(); generate() validates numel/dtype per input.
  std::map<std::string, std::vector<float>> noise_inputs;
  for (const auto &kv : noise_paths) {
    std::ifstream f(kv.second, std::ios::binary | std::ios::ate);
    if (!f) {
      std::fprintf(stderr, "FAIL: could not read noise file %s\n",
                   kv.second.c_str());
      return 1;
    }
    std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> v(bytes / (std::streamsize)sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()),
           (std::streamsize)(v.size() * sizeof(float)));
    std::printf("noise:    %s = %s  (%zu floats)\n", kv.first.c_str(),
                kv.second.c_str(), v.size());
    noise_inputs[kv.first] = std::move(v);
  }
  if (!noise_inputs.empty())
    GenRunner::bind_noise(inputs, noise_inputs, supplied);

  std::vector<float> audio;
  if (runner.generate(method, supplied, attr_values, (uint64_t)seed, audio)) {
    std::fprintf(stderr, "FAIL: generate() error\n");
    return 1;
  }

  OutputSpec spec = runner.output_spec(method);
  int channels = spec.channels, length = spec.length, sr = spec.sample_rate;
  // Fall back to spec.shape if the descriptors are absent.
  if (channels <= 0 || length <= 0) {
    // Assume (1, channels, length) or (channels, length).
    if (spec.shape.size() == 3) {
      channels = spec.shape[1];
      length = spec.shape[2];
    } else if (spec.shape.size() == 2) {
      channels = spec.shape[0];
      length = spec.shape[1];
    }
  }
  if (sr <= 0)
    sr = 44100;

  double mn = 1e30, mx = -1e30;
  int nonfinite = 0;
  for (float v : audio) {
    if (!std::isfinite(v))
      nonfinite++;
    mn = std::min(mn, (double)v);
    mx = std::max(mx, (double)v);
  }
  std::printf("output:   %zu floats  (%d ch x %d)  %.3f s @ %d Hz\n",
              audio.size(), channels, length, (double)length / sr, sr);
  std::printf("          min=%.5f max=%.5f nonfinite=%d\n", mn, mx, nonfinite);
  if (nonfinite) {
    std::fprintf(stderr, "FAIL: non-finite output\n");
    return 1;
  }

  if (!out_raw.empty()) {
    std::ofstream f(out_raw, std::ios::binary);
    f.write(reinterpret_cast<const char *>(audio.data()),
            (std::streamsize)(audio.size() * sizeof(float)));
    if (!f) {
      std::fprintf(stderr, "FAIL: could not write %s\n", out_raw.c_str());
      return 1;
    }
    std::printf("wrote %s\n", out_raw.c_str());
  }

  if (!write_wav(out, audio, channels, length, sr)) {
    std::fprintf(stderr, "FAIL: could not write %s\n", out.c_str());
    return 1;
  }
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
