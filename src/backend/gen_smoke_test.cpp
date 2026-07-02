// Standalone test harness for the generative (kind: "generative") Backend path,
// the counterpart to et_smoke_test for signal models. Used two ways:
//
//   ./gen_smoke_test <model.pte> [method]
//       Structural smoke test: print the parsed generative spec (typed inputs,
//       output, tokenizer), build ZERO-filled inputs of the right dtype/shape,
//       run generate(), and report output stats (shape, finite, range).
//
//   ./gen_smoke_test <model.pte> <method> --io <dir>
//       Parity gate: read one raw little-endian blob per input from
//       <dir>/<input-name>.bin (dtype/shape from the metadata), run generate(),
//       and compare the audio output against <dir>/audio_ref.bin. PASS if
//       max|Δ| <= 1e-4 (the protocol's C++-vs-Python-.pte gate, §7.6).
//
// Exercises the same Backend::generate() path neural.gen~ will use, without Max.

#include "backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static const char *role_name(Role r) {
  switch (r) {
  case Role::Signal:
    return "signal";
  case Role::Condition:
    return "condition";
  case Role::Noise:
    return "noise";
  case Role::Attribute:
    return "attribute";
  case Role::Buffer:
    return "buffer";
  }
  return "?";
}

static const char *dtype_name(executorch::aten::ScalarType t) {
  using executorch::aten::ScalarType;
  if (t == ScalarType::Long)
    return "int64";
  if (t == ScalarType::Int)
    return "int32";
  if (t == ScalarType::Float)
    return "float32";
  return "?";
}

static size_t dtype_size(executorch::aten::ScalarType t) {
  using executorch::aten::ScalarType;
  if (t == ScalarType::Long)
    return 8;
  return 4; // Int / Float
}

static size_t numel_of(const std::vector<int> &shape) {
  size_t n = 1;
  for (int d : shape)
    n *= (size_t)d;
  return n;
}

static std::string shape_str(const std::vector<int> &shape) {
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); i++)
    s += std::to_string(shape[i]) + (i + 1 < shape.size() ? ", " : "");
  return s + ")";
}

static std::vector<char> read_bytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<char> v;
  if (!f)
    return v;
  auto bytes = f.tellg();
  f.seekg(0);
  v.resize((size_t)bytes);
  f.read(v.data(), v.size());
  return v;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.pte> [method] [--io <dir>]\n", argv[0]);
    return 2;
  }
  std::string path = argv[1];
  std::string method = (argc > 2 && argv[2][0] != '-') ? argv[2] : "forward";
  bool io_mode = false;
  std::string io_dir;
  for (int i = 2; i < argc; i++) {
    if (std::string(argv[i]) == "--io" && i + 1 < argc) {
      io_mode = true;
      io_dir = argv[i + 1];
    }
  }

  Backend backend;
  if (backend.load(path)) {
    std::fprintf(stderr, "FAIL: load(\"%s\") returned error\n", path.c_str());
    return 1;
  }

  std::printf("loaded: %s\n  generative=%d\n", path.c_str(),
              (int)backend.is_generative());
  if (!backend.is_generative()) {
    std::fprintf(stderr,
                 "FAIL: \"%s\" is not a generative model (use et_smoke_test)\n",
                 path.c_str());
    return 1;
  }
  if (!backend.has_method(method)) {
    std::fprintf(stderr, "FAIL: method \"%s\" not found\n", method.c_str());
    return 1;
  }

  const auto &inputs = backend.method_inputs(method);
  OutputSpec out = backend.method_output(method);

  std::printf("  method %s: %zu inputs\n", method.c_str(), inputs.size());
  for (const auto &in : inputs)
    std::printf("    - %-16s role=%-14s dtype=%-7s shape=%s\n", in.name.c_str(),
                role_name(in.role), dtype_name(in.dtype),
                shape_str(in.shape).c_str());
  std::printf("  output: %s dtype=%s shape=%s  channels=%d length=%d sr=%d "
              "(channel-major)\n",
              out.name.c_str(), dtype_name(out.dtype),
              shape_str(out.shape).c_str(), out.channels, out.length,
              out.sample_rate);
  std::printf("  seed=%d\n", backend.gen_seed());

  // Assemble input buffers (kept alive for the generate() call).
  std::vector<std::vector<char>> storage(inputs.size());
  std::vector<const void *> in_ptrs(inputs.size());

  if (io_mode) {
    for (size_t i = 0; i < inputs.size(); i++) {
      const auto &spec = inputs[i];
      std::string p = io_dir + "/" + spec.name + ".bin";
      storage[i] = read_bytes(p);
      size_t want = numel_of(spec.shape) * dtype_size(spec.dtype);
      if (storage[i].size() != want) {
        std::fprintf(stderr,
                     "FAIL: %s is %zu bytes, expected %zu (%s %s)\n", p.c_str(),
                     storage[i].size(), want, dtype_name(spec.dtype),
                     shape_str(spec.shape).c_str());
        return 1;
      }
      in_ptrs[i] = storage[i].data();
    }
  } else {
    for (size_t i = 0; i < inputs.size(); i++) {
      const auto &spec = inputs[i];
      storage[i].assign(numel_of(spec.shape) * dtype_size(spec.dtype), 0);
      in_ptrs[i] = storage[i].data();
    }
  }

  std::printf("  running generate(\"%s\")%s ...\n", method.c_str(),
              io_mode ? " on golden inputs" : " on zero inputs");
  std::vector<float> audio;
  if (backend.generate(method, in_ptrs, audio)) {
    std::fprintf(stderr, "FAIL: generate(\"%s\") returned error\n",
                 method.c_str());
    return 1;
  }

  // Output stats.
  double mn = 1e30, mx = -1e30, sum = 0.0;
  int nonfinite = 0;
  for (float v : audio) {
    if (!std::isfinite(v))
      nonfinite++;
    mn = std::min(mn, (double)v);
    mx = std::max(mx, (double)v);
    sum += v;
  }
  std::printf("  output: %zu floats  min=%.5f max=%.5f mean=%.6f nonfinite=%d\n",
              audio.size(), mn, mx, sum / (double)std::max<size_t>(1, audio.size()),
              nonfinite);
  if (nonfinite) {
    std::fprintf(stderr, "FAIL: %d non-finite output samples\n", nonfinite);
    return 1;
  }

  if (io_mode) {
    std::vector<char> refb = read_bytes(io_dir + "/audio_ref.bin");
    if (refb.size() != audio.size() * sizeof(float)) {
      std::fprintf(stderr,
                   "FAIL: audio_ref.bin is %zu bytes, output is %zu floats "
                   "(%zu bytes)\n",
                   refb.size(), audio.size(), audio.size() * sizeof(float));
      return 1;
    }
    const float *ref = reinterpret_cast<const float *>(refb.data());
    double max_abs = 0.0;
    for (size_t i = 0; i < audio.size(); i++)
      max_abs = std::max(max_abs, (double)std::fabs(audio[i] - ref[i]));
    std::printf("  max|Δ| vs Python .pte: %.3e  %s\n", max_abs,
                max_abs <= 1e-4 ? "<= 1e-4 ✓" : "> 1e-4 ✗");
    if (max_abs > 1e-4) {
      std::fprintf(stderr, "PARITY FAIL\n");
      return 1;
    }
    std::printf("PARITY OK\n");
    return 0;
  }

  std::printf("SMOKE TEST OK\n");
  return 0;
}
