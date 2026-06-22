// Standalone test harness for the ExecuTorch Backend, used two ways:
//
//   ./et_smoke_test <model.pte> [method]
//       Smoke test: run `method` (default forward) on a synthetic block and
//       report output statistics (finite, range).
//
//   ./et_smoke_test <model.pte> <method> --io <in.f32> <out.f32>
//       Parity mode: read raw float32 mono samples from <in.f32>, process them
//       block-by-block through Backend (fresh load => zero streaming state),
//       and write the raw float32 output to <out.f32>. Used to compare the C++
//       MLX runtime against the eager PyTorch reference (only in_dim==out_dim==1,
//       i.e. `forward`).
//
// Exercises the same Backend path the Max externals use, without needing Max.

#include "backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<float> v;
  if (!f) return v;
  auto bytes = f.tellg();
  f.seekg(0);
  v.resize(bytes / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float));
  return v;
}

static void write_f32(const std::string &path, const std::vector<float> &v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.pte> [method] [--io <in.f32> <out.f32>]\n",
                 argv[0]);
    return 2;
  }
  std::string path = argv[1];
  std::string method = (argc > 2 && argv[2][0] != '-') ? argv[2] : "forward";
  bool io_mode = false;
  std::string in_path, out_path;
  for (int i = 2; i < argc; i++) {
    if (std::string(argv[i]) == "--io" && i + 2 < argc) {
      io_mode = true;
      in_path = argv[i + 1];
      out_path = argv[i + 2];
    }
  }

  Backend backend;
  if (backend.load(path)) {
    std::fprintf(stderr, "FAIL: load(\"%s\") returned error\n", path.c_str());
    return 1;
  }

  auto params = backend.get_method_params(method);
  if (params.size() < 4) {
    std::fprintf(stderr, "FAIL: method \"%s\" not found\n", method.c_str());
    return 1;
  }
  const int in_dim = params[0], out_dim = params[2];
  const int n = backend.get_buffer_size();

  if (io_mode) {
    if (in_dim != 1 || out_dim != 1) {
      std::fprintf(stderr, "FAIL: --io mode supports 1->1 methods only "
                           "(got in_dim=%d out_dim=%d)\n",
                   in_dim, out_dim);
      return 1;
    }
    std::vector<float> input = read_f32(in_path);
    if (input.empty()) {
      std::fprintf(stderr, "FAIL: could not read %s\n", in_path.c_str());
      return 1;
    }
    const int blocks = (int)input.size() / n;
    std::vector<float> output((size_t)blocks * n, 0.f);
    std::vector<float> blk_out(n, 0.f);
    for (int b = 0; b < blocks; b++) {
      std::vector<float *> in_buf{input.data() + (size_t)b * n};
      std::vector<float *> out_buf{blk_out.data()};
      backend.perform(in_buf, out_buf, n, method, 1);
      std::copy(blk_out.begin(), blk_out.end(), output.begin() + (size_t)b * n);
    }
    write_f32(out_path, output);
    std::printf("wrote %d blocks (%zu samples) of %s output to %s\n", blocks,
                output.size(), method.c_str(), out_path.c_str());
    return 0;
  }

  // --- smoke (stats) mode ---
  std::printf("loaded: %s\n  buffer_size=%d\n", path.c_str(), n);
  std::printf("  methods:");
  for (const auto &m : backend.get_available_methods())
    std::printf(" %s", m.c_str());
  std::printf("\n  method %s: in_dim=%d in_ratio=%d out_dim=%d out_ratio=%d "
              "n_vec=%d\n",
              method.c_str(), params[0], params[1], params[2], params[3], n);
  if (n <= 0) {
    std::fprintf(stderr, "FAIL: buffer_size is %d\n", n);
    return 1;
  }

  std::vector<std::vector<float>> in_store(in_dim, std::vector<float>(n));
  std::vector<float *> in_buf;
  for (int c = 0; c < in_dim; c++) {
    for (int i = 0; i < n; i++)
      in_store[c][i] = 0.1f * std::sin(0.01f * i + c);
    in_buf.push_back(in_store[c].data());
  }
  std::vector<std::vector<float>> out_store(out_dim, std::vector<float>(n, 0.f));
  std::vector<float *> out_buf;
  for (int c = 0; c < out_dim; c++)
    out_buf.push_back(out_store[c].data());

  for (int b = 0; b < 4; b++)
    backend.perform(in_buf, out_buf, n, method, 1);

  double mn = 1e30, mx = -1e30, sum = 0.0;
  int nonfinite = 0;
  for (int c = 0; c < out_dim; c++)
    for (int i = 0; i < n; i++) {
      float v = out_store[c][i];
      if (!std::isfinite(v)) nonfinite++;
      mn = std::min(mn, (double)v);
      mx = std::max(mx, (double)v);
      sum += v;
    }
  std::printf("  output: min=%.5f max=%.5f mean=%.6f nonfinite=%d\n", mn, mx,
              sum / (double)(out_dim * n), nonfinite);
  if (nonfinite) {
    std::fprintf(stderr, "FAIL: %d non-finite output samples\n", nonfinite);
    return 1;
  }
  if (mx == mn)
    std::printf("  WARN: output is constant (%.5f)\n", mn);
  std::printf("SMOKE TEST OK\n");
  return 0;
}
