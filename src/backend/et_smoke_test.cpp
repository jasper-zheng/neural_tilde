// Standalone test harness for the ExecuTorch Backend, used two ways:
//
//   ./et_smoke_test <model.pte> [method] [--data <model.ptd>]
//       Smoke test: run `method` (default forward) on a synthetic block and
//       report output statistics (finite, range).
//
//   ./et_smoke_test <model.pte> <method> --io <in.f32> <out.f32> [checks]
//       Parity mode: read raw float32 mono samples from <in.f32>, process them
//       block-by-block through Backend (fresh load => initial streaming state),
//       and write the raw float32 output to <out.f32>. Used to compare the C++
//       runtime against the eager PyTorch reference (only in_dim==out_dim==1,
//       i.e. `forward`). Optional checks on top:
//         --ref <ref.f32> [--tol 1e-4]  max|out - ref| <= tol
//         --stateless <st.f32>          output differs from a state-free run at
//                                       block edges (streaming state persists)
//         --reset-check                 reset() (and reload()) then re-stream:
//                                       identical to the first run
//         --threads <K>                 K Backends streaming on K threads, each
//                                       must match --ref
//
//   ./et_smoke_test <model.pte> [method] --bench <iters>
//       perform() latency per block (median / p99).
//
//   ./et_smoke_test <model.pte> [method] --load-loop <N>
//       Load + run + destroy N times; reports GPU free memory (CUDA models).
//
// The weights file (.ptd) is resolved by Backend from the metadata unless
// --data overrides it. Exercises the same Backend path the Max externals use,
// without needing Max. Exit status: 0 ok, 1 failure, 3 model disabled (the
// load-time guards refused it; the reason is logged).

#include "backend.h"
#include "nn_platform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<float> read_f32(const std::string &path) {
  std::ifstream f(nn::fs_path(path), std::ios::binary | std::ios::ate);
  std::vector<float> v;
  if (!f) return v;
  auto bytes = f.tellg();
  f.seekg(0);
  v.resize(bytes / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float));
  return v;
}

static void write_f32(const std::string &path, const std::vector<float> &v) {
  std::ofstream f(nn::fs_path(path), std::ios::binary);
  f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
}

static double max_abs_diff(const std::vector<float> &a,
                           const std::vector<float> &b, size_t from = 0,
                           size_t to = SIZE_MAX) {
  double m = 0;
  to = std::min({to, a.size(), b.size()});
  if (a.size() != b.size()) m = INFINITY;
  for (size_t i = from; i < to; i++)
    m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

// Stream a mono signal block by block through `backend` (1 -> 1 methods).
static std::vector<float> stream(Backend &backend, const std::string &method,
                                 const std::vector<float> &input, int n) {
  const int blocks = (int)input.size() / n;
  std::vector<float> output((size_t)blocks * n, 0.f);
  std::vector<float> blk_in(n), blk_out(n, 0.f);
  for (int b = 0; b < blocks; b++) {
    std::copy(input.begin() + (size_t)b * n, input.begin() + (size_t)(b + 1) * n,
              blk_in.begin());
    std::vector<float *> in_buf{blk_in.data()};
    std::vector<float *> out_buf{blk_out.data()};
    backend.perform(in_buf, out_buf, n, method, 1);
    std::copy(blk_out.begin(), blk_out.end(), output.begin() + (size_t)b * n);
  }
  return output;
}

static void print_gpu_mem(const char *when) {
  size_t free_b = 0, total_b = 0;
  if (nn::cuda_mem_info(free_b, total_b))
    std::printf("  GPU memory %s: %.1f MB free of %.1f MB\n", when,
                free_b / 1048576.0, total_b / 1048576.0);
}

int main(int argc, char **argv) {
  const std::vector<std::string> args = nn::utf8_args(argc, argv);
  if (args.size() < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.pte> [method] [--data <ptd>] "
                 "[--io <in.f32> <out.f32> [--ref <r.f32>] [--tol <t>] "
                 "[--stateless <s.f32>] [--reset-check] [--threads <K>]] "
                 "[--bench <iters>] [--load-loop <N>]\n",
                 argv[0]);
    return 2;
  }
  const std::string path = args[1];
  std::string method =
      (args.size() > 2 && args[2][0] != '-') ? args[2] : "forward";
  std::string data_path, in_path, out_path, ref_path, st_path;
  double tol = 1e-4;
  int bench = 0, threads = 0, load_loop = 0;
  bool reset_check = false;
  for (size_t i = 2; i < args.size(); i++) {
    const std::string &a = args[i];
    const bool has1 = i + 1 < args.size();
    if (a == "--io" && i + 2 < args.size()) {
      in_path = args[++i];
      out_path = args[++i];
    } else if (a == "--data" && has1) data_path = args[++i];
    else if (a == "--ref" && has1) ref_path = args[++i];
    else if (a == "--tol" && has1) tol = std::atof(args[++i].c_str());
    else if (a == "--stateless" && has1) st_path = args[++i];
    else if (a == "--bench" && has1) bench = std::atoi(args[++i].c_str());
    else if (a == "--threads" && has1) threads = std::atoi(args[++i].c_str());
    else if (a == "--load-loop" && has1) load_loop = std::atoi(args[++i].c_str());
    else if (a == "--reset-check") reset_check = true;
  }

  const nn::RuntimeVersion rt = nn::vc_runtime_version();
  if (rt.major > 0) {
    const nn::RuntimeVersion need = nn::vc_runtime_required();
    std::printf("  C++ runtime %d.%d (built for %d.%d+)\n", rt.major, rt.minor,
                need.major, need.minor);
    const std::string problem = nn::vc_runtime_problem();
    if (!problem.empty())
      std::fprintf(stderr, "WARNING: %s\n", problem.c_str());
  }

  auto t0 = Clock::now();
  Backend backend;
  if (backend.load(path, data_path)) {
    std::fprintf(stderr, "FAIL: load(\"%s\") returned error\n", path.c_str());
    return 1;
  }
  if (!backend.is_loaded()) {
    std::fprintf(stderr, "DISABLED: \"%s\" cannot run here (see above)\n",
                 path.c_str());
    return 3;
  }
  auto params = backend.get_method_params(method);
  if (params.size() < 4) {
    std::fprintf(stderr, "FAIL: method \"%s\" not found\n", method.c_str());
    return 1;
  }
  if (backend.prepare(method)) {
    std::fprintf(stderr, "FAIL: prepare(\"%s\") returned error\n",
                 method.c_str());
    return 1;
  }
  const double load_ms = ms_since(t0);
  const int in_dim = params[0], out_dim = params[2];
  const int n = backend.get_buffer_size();
  std::printf("loaded: %s\n  device: %s (delegate %s)  load+prepare %.1f ms\n",
              path.c_str(), backend.device_description().c_str(),
              backend.delegate().c_str(), load_ms);
  if (n <= 0) {
    std::fprintf(stderr, "FAIL: buffer_size is %d\n", n);
    return 1;
  }

  int fails = 0;
  if (!in_path.empty()) {
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
    std::vector<float> output = stream(backend, method, input, n);
    write_f32(out_path, output);
    std::printf("wrote %zu blocks (%zu samples) of %s output to %s\n",
                output.size() / n, output.size(), method.c_str(),
                out_path.c_str());

    std::vector<float> ref;
    if (!ref_path.empty()) {
      ref = read_f32(ref_path);
      const double err = max_abs_diff(output, ref);
      const bool ok = err <= tol;
      fails += !ok;
      std::printf("  parity vs reference: max|err|=%.3g (tol %.1g) %s\n", err,
                  tol, ok ? "PASS" : "FAIL");
    }
    if (!st_path.empty()) {
      // If state were lost between blocks, the output would match the
      // state-free signature at each block's first samples.
      const std::vector<float> st = read_f32(st_path);
      double edge = 0;
      for (size_t b = 1; b < output.size() / n; b++)
        edge = std::max(edge, max_abs_diff(output, st, b * n, b * n + 8));
      const bool ok = edge > 1e-3;
      fails += !ok;
      std::printf("  vs stateless run at block edges: max|diff|=%.3g -> state "
                  "%s\n",
                  edge, ok ? "PERSISTS (PASS)" : "LOST (FAIL)");
    }
    if (reset_check) {
      // Each must bring the model back to its initial state: re-streaming the
      // same input reproduces the first run (allowing only nondeterministic
      // GPU reductions).
      for (int pass = 0; pass < 2; pass++) {
        auto tr = Clock::now();
        if (pass == 0)
          backend.reset();
        else
          backend.reload(), backend.prepare(method);
        const double ms = ms_since(tr);
        const double err = max_abs_diff(output, stream(backend, method, input, n));
        const bool ok = err <= 1e-6;
        fails += !ok;
        std::printf("  %s: %.1f ms, re-run max|diff| vs first run=%.3g %s\n",
                    pass == 0 ? "reset()" : "reload()", ms, err,
                    ok ? "PASS" : "FAIL");
      }
    }
    if (threads > 0) {
      // Independent Backends on concurrent threads (as with several objects in
      // a patch): each streams the input and must match the reference.
      if (ref.empty()) ref = output;
      std::vector<double> errs(threads, INFINITY);
      std::vector<std::thread> pool;
      for (int k = 0; k < threads; k++)
        pool.emplace_back([&, k] {
          Backend b;
          if (b.load(path, data_path) || !b.is_loaded() || b.prepare(method))
            return;
          errs[k] = max_abs_diff(stream(b, method, input, n), ref);
        });
      for (auto &t : pool) t.join();
      double worst = 0;
      for (double e : errs) worst = std::max(worst, e);
      const bool ok = worst <= tol;
      fails += !ok;
      std::printf("  %d threads: worst max|err| vs reference=%.3g %s\n",
                  threads, worst, ok ? "PASS" : "FAIL");
    }
  }

  if (bench > 0) {
    std::vector<std::vector<float>> in_store(in_dim, std::vector<float>(n));
    std::vector<std::vector<float>> out_store(out_dim, std::vector<float>(n));
    std::vector<float *> in_buf, out_buf;
    for (int c = 0; c < in_dim; c++) {
      for (int i = 0; i < n; i++) in_store[c][i] = 0.1f * std::sin(0.01f * i + c);
      in_buf.push_back(in_store[c].data());
    }
    for (int c = 0; c < out_dim; c++) out_buf.push_back(out_store[c].data());
    for (int i = 0; i < 20; i++) backend.perform(in_buf, out_buf, n, method, 1);
    std::vector<double> us(bench);
    for (int i = 0; i < bench; i++) {
      auto tb = Clock::now();
      backend.perform(in_buf, out_buf, n, method, 1);
      us[i] = ms_since(tb) * 1000.0;
    }
    std::sort(us.begin(), us.end());
    const double med = us[bench / 2];
    const double p99 = us[std::min(bench - 1, (int)(bench * 0.99))];
    const double block_us = 1e6 * n / 48000.0;
    std::printf("  bench n=%d: median %.1f us, p99 %.1f us; block @48k = %.0f "
                "us -> RTF %.4f (p99 %.4f)\n",
                n, med, p99, block_us, med / block_us, p99 / block_us);
  }

  if (load_loop > 0) {
    // Load + one block + destroy, N times: GPU memory must come back.
    std::vector<float> blk_in((size_t)in_dim * n, 0.05f), blk_out((size_t)out_dim * n);
    std::vector<float *> in_buf, out_buf;
    for (int c = 0; c < in_dim; c++) in_buf.push_back(blk_in.data() + (size_t)c * n);
    for (int c = 0; c < out_dim; c++) out_buf.push_back(blk_out.data() + (size_t)c * n);
    print_gpu_mem("before loop");
    auto tl = Clock::now();
    for (int i = 0; i < load_loop; i++) {
      auto b = std::make_unique<Backend>();
      if (b->load(path, data_path) || !b->is_loaded() || b->prepare(method)) {
        std::fprintf(stderr, "FAIL: load %d failed\n", i);
        return 1;
      }
      b->perform(in_buf, out_buf, n, method, 1);
      if (i == 0) print_gpu_mem("with one extra instance");
    }
    std::printf("  load-loop: %d x load+prepare+run+destroy, %.1f ms each\n",
                load_loop, ms_since(tl) / load_loop);
    print_gpu_mem("after loop");
  }

  if (!in_path.empty() || bench > 0 || load_loop > 0) {
    if (fails) {
      std::fprintf(stderr, "FAIL: %d check(s) failed\n", fails);
      return 1;
    }
    std::printf("CHECKS OK\n");
    return 0;
  }

  // --- smoke (stats) mode ---
  std::printf("  methods:");
  for (const auto &m : backend.get_available_methods())
    std::printf(" %s", m.c_str());
  std::printf("\n  method %s: in_dim=%d in_ratio=%d out_dim=%d out_ratio=%d "
              "n_vec=%d\n",
              method.c_str(), params[0], params[1], params[2], params[3], n);

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
