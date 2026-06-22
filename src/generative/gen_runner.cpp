#include "gen_runner.h"

#include "../backend/noise_seed.h" // noise_substream_seed (shared with Backend)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>

using executorch::aten::ScalarType;

// ScalarType -> element size in bytes (Long is 8; Int / Float are 4).
static size_t dtype_size(ScalarType t) {
  if (t == ScalarType::Long)
    return 8;
  return 4;
}

static size_t numel_of(const std::vector<int> &shape) {
  size_t n = 1;
  for (int d : shape)
    n *= (size_t)d;
  return n;
}

void GenRunner::bind_condition(
    const std::vector<InputSpec> &inputs,
    const std::map<std::string, std::vector<char>> &cond,
    std::map<std::string, SuppliedInput> &supplied) {
  for (const auto &in : inputs) {
    if (in.role != Role::Condition)
      continue;
    auto it = cond.find(in.name);
    if (it == cond.end())
      continue; // missing here => generate() reports it as unsupplied
    supplied[in.name] = {in.dtype, it->second.data(),
                         it->second.size() / dtype_size(in.dtype)};
  }
}

void GenRunner::bind_init_audio(const std::vector<InputSpec> &inputs,
                                const std::vector<float> &init,
                                std::map<std::string, SuppliedInput> &supplied) {
  if (init.empty())
    return; // nothing supplied => generate() fills buffer input with silence
  for (const auto &in : inputs)
    if (in.role == Role::Buffer)
      supplied[in.name] = {ScalarType::Float, init.data(), init.size()};
}

void GenRunner::bind_noise(
    const std::vector<InputSpec> &inputs,
    const std::map<std::string, std::vector<float>> &noise,
    std::map<std::string, SuppliedInput> &supplied) {
  for (const auto &in : inputs) {
    if (in.role != Role::Noise)
      continue;
    auto it = noise.find(in.name);
    if (it == noise.end())
      continue; // not supplied => generate() seeds it
    supplied[in.name] = {ScalarType::Float, it->second.data(),
                         it->second.size()};
  }
}

void GenRunner::prepare_init_audio(const std::vector<float> &src, int src_ch,
                                   size_t src_frames, double src_sr, int dst_ch,
                                   size_t dst_len, double dst_sr,
                                   std::vector<float> &out) {
  out.assign((size_t)dst_ch * dst_len, 0.0f);
  if (src_ch <= 0 || dst_ch <= 0 || src_frames == 0 || src_sr <= 0.0 ||
      dst_sr <= 0.0)
    return; // leave silence

  const double ratio = src_sr / dst_sr; // source frames advanced per dst frame
  for (int c = 0; c < dst_ch; c++) {
    // mono source duplicates to every dst channel; otherwise map 1:1 and clamp
    // (a source with more channels than dst keeps only the first dst_ch).
    const int sc = std::min(c, src_ch - 1);
    const float *srcc = src.data() + (size_t)sc * src_frames;
    float *dstc = out.data() + (size_t)c * dst_len;
    for (size_t t = 0; t < dst_len; t++) {
      const double pos = (double)t * ratio;
      const size_t i0 = (size_t)pos;
      if (i0 + 1 < src_frames) {
        const float frac = (float)(pos - (double)i0);
        dstc[t] = srcc[i0] * (1.0f - frac) + srcc[i0 + 1] * frac;
      } else if (i0 < src_frames) {
        dstc[t] = srcc[i0]; // last source sample, no successor to interp with
      }
      // else: past the source end => leave the pre-zeroed pad
    }
  }
}

int GenRunner::load(const std::string &pte_path) {
  m_ready = false;
  if (m_backend.load(pte_path)) {
    std::cerr << "neural: could not load " << pte_path << std::endl;
    return 1;
  }
  if (!m_backend.is_generative()) {
    std::cerr << "neural: " << pte_path
              << " is not a generative model (kind != \"generative\")\n";
    return 1;
  }
  m_ready = true;
  return 0;
}

int GenRunner::generate(const std::string &method,
                        const std::map<std::string, SuppliedInput> &supplied,
                        const std::map<std::string, double> &attr_values,
                        uint64_t seed, std::vector<float> &out_audio) {
  if (!m_ready)
    return 1;

  const auto &inputs = m_backend.method_inputs(method);
  if (inputs.empty()) {
    std::cerr << "neural: no generative inputs for method \"" << method
              << "\"\n";
    return 1;
  }

  // Build each input buffer in call order (kept alive across execute()).
  std::vector<std::vector<char>> storage(inputs.size());
  std::vector<const void *> ptrs(inputs.size());

  for (size_t i = 0; i < inputs.size(); i++) {
    const InputSpec &spec = inputs[i];
    const size_t n = numel_of(spec.shape);
    storage[i].resize(n * dtype_size(spec.dtype));
    void *dst = storage[i].data();

    // Conditioning supplied by the patch (e.g. tokens) takes priority by name.
    auto it = supplied.find(spec.name);
    if (it != supplied.end()) {
      const SuppliedInput &s = it->second;
      if (s.numel != n) {
        std::cerr << "neural: supplied \"" << spec.name << "\" has " << s.numel
                  << " elems != expected " << n << "\n";
        return 1;
      }
      if (s.dtype != spec.dtype) {
        std::cerr << "neural: supplied \"" << spec.name
                  << "\" dtype does not match the model input\n";
        return 1;
      }
      std::memcpy(dst, s.data, n * dtype_size(spec.dtype));
      ptrs[i] = dst;
      continue;
    }

    switch (spec.role) {
    case Role::Attribute: {
      // Scalar attribute: the caller's value by name, else the sidecar default.
      auto av = attr_values.find(spec.name);
      double raw = (av != attr_values.end()) ? av->second : spec.def;
      double lo = spec.min, hi = spec.max;
      if (hi > lo)
        raw = std::min(std::max(raw, lo), hi);
      float *f = reinterpret_cast<float *>(dst);
      for (size_t k = 0; k < n; k++)
        f[k] = (float)raw;
      break;
    }
    case Role::Noise: {
      // Independent per-input stream keyed by (seed, name): enabling a matrix
      // for one noise input never perturbs the seeded values of the others.
      std::mt19937_64 nrng(noise_substream_seed(seed, spec.name));
      std::normal_distribution<float> ngauss(0.0f, 1.0f);
      float *f = reinterpret_cast<float *>(dst);
      for (size_t k = 0; k < n; k++)
        f[k] = ngauss(nrng);
      break;
    }
    case Role::Buffer: {
      // No init waveform was supplied: feed silence. With the model's strength
      // attribute (e.g. init_noise_level) at its max this reduces to pure
      // text-to-audio; otherwise the host is expected to supply via bind_init_audio.
      std::memset(dst, 0, n * sizeof(float));
      break;
    }
    case Role::Condition:
    default:
      std::cerr << "neural: required conditioning input \"" << spec.name
                << "\" was not supplied (connect a conditioning source, e.g. "
                   "neural.tokenizer)\n";
      return 1;
    }
    ptrs[i] = dst;
  }

  return m_backend.generate(method, ptrs, out_audio);
}
