#include "gen_runner.h"

#include "../backend/noise_seed.h" // noise_substream_seed (shared with Backend)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

void GenRunner::bind_buffers(
    const std::vector<InputSpec> &inputs,
    const std::map<std::string, std::vector<float>> &buffers,
    std::map<std::string, SuppliedInput> &supplied) {
  for (const auto &in : inputs) {
    if (in.role != Role::Buffer)
      continue;
    auto it = buffers.find(in.name);
    if (it == buffers.end() || it->second.empty())
      continue; // not supplied => generate() fills it with pad_value
    supplied[in.name] = {ScalarType::Float, it->second.data(),
                         it->second.size()};
  }
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
                                   std::vector<float> &out,
                                   double src_offset_frames, float pad_value) {
  out.assign((size_t)dst_ch * dst_len, pad_value);
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
      const double pos = src_offset_frames + (double)t * ratio;
      const size_t i0 = (size_t)pos;
      if (i0 + 1 < src_frames) {
        const float frac = (float)(pos - (double)i0);
        dstc[t] = srcc[i0] * (1.0f - frac) + srcc[i0 + 1] * frac;
      } else if (i0 < src_frames) {
        dstc[t] = srcc[i0]; // last source sample, no successor to interp with
      }
      // else: past the source end => leave the pre-filled pad
    }
  }
}

void GenRunner::copy_buffer_asis(const std::vector<float> &src, int src_ch,
                                 size_t src_frames, int dst_ch, size_t dst_len,
                                 std::vector<float> &out, size_t offset_frames,
                                 float pad_value) {
  out.assign((size_t)dst_ch * dst_len, pad_value);
  if (src_ch <= 0 || dst_ch <= 0 || offset_frames >= src_frames)
    return; // nothing to copy => all pad
  const size_t n = std::min(dst_len, src_frames - offset_frames);
  for (int c = 0; c < dst_ch; c++) {
    // Same channel map as prepare_init_audio: mono duplicates, extra src
    // channels beyond dst_ch are dropped.
    const int sc = std::min(c, src_ch - 1);
    const float *srcc = src.data() + (size_t)sc * src_frames + offset_frames;
    std::copy(srcc, srcc + n, out.data() + (size_t)c * dst_len);
  }
}

GenRunner::BufferGeom GenRunner::buffer_geometry(const InputSpec &spec,
                                                 int fallback_sr) {
  BufferGeom g;
  const auto &sh = spec.shape;
  g.ch = sh.size() >= 2 ? sh[sh.size() - 2] : 1;
  g.len = sh.size() >= 1 ? sh[sh.size() - 1] : 0;
  g.sr = spec.sample_rate > 0 ? spec.sample_rate : fallback_sr;
  if (g.sr <= 0)
    g.sr = 44100;
  return g;
}

void GenRunner::prepare_buffer_input(const InputSpec &spec, int fallback_sr,
                                     const std::vector<float> &src, int src_ch,
                                     size_t src_frames, double src_sr,
                                     double offset_ms, std::vector<float> &out) {
  const BufferGeom g = buffer_geometry(spec, fallback_sr);
  const float pad = (float)spec.pad_value;
  offset_ms = std::max(0.0, offset_ms);
  if (!spec.resample) {
    // Control curve: the ms offset is converted at the DECLARED rate, since the
    // buffer~'s own rate is ignored in this mode.
    const size_t off = (size_t)std::llround(offset_ms * 0.001 * g.sr);
    copy_buffer_asis(src, src_ch, src_frames, g.ch, (size_t)g.len, out, off,
                     pad);
  } else {
    // Audio: the offset is in the source's own frames (0.0.4 behaviour).
    const double off = offset_ms * 0.001 * src_sr;
    prepare_init_audio(src, src_ch, src_frames, src_sr, g.ch, (size_t)g.len,
                       (double)g.sr, out, off, pad);
  }
}

int GenRunner::load(const std::string &pte_path) {
  m_ready = false;
  if (m_backend.load(pte_path)) {
    log_error("neural: could not load " + pte_path);
    return 1;
  }
  if (!m_backend.is_generative()) {
    log_error("neural: " + pte_path +
              " is not a generative model (kind != \"gen\")");
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
    log_error("neural: no generative inputs for method \"" + method + "\"");
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
        log_error("neural: supplied \"" + spec.name + "\" has " +
                  std::to_string(s.numel) + " elems != expected " +
                  std::to_string(n));
        return 1;
      }
      if (s.dtype != spec.dtype) {
        log_error("neural: supplied \"" + spec.name +
                  "\" dtype does not match the model input");
        return 1;
      }
      std::memcpy(dst, s.data, n * dtype_size(spec.dtype));
      ptrs[i] = dst;
      continue;
    }

    switch (spec.role) {
    case Role::Attribute: {
      // Scalar attribute: the caller's value by name, else the metadata default.
      auto av = attr_values.find(spec.name);
      double raw = (av != attr_values.end()) ? av->second : spec.def;
      double lo = spec.min, hi = spec.max;
      if (hi > lo)
        raw = std::min(std::max(raw, lo), hi);
      if (spec.dtype == ScalarType::Long) {
        // int64 attribute: round to the integer the model expects (same as the
        // live path in Backend::perform).
        int64_t *l = reinterpret_cast<int64_t *>(dst);
        for (size_t k = 0; k < n; k++)
          l[k] = (int64_t)std::llround(raw);
      } else {
        float *f = reinterpret_cast<float *>(dst);
        for (size_t k = 0; k < n; k++)
          f[k] = (float)raw;
      }
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
      // No buffer was supplied: fill with the input's pad_value (0 => silence).
      // With the model's strength attribute (e.g. init_noise_level) at its max
      // this reduces to pure text-to-audio; otherwise the host is expected to
      // supply it via bind_buffers.
      float *f = reinterpret_cast<float *>(dst);
      std::fill(f, f + n, (float)spec.pad_value);
      break;
    }
    case Role::Condition:
    default:
      log_error("neural: required conditioning input \"" + spec.name +
                "\" was not supplied (connect a conditioning source, e.g. "
                "neural.tokenizer)");
      return 1;
    }
    ptrs[i] = dst;
  }

  return m_backend.generate(method, ptrs, out_audio);
}
