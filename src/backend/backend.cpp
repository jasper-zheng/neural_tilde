#include "backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

#include "json.hpp"       // nlohmann/json single-header, vendored into src/backend/
#include "noise_seed.h"   // noise_substream_seed (shared with GenRunner)

// ---------------------------------------------------------------------------
// ExecuTorch C++ API NOTES (verify against the installed headers when wiring
// up the build in Workstream C/D):
//   - executorch::extension::Module(path)               load a .pte
//   - module.execute(method, tensor) -> Result<vector<EValue>>
//   - executorch::extension::from_blob(ptr, sizes, ScalarType::Float)
//   - result.ok(); result->at(0).toTensor(); t.size(i); t.const_data_ptr<float>()
//   - module.method_names() -> Result<set<string>>  (best-effort cross-check)
// The decimation/upsampling around the model call is plain-C++ buffer math
// (replaces the old torch::cat/reshape/select/permute/repeat_interleave path).
// ---------------------------------------------------------------------------

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using json = nlohmann::json;

// --- Generative-profile helpers (see EXECUTORCH_PROTOCOL.md §7) -------------

// Map a sidecar dtype string to an ExecuTorch ScalarType. Defaults to Float.
static ScalarType parse_dtype(const std::string &s) {
  if (s == "int64" || s == "long")
    return ScalarType::Long;
  if (s == "int32" || s == "int")
    return ScalarType::Int;
  if (s == "float32" || s == "float")
    return ScalarType::Float;
  std::cerr << "neural: unknown dtype \"" << s << "\", assuming float32\n";
  return ScalarType::Float;
}

static size_t et_dtype_size(ScalarType t) {
  switch (t) {
  case ScalarType::Long:
    return 8;
  case ScalarType::Int:
    return 4;
  case ScalarType::Float:
    return 4;
  default:
    return 4;
  }
}

static Role parse_role(const std::string &s) {
  // "signal" => (live) the per-block multi-channel audio from the signal inlets;
  // "audio" is accepted as an alias for the audio output's role.
  if (s == "signal" || s == "audio")
    return Role::Signal;
  // "condition" => externally-supplied conditioning, bound by name (e.g. token
  // ids / attention mask from a tokenizer, or a held control vector for live).
  // dtype comes from the sidecar; no back-compat aliases.
  if (s == "condition")
    return Role::Condition;
  if (s == "noise")
    return Role::Noise;
  if (s == "buffer")
    return Role::Buffer;
  // "attribute" (and any unrecognized role) => a scalar attribute the consumer
  // exposes as a Max attribute named by the input's `name`.
  return Role::Attribute;
}

Backend::Backend()
    : m_loaded(0), m_metadata_loaded(false), m_buffer_size(0),
      m_generative(false), m_seed(0) {}

bool Backend::load_metadata(const std::string &pte_path) {
  // Sidecar lives next to the .pte with the same basename and a .json suffix.
  std::string json_path = pte_path;
  auto dot = json_path.find_last_of('.');
  if (dot != std::string::npos)
    json_path = json_path.substr(0, dot);
  json_path += ".json";

  std::ifstream f(json_path);
  if (!f.is_open()) {
    std::cerr << "neural: missing sidecar metadata " << json_path << std::endl;
    return false;
  }

  json j;
  try {
    f >> j;
  } catch (const std::exception &e) {
    std::cerr << "neural: failed to parse " << json_path << ": " << e.what()
              << std::endl;
    return false;
  }

  m_methods.clear();
  m_available_methods.clear();
  m_buffer_size = j.value("buffer_size", 0);

  // Top-level model kind: REQUIRED. "live" (streaming, §2) vs "gen" (generative,
  // §3). No default and no back-compat — a missing or unknown kind is an error.
  if (!j.contains("kind")) {
    std::cerr << "neural: sidecar " << json_path
              << " missing required \"kind\" (expected \"live\" or \"gen\")"
              << std::endl;
    return false;
  }
  const std::string kind = j.value("kind", std::string());
  if (kind != "live" && kind != "gen") {
    std::cerr << "neural: sidecar " << json_path << " has unknown kind \"" << kind
              << "\" (expected \"live\" or \"gen\")" << std::endl;
    return false;
  }
  m_generative = (kind == "gen");
  // Default RNG seed for "noise" inputs (both kinds; absent => 0). gen reads it
  // per generate(); live seeds its per-input streams from it (reseed on load).
  m_seed = j.value("seed", 0);
  m_reseed_noise.store(true, std::memory_order_release);

  if (!j.contains("methods")) {
    std::cerr << "neural: no \"methods\" in " << json_path << std::endl;
    return false;
  }

  // Both kinds describe a method as an ordered "inputs" list + a single audio
  // "output" (see EXECUTORCH_PROTOCOL.md). The parse is shared; only the role
  // taxonomy differs (live: exactly one "signal" input, no "buffer"; gen: no
  // "signal").
  for (auto &kv : j["methods"].items()) {
    const std::string &name = kv.key();
    const json &m = kv.value();
    MethodInfo info;
    info.dynamic_time = m.value("dynamic_time", false);
    info.batch = m.contains("batch") && m["batch"].is_null()
                     ? -1
                     : m.value("batch", 1);

    if (!m.contains("inputs") || !m["inputs"].is_array()) {
      std::cerr << "neural: method \"" << name << "\" has no \"inputs\" array in "
                << json_path << std::endl;
      return false;
    }
    int signal_count = 0, buffer_count = 0;
    for (const auto &in : m["inputs"]) {
      InputSpec spec;
      spec.name = in.value("name", std::string());
      spec.role = parse_role(in.value("role", std::string("attribute")));
      spec.dtype = parse_dtype(in.value("dtype", std::string("float32")));
      spec.shape = in.value("shape", std::vector<int>{});
      spec.min = in.value("min", 0.0);
      spec.max = in.value("max", 0.0);
      spec.def = in.value("default", 0.0);
      spec.description = in.value("description", std::string());
      spec.sample_rate = in.value("sample_rate", 0);
      spec.channels = in.value("channels", 0);
      spec.ratio = in.value("ratio", 1);
      if (in.contains("labels"))
        spec.labels = in["labels"].get<std::vector<std::string>>();
      if (spec.role == Role::Signal) {
        signal_count++;
        // Default per-channel inlet labels when none were declared.
        for (int c = (int)spec.labels.size(); c < spec.channels; c++)
          spec.labels.push_back("(signal) model input " + std::to_string(c));
      } else if (spec.role == Role::Buffer) {
        buffer_count++;
      }
      info.inputs.push_back(std::move(spec));
    }

    const json &out = m.contains("output") ? m["output"] : json::object();
    info.output.name = out.value("name", std::string());
    info.output.role = parse_role(out.value("role", std::string("signal")));
    info.output.dtype = parse_dtype(out.value("dtype", std::string("float32")));
    info.output.shape = out.value("shape", std::vector<int>{});
    info.output.channels = out.value("channels", 0);
    info.output.length = out.value("length", 0);
    info.output.sample_rate = out.value("sample_rate", 0);
    info.output.ratio = out.value("ratio", 1);
    if (out.contains("labels"))
      info.output.labels = out["labels"].get<std::vector<std::string>>();
    for (int c = (int)info.output.labels.size(); c < info.output.channels; c++)
      info.output.labels.push_back("(signal) model output " + std::to_string(c));

    // Kind-specific shape of the inputs list.
    if (!m_generative && signal_count != 1) {
      std::cerr << "neural: live method \"" << name
                << "\" must declare exactly one \"signal\" input (got "
                << signal_count << ")" << std::endl;
      return false;
    }
    if (!m_generative && buffer_count != 0) {
      std::cerr << "neural: live method \"" << name
                << "\" must not declare a \"buffer\" input (gen-only)" << std::endl;
      return false;
    }
    if (m_generative && signal_count != 0) {
      std::cerr << "neural: generative method \"" << name
                << "\" must not declare a \"signal\" input" << std::endl;
      return false;
    }

    m_methods[name] = std::move(info);
    m_available_methods.push_back(name);
  }
  return true;
}

int Backend::load(std::string path) {
  // Parse the sidecar FIRST: without it we know nothing about the model's I/O
  // and cannot even build the host's inlets/attributes. A failure here is the
  // only hard error — the caller has nothing to set up.
  m_loaded = 0;
  m_metadata_loaded = false;
  if (!load_metadata(path)) {
    std::cerr << "neural: could not load metadata for " << path << std::endl;
    return 1;
  }
  m_metadata_loaded = true;
  m_path = path;

  // Is the program (.pte) actually on disk? A sidecar MAY ship without it; we
  // keep the parsed metadata (so the host can still create inlets/attributes)
  // but mark the model non-runnable and report it. is_loaded() stays false.
  std::ifstream pte_file(path, std::ios::binary);
  if (!pte_file.good()) {
    std::cerr << "neural: program \"" << path
              << "\" not found — metadata loaded, but the model is disabled "
                 "(provide the .pte to enable it)"
              << std::endl;
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    m_module.reset();
    return 0; // success enough to set up I/O; not runnable
  }

  try {
    auto module = std::make_unique<Module>(path);

    // Best-effort: warn if the .pte's runnable methods diverge from the JSON.
    try {
      auto names = module->method_names();
      if (names.ok()) {
        for (const auto &name : m_available_methods) {
          if (!names->count(name))
            std::cerr << "neural: method \"" << name
                      << "\" listed in JSON but absent from .pte" << std::endl;
        }
      }
    } catch (...) {
    }

    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    m_module = std::move(module);
    m_loaded = 1;
    model_lock.unlock();
    return 0;
  } catch (const std::exception &e) {
    // The metadata is still usable, so treat a program-load failure like a
    // missing program: set up I/O but stay disabled.
    std::cerr << e.what() << '\n';
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    m_module.reset();
    return 0;
  }
}

int Backend::reload() { return load(m_path); }

bool Backend::is_loaded() { return m_loaded; }

bool Backend::has_metadata() { return m_metadata_loaded; }

int Backend::get_buffer_size() { return m_buffer_size; }

// --- Generative profile (kind: "gen", see EXECUTORCH_PROTOCOL.md §3) ---------

bool Backend::is_generative() { return m_generative; }

const std::vector<InputSpec> &Backend::method_inputs(const std::string &method) {
  static const std::vector<InputSpec> empty;
  auto it = m_methods.find(method);
  return it != m_methods.end() ? it->second.inputs : empty;
}

OutputSpec Backend::method_output(const std::string &method) {
  auto it = m_methods.find(method);
  return it != m_methods.end() ? it->second.output : OutputSpec{};
}

int Backend::gen_seed() { return m_seed; }

void Backend::set_seed(int seed) {
  // Write the seed before flagging the reseed so perform() (which consumes the
  // flag with acquire) observes the new value. The streams are rebuilt lazily on
  // the next block; gen reads m_seed directly via gen_seed().
  m_seed = seed;
  m_reseed_noise.store(true, std::memory_order_release);
}

int Backend::generate(const std::string &method,
                      const std::vector<const void *> &input_data,
                      std::vector<float> &out_audio) {
  if (!m_loaded)
    return 1;

  auto it = m_methods.find(method);
  if (it == m_methods.end()) {
    std::cerr << "neural: generative method \"" << method << "\" not found\n";
    return 1;
  }
  const MethodInfo &info = it->second;
  if (input_data.size() != info.inputs.size()) {
    std::cerr << "neural: generate(\"" << method << "\") expected "
              << info.inputs.size() << " inputs, got " << input_data.size()
              << "\n";
    return 1;
  }

  // Build typed tensors from the host-owned buffers, in call order. The
  // TensorPtrs (and the host buffers behind them) must outlive execute().
  std::vector<TensorPtr> tensors;
  tensors.reserve(info.inputs.size());
  std::vector<EValue> inputs;
  inputs.reserve(info.inputs.size());
  for (size_t i = 0; i < info.inputs.size(); i++) {
    const InputSpec &spec = info.inputs[i];
    std::vector<executorch::aten::SizesType> sizes(spec.shape.begin(),
                                                   spec.shape.end());
    tensors.push_back(
        from_blob(const_cast<void *>(input_data[i]), sizes, spec.dtype));
    inputs.emplace_back(*tensors.back());
  }

  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  auto result = m_module->execute(method, inputs);
  model_lock.unlock();

  if (!result.ok()) {
    std::cerr << "neural: execute(\"" << method << "\") failed (error "
              << (int)result.error() << ")\n";
    return 1;
  }

  auto output = result->at(0).toTensor();
  const size_t numel = (size_t)output.numel();
  const float *ptr = output.const_data_ptr<float>();
  out_audio.assign(ptr, ptr + numel);
  return 0;
}

void Backend::reset() {
  // Re-initialise a model's persistent internal state to zero. Safe fallback:
  // reload the program, which re-instantiates the methods with zero-initialised
  // mutable buffers (see EXECUTORCH_PROTOCOL.md). The caller must quiesce the
  // worker thread first (the frontends' reload/reset messages stop it before
  // calling).
  // TODO: in-place zero of the named mutable buffers once the ExecuTorch 1.3.1
  // mutable-buffer API is wired, to avoid a full reload.
  // Also restart the per-input noise streams so reset replays the same seeded
  // noise deterministically (a reload re-runs load_metadata, which already flags
  // this; set it here too for the in-place path above).
  m_reseed_noise.store(true, std::memory_order_release);
  if (m_loaded)
    reload();
}

void Backend::perform(std::vector<float *> in_buffer,
                      std::vector<float *> out_buffer, int n_vec,
                      std::string method, int n_batches,
                      const std::vector<float> &attr_values,
                      const std::map<std::string, SuppliedInput> &supplied) {
  if (!m_loaded)
    return;

  auto it = m_methods.find(method);
  if (it == m_methods.end())
    return;
  const MethodInfo &info = it->second;

  // The single signal input carries the audio-in geometry; the output spec
  // carries the audio-out geometry (channels + temporal ratio).
  const InputSpec *sig = nullptr;
  for (const auto &in : info.inputs)
    if (in.role == Role::Signal) {
      sig = &in;
      break;
    }
  if (!sig) {
    std::cerr << "neural: method \"" << method << "\" has no signal input\n";
    return;
  }

  const int in_dim = sig->channels;
  const int in_ratio = sig->ratio;
  const int out_ratio = info.output.ratio;
  const int t_in = n_vec / in_ratio;
  const int t_out = n_vec / out_ratio;

  if ((int)in_buffer.size() != in_dim * n_batches) {
    std::cerr << "neural: bad in_buffer size, expected " << in_dim * n_batches
              << " buffers, got " << in_buffer.size() << "!\n";
    return;
  }

  // DECIMATE input by in_ratio into a contiguous [n_batches, in_dim, t_in]
  // scratch buffer. Input buffer index i maps channel-major: d = i/n_batches,
  // b = i%n_batches (matches the previous torch reshape ordering). m_in_scratch
  // is a reused member: resize() keeps its capacity, so a steady stream of
  // same-shape blocks does no heap allocation here.
  m_in_scratch.resize((size_t)n_batches * in_dim * t_in);
  for (int i = 0; i < (int)in_buffer.size(); i++) {
    const int d = i / n_batches;
    const int b = i % n_batches;
    float *dst = m_in_scratch.data() + ((size_t)b * in_dim + d) * t_in;
    const float *src = in_buffer[i];
    for (int t = 0; t < t_in; t++)
      dst[t] = src[t * in_ratio + (in_ratio - 1)];
  }

  auto signal = from_blob(m_in_scratch.data(), {n_batches, in_dim, t_in},
                          executorch::aten::ScalarType::Float);

  // Build the positional input list by role, in declared order. Backing storage
  // and the TensorPtrs must outlive execute(); all are reused members so a steady
  // same-shape stream does no per-block heap allocation. The signal TensorPtr is
  // a local that also outlives execute (function scope).
  size_t n_attr = 0, n_noise = 0, n_cond = 0;
  for (const auto &in : info.inputs) {
    if (in.role == Role::Attribute)
      n_attr++;
    else if (in.role == Role::Noise)
      n_noise++;
    else if (in.role == Role::Condition)
      n_cond++;
  }
  m_attr_f.clear();
  m_attr_l.clear();
  m_attr_f.reserve(n_attr); // reserve so .back() addresses stay valid
  m_attr_l.reserve(n_attr);
  m_noise_bufs.resize(n_noise);
  m_cond_bufs.resize(n_cond);

  // (Re)seed the per-noise-input RNG streams when the seed/model changed (or on
  // first run): one persistent mt19937_64 per noise input, keyed by (seed, name)
  // so each is independent of input order and of whether siblings are supplied.
  // Otherwise the streams persist across blocks (advanced below) so the seeded
  // noise evolves block-to-block yet replays identically for a given seed.
  if (m_reseed_noise.exchange(false, std::memory_order_acquire) ||
      m_noise_streams.size() != n_noise) {
    m_noise_streams.clear();
    m_noise_streams.reserve(n_noise);
    for (const auto &in : info.inputs)
      if (in.role == Role::Noise)
        m_noise_streams.emplace_back(
            noise_substream_seed((uint64_t)m_seed, in.name));
  }
  m_input_tensors.clear();
  m_input_tensors.reserve(info.inputs.size());
  m_inputs.clear();
  m_inputs.reserve(info.inputs.size());

  // Live noise/condition tensors are [n_batches] + declared_shape[1:] — the
  // sidecar's leading dim is a batch placeholder of 1, replicated to n_batches.
  size_t attr_i = 0, noise_i = 0, cond_i = 0;
  for (const auto &spec : info.inputs) {
    switch (spec.role) {
    case Role::Signal:
      m_inputs.emplace_back(*signal);
      break;
    case Role::Attribute: {
      // Scalar [1] control: the caller's value by declared-order index, else the
      // sidecar default (so attribute-less callers like mc/mcs pass nothing).
      const double v = attr_i < attr_values.size()
                           ? (double)attr_values[attr_i]
                           : spec.def;
      attr_i++;
      TensorPtr t;
      if (spec.dtype == ScalarType::Long) {
        m_attr_l.push_back((int64_t)std::llround(v));
        t = from_blob(&m_attr_l.back(), {1}, ScalarType::Long);
      } else {
        m_attr_f.push_back((float)v);
        t = from_blob(&m_attr_f.back(), {1}, ScalarType::Float);
      }
      m_input_tensors.push_back(t);
      m_inputs.emplace_back(*t);
      break;
    }
    case Role::Noise: {
      std::vector<executorch::aten::SizesType> sizes;
      sizes.push_back(n_batches);
      size_t per_batch = 1;
      for (size_t k = 1; k < spec.shape.size(); k++) {
        sizes.push_back(spec.shape[k]);
        per_batch *= (size_t)spec.shape[k];
      }
      const size_t total = (size_t)n_batches * per_batch;
      std::vector<float> &buf = m_noise_bufs[noise_i];
      buf.resize(total);
      // If the host supplied this noise input by name (e.g. a Jitter matrix on a
      // live matrix-noise inlet), use it — one per-batch matrix replicated across
      // the batch. Otherwise draw a fresh per-block N(0,1) from this input's
      // persistent (seed, name) stream so seeded noise evolves yet is reproducible.
      auto sup = supplied.find(spec.name);
      if (sup != supplied.end() && sup->second.dtype == ScalarType::Float &&
          sup->second.numel == per_batch) {
        const float *src = static_cast<const float *>(sup->second.data);
        for (int b = 0; b < n_batches; b++)
          std::memcpy(buf.data() + (size_t)b * per_batch, src,
                      per_batch * sizeof(float));
      } else {
        // Fresh distribution per input (its cached pair-state must not bleed
        // across this method's other noise streams); the stream itself persists.
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::mt19937_64 &rng = m_noise_streams[noise_i];
        for (size_t k = 0; k < total; k++)
          buf[k] = gauss(rng);
      }
      noise_i++;
      TensorPtr t = from_blob(buf.data(), sizes, ScalarType::Float);
      m_input_tensors.push_back(t);
      m_inputs.emplace_back(*t);
      break;
    }
    case Role::Condition: {
      // Externally-supplied held control vector, matched by name; zero-filled
      // until first set, then replicated across the batch.
      std::vector<executorch::aten::SizesType> sizes;
      sizes.push_back(n_batches);
      size_t per_batch = 1;
      for (size_t k = 1; k < spec.shape.size(); k++) {
        sizes.push_back(spec.shape[k]);
        per_batch *= (size_t)spec.shape[k];
      }
      const size_t esize = et_dtype_size(spec.dtype);
      std::vector<char> &buf = m_cond_bufs[cond_i++];
      buf.assign((size_t)n_batches * per_batch * esize, 0);
      auto sup = supplied.find(spec.name);
      if (sup != supplied.end()) {
        const SuppliedInput &s = sup->second;
        if (s.numel == per_batch && s.dtype == spec.dtype) {
          for (int b = 0; b < n_batches; b++)
            std::memcpy(buf.data() + (size_t)b * per_batch * esize, s.data,
                        per_batch * esize);
        } else {
          std::cerr << "neural: supplied condition \"" << spec.name
                    << "\" numel/dtype mismatch (expected " << per_batch
                    << "), zero-filled\n";
        }
      }
      TensorPtr t = from_blob(buf.data(), sizes, spec.dtype);
      m_input_tensors.push_back(t);
      m_inputs.emplace_back(*t);
      break;
    }
    default:
      std::cerr << "neural: input \"" << spec.name
                << "\" has a role unsupported by neural.live~ (buffer is gen-only)\n";
      return;
    }
  }

  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  auto result = m_module->execute(method, m_inputs);
  model_lock.unlock();

  if (!result.ok()) {
    std::cerr << "neural: execute(\"" << method << "\") failed (error "
              << (int)result.error() << ")\n";
    return;
  }

  auto output = result->at(0).toTensor();
  const int out_batches = output.size(0);
  const int out_channels = output.size(1);
  const int out_n_vec = output.size(2);

  if (out_batches * out_channels != (int)out_buffer.size()) {
    std::cerr << "neural: bad out_buffer size, expected "
              << out_batches * out_channels << " buffers, got "
              << out_buffer.size() << "!\n";
    return;
  }
  if (out_n_vec != t_out) {
    std::cerr << "neural: model output size inconsistent, expected " << t_out
              << " samples, got " << out_n_vec << "!\n";
    return;
  }

  // UPSAMPLE by out_ratio (repeat-interleave). Output tensor is contiguous
  // [out_batches, out_dim, t_out]; output buffer index j maps batch-major
  // (j = b*out_dim + d), matching the previous torch reshape ordering.
  const float *out_ptr = output.const_data_ptr<float>();
  for (int j = 0; j < (int)out_buffer.size(); j++) {
    const float *src = out_ptr + (size_t)j * t_out;
    float *dst = out_buffer[j];
    for (int t = 0; t < t_out; t++) {
      const float v = src[t];
      for (int r = 0; r < out_ratio; r++)
        dst[t * out_ratio + r] = v;
    }
  }
}

std::vector<std::string> Backend::get_available_methods() {
  return m_available_methods;
}

// [in_channels, in_ratio, out_channels, out_ratio] derived from the method's
// signal input + audio output. Empty for an unknown method (the frontends treat
// that as "method not found"). Gen methods have no signal input => empty.
std::vector<int> Backend::get_method_params(std::string method) {
  std::vector<int> params;
  auto it = m_methods.find(method);
  if (it == m_methods.end())
    return params;
  const MethodInfo &info = it->second;
  const InputSpec *sig = nullptr;
  for (const auto &in : info.inputs)
    if (in.role == Role::Signal) {
      sig = &in;
      break;
    }
  if (!sig)
    return params; // not a streaming method (e.g. a gen model)
  params.push_back(sig->channels);
  params.push_back(sig->ratio);
  params.push_back(info.output.channels);
  params.push_back(info.output.ratio);
  return params;
}

int Backend::get_higher_ratio() {
  int higher_ratio = 1;
  for (const auto &kv : m_methods) {
    for (const auto &in : kv.second.inputs)
      if (in.role == Role::Signal)
        higher_ratio = std::max(higher_ratio, in.ratio);
    higher_ratio = std::max(higher_ratio, kv.second.output.ratio);
  }
  return higher_ratio;
}

std::vector<std::string> Backend::get_input_labels(std::string method) {
  auto it = m_methods.find(method);
  if (it != m_methods.end())
    for (const auto &in : it->second.inputs)
      if (in.role == Role::Signal)
        return in.labels;
  return {};
}

std::vector<std::string> Backend::get_output_labels(std::string method) {
  auto it = m_methods.find(method);
  if (it != m_methods.end())
    return it->second.output.labels;
  return {};
}

// The Attribute-role inputs a method declares, in declared order (the consumer
// builds one dynamic Max attribute per entry). Returned by value (filtered).
std::vector<InputSpec> Backend::method_attributes(const std::string &method) {
  std::vector<InputSpec> attrs;
  auto it = m_methods.find(method);
  if (it != m_methods.end())
    for (const auto &in : it->second.inputs)
      if (in.role == Role::Attribute)
        attrs.push_back(in);
  return attrs;
}

bool Backend::has_method(std::string method_name) {
  // Covers both kinds: load_metadata populates m_available_methods for each.
  return std::find(m_available_methods.begin(), m_available_methods.end(),
                   method_name) != m_available_methods.end();
}

// --- Settable attributes: unsupported in v1 -------------------------------

bool Backend::has_settable_attribute(std::string) { return false; }

std::vector<std::string> Backend::get_settable_attributes() { return {}; }

std::string Backend::get_attribute_as_string(std::string attribute_name) {
  throw std::string("settable attributes are not supported in this "
                    "ExecuTorch build (requested \"" +
                    attribute_name + "\")");
}

void Backend::set_attribute(std::string attribute_name,
                            std::vector<std::string>) {
  throw std::string("settable attributes are not supported in this "
                    "ExecuTorch build (requested \"" +
                    attribute_name + "\")");
}
