#include "backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/backend/interface.h>

#include "json.hpp"       // nlohmann/json single-header, vendored into src/backend/
#include "noise_seed.h"   // noise_substream_seed (shared with GenRunner)
#include "nn_platform.h"  // UTF-8 paths, CUDA device query

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

// Map a metadata dtype string to an ExecuTorch ScalarType. Unknown => Float,
// reported through `unknown` so the caller can log it.
static ScalarType parse_dtype(const std::string &s, std::string &unknown) {
  if (s == "int64" || s == "long")
    return ScalarType::Long;
  if (s == "int32" || s == "int")
    return ScalarType::Int;
  if (s == "float32" || s == "float")
    return ScalarType::Float;
  unknown = s;
  return ScalarType::Float;
}

// The ExecuTorch backend id a JSON `delegate` lowers to ("" => no delegate
// to check, e.g. portable or an older export without the field).
static const char *backend_id(const std::string &delegate) {
  if (delegate == "xnnpack")
    return "XnnpackBackend";
  if (delegate == "cuda")
    return "CudaBackend";
  if (delegate == "coreml")
    return "CoreMLBackend";
  if (delegate == "mlx")
    return "MLXBackend";
  if (delegate == "metal")
    return "MetalBackend";
  if (delegate == "mps")
    return "MPSBackend";
  return "";
}

// Delegates whose compiled programs only exist for Apple hosts.
static bool apple_only(const std::string &delegate) {
  return delegate == "coreml" || delegate == "mlx" || delegate == "metal" ||
         delegate == "mps";
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
  // "signal" => (live) the per-block multi-channel audio from the signal
  // inlets, and (live) the method's output role.
  if (s == "signal")
    return Role::Signal;
  // "condition" => externally-supplied conditioning, bound by name (e.g. token
  // ids / attention mask from a tokenizer, or a held control vector for live).
  // dtype comes from the metadata; no back-compat aliases.
  if (s == "condition")
    return Role::Condition;
  if (s == "noise")
    return Role::Noise;
  // "buffer" => (gen) an init waveform read from a buffer~ (input), and
  // (gen) the method's output role (written into a buffer~).
  if (s == "buffer")
    return Role::Buffer;
  // "attribute" (and any unrecognized role) => a scalar attribute the consumer
  // exposes as a Max attribute named by the input's `name`.
  return Role::Attribute;
}

Backend::Backend()
    : m_loaded(0), m_metadata_loaded(false), m_buffer_size(0),
      m_generative(false), m_seed(0) {
  nn::install_executorch_log_forwarding();
}

void Backend::log(nn::LogLevel level, const std::string &msg) const {
  nn::log(m_log_sink, level, msg);
}

void Backend::set_log_sink(nn::LogSink sink) { m_log_sink = std::move(sink); }

void Backend::log_block_error(const std::string &msg) {
  if (!m_exec_error_logged.exchange(true, std::memory_order_relaxed))
    log_error(msg + " (further errors from this model are not reported "
                    "until it is reloaded)");
}

bool Backend::load_metadata(const std::string &pte_path) {
  // Metadata lives next to the .pte with the same basename and a .json suffix.
  std::string json_path = pte_path;
  auto dot = json_path.find_last_of('.');
  if (dot != std::string::npos)
    json_path = json_path.substr(0, dot);
  json_path += ".json";

  std::ifstream f(nn::fs_path(json_path));
  if (!f.is_open()) {
    log_error("neural: missing metadata " + json_path);
    return false;
  }

  json j;
  try {
    f >> j;
  } catch (const std::exception &e) {
    log_error("neural: failed to parse " + json_path + ": " + e.what());
    return false;
  }

  m_methods.clear();
  m_available_methods.clear();
  m_buffer_size = j.value("buffer_size", 0);

  // Top-level model kind: REQUIRED. "live" (streaming, §2) vs "gen" (generative,
  // §3). No default and no back-compat — a missing or unknown kind is an error.
  if (!j.contains("kind")) {
    log_error("neural: metadata " + json_path +
              " missing required \"kind\" (expected \"live\" or \"gen\")");
    return false;
  }
  const std::string kind = j.value("kind", std::string());
  if (kind != "live" && kind != "gen") {
    log_error("neural: metadata " + json_path + " has unknown kind \"" + kind +
              "\" (expected \"live\" or \"gen\")");
    return false;
  }

  // Export provenance + runtime requirements (all optional: older exports and
  // CPU delegates omit the platform/CUDA fields, which skips those checks).
  m_delegate = j.value("delegate", std::string());
  m_platform = j.value("platform", std::string());
  m_data_file = j.value("data_file", std::string());
  m_executorch_version = j.value("executorch_version", std::string());
  m_cuda_min_arch = j.value("cuda_min_arch", 0);
  m_cuda_archs = j.value("cuda_archs", std::vector<int>{});
  m_generative = (kind == "gen");
  // Default RNG seed for "noise" inputs (both kinds; absent => 0). gen reads it
  // per generate(); live seeds its per-input streams from it (reseed on load).
  m_seed = j.value("seed", 0);
  m_reseed_noise.store(true, std::memory_order_release);

  if (!j.contains("methods")) {
    log_error("neural: no \"methods\" in " + json_path);
    return false;
  }
  std::string unknown_dtype;

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
      log_error("neural: method \"" + name + "\" has no \"inputs\" array in " +
                json_path);
      return false;
    }
    int signal_count = 0, buffer_count = 0;
    for (const auto &in : m["inputs"]) {
      InputSpec spec;
      spec.name = in.value("name", std::string());
      spec.role = parse_role(in.value("role", std::string("attribute")));
      spec.dtype =
          parse_dtype(in.value("dtype", std::string("float32")), unknown_dtype);
      spec.shape = in.value("shape", std::vector<int>{});
      spec.min = in.value("min", 0.0);
      spec.max = in.value("max", 0.0);
      spec.def = in.value("default", 0.0);
      spec.description = in.value("description", std::string());
      spec.sample_rate = in.value("sample_rate", 0);
      spec.resample = in.value("resample", true);
      spec.pad_value = in.value("pad_value", 0.0);
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
    info.output.dtype =
        parse_dtype(out.value("dtype", std::string("float32")), unknown_dtype);
    if (!unknown_dtype.empty()) {
      log(nn::LogLevel::Warning, "neural: unknown dtype \"" + unknown_dtype +
                                     "\" in method \"" + name +
                                     "\", assuming float32");
      unknown_dtype.clear();
    }
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
      log_error("neural: live method \"" + name +
                "\" must declare exactly one \"signal\" input (got " +
                std::to_string(signal_count) + ")");
      return false;
    }
    if (!m_generative && buffer_count != 0) {
      log_error("neural: live method \"" + name +
                "\" must not declare a \"buffer\" input (gen-only)");
      return false;
    }
    if (m_generative && signal_count != 0) {
      log_error("neural: generative method \"" + name +
                "\" must not declare a \"signal\" input");
      return false;
    }

    m_methods[name] = std::move(info);
    m_available_methods.push_back(name);
  }
  return true;
}

static std::string error_code(executorch::runtime::Error e) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%x", (unsigned)e);
  return buf;
}

std::string Backend::compatibility_problem() {
  // 1. Is the delegate the program was lowered for compiled into this build?
  // Covers macOS-only models on Windows and CUDA models on macOS alike.
  const char *id = backend_id(m_delegate);
  if (*id && executorch::runtime::get_backend_class(id) == nullptr) {
    std::string msg = "it uses the \"" + m_delegate +
                      "\" backend, which this build of neural_tilde doesn't "
                      "include";
    if (apple_only(m_delegate))
      msg += " (it is macOS-only)";
    msg += "; re-export it with delegate=\"xnnpack\" (CPU)";
    if (std::string(nn::host_platform()) == "windows")
      msg += " or delegate=\"cuda\" (NVIDIA GPU)";
    return msg;
  }
  if (m_delegate != "cuda")
    return "";

  // 2. AOTI programs embed a native library compiled for one OS.
  const std::string host = nn::host_platform();
  if (!m_platform.empty() && m_platform != host)
    return "it was exported for CUDA on " + m_platform + ", but this is " +
           host + "; re-export it on " + host;

  // 3. A CUDA device and a driver new enough for the runtime's CUDA version.
  const nn::CudaDeviceInfo &dev = nn::cuda_device_info();
#if defined(NN_CUDA_REQUIRED_VERSION) && NN_CUDA_REQUIRED_VERSION > 0
  if (dev.driver_version > 0 && dev.driver_version < NN_CUDA_REQUIRED_VERSION)
    return "the NVIDIA driver supports CUDA " +
           nn::cuda_version_string(dev.driver_version) +
           ", but this build of neural_tilde needs CUDA " +
           nn::cuda_version_string(NN_CUDA_REQUIRED_VERSION) +
           " or newer; update the NVIDIA driver";
#endif
  if (!dev.ok)
    return dev.error + "; this model needs an NVIDIA GPU (use a "
                       "delegate=\"xnnpack\" export to run on the CPU)";

  // 4. The GPU is at least the oldest architecture the kernels were built for.
  if (m_cuda_min_arch > 0 && dev.cc < m_cuda_min_arch)
    return "the GPU (" + dev.name + ") is sm_" + std::to_string(dev.cc) +
           ", but this model needs sm_" + std::to_string(m_cuda_min_arch) +
           " or newer";
  return "";
}

std::string Backend::resolve_data_file(const std::string &pte_path,
                                       bool &missing) {
  missing = false;
  if (!m_data_override.empty()) {
    missing = !nn::file_exists(m_data_override);
    return m_data_override;
  }
  const std::filesystem::path pte = nn::fs_path(pte_path);
  if (!m_data_file.empty()) {
    // Declared by the exporter, relative to the .pte's folder.
    const std::string ptd =
        nn::utf8_string(pte.parent_path() / nn::fs_path(m_data_file));
    missing = !nn::file_exists(ptd);
    return ptd;
  }
  // Undeclared: pick up "<basename>.ptd" beside the .pte if one exists.
  std::filesystem::path guess = pte;
  guess.replace_extension(".ptd");
  const std::string ptd = nn::utf8_string(guess);
  return nn::file_exists(ptd) ? ptd : std::string();
}

std::unique_ptr<Module>
Backend::make_module(const std::string &pte_path, const std::string &ptd_path,
                     std::vector<uint8_t> &program_bytes,
                     std::vector<uint8_t> &data_bytes) {
  using executorch::extension::BufferDataLoader;
  const std::string pte = nn::loader_path(pte_path);
  const std::string ptd = ptd_path.empty() ? "" : nn::loader_path(ptd_path);
  if (!pte.empty() && (ptd_path.empty() || !ptd.empty()))
    return ptd_path.empty() ? std::make_unique<Module>(pte)
                            : std::make_unique<Module>(pte, ptd);

  // ExecuTorch's file loaders take narrow (ANSI code page) paths on Windows; a
  // path with no ASCII form is read into memory here instead.
  if (!nn::read_file(pte_path, program_bytes)) {
    log_error("neural: could not read \"" + pte_path + "\"");
    return nullptr;
  }
  std::unique_ptr<executorch::runtime::DataLoader> data_loader;
  if (!ptd_path.empty()) {
    if (!nn::read_file(ptd_path, data_bytes)) {
      log_error("neural: could not read \"" + ptd_path + "\"");
      return nullptr;
    }
    data_loader =
        std::make_unique<BufferDataLoader>(data_bytes.data(), data_bytes.size());
  }
  return std::make_unique<Module>(
      std::make_unique<BufferDataLoader>(program_bytes.data(),
                                         program_bytes.size()),
      nullptr, nullptr, nullptr, std::move(data_loader));
}

static std::string describe_device(const std::string &delegate) {
  if (delegate == "cuda") {
    const nn::CudaDeviceInfo &dev = nn::cuda_device_info();
    return "CUDA: " + dev.name + " (sm_" + std::to_string(dev.cc) + ")";
  }
  if (delegate == "xnnpack")
    return "CPU (XNNPACK)";
  if (delegate == "coreml")
    return "Core ML";
  if (delegate == "mlx")
    return "GPU (MLX)";
  if (delegate == "metal")
    return "GPU (Metal)";
  if (delegate == "mps")
    return "GPU (MPS)";
  return "CPU";
}

int Backend::load(std::string path, std::string data_path) {
  nn::LogScope log_scope(&m_log_sink);
  m_data_override = std::move(data_path);
  // Parse the metadata FIRST: without it we know nothing about the model's I/O
  // and cannot even build the host's inlets/attributes. A failure here is the
  // only hard error — the caller has nothing to set up.
  m_loaded = 0;
  m_metadata_loaded = false;
  bool metadata_ok = false;
  try {
    metadata_ok = load_metadata(path);
  } catch (const std::exception &e) { // e.g. a field of the wrong JSON type
    log_error("neural: invalid metadata for " + path + ": " + e.what());
  }
  if (!metadata_ok) {
    log_error("neural: could not load metadata for " + path);
    return 1;
  }
  m_metadata_loaded = true;
  m_path = path;
  m_device_desc.clear();
  m_exec_error_logged.store(false, std::memory_order_relaxed);

  // Every reason the program can't run keeps the parsed metadata (so the host
  // can still create inlets/attributes) but marks the model non-runnable and
  // reports why. is_loaded() stays false.
  auto disable = [&](const std::string &why) {
    log_error("neural: " + why +
              " — metadata loaded, but the model is disabled");
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    m_module.reset();
    m_program_bytes.clear();
    m_data_bytes.clear();
    return 0; // success enough to set up I/O; not runnable
  };

  // Is the program (.pte) actually on disk? Metadata MAY ship without it.
  if (!nn::file_exists(path))
    return disable("program \"" + path +
                   "\" not found (provide the .pte to enable it)");

  const std::string problem = compatibility_problem();
  if (!problem.empty())
    return disable("cannot run \"" + path + "\": " + problem);

  bool data_missing = false;
  const std::string ptd = resolve_data_file(path, data_missing);
  if (data_missing)
    return disable("weights file \"" + ptd +
                   "\" not found (it must sit beside the .pte)");

  if (m_delegate == "cuda")
    nn::sweep_stale_cuda_kernel_files(); // left by earlier sessions (Windows)

  try {
    std::vector<uint8_t> program_bytes, data_bytes;
    auto module = make_module(path, ptd, program_bytes, data_bytes);
    if (!module)
      return disable("could not open \"" + path + "\"");

    // Parses the program: a corrupt or incompatible .pte fails here, not on
    // the first audio block. Also warn if its methods diverge from the JSON.
    auto names = module->method_names();
    if (!names.ok())
      return disable("could not read program \"" + path + "\" (error " +
                     error_code(names.error()) +
                     "; was it exported with a different ExecuTorch version? "
                     "metadata says " +
                     (m_executorch_version.empty() ? "unknown"
                                                   : m_executorch_version) +
                     ")");
    for (const auto &name : m_available_methods)
      if (!names->count(name))
        log(nn::LogLevel::Warning, "neural: method \"" + name +
                                       "\" listed in JSON but absent from .pte");

    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    // The old module goes first, then the buffers it may have read from.
    m_module = std::move(module);
    m_program_bytes = std::move(program_bytes);
    m_data_bytes = std::move(data_bytes);
    m_loaded = 1;
    model_lock.unlock();
  } catch (const std::exception &e) {
    return disable("could not load \"" + path + "\": " + e.what());
  }

  m_device_desc = describe_device(m_delegate);
  if (m_delegate == "cuda" && !m_cuda_archs.empty()) {
    // SASS for sm_XY runs on sm_XZ (Z >= Y); otherwise the driver JIT-compiles
    // the embedded PTX, which works but makes the first load slower.
    const int cc = nn::cuda_device_info().cc;
    bool native = false;
    for (int a : m_cuda_archs)
      native |= (a / 10 == cc / 10) && (a % 10 <= cc % 10);
    if (!native)
      log(nn::LogLevel::Info,
          "neural: no native kernels for sm_" + std::to_string(cc) +
              " in this model; the NVIDIA driver compiles them on first load "
              "(slower first load)");
  }
  return 0;
}

int Backend::reload() { return load(m_path, m_data_override); }

int Backend::prepare(const std::string &method) {
  if (!m_loaded)
    return 1;
  nn::LogScope log_scope(&m_log_sink);
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  const auto err = m_module->load_method(method);
  model_lock.unlock();
  if (err != executorch::runtime::Error::Ok) {
    log_error("neural: could not load method \"" + method + "\" of \"" +
              m_path + "\" (error " + error_code(err) + ")");
    if (m_delegate == "cuda" && err == executorch::runtime::Error::AccessFailed) {
      std::error_code ec;
      auto tmp = std::filesystem::temp_directory_path(ec);
      if (!tmp.has_filename()) // "C:\...\Temp\" on Windows
        tmp = tmp.parent_path();
      log_error("neural: the CUDA runtime extracts the model's kernels to " +
                (ec ? std::string("the temp folder") : nn::utf8_string(tmp)) +
                "; check that the folder is writable and not blocked by "
                "antivirus software");
    }
    return 1;
  }
  return 0;
}

std::string Backend::device_description() { return m_device_desc; }

std::string Backend::delegate() { return m_delegate; }

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
    log_error("neural: generative method \"" + method + "\" not found");
    return 1;
  }
  const MethodInfo &info = it->second;
  if (input_data.size() != info.inputs.size()) {
    log_error("neural: generate(\"" + method + "\") expected " +
              std::to_string(info.inputs.size()) + " inputs, got " +
              std::to_string(input_data.size()));
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

  nn::LogScope log_scope(&m_log_sink);
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  auto result = m_module->execute(method, inputs);
  model_lock.unlock();

  if (!result.ok()) {
    // generate() is user-triggered, so every failure is reported.
    log_error("neural: execute(\"" + method + "\") failed (error " +
              error_code(result.error()) + ")");
    return 1;
  }

  auto output = result->at(0).toTensor();
  const size_t numel = (size_t)output.numel();
  const float *ptr = output.const_data_ptr<float>();
  out_audio.assign(ptr, ptr + numel);
  return 0;
}

void Backend::reset() {
  // Re-initialise a model's persistent internal state. Each loaded method is
  // unloaded and loaded again, which re-instantiates it with freshly
  // initialised mutable buffers (see EXECUTORCH_PROTOCOL.md) while keeping the
  // parsed program and the open weights file. Fallback: a full reload. The
  // caller must quiesce the worker thread first (the frontends' reload/reset
  // messages stop it before calling).
  // Also restart the per-input noise streams so reset replays the same seeded
  // noise deterministically (a reload re-runs load_metadata, which already
  // flags this; set it here too for the in-place path).
  m_reseed_noise.store(true, std::memory_order_release);
  if (!m_loaded)
    return;
  {
    nn::LogScope log_scope(&m_log_sink);
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    bool ok = true;
    for (const auto &name : m_available_methods) {
      if (!m_module->is_method_loaded(name))
        continue; // never run: still in its initial state
      m_module->unload_method(name);
      if (m_module->load_method(name) != executorch::runtime::Error::Ok) {
        ok = false;
        break;
      }
    }
    if (ok)
      return;
  }
  log(nn::LogLevel::Warning,
      "neural: in-place reset failed, reloading \"" + m_path + "\"");
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
    log_block_error("neural: method \"" + method + "\" has no signal input");
    return;
  }

  const int in_dim = sig->channels;
  const int in_ratio = sig->ratio;
  const int out_ratio = info.output.ratio;
  const int t_in = n_vec / in_ratio;
  const int t_out = n_vec / out_ratio;

  if ((int)in_buffer.size() != in_dim * n_batches) {
    log_block_error("neural: bad in_buffer size, expected " +
                    std::to_string(in_dim * n_batches) + " buffers, got " +
                    std::to_string(in_buffer.size()));
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
  // metadata's leading dim is a batch placeholder of 1, replicated to n_batches.
  size_t attr_i = 0, noise_i = 0, cond_i = 0;
  for (const auto &spec : info.inputs) {
    switch (spec.role) {
    case Role::Signal:
      m_inputs.emplace_back(*signal);
      break;
    case Role::Attribute: {
      // Scalar [1] control: the caller's value by declared-order index, else the
      // metadata default (so attribute-less callers like mc/mcs pass nothing).
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
          log_block_error("neural: supplied condition \"" + spec.name +
                          "\" numel/dtype mismatch (expected " +
                          std::to_string(per_batch) + "), zero-filled");
        }
      }
      TensorPtr t = from_blob(buf.data(), sizes, spec.dtype);
      m_input_tensors.push_back(t);
      m_inputs.emplace_back(*t);
      break;
    }
    default:
      log_block_error("neural: input \"" + spec.name +
                      "\" has a role unsupported by neural.live~ (buffer is "
                      "gen-only)");
      return;
    }
  }

  // ExecuTorch's own messages follow the same once-per-load rule as ours.
  static const nn::LogSink drop = [](nn::LogLevel, const std::string &) {};
  nn::LogScope log_scope(
      m_exec_error_logged.load(std::memory_order_relaxed) ? &drop : &m_log_sink);
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  auto result = m_module->execute(method, m_inputs);
  model_lock.unlock();

  if (!result.ok()) {
    log_block_error("neural: execute(\"" + method + "\") failed (error " +
                    error_code(result.error()) + ")");
    return;
  }

  auto output = result->at(0).toTensor();
  const int out_batches = output.size(0);
  const int out_channels = output.size(1);
  const int out_n_vec = output.size(2);

  if (out_batches * out_channels != (int)out_buffer.size()) {
    log_block_error("neural: bad out_buffer size, expected " +
                    std::to_string(out_batches * out_channels) +
                    " buffers, got " + std::to_string(out_buffer.size()));
    return;
  }
  if (out_n_vec != t_out) {
    log_block_error("neural: model output size inconsistent, expected " +
                    std::to_string(t_out) + " samples, got " +
                    std::to_string(out_n_vec));
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
