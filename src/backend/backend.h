#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include "nn_log.h"

// Name printed in the boot banner of the Max externals (replaces TORCH_VERSION).
#ifndef NN_ENGINE_NAME
#define NN_ENGINE_NAME "ExecuTorch"
#endif

// --- Shared input-role mechanism (see PROTOCOL.md) ---------------
// Both model kinds describe a method as an ordered, positional list of typed
// inputs, each tagged with a role, plus a single audio output. The host sizes &
// fills every input by role without hardcoding any model.
//
//   Signal    — (live only) the per-block multi-channel audio read from the
//               signal inlets; carries channels + temporal ratio. EXACTLY ONE
//               per method. The audio tensor is [batch, channels, n_vec/ratio].
//   Condition — externally supplied, matched by name (token ids / masks for gen;
//               a held control vector for live). Zero-filled if unsupplied.
//   Attribute — a scalar with shape [1] control exposed as a Max attribute,
//               clamped to [min,max], defaulting to `def`.
//   Noise     — host-filled N(0,1), keyed by (seed, name). gen re-derives the
//               stream each generate(); live seeds one persistent stream per
//               input at load/reset, then advances it per block (still evolves).
//               A host MAY instead drive a noise input from a Jitter matrix.
//   Buffer    — (gen only) an init waveform read from a Max buffer~ (resampled,
//               or copied 1:1 when the metadata says "resample": false).
enum class Role { Signal, Condition, Attribute, Noise, Buffer };

// One declared input of a method's positional forward signature.
struct InputSpec {
  std::string name;
  Role role;
  std::vector<int> shape; // condition/noise/buffer/attribute tensors
  executorch::aten::ScalarType dtype;
  // Attribute-only: clamp range + default; `description` => Max inspector label.
  double min = 0, max = 0, def = 0;
  std::string description;
  // Buffer-only (gen): target sample rate the host resamples the init clip to
  // (0 => fall back to the output's sample_rate). channels/length come from
  // `shape`.
  int sample_rate = 0;
  // Buffer-only (gen): `resample` false => copy the buffer~ 1:1 at the declared
  // geometry, ignoring its sample rate (control curves, e.g. a 50 Hz F0); true
  // (default) => linear resample as audio. `pad_value` fills past the source end
  // (and an unsupplied buffer input). Both optional; defaults = 0.0.4 behaviour.
  bool resample = true;
  double pad_value = 0.0;
  // Signal-only (live): channel count + temporal compression ratio.
  int channels = 0, ratio = 1;
  // Signal-only (live): per-channel inlet labels (optional; auto-generated when
  // absent).
  std::vector<std::string> labels;
};

// The single audio output of a method.
struct OutputSpec {
  std::string name;
  Role role; // "buffer" (gen) | "signal" (live); cosmetic, never switched on
  std::vector<int> shape;
  executorch::aten::ScalarType dtype;
  // gen: channel-major [1, channels, length] at sample_rate (no layout field).
  int channels = 0, length = 0, sample_rate = 0;
  // live: temporal compression ratio + per-channel outlet labels (optional).
  int ratio = 1;
  std::vector<std::string> labels;
};

// Per-method metadata parsed from the model's JSON, shared by both kinds.
struct MethodInfo {
  std::vector<InputSpec> inputs; // positional call signature
  OutputSpec output;
  int batch = 1; // fixed batch baked into the .pte, or -1 if dynamic
  bool dynamic_time = false;
};

// Pre-computed input handed to the runtime by name (matching an InputSpec.name).
// The consumer copies `numel` elements of `dtype` from `data`; `data` must
// outlive the call (the caller owns it). Shared by the generative runner
// (GenRunner) and the streaming consumer (Backend::perform, for conditions).
struct SuppliedInput {
  executorch::aten::ScalarType dtype;
  const void *data;
  size_t numel;
};

// Backend wraps an ExecuTorch program (.pte) plus its JSON metadata.
// The public API mirrors the previous TorchScript backend so the Max/Pd
// frontends are essentially unchanged. A live method MAY carry internal mutable
// state that persists across calls; the host loads one instance per object and
// reuses it (see EXECUTORCH_PROTOCOL.md).
class Backend {
protected:
  // In-memory program/weights, used only when a path can't be opened by
  // ExecuTorch's narrow-path loaders (non-ASCII on Windows). Declared before
  // m_module so they are destroyed after it.
  std::vector<uint8_t> m_program_bytes, m_data_bytes;
  std::unique_ptr<executorch::extension::Module> m_module;
  int m_loaded;
  // True once the metadata JSON has been parsed (the model's I/O is known),
  // regardless of whether the .pte program is present. A model that ships only
  // its metadata is metadata_loaded but not loaded: the host can build its
  // inlets/attributes, but it cannot run.
  bool m_metadata_loaded{false};
  std::string m_path;
  std::mutex m_model_mutex;

  // Per-method metadata for BOTH kinds, keyed by method name. m_generative
  // selects perform() (live) vs generate() (gen) at runtime.
  std::map<std::string, MethodInfo> m_methods;
  std::vector<std::string> m_available_methods;
  int m_buffer_size; // top-level buffer_size the .pte was exported at (live)

  bool m_generative; // true if the metadata declared kind: "gen"
  int m_seed;        // default RNG seed for gen "noise" inputs

  // Reusable scratch for perform() so the audio/worker hot path does no per-block
  // heap allocation: a live model's shapes are fixed across calls, so these grow
  // once and are then reused (resize()/clear() keep the capacity). Only touched
  // by the single perform() caller (the worker thread); load/reload quiesce it.
  std::vector<float> m_in_scratch;                 // decimated [batch, in_dim, t_in]
  std::vector<float> m_attr_f;                      // float attribute backing
  std::vector<int64_t> m_attr_l;                    // long attribute backing
  std::vector<std::vector<float>> m_noise_bufs;     // per noise-role input
  std::vector<std::vector<char>> m_cond_bufs;       // per condition-role input
  std::vector<executorch::extension::TensorPtr> m_input_tensors;
  std::vector<executorch::runtime::EValue> m_inputs;
  // One persistent N(0,1) stream per noise-role input (declared order), seeded
  // from noise_substream_seed(m_seed, name) and advanced per block, so live
  // seeded noise is reproducible from `seed` yet still evolves. Rebuilt when the
  // noise-input count changes or a reseed is requested (set_seed / reset).
  std::vector<std::mt19937_64> m_noise_streams;
  std::atomic<bool> m_reseed_noise{true};

  // Top-level export metadata (see PROTOCOL.md): which delegate the program was
  // lowered for, the platform its compiled kernels target (AOTI delegates), the
  // external weights file, and the CUDA architectures it carries.
  std::string m_delegate, m_platform, m_data_file, m_executorch_version;
  std::string m_data_override; // load()'s data_path
  int m_cuda_min_arch = 0;
  std::vector<int> m_cuda_archs;
  std::string m_device_desc;

  // Diagnostics sink (null => nn's default sink, i.e. stderr unless the host
  // installed one). An execute() failure is reported once per load, not on
  // every audio block.
  nn::LogSink m_log_sink;
  std::atomic<bool> m_exec_error_logged{false};
  void log(nn::LogLevel level, const std::string &msg) const;
  void log_error(const std::string &msg) const {
    log(nn::LogLevel::Error, msg);
  }
  // For errors raised per audio block in perform(): only the first since the
  // last load() is reported.
  void log_block_error(const std::string &msg);

  // Parse "<basename>.json" beside the .pte into m_methods / m_buffer_size.
  bool load_metadata(const std::string &pte_path);
  // Why this host can't run the model ("" if it can): the delegate is compiled
  // into this build, the platform matches, and for CUDA a capable GPU exists.
  std::string compatibility_problem();
  // The program's external weights file (absolute, UTF-8), "" if it has none.
  // Sets `missing` when the metadata names one that isn't on disk.
  std::string resolve_data_file(const std::string &pte_path, bool &missing);
  // Build the ExecuTorch Module, falling back to in-memory loaders for paths
  // ExecuTorch can't open. Null on failure (already logged).
  std::unique_ptr<executorch::extension::Module>
  make_module(const std::string &pte_path, const std::string &ptd_path,
              std::vector<uint8_t> &program_bytes,
              std::vector<uint8_t> &data_bytes);

public:
  Backend();

  // Route this instance's diagnostics (and ExecuTorch errors raised while it
  // calls into the runtime) to `sink`. Null => the process default (stderr).
  // Set before load(); not synchronised with a running perform().
  void set_log_sink(nn::LogSink sink);

  // Load `method` now (kernel/weights upload for GPU delegates) instead of on
  // the first perform()/generate(), so the cost and any error land here, off
  // the audio path. Returns 0 on success.
  int prepare(const std::string &method);

  // Where the model runs, for banners / neural.info: e.g.
  // "CUDA: NVIDIA GeForce RTX 4070 Laptop GPU (sm_89)" or "CPU (XNNPACK)".
  // Empty before a successful load.
  std::string device_description();
  // The JSON `delegate` ("xnnpack", "cuda", ...; "" if absent).
  std::string delegate();

  // attr_values (optional): current value of each Attribute-role input, in
  // declared order; a value not supplied (shorter list / empty) falls back to
  // the metadata default, so attribute-less callers (mc/mcs) pass nothing.
  // supplied (optional): externally-supplied inputs by name (Condition-role for
  // live); an unsupplied condition is zero-filled.
  void perform(std::vector<float *> in_buffer, std::vector<float *> out_buffer,
               int n_vec, std::string method, int n_batches,
               const std::vector<float> &attr_values = {},
               const std::map<std::string, SuppliedInput> &supplied = {});

  // `data_path` (optional) overrides the external weights file the metadata
  // declares (or "<basename>.ptd" beside the .pte); reload() keeps it.
  int load(std::string path, std::string data_path = "");
  int reload();
  // True only when the .pte program is present and the model can run. Gates
  // execution (perform / generate).
  bool is_loaded();
  // True once the metadata JSON is parsed (the model's I/O is known), even if the
  // .pte program is missing. Gates inlet/attribute setup in the frontends, so a
  // metadata-only model still creates its I/O while staying disabled.
  bool has_metadata();

  std::vector<std::string> get_available_methods();
  std::vector<int> get_method_params(std::string method);
  int get_higher_ratio();

  // Fixed-shape ExecuTorch programs require the host block size to match the
  // size the .pte was exported at. The frontends adopt this.
  int get_buffer_size();

  // --- Generative profile (kind: "gen") ------------------------------------
  bool is_generative();
  // Typed, ordered input/output specs for a method (data-driven; the host sizes
  // & fills each input by role without hardcoding any model). Shared by live and
  // gen, though only gen drives generate().
  const std::vector<InputSpec> &method_inputs(const std::string &method);
  OutputSpec method_output(const std::string &method);
  int gen_seed();
  // Set the base RNG seed for noise inputs (the metadata `seed` is the default).
  // For a live model this reseeds each noise input's persistent stream from
  // (seed, name) on the next perform(), so seeded noise restarts deterministically;
  // gen reads it via gen_seed() per generate(). Safe to call from a control thread.
  void set_seed(int seed);
  // Run a generative method. input_data[i] is a host-owned buffer matching
  // method_inputs(method)[i] (shape x dtype); it must outlive the call. On
  // success (returns 0) out_audio holds the output tensor's floats (channel-major).
  int generate(const std::string &method,
               const std::vector<const void *> &input_data,
               std::vector<float> &out_audio);

  // Reset a model's persistent internal state to its initial value. Re-creates
  // the loaded methods in place (keeping the program and weights file open),
  // falling back to a full reload; callable from a frontend `reset` message.
  void reset();

  // Inlet/outlet names (derived from the signal input / output specs).
  std::vector<std::string> get_input_labels(std::string method);
  std::vector<std::string> get_output_labels(std::string method);

  // The Attribute-role inputs a method declares (the consumer builds one dynamic
  // Max attribute per entry). Empty for methods/models without attributes.
  std::vector<InputSpec> method_attributes(const std::string &method);

  bool has_method(std::string method_name);

  // --- Settable attributes: unsupported in v1 (stubs kept for the frontends) ---
  bool has_settable_attribute(std::string attribute);
  std::vector<std::string> get_settable_attributes();
  std::string get_attribute_as_string(std::string attribute_name);
  void set_attribute(std::string attribute_name,
                     std::vector<std::string> attribute_args);
};
