#pragma once
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

// Name printed in the boot banner of the Max externals (replaces TORCH_VERSION).
#ifndef NN_ENGINE_NAME
#define NN_ENGINE_NAME "ExecuTorch"
#endif

// --- Shared input-role mechanism (see EXECUTORCH_PROTOCOL.md) ---------------
// Both model kinds describe a method as an ordered, positional list of typed
// inputs, each tagged with a role, plus a single audio output. The host sizes &
// fills every input by role without hardcoding any model.
//
//   Signal    — (live only) the per-block multi-channel audio read from the
//               signal inlets; carries channels + temporal ratio. EXACTLY ONE
//               per method. The audio tensor is [batch, channels, n_vec/ratio].
//   Condition — externally supplied, matched by name (token ids / masks for gen;
//               a held control vector for live). Zero-filled if unsupplied.
//   Attribute — a scalar [1] control the consumer exposes as a Max attribute,
//               clamped to [min,max], defaulting to `def`.
//   Noise     — host-filled N(0,1), keyed by (seed, name). gen re-derives the
//               stream each generate(); live seeds one persistent stream per
//               input at load/reset, then advances it per block (still evolves).
//               A host MAY instead drive a noise input from a Jitter matrix.
//   Buffer    — (gen only) an init waveform read from a Max buffer~ (resampled).
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
  // Signal-only (live): channel count + temporal compression ratio.
  int channels = 0, ratio = 1;
  // Signal-only (live): per-channel inlet labels (optional; auto-generated when
  // absent).
  std::vector<std::string> labels;
};

// The single audio output of a method.
struct OutputSpec {
  std::string name;
  Role role; // "audio" (gen) | "signal" (live); cosmetic, never switched on
  std::vector<int> shape;
  executorch::aten::ScalarType dtype;
  // gen: channel-major [1, channels, length] at sample_rate (no layout field).
  int channels = 0, length = 0, sample_rate = 0;
  // live: temporal compression ratio + per-channel outlet labels (optional).
  int ratio = 1;
  std::vector<std::string> labels;
};

// Per-method metadata parsed from the sidecar JSON, shared by both kinds.
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

// Backend wraps an ExecuTorch program (.pte) plus its sidecar JSON metadata.
// The public API mirrors the previous TorchScript backend so the Max/Pd
// frontends are essentially unchanged. A live method MAY carry internal mutable
// state that persists across calls; the host loads one instance per object and
// reuses it (see EXECUTORCH_PROTOCOL.md).
class Backend {
protected:
  std::unique_ptr<executorch::extension::Module> m_module;
  int m_loaded;
  // True once the sidecar JSON has been parsed (the model's I/O is known),
  // regardless of whether the .pte program is present. A model that ships only
  // its sidecar is metadata_loaded but not loaded: the host can build its
  // inlets/attributes, but it cannot run.
  bool m_metadata_loaded{false};
  std::string m_path;
  std::mutex m_model_mutex;

  // Per-method metadata for BOTH kinds, keyed by method name. m_generative
  // selects perform() (live) vs generate() (gen) at runtime.
  std::map<std::string, MethodInfo> m_methods;
  std::vector<std::string> m_available_methods;
  int m_buffer_size; // top-level buffer_size the .pte was exported at (live)

  bool m_generative; // true if the sidecar declared kind: "gen"
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

  // Parse "<basename>.json" beside the .pte into m_methods / m_buffer_size.
  bool load_metadata(const std::string &pte_path);

public:
  Backend();

  // attr_values (optional): current value of each Attribute-role input, in
  // declared order; a value not supplied (shorter list / empty) falls back to
  // the sidecar default, so attribute-less callers (mc/mcs) pass nothing.
  // supplied (optional): externally-supplied inputs by name (Condition-role for
  // live); an unsupplied condition is zero-filled.
  void perform(std::vector<float *> in_buffer, std::vector<float *> out_buffer,
               int n_vec, std::string method, int n_batches,
               const std::vector<float> &attr_values = {},
               const std::map<std::string, SuppliedInput> &supplied = {});

  int load(std::string path);
  int reload();
  // True only when the .pte program is present and the model can run. Gates
  // execution (perform / generate).
  bool is_loaded();
  // True once the sidecar JSON is parsed (the model's I/O is known), even if the
  // .pte program is missing. Gates inlet/attribute setup in the frontends, so a
  // sidecar-only model still creates its I/O while staying disabled.
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
  // Set the base RNG seed for noise inputs (the sidecar `seed` is the default).
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

  // Reset a model's persistent internal state to its zero initial value. Safe
  // fallback reloads the program; callable from a frontend `reset` message.
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
